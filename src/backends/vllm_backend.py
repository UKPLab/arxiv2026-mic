"""vLLM backend — shared entry point for all open-source VLMs + Ours.

ROCm MIOpen cache environment defaults are set before importing vLLM.
"""

import os

os.environ.setdefault("MIOPEN_USER_DB_PATH", os.path.expanduser("~/.cache/miopen_user_db"))
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", os.path.expanduser("~/.cache/miopen_cache"))

import base64
import json
import math
import re
from pathlib import Path
from typing import get_args


def _image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_media_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}.get(ext, "jpeg")


def extract_verdict_inconsistent_prob(output_obj, tokenizer) -> dict | None:
    """Approximate softmax(INCONSISTENT vs CONSISTENT) at the label token.

    `output_obj` is a single vllm CompletionOutput. Requires SamplingParams to
    have been called with `logprobs >= 5`. Returns None if the verdict tag
    cannot be located, otherwise a dict with diagnostic fields. Top-k token
    alternatives do not give calibrated full-label sequence probabilities.
    """
    text = output_obj.text or ""
    token_ids = list(output_obj.token_ids or [])
    step_logprobs = list(output_obj.logprobs or [])
    if not token_ids or not step_logprobs:
        return None

    # Ignore verdict tags quoted inside reasoning and locate the label itself;
    # whitespace after the opening tag can occupy a separate token.
    final_start = text.lower().rfind("</think>")
    final_start = final_start + len("</think>") if final_start >= 0 else 0
    m = re.search(r"<verdict>\s*(CONSISTENT|INCONSISTENT)\b", text[final_start:], re.IGNORECASE)
    if m is None:
        return None
    target_char = final_start + m.start(1)

    # Decode prefixes together: byte-level tokenizers can split a Unicode
    # character across tokens, so joining individually decoded tokens changes
    # character offsets. Binary search avoids decoding every growing prefix.
    low, high = 0, len(token_ids)
    while low < high:
        middle = (low + high) // 2
        if len(tokenizer.decode(token_ids[:middle + 1])) > target_char:
            high = middle
        else:
            low = middle + 1
    target_step = low
    if target_step >= len(step_logprobs) or target_step >= len(token_ids):
        return None

    step = step_logprobs[target_step]
    if not step:
        return None

    # Accept label prefixes; unrelated alternatives such as "Image" or
    # "Clearly" must not contribute to a verdict probability.
    cons_lp = -math.inf
    incon_lp = -math.inf
    cons_choice = None
    incon_choice = None
    for tid, lp in step.items():
        try:
            txt = tokenizer.decode([tid])
        except Exception:
            continue
        stripped = txt.strip().upper()
        if not stripped:
            continue
        lpv = float(getattr(lp, "logprob", lp))
        if "INCONSISTENT".startswith(stripped):
            if lpv > incon_lp:
                incon_lp = lpv
                incon_choice = (tid, txt, lpv)
        elif "CONSISTENT".startswith(stripped):
            if lpv > cons_lp:
                cons_lp = lpv
                cons_choice = (tid, txt, lpv)

    if cons_lp == -math.inf and incon_lp == -math.inf:
        return None
    # If only one side was found in the top-k, treat the missing side as the
    # k-th rank logprob (a conservative lower bound).
    fallback = min((float(getattr(lp, "logprob", lp)) for lp in step.values()), default=-50.0)
    if cons_lp == -math.inf:
        cons_lp = fallback
    if incon_lp == -math.inf:
        incon_lp = fallback

    max_lp = max(cons_lp, incon_lp)
    e_c = math.exp(cons_lp - max_lp)
    e_i = math.exp(incon_lp - max_lp)
    p_incon = e_i / (e_c + e_i)
    return {
        "p_inconsistent": float(p_incon),
        "method": "top_k_label_prefix_approximation",
        "logp_consistent": float(cons_lp),
        "logp_inconsistent": float(incon_lp),
        "verdict_token_step": int(target_step),
        "verdict_token_text": tokenizer.decode([token_ids[target_step]]),
    }


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks injected by thinking models outside our schema.

    Our own <think> tag is part of the output schema, so we only strip blocks
    that look like the model's own internal chain-of-thought wrapper, which
    tends to appear before the real <verdict>. Conservative: only strips if
    a </think> precedes the first <verdict>.
    """
    # Only strip a prefix <think>...</think> that comes before any <verdict>
    m = re.search(r"<verdict>", text, flags=re.IGNORECASE)
    if not m:
        return text
    head = text[: m.start()]
    if head.count("</think>") > 1:
        # collapse all but the last </think> — keep the final thinking block
        # so downstream parser can pick it up as our schema's think field.
        parts = head.rsplit("</think>", 1)
        cleaned_head = parts[1] if len(parts) == 2 else head
        return cleaned_head + text[m.start():]
    return text


class VLLMBackend:
    def __init__(
        self,
        model_name: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.6,
        tensor_parallel_size: int = 1,
        adapter_path: str | None = None,
        return_logprobs: bool = False,
    ):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.adapter_path = adapter_path
        self.return_logprobs = return_logprobs
        self.llm = None
        self._tokenizer = None

    def load(self):
        os.makedirs(os.environ["MIOPEN_USER_DB_PATH"], exist_ok=True)
        os.makedirs(os.environ["MIOPEN_CUSTOM_CACHE_DIR"], exist_ok=True)
        from vllm import LLM

        mn = self.model_name.lower()
        extra = {}
        if "glm-4.1v" in mn or "glm-4v" in mn:
            extra = dict(
                enforce_eager=True,
                max_num_seqs=2,
                limit_mm_per_prompt={"image": 1},
                mm_processor_kwargs={
                    "size": {"shortest_edge": 12544, "longest_edge": 47040000},
                    "fps": 1,
                },
            )
        if self.adapter_path:
            extra["enable_lora"] = True
            config = json.loads((Path(self.adapter_path) / "adapter_config.json").read_text(encoding="utf-8"))
            ranks = [config["r"], *config.get("rank_pattern", {}).values()]
            if any(type(rank) is not int or rank < 1 for rank in ranks):
                raise ValueError("Adapter ranks must be positive integers")
            try:
                from vllm.config.lora import MaxLoRARanks
                allowed_ranks = sorted(get_args(MaxLoRARanks))
            except ImportError:
                allowed_ranks = [1, 8, 16, 32, 64, 128, 256, 320, 512]
            rank = next((value for value in allowed_ranks if value >= max(ranks)), None)
            if rank is None:
                raise ValueError(f"Adapter rank {max(ranks)} exceeds vLLM's supported ranks")
            extra["max_lora_rank"] = rank

        print(f"[vllm] loading {self.model_name} tp={self.tensor_parallel_size}")
        self.llm = LLM(
            model=self.model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            **extra,
        )
        print("[vllm] loaded")
        if self.return_logprobs:
            try:
                self._tokenizer = self.llm.get_tokenizer()
            except Exception:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            print(f"[vllm] tokenizer ready for logprob extraction")

    def unload(self):
        import gc
        del self.llm
        self.llm = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def run(self, image_path: str, prompt: str):
        """Generate one (image, prompt) -> text. If `return_logprobs` was
        set at backend construction, also returns a verdict-score dict.

        Return value:
          - text (str) when return_logprobs is False
          - (text, verdict_score_dict_or_None) when return_logprobs is True
        """
        from vllm import SamplingParams

        mn = self.model_name.lower()
        is_thinking = "thinking" in mn
        sp_extra = {}
        if self.return_logprobs:
            # Top-20 alternatives at each step — enough to catch both
            # CONSISTENT and INCONSISTENT in the verdict-label position.
            sp_extra["logprobs"] = 20
        if is_thinking:
            sp = SamplingParams(
                max_tokens=32768,
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                presence_penalty=1.5,
                **sp_extra,
            )
        else:
            # repetition_penalty=1.05 is required for Qwen3-VL-8B / InternVL3-8B
            # to avoid degenerate loops at temp=0; Qwen3-VL-4B (MIC base),
            # Qwen3-VL-4B-Instruct, Qwen3-VL-*-Thinking, and LLaVA-OneVision-7B
            # do not need it. When collecting logprobs, drop rep_penalty so that
            # the model's chosen token equals the argmax of the reported logprobs
            # (otherwise rep_penalty silently reranks and breaks the verdict-score).
            needs_rep_penalty = ("qwen3-vl-8b" in mn or "internvl3-8b" in mn) and not is_thinking
            rp = 1.05 if (needs_rep_penalty and not self.return_logprobs) else 1.0
            sp = SamplingParams(
                max_tokens=4096,
                temperature=0,
                repetition_penalty=rp,
                **sp_extra,
            )

        media = _image_media_type(image_path)
        b64 = _image_to_base64(image_path)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/{media};base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}]

        chat_kwargs = dict(messages=messages, sampling_params=sp)
        if self.adapter_path:
            from vllm.lora.request import LoRARequest
            chat_kwargs["lora_request"] = LoRARequest("adapter", 1, self.adapter_path)

        outputs = self.llm.chat(**chat_kwargs)
        out0 = outputs[0].outputs[0]
        text = out0.text

        if "glm-4.1v" in mn:
            text = _strip_thinking(text)

        if not self.return_logprobs:
            return text

        score = None
        if self._tokenizer is not None:
            try:
                score = extract_verdict_inconsistent_prob(out0, self._tokenizer)
            except Exception as e:
                score = {"error": f"{type(e).__name__}: {e}"}
        return text, score
