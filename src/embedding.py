"""Qwen3-Embedding-0.6B wrapper for ExplSim cosine similarity.

Uses last-token pooling + L2 normalization, per the Qwen3-Embedding spec.
Cosine of L2-normalized vectors = dot product, clipped to [0, 1] for safety.
"""

import math

import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _clip_similarity(value: float) -> float:
    if not math.isfinite(value):
        raise FloatingPointError("Embedding similarity is not finite; check the embedding model and dtype")
    return max(0.0, min(1.0, value))


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Last-token pooling honoring left/right padding."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch, device=last_hidden_states.device), sequence_lengths]


class Qwen3Embedder:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 512,
        dtype: str = "bfloat16",
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        torch_dtype = getattr(torch, dtype) if dtype != "float32" else torch.float32
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return torch.empty((0, self.model.config.hidden_size))
        out_chunks: list[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            h = self.model(**batch).last_hidden_state
            emb = _last_token_pool(h, batch["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out_chunks.append(emb.float().cpu())
        return torch.cat(out_chunks, dim=0)

    def cosine(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        e = self.encode([a, b])
        sim = float((e[0] * e[1]).sum().item())
        return _clip_similarity(sim)

    def cosine_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batch-encode N pairs → N cosines. Empties → 0.0."""
        out: list[float] = [0.0] * len(pairs)
        valid_idx: list[int] = []
        texts: list[str] = []
        for i, (a, b) in enumerate(pairs):
            if a and b:
                valid_idx.append(i)
                texts.append(a)
                texts.append(b)
        if not valid_idx:
            return out
        emb = self.encode(texts)
        for k, idx in enumerate(valid_idx):
            ea, eb = emb[2 * k], emb[2 * k + 1]
            sim = float((ea * eb).sum().item())
            out[idx] = _clip_similarity(sim)
        return out
