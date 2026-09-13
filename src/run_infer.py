"""Stage 1 — inference CLI.

Reads a split, runs one (model, prompt) on every sample, writes predictions JSONL.
No metrics, no scoring. Score layer (Stage 2) is separate.

Usage:
    python -m src.run_infer \\
        --model Qwen/Qwen3-VL-4B-Instruct \\
        --backend vllm \\
        --split test_id_edit \\
        --prompt canonical \\
        --run-name Qwen3-VL-4B__test_id_edit__canonical \\
        --limit 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_split
from src.prompts import build_prompt, PROMPT_VARIANTS
from src.parsers.cot_tagged import parse
from src.records import read_predictions
from src.schema import Prediction


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "predictions"


def _load_existing(out_path: Path, expected=None, samples=None, inference_config=None) -> set[str]:
    """Resume only complete rows belonging to the requested run and samples."""
    if not out_path.exists():
        return set()
    rows = read_predictions(out_path)
    by_id = {sample.sample_id: sample for sample in samples} if samples is not None else None
    for row in rows:
        if expected and any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing predictions use different run/model/split/prompt metadata; use a new --run-name")
        if inference_config is not None and row.get("inference_config") != inference_config:
            raise ValueError("Existing predictions have missing or different inference settings; use a new --run-name")
        if by_id is not None and row["sample_id"] not in by_id:
            raise ValueError("Existing predictions contain IDs outside the selected samples; use a new --run-name")
    return {row["sample_id"] for row in rows}


def _shard_samples(samples, shard_id: int, num_shards: int):
    if num_shards <= 1:
        return samples
    return [s for i, s in enumerate(samples) if i % num_shards == shard_id]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id / local ckpt / API model name")
    p.add_argument("--model-display-name", default=None,
                   help="model value written to predictions; useful when loading from a local snapshot")
    p.add_argument("--backend", default="vllm", choices=["vllm", "api"])
    p.add_argument("--provider", default=None, choices=["openai"],
                   help="Required when --backend api")
    p.add_argument("--api-workers", type=int, default=8,
                   help="Concurrency for --backend api (ignored for vllm)")
    p.add_argument("--split", required=True, help="test_id_edit / test_ood_edit / train / val")
    p.add_argument("--prompt", default="canonical", choices=list(PROMPT_VARIANTS))
    p.add_argument("--run-name", required=True, help="filename stem for predictions JSONL")
    p.add_argument("--generator", default=None,
                   help="Override generator metadata; otherwise use each source record")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--edited-only", action="store_true",
                   help="Run only edited samples from the split. Useful for edited-only tables.")
    p.add_argument("--balanced", type=int, default=None,
                   help="Sample N/2 original + N/2 edited (paired order). Overrides --limit.")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--collect-verdict-logprobs", action="store_true",
                   help="Capture top-20 logprobs and emit a verdict_score field (vllm backend only)")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--split-root", type=Path)
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        p.error("Require num-shards >= 1 and 0 <= shard-id < num-shards")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    if not args.run_name.strip() or args.run_name in {".", ".."} or Path(args.run_name).name != args.run_name:
        p.error("--run-name must be a filename stem")
    if args.backend == "api" and not args.provider:
        p.error("--backend api requires --provider openai")
    if args.api_workers < 1:
        p.error("--api-workers must be positive")
    if args.backend == "api" and (args.adapter_path or args.collect_verdict_logprobs):
        p.error("--adapter-path and --collect-verdict-logprobs require --backend vllm")
    if args.max_model_len < 1 or args.tensor_parallel_size < 1 or not 0 < args.gpu_memory_utilization <= 1:
        p.error("Model length and tensor parallel size must be positive; GPU memory utilization must be in (0, 1]")

    if args.edited_only and args.balanced is not None:
        raise SystemExit("--edited-only cannot be combined with --balanced")

    samples = load_split(args.split, args.data_root, args.split_root)
    if args.edited_only:
        samples = [s for s in samples if s.is_edited]
    if args.balanced is not None:
        if args.balanced < 2 or args.balanced % 2:
            p.error("--balanced must be a positive even number")
        pairs = {}
        for sample in samples:
            pairs.setdefault(sample.article_id, {})[sample.is_edited] = sample
        complete = [pair for pair in pairs.values() if set(pair) == {False, True}]
        if len(complete) < args.balanced // 2:
            p.error("Not enough complete pairs for --balanced")
        samples = [sample for pair in complete[:args.balanced // 2] for sample in (pair[False], pair[True])]
    elif args.limit is not None:
        samples = samples[: args.limit]
    samples = _shard_samples(samples, args.shard_id, args.num_shards)
    print(f"[run_infer] split={args.split} samples={len(samples)} "
          f"shard={args.shard_id}/{args.num_shards}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_shards > 1:
        out_path = args.output_dir / f"{args.run_name}.shard{args.shard_id}of{args.num_shards}.jsonl"
    else:
        out_path = args.output_dir / f"{args.run_name}.jsonl"

    done_ids: set[str] = set()
    inference_config = {
        "model": args.model, "backend": args.backend, "provider": args.provider,
        "adapter_path": str(Path(args.adapter_path).resolve()) if args.adapter_path else None,
        "max_model_len": args.max_model_len,
        "collect_verdict_logprobs": args.collect_verdict_logprobs,
        "generator_override": args.generator,
    }
    if args.skip_existing:
        expected = {"run_id": args.run_name, "model": args.model_display_name or args.model,
                    "split": args.split, "prompt_variant": args.prompt}
        done_ids = _load_existing(out_path, expected, samples, inference_config)
        print(f"[run_infer] existing predictions: {len(done_ids)}  (will skip)")

    remaining = [s for s in samples if s.sample_id not in done_ids]
    if not remaining:
        print("[run_infer] nothing to do, exiting")
        return
    missing_images = [s.image_path for s in remaining if not Path(s.image_path).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} input images, including {missing_images[0]}")

    # Load backend
    if args.backend == "vllm":
        from src.backends.vllm_backend import VLLMBackend
        backend = VLLMBackend(
            model_name=args.model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            adapter_path=args.adapter_path,
            return_logprobs=args.collect_verdict_logprobs,
        )
        backend.load()
    elif args.backend == "api":
        if not args.provider:
            raise SystemExit("--backend api requires --provider openai")
        from src.backends.api_backend import APIBackend
        backend = APIBackend(provider=args.provider, api_model=args.model)
        backend.load()
    else:
        raise NotImplementedError(args.backend)

    def _process_one(s):
        """Infer one sample, return Prediction dict ready to write."""
        prompt = build_prompt(s.claim, variant=args.prompt)
        result = backend.run(s.image_path, prompt)
        verdict_score = None
        if isinstance(result, tuple):
            raw, verdict_score = result
        else:
            raw = result
        parsed = parse(raw)
        d = Prediction(
            sample_id=s.sample_id,
            run_id=args.run_name,
            model=args.model_display_name or args.model,
            split=args.split,
            prompt_variant=args.prompt,
            generator=args.generator or s.generator,
            raw_text=raw,
            pred_verdict=parsed["verdict"],
            pred_type=parsed["type"],
            pred_visual=parsed["visual"],
            pred_explanation=parsed["explanation"],
            pred_think=parsed["think"],
            parse_ok=parsed["parse_ok"],
            parse_errors=parsed["parse_errors"],
        ).to_dict()
        d["inference_config"] = inference_config
        if verdict_score is not None:
            d["verdict_score"] = verdict_score
        return d

    t0 = time.time()
    failed = 0
    needs_newline = False
    if args.skip_existing and out_path.exists() and out_path.stat().st_size:
        with open(out_path, "rb") as existing:
            existing.seek(-1, 2)
            needs_newline = existing.read(1) != b"\n"
    with open(out_path, "a" if args.skip_existing else "w", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        if args.backend == "api" and args.api_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            written = 0
            with ThreadPoolExecutor(max_workers=args.api_workers) as pool:
                futs = {pool.submit(_process_one, s): s for s in remaining}
                for fut in as_completed(futs):
                    s = futs[fut]
                    try:
                        d = fut.result()
                    except Exception as e:
                        failed += 1
                        print(f"[err] {s.sample_id}: {type(e).__name__}: {e}")
                        continue
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
                    f.flush()
                    written += 1
                    if written % 20 == 0 or written == len(remaining):
                        rate = written / (time.time() - t0)
                        print(f"  [{written}/{len(remaining)}]  {rate:.2f} samp/s  "
                              f"last={s.sample_id} verdict={d.get('pred_verdict')}")
        else:
            for i, s in enumerate(remaining):
                try:
                    d = _process_one(s)
                except Exception as e:
                    failed += 1
                    print(f"[err] {s.sample_id}: {type(e).__name__}: {e}")
                    continue
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
                    rate = (i + 1) / (time.time() - t0)
                    print(f"  [{i + 1}/{len(remaining)}]  {rate:.2f} samp/s  "
                          f"last={s.sample_id} verdict={d.get('pred_verdict')}")

    backend.unload()
    print(f"[run_infer] done → {out_path}")
    if failed:
        raise SystemExit(f"{failed} samples failed; rerun the same command to resume")


if __name__ == "__main__":
    main()
