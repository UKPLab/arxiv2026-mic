"""Recreate published image pairs without rerunning prompt construction.

All images and run records go into a separate output root. Published reviews
describe the historical images; regenerated counterparts still need review.
"""

import argparse
import base64
import hashlib
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from data_construction.common import (
    create_client, nonnegative_int, positive_int, read_json, write_bytes, write_json,
)
from data_construction.images import valid_image
from src import REPO_ROOT
from src.data import image_path, normalize_record, validate_pairs

SPLITS = ("train", "val", "test_id", "test_ood")
MODELS = {"gpt-image": "gpt-image-1.5", "flux2": "black-forest-labs/FLUX.2-klein-9B"}


def letterbox_png(source):
    """Match the released 768 x 512 RGB PNG geometry, with black padding."""
    with Image.open(source) as image:
        image = image.convert("RGB")
        scale = min(768 / image.width, 512 / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        resized = image.resize(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (768, 512), (0, 0, 0))
        canvas.paste(resized, ((768 - size[0]) // 2, (512 - size[1]) // 2))
        output = io.BytesIO()
        canvas.save(output, format="PNG", compress_level=6)
        return output.getvalue()


def load_editing_prompts(path, output_root, backend):
    """Read generation inputs alongside the original editing proposal details."""
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("The editing prompts file must be a nonempty JSON array")
    seen, outputs = set(), set()
    selected = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each editing prompt record must be a JSON object")
        proposal = row.get("edit_result")
        if (not isinstance(proposal, dict)
                or not isinstance(proposal.get("editing_prompt"), str)
                or not proposal["editing_prompt"].strip()):
            raise ValueError("Editing prompt record is missing edit_result.editing_prompt")
        row = dict(row, editing_prompt=proposal["editing_prompt"])
        for key in ("_id", "split", "source_url", "source_image", "original_image", "edited_image", "editing_prompt"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Editing prompt record is missing {key}")
        if row["_id"] in seen or row["split"] not in SPLITS:
            raise ValueError(f"Duplicate ID or unknown split: {row['_id']}")
        seen.add(row["_id"])
        if backend == "flux2" and not row.get("flux2_edited_image"):
            continue
        if backend == "flux2":
            if not isinstance(row.get("flux2_editing_prompt"), str) or not row["flux2_editing_prompt"].strip():
                raise ValueError(f"Missing FLUX.2 editing prompt: {row['_id']}")
            row["edited_image"] = row["flux2_edited_image"]
            row["editing_prompt"] = row["flux2_editing_prompt"]
        for key in ("source_image", "original_image", "edited_image"):
            if Path(row[key]).is_absolute():
                raise ValueError(f"Editing prompt image paths must be relative: {row[key]}")
            image_path(output_root, row[key])
        if row["edited_image"] in outputs:
            raise ValueError(f"Duplicate edited output: {row['edited_image']}")
        if len({row[key] for key in ("source_image", "original_image", "edited_image")}) != 3:
            raise ValueError(f"Source and output image paths must differ: {row['_id']}")
        outputs.add(row["edited_image"])
        selected.append(row)
    return selected


def ensure_source(row, output_root, source_image_dir=None, *, require_raw=True, refresh_original=False):
    """Download exact source filenames, retaining the raw input used by GPT Image."""
    import requests

    source = image_path(output_root, row["source_image"])
    original = image_path(output_root, row["original_image"])
    if not require_raw and not refresh_original and original.is_file() and valid_image(original):
        return source, original
    if not source.is_file() or not valid_image(source):
        local = Path(source_image_dir) / source.name if source_image_dir else None
        if local is not None and local.is_file() and valid_image(local):
            write_bytes(source, local.read_bytes())
        else:
            urls = list(dict.fromkeys(filter(None, [row["source_url"], row.get("source_fallback_url")])))
            failures = []
            for url in urls:
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    if not valid_image(io.BytesIO(response.content)):
                        raise ValueError("Response is not a readable image")
                    write_bytes(source, response.content)
                    break
                except (requests.RequestException, OSError, ValueError) as error:
                    failures.append(str(error))
            else:
                raise RuntimeError("Source download failed: " + "; ".join(failures))
    if refresh_original or not original.is_file() or not valid_image(original):
        write_bytes(original, letterbox_png(source))
    return source, original


class GPTImageEditor:
    def __init__(self, args):
        self.client = create_client()
        self.model, self.quality = args.model, args.quality

    def generate(self, row, source, original):
        with source.open("rb") as image:
            response = self.client.images.edit(
                model=self.model, image=image, prompt=row["editing_prompt"],
                quality=self.quality, size="auto",
            )
        content = base64.b64decode(response.data[0].b64_json, validate=True)
        return letterbox_png(io.BytesIO(content))

    def close(self):
        self.client.close()


class Flux2Editor:
    def __init__(self, args):
        import torch
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as error:
            raise RuntimeError("Run 'pip install -r requirements.txt' to use the FLUX.2 backend") from error

        if not torch.cuda.is_available():
            raise RuntimeError("FLUX.2 reproduction requires a CUDA GPU")
        self.torch, self.seed, self.steps = torch, args.seed, args.num_steps
        self.pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        if args.cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to("cuda")

    def generate(self, row, source, original):
        with Image.open(original) as image:
            reference = image.convert("RGB")
        result = self.pipe(
            prompt=row["editing_prompt"], image=[reference], height=reference.height,
            width=reference.width, num_inference_steps=self.steps,
            generator=self.torch.Generator(device="cuda").manual_seed(self.seed),
        ).images[0]
        output = io.BytesIO()
        result.save(output, format="PNG")
        return letterbox_png(io.BytesIO(output.getvalue()))

    def close(self):
        del self.pipe


def pair_complete(row, output_root):
    return all((path := image_path(output_root, row[key])).is_file() and valid_image(path)
               for key in ("original_image", "edited_image"))


def load_split_records(data_root, tasks):
    """Validate inherited labels and task membership before generating images."""
    splits, memberships = {}, {}
    for split in SPLITS:
        path = data_root / "splits" / f"{split}.json"
        records = read_json(path)
        if not isinstance(records, list):
            raise ValueError(f"Split records must be a JSON array: {path}")
        for record in records:
            normalize_record(record)
            image_path(data_root, record["local_image"])
        pairs = validate_pairs(records)
        for claim_id, pair in pairs.items():
            if claim_id in memberships:
                raise ValueError(f"Claim appears in multiple splits: {claim_id}")
            memberships[claim_id] = (split, pair)
        splits[split] = records
    for task in tasks:
        split, pair = memberships.get(task["_id"], (None, None))
        if split != task["split"]:
            raise ValueError(f"Editing prompt has no matching pair in {task['split']}: {task['_id']}")
        if pair["original"]["local_image"] != task["original_image"]:
            raise ValueError(f"Editing prompt and split reference different originals: {task['_id']}")
        claim = task.get("claim", task.get("caption"))
        if claim is not None and claim != pair["original"]["claim"]:
            raise ValueError(f"Editing prompt and split have different claims: {task['_id']}")
    return splits


def export_splits(split_records, output_root, rows, model):
    """Export complete regenerated pairs, explicitly retaining labels as provisional."""
    complete = {row["_id"]: row for row in rows if pair_complete(row, output_root)}
    count = 0
    for split in SPLITS:
        records = []
        for record in split_records[split]:
            task = complete.get(record["_id"])
            if task is None or task["split"] != split:
                continue
            edited = record["image_type"] == "edited"
            records.append(dict(
                record, local_image=task["edited_image" if edited else "original_image"],
                generator=model if edited else "none", review_required=edited,
                reproduction=True, annotation_provenance="inherited_from_released_image",
            ))
        write_json(output_root / "splits" / f"{split}.json", records)
        count += len(records) // 2
    return count


def reproduce_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data", help="Directory containing released text metadata")
    parser.add_argument("--output-root", type=Path, help="Separate output directory (default: DATA/reproduced/BACKEND)")
    parser.add_argument("--backend", choices=MODELS, default="gpt-image")
    parser.add_argument("--split", choices=("all", *SPLITS), default="all")
    parser.add_argument("--limit", type=nonnegative_int, help="Maximum pending pairs; 0 makes no requests")
    parser.add_argument("--dry-run", action="store_true", help="Show pending counts without downloads or model calls")
    parser.add_argument("--download-only", action="store_true", help="Fetch and normalize source images without generating edits")
    parser.add_argument("--source-image-dir", type=Path, help="Reuse raw source images from this directory by filename")
    parser.add_argument("--max-workers", type=positive_int, default=4, help="GPT/download workers; FLUX generation uses one worker")
    parser.add_argument("--model", help="Override the original generation model; use a new output root")
    parser.add_argument("--quality", choices=("low", "medium", "high", "auto"), default="medium")
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument("--num-steps", type=positive_int, default=28)
    parser.add_argument("--cpu-offload", action="store_true", help="Offload the FLUX model to CPU between GPU operations")
    args = parser.parse_args(argv)
    args.model = args.model or MODELS[args.backend]
    args.data_root = args.data_root.resolve()
    output = (args.output_root or args.data_root / "reproduced" / args.backend).resolve()
    if args.data_root.is_relative_to(output):
        parser.error("--output-root must be separate from the released data root")
    prompts_path = args.data_root / "editing_prompts.json"
    rows = load_editing_prompts(prompts_path, output, args.backend)
    selected = [row for row in rows if args.split == "all" or row["split"] == args.split]
    if not selected:
        parser.error("No released editing tasks match this backend and split")
    split_records = load_split_records(args.data_root, rows)

    profile = {
        "backend": args.backend, "model": args.model,
        "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "normalization": "768x512 RGB PNG, black letterbox, LANCZOS",
        "generation": ({"quality": args.quality, "size": "auto"} if args.backend == "gpt-image"
                       else {"seed": args.seed, "num_inference_steps": args.num_steps}),
        "review_required": True,
    }
    profile_path = output / "generation_config.json"
    if profile_path.exists():
        if read_json(profile_path) != profile:
            parser.error("Output contains a different model, settings, or prompts file; choose a new --output-root")
    elif any(image_path(output, row[key]).exists() for row in rows for key in ("original_image", "edited_image")):
        parser.error("Existing images have no reproduction provenance; choose a new --output-root")

    def completed(row):
        if args.download_only:
            return all((path := image_path(output, row[key])).is_file() and valid_image(path)
                       for key in ("source_image", "original_image"))
        return pair_complete(row, output)

    pending = [row for row in selected if not completed(row)]
    tasks = pending[:args.limit]
    print(f"{args.backend}: {len(selected)} selected, {len(selected) - len(pending)} cached, "
          f"{len(tasks)} pending this run; output: {output}")
    if args.dry_run or args.limit == 0:
        return
    write_json(profile_path, profile)
    checkpoint = output / "generation_results.json"
    results = {row["_id"]: row for row in read_json(checkpoint)} if checkpoint.exists() else {}
    editor = None
    needs_edits = any(not (path := image_path(output, row["edited_image"])).is_file()
                      or not valid_image(path) for row in tasks)
    if needs_edits and not args.download_only:
        editor = GPTImageEditor(args) if args.backend == "gpt-image" else Flux2Editor(args)

    def process(row):
        result = dict(row, generator=args.model, review_required=True)
        try:
            destination = image_path(output, row["edited_image"])
            needs_edit = not args.download_only and (not destination.is_file() or not valid_image(destination))
            source, original = ensure_source(
                row, output, args.source_image_dir,
                require_raw=args.download_only or (needs_edit and args.backend == "gpt-image"),
                refresh_original=needs_edit and args.backend == "gpt-image",
            )
            if not args.download_only:
                if needs_edit:
                    content = editor.generate(row, source, original)
                    if not valid_image(io.BytesIO(content)):
                        raise ValueError("Generation returned invalid image bytes")
                    write_bytes(destination, content)
                result["image_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
            result["status"] = "downloaded" if args.download_only else "generated"
        except Exception as error:
            result.update(status="error", error=str(error))
        return result

    def save():
        write_json(checkpoint, [results[key] for key in sorted(results)])

    futures = {}
    errors = 0
    try:
        workers = 1 if args.backend == "flux2" and not args.download_only else args.max_workers
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process, row): row for row in tasks}
            try:
                for done, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    results[result["_id"]] = result
                    errors += result["status"] == "error"
                    if result["status"] == "error":
                        print(f"  {result['_id']}: {result['error']}")
                    if done % 25 == 0 or done == len(tasks):
                        save()
                        print(f"[{done}/{len(tasks)}] errors={errors}")
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    finally:
        # Executor shutdown waits for active work: keep completed paid requests.
        for future in futures:
            if future.done() and not future.cancelled() and future.exception() is None:
                result = future.result()
                results[result["_id"]] = result
        save()
        if editor is not None:
            editor.close()
        pairs = export_splits(split_records, output, rows, args.model)
        print(f"Saved {pairs} complete pairs. Regenerated edits require review before using inherited labels.")
    if errors:
        raise SystemExit(1)
