"""Prepare paired train/validation data for SFT and GRPO.

Adapted from the research project's preparation scripts. Target tags
and type names follow the release's shared schema and canonical prompt.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from src.data import DATA_ROOT, image_path, read_records, sample_id, validate_pairs
from src.parsers.cot_tagged import parse
from src.prompts import build_prompt


def build_target(row):
    teacher = row.get("cot_response") or ""
    match = re.search(r"<think>(.*?)</think>", teacher, re.S | re.I)
    edited = row["image_type"] == "edited"
    verdict = "INCONSISTENT" if edited else "CONSISTENT"
    teacher_final = teacher[match.end():] if match else teacher
    teacher_verdicts = re.findall(r"<verdict>\s*(CONSISTENT|INCONSISTENT)\s*</verdict>",
                                 teacher_final, re.I)
    agrees_with_label = all(value.upper() == verdict for value in teacher_verdicts)
    if match and match.group(1).strip() and agrees_with_label:
        think = match.group(1).strip()
    elif edited:
        think = f"The visible cue is {row['gt_entity_fine']}. {row['gt_why_contradicts']}"
    else:
        think = "No specific visible element contradicts the accompanying claim."
    fields = {
        "think": think,
        "verdict": verdict,
        "type": row["edit_type"] if edited else "None",
        "visual": row["gt_entity_fine"] if edited else "None",
        "explanation": row["gt_why_contradicts"] if edited else "None",
    }
    target = "\n".join(f"<{key}>{value}</{key}>" for key, value in fields.items())
    parsed = parse(target)
    if not parsed["parse_ok"]:
        raise ValueError(f"{sample_id(row)}: invalid training target {parsed['parse_errors']}")
    return target


def sft_record(row, data_root):
    return {
        "messages": [
            {"role": "user", "content": "<image>" + build_prompt(row["claim"])},
            {"role": "assistant", "content": build_target(row)},
        ],
        "images": [str(image_path(data_root, row["local_image"]))],
    }


def grpo_record(row, split, data_root, store_image_bytes=True):
    path = image_path(data_root, row["local_image"])
    edited = row["image_type"] == "edited"
    gt = {
        "verdict": "INCONSISTENT" if edited else "CONSISTENT",
        "edit_type": row["edit_type"] if edited else None,
        "gt_entity_fine": row.get("gt_entity_fine") if edited else None,
        "gt_why_contradicts": row.get("gt_why_contradicts") if edited else None,
    }
    image = {"path": str(path), "max_pixels": 1003520, "min_pixels": 3136}
    if store_image_bytes:
        image["bytes"] = path.read_bytes()
    else:
        # qwen-vl-utils reads `image`; `path` alone is only dataset metadata.
        image["image"] = str(path)
    return {
        "data_source": "mic",
        "prompt": [{"role": "user", "content": "<image>" + build_prompt(row["claim"])}],
        "images": [image],
        "ability": "image_claim_inconsistency_detection",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
        "extra_info": {"sample_id": sample_id(row), "article_id": row["_id"], "split": split},
    }


def prepare(split_root, data_root, output_dir, mode, force=False, store_image_bytes=True):
    split_root, data_root, output_dir = map(Path, (split_root, data_root, output_dir))
    splits = {}
    for split in ("train", "val"):
        splits[split] = read_records(split_root / f"{split}.json", data_root, check_images=True)
        validate_pairs(splits[split])
    if {r["_id"] for r in splits["train"]} & {r["_id"] for r in splits["val"]}:
        raise ValueError("Claim overlap between train and validation")
    source_images = {
        split: {image_path(data_root, row["local_image"]) for row in rows if row["image_type"] == "original"}
        for split, rows in splits.items()
    }
    if source_images["train"] & source_images["val"]:
        raise ValueError("Source image overlap between train and validation; review the split manifests")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise ValueError(f"Output is not empty: {output_dir}; use --force to replace generated files")
    converted = {}
    for split, rows in splits.items():
        converted[split] = [sft_record(r, data_root) if mode == "sft" else
                            grpo_record(r, split, data_root, store_image_bytes) for r in rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in converted.items():
        if mode == "sft":
            (output_dir / f"{split}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        else:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.Table.from_pylist(rows), output_dir / f"{split}.parquet")
    if mode == "sft":
        info = {f"mic_{split}": {
            "file_name": f"{split}.json", "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        } for split in splits}
        (output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    manifest = {"format": mode, "prompt": "canonical", "splits": {
        split: {"rows": len(rows), "source_sha256": hashlib.sha256((split_root / f"{split}.json").read_bytes()).hexdigest()}
        for split, rows in splits.items()}, "test_data_read": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", choices=["sft", "grpo"], required=True)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--split-root", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--force", action="store_true")
    p.add_argument("--paths-only", action="store_true", help="GRPO only: omit image bytes")
    args = p.parse_args()
    print(json.dumps(prepare(args.split_root or args.data_root / "splits", args.data_root,
                             args.output_dir or Path("training/data") / args.format, args.format,
                             args.force, not args.paths_only), indent=2))


if __name__ == "__main__":
    main()
