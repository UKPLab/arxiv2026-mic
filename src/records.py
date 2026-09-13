"""Validate benchmark records and normalize legacy construction labels."""

import json
from pathlib import Path

from .schema import INCONSISTENCY_TYPES

LEGACY_TYPES = {
    "social_behavior": "gesture",
    "text_language": "signage",
    "ads_anachronism": "branding",
    "environmental": "environment",
}


def normalize_type(value):
    if value is None or str(value).strip().lower() in {"none", "null", ""}:
        return None
    value = str(value).strip().lower()
    return LEGACY_TYPES.get(value, value)


def sample_id(item):
    return f"{item['_id'].rsplit('/', 1)[-1]}__{item['image_type']}"


def image_path(data_root, local_image):
    root = Path(data_root).resolve()
    path = (root / local_image).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path must be inside the data root: {local_image}")
    return path


def normalize_record(item):
    if not isinstance(item, dict):
        raise ValueError("Each benchmark record must be a JSON object")
    row = dict(item)
    for key in ("_id", "image_type", "local_image", "claim"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Missing or empty record field: {key}")
    if row["image_type"] not in {"original", "edited"}:
        raise ValueError(f"Invalid image_type: {row['image_type']}")
    row["edit_type"] = normalize_type(row.get("edit_type"))
    if row["image_type"] == "edited":
        if row["edit_type"] not in INCONSISTENCY_TYPES:
            raise ValueError(f"{row['_id']}: unknown edit_type {row['edit_type']}")
        for key in ("gt_entity_fine", "gt_why_contradicts"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{row['_id']}: missing {key}")
    else:
        for key in ("gt_entity_fine", "gt_why_contradicts"):
            row[key] = None
    return row


def read_records(path, data_root=None, check_images=False):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected a non-empty JSON array: {path}")
    rows = [normalize_record(row) for row in raw]
    seen = set()
    for row in rows:
        sid = sample_id(row)
        if sid in seen:
            raise ValueError(f"Duplicate sample ID: {sid}")
        seen.add(sid)
        if data_root is not None:
            image = image_path(data_root, row["local_image"])
            if check_images and not image.is_file():
                raise FileNotFoundError(image)
    return rows


def validate_pairs(rows):
    pairs = {}
    for row in rows:
        pair = pairs.setdefault(row["_id"], {})
        if row["image_type"] in pair:
            raise ValueError(f"Duplicate image type for {row['_id']}")
        pair[row["image_type"]] = row
    for article_id, pair in pairs.items():
        if set(pair) != {"original", "edited"}:
            raise ValueError(f"Incomplete original/edited pair: {article_id}")
        if pair["original"]["claim"] != pair["edited"]["claim"]:
            raise ValueError(f"Claims differ within pair: {article_id}")
    return pairs


def read_predictions(path):
    """Read complete prediction rows, rejecting corrupt or mixed-run files."""
    rows, seen = [], set()
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            location = f"{path}:{lineno}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid prediction JSON; repair the file or start a new run") from error
            if not isinstance(row, dict):
                raise ValueError(f"{location}: prediction must be a JSON object")
            for key in ("sample_id", "run_id", "model", "split", "prompt_variant", "generator"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{location}: missing or empty prediction field {key}")
            if not isinstance(row.get("raw_text"), str) or type(row.get("parse_ok")) is not bool:
                raise ValueError(f"{location}: raw_text must be a string and parse_ok must be a boolean")
            errors = row.get("parse_errors")
            if not isinstance(errors, list) or any(not isinstance(error, str) for error in errors):
                raise ValueError(f"{location}: parse_errors must be a list of strings")
            for key in ("pred_verdict", "pred_type", "pred_visual", "pred_explanation"):
                if key not in row or (row[key] is not None and not isinstance(row[key], str)):
                    raise ValueError(f"{location}: {key} must be a string or null")
            if row["sample_id"] in seen:
                raise ValueError(f"{location}: predictions must have unique sample IDs: {row['sample_id']}")
            if rows and any(row[key] != rows[0][key] for key in ("run_id", "model", "split", "prompt_variant")):
                raise ValueError(f"{location}: mixed run/model/split/prompt metadata in predictions")
            seen.add(row["sample_id"])
            rows.append(row)
    return rows
