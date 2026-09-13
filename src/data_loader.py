"""Load MIC split manifests with configurable data and split locations."""

import os
from pathlib import Path

from .records import image_path, read_records, sample_id
from .schema import EvalSample

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("MIC_DATA_ROOT", REPO_ROOT / "data"))
SPLITS_ROOT = DATA_ROOT / "splits"


def load_split(split_name: str, data_root=None, split_root=None) -> list[EvalSample]:
    root = Path(data_root) if data_root is not None else DATA_ROOT
    splits = Path(split_root) if split_root is not None else root / "splits"
    rows = read_records(splits / f"{split_name}.json", root)
    samples = []
    for item in rows:
        edited = item["image_type"] == "edited"
        samples.append(EvalSample(
            sample_id=sample_id(item), article_id=item["_id"], split=split_name,
            image_path=str(image_path(root, item["local_image"])), claim=item["claim"],
            is_edited=edited, generator=item.get("generator", "unknown" if edited else "none"),
            year=item.get("year"), gt_edit_type=item["edit_type"] if edited else None,
            gt_entity_fine=item.get("gt_entity_fine"),
            gt_why_contradicts=item.get("gt_why_contradicts"),
        ))
    return samples
