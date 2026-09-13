"""TypeAcc — inconsistency type classification accuracy.

Only defined on edited samples (where gt_edit_type exists).
Originals (gt_edit_type=None) are excluded from the denominator.
"""


def score_one(is_edited: bool, gt_edit_type: str | None, pred_type: str | None) -> dict:
    if not is_edited or gt_edit_type is None:
        return {"type_correct": None, "type_applicable": 0}

    correct = int(pred_type == gt_edit_type)
    return {"type_correct": correct, "type_applicable": 1}


def aggregate(per_sample: list[dict]) -> dict:
    denom = sum(m.get("type_applicable", 0) for m in per_sample)
    numer = sum(m.get("type_correct", 0) for m in per_sample if m.get("type_correct") is not None)
    return {
        "n_edited": denom,
        "type_acc": numer / denom if denom else 0.0,
    }
