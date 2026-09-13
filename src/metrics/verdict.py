"""Verdict-level metrics: Accuracy / Precision / Recall / F1.

Positive class = INCONSISTENT.

Per-sample output (consumed by run_score):
    verdict_correct: 1 if pred_verdict == gt_verdict else 0
    verdict_tp / fp / fn / tn: one-hot flag for the confusion-matrix cell

Aggregation (Acc/P/R/F1) is implemented below and called by run_score.py.
"""


def score_one(gt_verdict: str, pred_verdict: str | None) -> dict:
    """Per-sample verdict counters.

    Parse failure on verdict (pred_verdict is None) is treated as a wrong
    prediction, bucketed conservatively:
      - gt INCONSISTENT + parse_fail → FN (model failed to detect positive)
      - gt CONSISTENT   + parse_fail → FP (model failed to clear negative)
    This keeps Recall honest — without it, parse-failed positives would drop
    out of the denominator and artificially inflate Recall.
    invalid_count is still tracked as a separate tag for reporting.
    """
    if gt_verdict not in {"CONSISTENT", "INCONSISTENT"}:
        raise ValueError(f"Invalid ground-truth verdict: {gt_verdict}")
    gt_pos = gt_verdict == "INCONSISTENT"

    if pred_verdict not in {"CONSISTENT", "INCONSISTENT"}:
        return {
            "verdict_correct": 0,
            "verdict_tp": 0,
            "verdict_fp": int(not gt_pos),
            "verdict_fn": int(gt_pos),
            "verdict_tn": 0,
            "verdict_invalid": 1,
        }

    pred_pos = pred_verdict == "INCONSISTENT"
    tp = int(gt_pos and pred_pos)
    fp = int((not gt_pos) and pred_pos)
    fn = int(gt_pos and (not pred_pos))
    tn = int((not gt_pos) and (not pred_pos))

    return {
        "verdict_correct": int(gt_pos == pred_pos),
        "verdict_tp": tp,
        "verdict_fp": fp,
        "verdict_fn": fn,
        "verdict_tn": tn,
        "verdict_invalid": 0,
    }


def aggregate(per_sample: list[dict], rows: list | None = None,
              edited_only: bool = False) -> dict:
    """Compute Acc/P/R/F1 from per-sample counters.

    edited_only=True: Table-2 view. Filter rows where is_edited=True first.
    Precision on edited-only degenerates to 1.0 (no negatives); F1 becomes
    2*recall/(1+recall). We still expose the number under the 'f1' key for
    consistent column naming downstream — the table caller is expected to
    know which view it asked for.
    """
    if edited_only:
        if rows is None:
            raise ValueError("edited_only=True requires rows (to filter by is_edited)")
        per_sample = [m for m, r in zip(per_sample, rows) if r.get("is_edited")]

    tp = sum(m.get("verdict_tp", 0) for m in per_sample)
    fp = sum(m.get("verdict_fp", 0) for m in per_sample)
    fn = sum(m.get("verdict_fn", 0) for m in per_sample)
    tn = sum(m.get("verdict_tn", 0) for m in per_sample)
    invalid = sum(m.get("verdict_invalid", 0) for m in per_sample)

    # invalid is already accounted for in fp/fn, don't double-count
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Negative-class (CONSISTENT) F1 and Macro-F1 over both classes.
    # Only meaningful when negatives are present (edited_only=False); on the
    # edited-only view tn=fp=0, so f1_con and macro_f1 degenerate and should
    # not be reported.
    prec_neg = tn / (tn + fn) if (tn + fn) else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) else 0.0
    f1_con = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) else 0.0
    macro_f1 = (f1 + f1_con) / 2

    return {
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "invalid": invalid,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f1_con": f1_con,
        "macro_f1": macro_f1,
        "detection_rate": rec,
    }
