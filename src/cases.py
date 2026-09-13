"""Single source of truth for (gt × pred) case classification.

Both similarity metrics (VisualSim, ExplSim) share the same 4-case partition.
Defining it once here prevents drift between run_score.py and the per-metric
scoring functions.

Cases:
  "tn_abstain"  — gt=CONSISTENT & pred=CONSISTENT (nothing to score; mean excludes)
  "miss"        — gt=INCONSISTENT & pred=CONSISTENT (model missed the edit → 0)
  "false_alarm" — gt=CONSISTENT & pred=INCONSISTENT (model hallucinated → 0)
  "tp"          — gt=INCONSISTENT & pred=INCONSISTENT (embedding cosine similarity)
"""


def classify(gt_verdict: str, pred_verdict: str | None) -> str:
    gt_pos = gt_verdict == "INCONSISTENT"
    pred_pos = pred_verdict == "INCONSISTENT"
    if not gt_pos and not pred_pos:
        return "tn_abstain"
    if gt_pos and not pred_pos:
        return "miss"
    if not gt_pos and pred_pos:
        return "false_alarm"
    return "tp"
