"""Shared verdict cases for visual-evidence and explanation scoring.

Correct negatives abstain, misses and false alarms receive zero, and correct
positives require text similarity. A missing prediction is treated as negative
for this case classification; verdict errors are scored separately.
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
