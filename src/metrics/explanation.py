"""ExplSim — Qwen3-Embedding cosine similarity between the ground-truth
world-knowledge reason (gt_why_contradicts) and the model's <explanation>.

Same 4-case structure as visual.py. Because embedding encoding is cheap in
batch, this module prefers a batch entry point: batch_score() computes
per-sample values in one pass over all samples that need similarity.
"""


from . import classify as classify_case


def score_trivial(case: str) -> dict | None:
    """Return score dict for non-tp cases; None for 'tp' (caller must batch-embed)."""
    if case == "tn_abstain":
        return {"explanation_score": None, "explanation_case": "tn_abstain"}
    if case == "miss":
        return {"explanation_score": 0.0, "explanation_case": "miss"}
    if case == "false_alarm":
        return {"explanation_score": 0.0, "explanation_case": "false_alarm"}
    return None


def score_one_trivial(gt_verdict: str, pred_verdict: str | None) -> dict | None:
    """Back-compat shim: classify + score_trivial in one call."""
    return score_trivial(classify_case(gt_verdict, pred_verdict))


def batch_score(samples_needing_sim: list[tuple[str, str]], embedder) -> list[float]:
    """samples_needing_sim: list of (gt_why_contradicts, pred_explanation) pairs.
    Returns list of cosine scores in [0, 1], same order."""
    if not samples_needing_sim:
        return []
    return embedder.cosine_pairs(samples_needing_sim)


def aggregate(per_sample: list[dict], rows: list, edited_only: bool = True) -> dict:
    scores = []
    for m, r in zip(per_sample, rows):
        if edited_only and not r.get("is_edited"):
            continue
        s = m.get("explanation_score")
        if s is None:
            continue
        scores.append(s)
    return {
        "n": len(scores),
        "explanation": sum(scores) / len(scores) if scores else 0.0,
    }
