"""VisualSim — Qwen3-Embedding cosine similarity between the ground-truth
inconsistent visual entity (gt_entity_fine) and the model's <visual> description.

GT is `gt_entity_fine`: the fine-grained visual entity, object, gesture,
clothing style, sign, landmark, or environment cue that was made inconsistent
with the claim.

4 (gt x pred) cases:
  gt=INCONSISTENT, pred=INCONSISTENT -> cosine similarity in [0, 1]
  gt=INCONSISTENT, pred=CONSISTENT   -> 0.0 (miss)
  gt=CONSISTENT,   pred=INCONSISTENT -> 0.0 (false alarm)
  gt=CONSISTENT,   pred=CONSISTENT   -> None (excluded from denominator — there
                                      is no visual element to describe)

Aggregation: mean over non-None scores. edited_only=True (Table 2 view) also
filters to is_edited=True, which naturally drops all tn_abstain samples.
"""


from ..cases import classify as classify_case


def score_trivial(case: str) -> dict | None:
    """Return score dict for non-tp cases; None for 'tp' (caller must batch-embed)."""
    if case == "tn_abstain":
        return {"visual_score": None, "visual_case": "tn_abstain"}
    if case == "miss":
        return {"visual_score": 0.0, "visual_case": "miss"}
    if case == "false_alarm":
        return {"visual_score": 0.0, "visual_case": "false_alarm"}
    return None


def score_one_trivial(gt_verdict: str, pred_verdict: str | None) -> dict | None:
    """Back-compat shim: classify + score_trivial in one call."""
    return score_trivial(classify_case(gt_verdict, pred_verdict))


def batch_score(samples_needing_sim: list[tuple[str, str]], embedder) -> list[float]:
    """samples_needing_sim: list of (gt_entity_fine, pred_visual) pairs.
    Returns list of cosine scores in [0, 1], same order."""
    if not samples_needing_sim:
        return []
    return embedder.cosine_pairs(samples_needing_sim)


def aggregate(per_sample: list[dict], rows: list, edited_only: bool = True) -> dict:
    scores = []
    for m, r in zip(per_sample, rows):
        if edited_only and not r.get("is_edited"):
            continue
        s = m.get("visual_score")
        if s is None:
            continue
        scores.append(s)
    return {
        "n": len(scores),
        "visual": sum(scores) / len(scores) if scores else 0.0,
    }
