"""Stage 2 — scoring CLI.

Reads predictions JSONL + its originating split, computes metric columns
per sample, writes scored JSONL. Prints Table 1 (pair-doubled) and Table 2
(edited-only) aggregate views.

Metrics:
  verdict + type_acc                 (local, cheap)
  VisualSim  (gt_entity_fine     vs <visual>)       Qwen3-Embedding cosine
  ExplSim    (gt_why_contradicts vs <explanation>)  Qwen3-Embedding cosine

Usage:
    python -m src.run_score \\
        --predictions results/predictions/<run>.jsonl \\
        --split test_id_edit
    # verdict + type only, no embedding model:
    python -m src.run_score --predictions ... --split ... --skip-embed
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cases import classify as classify_case
from src.data_loader import load_split
from src.parsers.cot_tagged import parse
from src.records import read_predictions
from src.schema import ScoredSample
from src.metrics import verdict as m_verdict
from src.metrics import type_acc as m_type
from src.metrics import visual as m_visual
from src.metrics import explanation as m_explanation


RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
SCORED_DIR = RESULTS_ROOT / "scored"


def _load_predictions(path: Path) -> list[dict]:
    return read_predictions(path)


def _prediction_for_scoring(p: dict) -> dict:
    """Apply the scoring policy for malformed model outputs.

    The predictions JSONL preserves extracted fields, but a response that fails
    schema validation receives no credit for any extracted field.
    """
    parsed = parse(p["raw_text"])
    fields_match = all(p.get(f"pred_{key}") == parsed[key]
                       for key in ("verdict", "type", "visual", "explanation"))
    if p["parse_ok"] and parsed["parse_ok"] and fields_match and not p["parse_errors"]:
        return p

    q = dict(p)
    q["parse_ok"] = False
    q["parse_errors"] = list(dict.fromkeys(p["parse_errors"] + parsed["parse_errors"]
                                          + ([] if fields_match else ["parsed_fields_mismatch"])))
    q["pred_verdict"] = None
    q["pred_type"] = None
    q["pred_visual"] = None
    q["pred_explanation"] = None
    return q


def score_run(
    pred_path: Path,
    split_name: str,
    embedder=None,
    data_root=None,
    split_root=None,
    allow_partial=False,
) -> tuple[list[ScoredSample], dict]:
    preds = _load_predictions(pred_path)
    samples = {s.sample_id: s for s in load_split(split_name, data_root, split_root)}
    ids = [p["sample_id"] for p in preds]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Predictions must be non-empty with unique sample IDs")
    unknown = set(ids) - samples.keys()
    missing = samples.keys() - set(ids)
    if unknown:
        raise ValueError(f"Unknown prediction IDs: {sorted(unknown)[:5]}")
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} predictions; use --allow-partial for an explicitly partial report")
    for pred in preds:
        if pred.get("split") != split_name:
            raise ValueError(f"Prediction split does not match {split_name}")

    per_sample: list[dict] = []
    pack: list[tuple[dict, object, dict]] = []  # (per_sample_metric, split_sample, pred)

    pending_visual: list[tuple[int, str, str]] = []       # (idx, gt_entity_fine, pred_visual)
    pending_explanation: list[tuple[int, str, str]] = []  # (idx, gt_why, pred_explanation)

    # ---------- Pass 1: local metrics + case classification ----------
    for p in preds:
        i = len(per_sample)
        sid = p["sample_id"]
        s = samples.get(sid)
        if s is None:
            continue
        p_score = _prediction_for_scoring(p)

        m: dict = {}
        m["parse_ok"] = p_score["parse_ok"]
        m["parse_failed_forced_zero"] = int(not p_score["parse_ok"])
        m["parse_error_count"] = len(p_score["parse_errors"])
        m.update(m_verdict.score_one(s.gt_verdict, p_score.get("pred_verdict")))
        m.update(m_type.score_one(s.is_edited, s.gt_edit_type, p_score.get("pred_type")))

        case = classify_case(s.gt_verdict, p_score.get("pred_verdict"))

        # ---- VisualSim ---- (GT = gt_entity_fine, the inconsistent visual entity)
        triv = m_visual.score_trivial(case)
        if triv is not None:
            m.update(triv)
        elif s.gt_entity_fine and p_score.get("pred_visual"):
            pending_visual.append((i, s.gt_entity_fine, p_score["pred_visual"]))
            m["visual_case"] = "tp_pending"
        else:
            m.update({"visual_score": 0.0, "visual_case": "tp_missing_text"})

        # ---- ExplSim ---- (GT = gt_why_contradicts, the world-knowledge reason)
        triv = m_explanation.score_trivial(case)
        if triv is not None:
            m.update(triv)
        elif s.gt_why_contradicts and p_score.get("pred_explanation"):
            pending_explanation.append((i, s.gt_why_contradicts, p_score["pred_explanation"]))
            m["explanation_case"] = "tp_pending"
        else:
            m.update({"explanation_score": 0.0, "explanation_case": "tp_missing_text"})

        per_sample.append(m)
        pack.append((m, s, p_score))

    # ---------- Pass 2: batched embedding for VisualSim ----------
    if embedder is not None and pending_visual:
        print(f"[score] VisualSim embedding: {len(pending_visual)} pairs")
        pairs = [(gt, pr) for _, gt, pr in pending_visual]
        sims = m_visual.batch_score(pairs, embedder)
        for (idx, _, _), sim in zip(pending_visual, sims):
            per_sample[idx].update({"visual_score": sim, "visual_case": "tp_sim"})

    # ---------- Pass 3: batched embedding for ExplSim ----------
    if embedder is not None and pending_explanation:
        print(f"[score] ExplSim embedding: {len(pending_explanation)} pairs")
        pairs = [(gt, pr) for _, gt, pr in pending_explanation]
        sims = m_explanation.batch_score(pairs, embedder)
        for (idx, _, _), sim in zip(pending_explanation, sims):
            per_sample[idx].update({"explanation_score": sim, "explanation_case": "tp_sim"})

    if embedder is None:
        for metrics in per_sample:
            metrics.update(visual_score=None, explanation_score=None,
                           visual_case="not_computed", explanation_case="not_computed")

    # ---------- Build ScoredSample list ----------
    scored: list[ScoredSample] = []
    for m, s, p in pack:
        scored.append(ScoredSample(
            sample_id=s.sample_id,
            run_id=p["run_id"],
            model=p["model"],
            split=p["split"],
            prompt_variant=p["prompt_variant"],
            generator=p["generator"],
            is_edited=s.is_edited,
            gt_verdict=s.gt_verdict,
            gt_edit_type=s.gt_edit_type,
            pred_verdict=p.get("pred_verdict"),
            pred_type=p.get("pred_type"),
            metrics=m,
        ))

    # ---------- Aggregate ----------
    rows = [{"is_edited": s.is_edited} for s in scored]
    agg = {
        "table1": m_verdict.aggregate(per_sample, rows, edited_only=False),
        "table2_verdict": m_verdict.aggregate(per_sample, rows, edited_only=True),
        "type_acc": m_type.aggregate(per_sample),
        "visual": m_visual.aggregate(per_sample, rows, edited_only=True),
        "explanation": m_explanation.aggregate(per_sample, rows, edited_only=True),
    }
    agg["coverage"] = {"expected": len(samples), "scored": len(scored), "missing": len(missing)}
    agg["embedding_computed"] = embedder is not None
    if embedder is None:
        agg["visual"] = {"n": 0, "visual": None}
        agg["explanation"] = {"n": 0, "explanation": None}
    return scored, agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--skip-embed", action="store_true",
                   help="skip Qwen3-Embedding (VisualSim/ExplSim left as None)")
    p.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--split-root", type=Path)
    p.add_argument("--allow-partial", action="store_true", help="Report coverage for a partial prediction file")
    args = p.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(f"predictions not found: {pred_path}")
    out_path = Path(args.out) if args.out else SCORED_DIR / pred_path.name
    summary_path = out_path.with_suffix(".summary.json")
    if len({pred_path.resolve(), out_path.resolve(), summary_path.resolve()}) != 3:
        p.error("Predictions, scored output, and summary must use different paths")

    embedder = None
    if not args.skip_embed:
        from src.embedding import Qwen3Embedder
        embedder = Qwen3Embedder(
            model_id=args.embed_model,
            device=os.environ.get("EVAL_EMBED_DEVICE"),
            dtype=os.environ.get("EVAL_EMBED_DTYPE", "bfloat16"),
        )

    scored, agg = score_run(pred_path, args.split, embedder=embedder, data_root=args.data_root,
                            split_root=args.split_root, allow_partial=args.allow_partial)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    print(f"[run_score] wrote {len(scored)} scored rows → {out_path}")

    v1 = agg["table1"]
    v2 = agg["table2_verdict"]
    t = agg["type_acc"]
    vis = agg["visual"]
    exp = agg["explanation"]
    print()
    print("=== Table 1 (pair-doubled) ===")
    print(f"  N={v1['n']}  TP={v1['tp']}  FP={v1['fp']}  FN={v1['fn']}  TN={v1['tn']}")
    if v1['invalid']:
        print(f"  (of which {v1['invalid']} are parse-failed, already bucketed into FP/FN)")
    print(f"  Accuracy  = {v1['accuracy']:.4f}")
    print(f"  Macro-F1  = {v1['macro_f1']:.4f}")
    print(f"  F1_inc    = {v1['f1']:.4f}   (positive = INCONSISTENT)")
    print(f"  F1_con    = {v1['f1_con']:.4f}   (positive = CONSISTENT)")
    print()
    print("=== Table 2 (edited-only) ===")
    print(f"  N={v2['n']}  TP={v2['tp']}  FN={v2['fn']}")
    if v2['invalid']:
        print(f"  (of which {v2['invalid']} are parse-failed, already bucketed into FN)")
    print(f"  F1        = {v2['f1']:.4f}")
    print(f"  TypeAcc   = {t['type_acc']:.4f}   (n_edited={t['n_edited']})")
    if args.skip_embed:
        print("  VisualSim / ExplSim: not computed (--skip-embed)")
    else:
        print(f"  VisualSim = {vis['visual']:.4f}  (n={vis['n']})")
        print(f"  ExplSim   = {exp['explanation']:.4f}  (n={exp['n']})")
    print(f"  Coverage: {agg['coverage']}")


if __name__ == "__main__":
    main()
