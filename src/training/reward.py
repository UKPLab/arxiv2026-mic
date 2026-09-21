# Reward function for MIC claim-grounded visual inconsistency detection
# Used by verl GRPO trainer (data_source="mic")
#
# Verifiable reward over five components (paper Section "Reward Functions Design"):
#   R_fmt  format validity        (binary, soft — not a gate)
#   R_ver  verdict exact match    (binary)
#   R_type type exact match       (binary)
#   R_desc visual-evidence sim     (Qwen3-Embedding cosine)
#   R_exp  explanation sim         (Qwen3-Embedding cosine)
# Inconsistent: weighted sum of all five. Consistent: only R_fmt + R_ver.
import json
import os
import re
from threading import Lock

_TAGS = ("think", "verdict", "type", "visual", "explanation")
_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
_FORMAT_WEIGHT = 0.10
_VERDICT_WEIGHT = 0.35
_TYPE_WEIGHT = 0.25
_VISUAL_WEIGHT = 0.15
_EXPLANATION_WEIGHT = 0.15


def _extract_tag_occurrences(text: str, tag: str) -> list[str]:
    return [match.strip() for match in re.findall(fr"<{tag}>(.*?)</{tag}>", text, re.DOTALL)]


def _schema_is_valid(text: str) -> bool:
    return all(len(_extract_tag_occurrences(text, tag)) == 1 for tag in _TAGS)


def _extract_unique_tag(text: str, tag: str) -> str | None:
    matches = _extract_tag_occurrences(text, tag)
    if len(matches) != 1:
        return None
    return matches[0]


def _normalize_free_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    if not cleaned:
        return None
    if cleaned.lower() in {"none", "n/a", "na", "null"}:
        return None
    return cleaned


def _normalize_verdict(value: str | None) -> str | None:
    cleaned = _normalize_free_text(value)
    if cleaned is None:
        return None
    upper = cleaned.upper()
    return upper if upper in {"CONSISTENT", "INCONSISTENT"} else None


def _normalize_type(value: str | None) -> str | None:
    cleaned = _normalize_free_text(value)
    if cleaned is None:
        return None

    from src.data import normalize_type
    from src.schema import INCONSISTENCY_TYPES
    normalized = normalize_type(cleaned)
    return normalized if normalized in INCONSISTENCY_TYPES else None


def _first_text(*values: str | None) -> str:
    """Return the first non-empty normalized text among candidate GT keys."""
    for value in values:
        cleaned = _normalize_free_text(value)
        if cleaned:
            return cleaned
    return ""


_embedder = None
_embedder_lock = Lock()


def _semantic_similarity(prediction, target):
    global _embedder
    if not prediction or not target:
        return 0.0
    if _embedder is None:
        # verl evaluates rewards in a thread pool; initialize one model per worker.
        with _embedder_lock:
            if _embedder is None:
                from src.embedding import Qwen3Embedder
                _embedder = Qwen3Embedder(
                    model_id=os.environ.get("MIC_REWARD_EMBED_MODEL", _EMBEDDING_MODEL),
                    device=os.environ.get("MIC_REWARD_EMBED_DEVICE", "cpu"),
                    dtype=os.environ.get("MIC_REWARD_EMBED_DTYPE", "float32"),
                )
    return _embedder.cosine(prediction, target)


def score_response(predict_str: str, ground_truth: str, **kwargs) -> float:
    """Compute the verifiable reward for MIC GRPO training.

    Inconsistent: lambda_fmt R_fmt + lambda_ver R_ver + lambda_type R_type
                  + lambda_desc R_desc + lambda_exp R_exp
    Consistent:   lambda_fmt R_fmt + lambda_ver R_ver  (inconsistency-specific
                  rewards excluded).
    """
    # R_fmt — soft format reward (no longer a hard gate)
    format_score = 1.0 if _schema_is_valid(predict_str) else 0.0

    try:
        gt = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        gt = {"verdict": ground_truth}

    if not isinstance(gt, dict):
        raise ValueError("Reward ground truth must be a JSON object")
    gt_verdict = _normalize_verdict(gt.get("verdict"))
    if gt_verdict is None:
        raise ValueError("Missing or invalid ground-truth verdict")
    extra_info = kwargs.get("extra_info") or {}

    # R_ver — verdict exact match
    pred_verdict = _normalize_verdict(_extract_unique_tag(predict_str, "verdict"))
    verdict_score = 1.0 if pred_verdict == gt_verdict else 0.0

    # Consistent examples: only format + verdict
    if gt_verdict == "CONSISTENT":
        return _FORMAT_WEIGHT * format_score + _VERDICT_WEIGHT * verdict_score

    # ---- Inconsistent examples: all five components ----
    # R_type — inconsistency-type exact match
    gt_type = _normalize_type(gt.get("edit_type")) or _normalize_type(extra_info.get("edit_type"))
    pred_type = _normalize_type(_extract_unique_tag(predict_str, "type"))
    type_score = 1.0 if gt_type is not None and pred_type == gt_type else 0.0

    # R_desc — visual-evidence description similarity
    gt_visual = _first_text(gt.get("gt_entity_fine"), gt.get("entity_fine"),
                            gt.get("visual"), extra_info.get("entity_fine"))
    pred_visual = _normalize_free_text(_extract_unique_tag(predict_str, "visual"))
    visual_score = _semantic_similarity(pred_visual, gt_visual)

    # R_exp — world-knowledge explanation similarity
    gt_explanation = _first_text(gt.get("gt_why_contradicts"), gt.get("why_contradicts"), gt.get("explanation"),
                                 extra_info.get("why_contradicts"))
    pred_explanation = _normalize_free_text(_extract_unique_tag(predict_str, "explanation"))
    explanation_score = _semantic_similarity(pred_explanation, gt_explanation)

    return (
        _FORMAT_WEIGHT * format_score
        + _VERDICT_WEIGHT * verdict_score
        + _TYPE_WEIGHT * type_score
        + _VISUAL_WEIGHT * visual_score
        + _EXPLANATION_WEIGHT * explanation_score
    )


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """verl custom-reward entry point; return a scalar component-weighted reward."""
    return score_response(solution_str, ground_truth, extra_info=extra_info)
