"""Prompt template for consistency detection.

Five tags in the output schema, each mapped 1:1 to a metric:
  <think>        — chain-of-thought reasoning (not scored)
  <verdict>      — Macro-F1 / Accuracy (Table 1)
  <type>         — TypeAcc
  <visual>       — VisualSim: cosine similarity to gt_entity_fine (Qwen3-Embedding)
  <explanation>  — ExplSim:   cosine similarity to gt_why_contradicts (Qwen3-Embedding)

VisualSim and ExplSim are independent embedding similarities, each over its own
tag; there is no derived/concatenated metric.
"""

from .schema import INCONSISTENCY_TYPES


_TYPE_LIST = ", ".join(INCONSISTENCY_TYPES)

_INTRO = (
    "You are an expert fact-checker analyzing whether a news image is "
    "consistent with its claimed context."
)
_CLAIM = 'Claim: "{claim}"'

_VISUAL_CANONICAL = (
    "Carefully inspect the image step by step. Check multiple visual aspects "
    "including people, flags, text, architecture, vegetation, technology, "
    "social behavior, and other contextual elements."
)

_SCHEMA_CANONICAL = f"""<think>
Step 1: [Extract one checkable aspect from the claim]
-> Region: [specific area of the image you are examining]
-> Observation: [what you actually see]
-> Assessment: CONSISTENT or INCONSISTENT - [brief justification]

Step 2: ...
(Continue for 3-5 steps, covering different visual aspects)
</think>

<verdict>CONSISTENT or INCONSISTENT</verdict>

<type>CONSISTENT -> None; INCONSISTENT -> exactly one of: {_TYPE_LIST}</type>

<visual>CONSISTENT -> None; INCONSISTENT -> concise description of the specific entity (object / person / flag / sign / gesture / behavior / clothing style / landmark) in the image that is inconsistent with the claim</visual>

<explanation>CONSISTENT -> None; INCONSISTENT -> concise world-knowledge reason explaining why this contradicts the claim</explanation>"""

_RULES = """Rules:
- Output each tag exactly once.
- If the image is inconsistent, choose exactly one inconsistency type from the allowed list.
- If the image is consistent, write None for type, visual, and explanation.
- Do not add any extra sections outside these tags.
- Do not invent details that are not visible in the image."""


def _assemble(*blocks: str) -> str:
    return "\n\n".join(blocks) + "\n"


_TEMPLATES = {
    "canonical": _assemble(
        _INTRO, _CLAIM, _VISUAL_CANONICAL,
        "Return your answer using exactly this format:", _SCHEMA_CANONICAL, _RULES,
    ),
}

PROMPT_VARIANTS = tuple(_TEMPLATES.keys())


def build_prompt(claim: str, variant: str = "canonical") -> str:
    if variant not in _TEMPLATES:
        raise ValueError(f"Unknown prompt variant: {variant}. Expected one of {PROMPT_VARIANTS}")
    return _TEMPLATES[variant].format(claim=claim)
