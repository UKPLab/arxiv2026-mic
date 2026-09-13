"""Pipeline dataclasses — single source of truth for all cross-stage I/O."""

from dataclasses import dataclass, field, asdict
from typing import Optional


INCONSISTENCY_TYPES = (
    "clothing",
    "flag",
    "gesture",
    "signage",
    "architecture",
    "infrastructure",
    "technology",
    "branding",
    "environment",
)


@dataclass
class EvalSample:
    """One (image, claim) input to the pipeline.

    sample_id: unique per (split, article_id, image_type). Drives skip_existing.
    is_edited: True → gt verdict INCONSISTENT; False → CONSISTENT.
    gt_edit_type: only meaningful for edited samples; None for originals.
    """

    sample_id: str
    article_id: str
    split: str
    image_path: str
    claim: str
    is_edited: bool
    generator: str
    year: Optional[int] = None

    gt_edit_type: Optional[str] = None       # → TypeAcc
    gt_entity_fine: Optional[str] = None      # → VisualSim (inconsistent visual entity)
    gt_why_contradicts: Optional[str] = None  # → ExplSim (world-knowledge reason)

    @property
    def gt_verdict(self) -> str:
        return "INCONSISTENT" if self.is_edited else "CONSISTENT"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Prediction:
    """Stage 1 output — one row in predictions/*.jsonl."""

    sample_id: str
    run_id: str
    model: str
    split: str
    prompt_variant: str
    generator: str

    raw_text: str
    pred_verdict: Optional[str] = None
    pred_type: Optional[str] = None
    pred_visual: Optional[str] = None
    pred_explanation: Optional[str] = None
    pred_think: Optional[str] = None

    parse_ok: bool = True
    parse_errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoredSample:
    """Stage 2 output — one row in scored/*.jsonl.

    Holds per-sample metric values + the gt labels it was scored against,
    so that tables can be regenerated without re-joining to split files.
    """

    sample_id: str
    run_id: str
    model: str
    split: str
    prompt_variant: str
    generator: str

    is_edited: bool
    gt_verdict: str
    gt_edit_type: Optional[str]

    pred_verdict: Optional[str]
    pred_type: Optional[str]

    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
