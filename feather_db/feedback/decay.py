"""Feedback-aware decay modifier.

Phase 9.1 Week 3. Layered ON TOP OF the existing recall_count-based
adaptive decay (see include/scoring.h). The base decay rewards
frequently-accessed items; this modifier adds explicit user signal:

    final_weight = base_weight * feedback_modifier(target_id)

Behavior:
    retracted          → 0.0  (kill weight entirely)
    corrected (N times)→ 0.5^min(N,3)            (cap −87.5%)
    endorsed  (N times)→ 1 + 0.2*min(N,5)        (cap +100%)
    retrieval_down     → 0.9^min(N,5)            (cap −41%)
    retrieval_up       → 1 + 0.05*min(N,10)      (cap +50%)

These are intentionally conservative — feedback should nudge ranking,
not dominate it. Tune via the `weights` parameter once we have real
correction data (Phase 11).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from .log import FeedbackLog


@dataclass(frozen=True)
class FeedbackWeights:
    """Tuning knobs for the feedback decay modifier."""
    correction_decay: float = 0.5      # per-correction multiplier
    correction_cap: int = 3
    endorsement_boost: float = 0.2     # per-endorse increment
    endorsement_cap: int = 5
    retrieval_down_decay: float = 0.9
    retrieval_down_cap: int = 5
    retrieval_up_boost: float = 0.05
    retrieval_up_cap: int = 10


DEFAULT_WEIGHTS = FeedbackWeights()


def feedback_decay_modifier(target_id: int,
                            log: FeedbackLog,
                            *,
                            namespace_id: Optional[str] = None,
                            weights: FeedbackWeights = DEFAULT_WEIGHTS
                            ) -> float:
    """Compute a multiplicative weight modifier for a target.

    Returns 1.0 if there is no feedback for this target. Returns 0.0
    if the target has been retracted by any user.

    Designed to be applied as:
        final_weight = base_score * feedback_decay_modifier(...)
    """
    summary = log.aggregate_for_target(target_id, namespace_id=namespace_id)
    if summary["retracted"] > 0:
        return 0.0
    mod = 1.0
    if summary["corrected"]:
        n = min(summary["corrected"], weights.correction_cap)
        mod *= weights.correction_decay ** n
    if summary["endorsed"]:
        n = min(summary["endorsed"], weights.endorsement_cap)
        mod *= 1.0 + weights.endorsement_boost * n
    if summary["retrieval_down"]:
        n = min(summary["retrieval_down"], weights.retrieval_down_cap)
        mod *= weights.retrieval_down_decay ** n
    if summary["retrieval_up"]:
        n = min(summary["retrieval_up"], weights.retrieval_up_cap)
        mod *= 1.0 + weights.retrieval_up_boost * n
    return mod
