"""User feedback / correction signal logging.

Phase 9.1 Week 3. Detect-and-log only — no auto-action. Layered on
top of the existing recall_count adaptive decay; surfaces a
multiplicative modifier so downstream scoring can incorporate user
signal without rewriting the C++ scorer.
"""
from .events import (
    FeedbackEvent,
    ALL_KINDS,
    KIND_FACT_CORRECTED,
    KIND_FACT_ENDORSED,
    KIND_FACT_RETRACTED,
    KIND_RETRIEVAL_UP,
    KIND_RETRIEVAL_DOWN,
    KIND_ENTITY_MERGED,
    KIND_CONTRADICTION_RESOLVED,
)
from .log import FeedbackLog
from .decay import (
    FeedbackWeights,
    DEFAULT_WEIGHTS,
    feedback_decay_modifier,
)

__all__ = [
    "FeedbackEvent",
    "FeedbackLog",
    "FeedbackWeights",
    "DEFAULT_WEIGHTS",
    "feedback_decay_modifier",
    "ALL_KINDS",
    "KIND_FACT_CORRECTED",
    "KIND_FACT_ENDORSED",
    "KIND_FACT_RETRACTED",
    "KIND_RETRIEVAL_UP",
    "KIND_RETRIEVAL_DOWN",
    "KIND_ENTITY_MERGED",
    "KIND_CONTRADICTION_RESOLVED",
]
