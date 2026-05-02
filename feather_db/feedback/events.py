"""FeedbackEvent dataclass + canonical event-kind constants.

Phase 9.1 Week 3. Detect-and-log only — no auto-action.

The data flywheel: every correction, endorsement, contradiction
resolution, and retrieval thumbs-up/down is logged with provenance. In
6 months we have ground-truth signal to bootstrap auto-evolving
ontology + self-alignment loops without guessing.
"""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ── Canonical event kinds ────────────────────────────────────────────

KIND_FACT_CORRECTED = "fact_corrected"
"""User corrected a fact's object value.
payload: {"old_object": str, "new_object": str, "fact_subject": str,
          "fact_predicate": str}"""

KIND_FACT_ENDORSED = "fact_endorsed"
"""User confirmed a fact is correct.
payload: {} (target_id is enough)"""

KIND_FACT_RETRACTED = "fact_retracted"
"""User marked a fact as no longer true / mistakenly extracted.
payload: {"reason": str}"""

KIND_RETRIEVAL_UP = "retrieval_thumbs_up"
"""User thumbs-upped a retrieval result.
payload: {"query": str, "rank": int, "session_id": Optional[str]}"""

KIND_RETRIEVAL_DOWN = "retrieval_thumbs_down"
"""User thumbs-downed a retrieval result.
payload: same as RETRIEVAL_UP, plus optional "reason"."""

KIND_ENTITY_MERGED = "entity_merged"
"""User merged two canonical entities (e.g. 'Acme Corp' = 'ACME Inc').
payload: {"merged_with_id": int}"""

KIND_CONTRADICTION_RESOLVED = "contradiction_resolved"
"""User reviewed a ContradictionFinding and chose a resolution.
payload: {"finding_target_id": int,
          "resolution": "kept_new" | "kept_old" | "kept_both" |
                         "merged" | "rejected_both",
          "rationale": Optional[str]}"""

ALL_KINDS = frozenset({
    KIND_FACT_CORRECTED,
    KIND_FACT_ENDORSED,
    KIND_FACT_RETRACTED,
    KIND_RETRIEVAL_UP,
    KIND_RETRIEVAL_DOWN,
    KIND_ENTITY_MERGED,
    KIND_CONTRADICTION_RESOLVED,
})


# ── The event ────────────────────────────────────────────────────────

@dataclass
class FeedbackEvent:
    """One user feedback signal.

    Keep payload schema-loose — different kinds carry different fields.
    Validation lives at append-time in FeedbackLog (so callers get
    immediate feedback, but we don't reject historical reads on schema
    drift).
    """
    kind: str
    target_id: int
    namespace_id: str
    timestamp: int = 0
    user_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None
    event_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = int(time.time())
        if not self.event_id:
            self.event_id = self._make_id()

    def _make_id(self) -> str:
        """Deterministic id from kind + target + ts + user. 12 hex chars
        is enough collision resistance for an audit log."""
        h = hashlib.sha256()
        h.update(self.kind.encode("utf-8"))
        h.update(b"|")
        h.update(str(self.target_id).encode("utf-8"))
        h.update(b"|")
        h.update(str(self.timestamp).encode("utf-8"))
        h.update(b"|")
        h.update((self.user_id or "").encode("utf-8"))
        h.update(b"|")
        h.update(json.dumps(self.payload, sort_keys=True,
                            default=str).encode("utf-8"))
        return h.hexdigest()[:12]

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, s: str) -> "FeedbackEvent":
        d = json.loads(s)
        return cls(**d)
