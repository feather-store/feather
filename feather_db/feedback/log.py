"""FeedbackLog — append-only JSONL store with in-memory indexing.

Phase 9.1 Week 3. Single-writer, single-process. Multi-tenant Cloud
will swap this for a sharded / DB-backed implementation; the public
API is intentionally narrow so the swap is transparent.
"""
from __future__ import annotations
import os
from collections import Counter, defaultdict
from typing import Iterator, Optional

from .events import FeedbackEvent, ALL_KINDS


class FeedbackLog:
    """Append-only feedback event log.

    Args:
        path:           file path. Created on first append if missing.
        validate_kinds: if True (default), reject events whose kind is
                        not in ALL_KINDS. Set False to ingest legacy
                        formats.
    """

    def __init__(self, path: str, *, validate_kinds: bool = True):
        self._path = path
        self._validate = validate_kinds

    @property
    def path(self) -> str:
        return self._path

    # ── Write ──────────────────────────────────────────────────────

    def append(self, event: FeedbackEvent) -> str:
        """Append an event. Returns its event_id.

        Raises ValueError if kind validation is on and the kind is
        unknown.
        """
        if self._validate and event.kind not in ALL_KINDS:
            raise ValueError(
                f"Unknown feedback kind: {event.kind!r}. "
                f"Allowed: {sorted(ALL_KINDS)}"
            )
        # Atomic-ish append: open in 'a' mode each time. Keeps the API
        # simple at the cost of fsync-per-event. Fine for the volumes
        # we're handling (manual-review-cadence events, not telemetry).
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(event.to_json())
            fh.write("\n")
        return event.event_id

    # ── Read ───────────────────────────────────────────────────────

    def iter_events(self, *,
                    namespace_id: Optional[str] = None,
                    kind: Optional[str] = None,
                    target_id: Optional[int] = None,
                    since: Optional[int] = None,
                    until: Optional[int] = None,
                    user_id: Optional[str] = None
                    ) -> Iterator[FeedbackEvent]:
        """Yield matching events in chronological (file) order."""
        if not os.path.exists(self._path):
            return
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = FeedbackEvent.from_json(line)
                except Exception:
                    continue  # skip corrupted lines silently
                if namespace_id is not None and ev.namespace_id != namespace_id:
                    continue
                if kind is not None and ev.kind != kind:
                    continue
                if target_id is not None and ev.target_id != target_id:
                    continue
                if user_id is not None and ev.user_id != user_id:
                    continue
                if since is not None and ev.timestamp < since:
                    continue
                if until is not None and ev.timestamp > until:
                    continue
                yield ev

    def count(self, **filters) -> int:
        return sum(1 for _ in self.iter_events(**filters))

    # ── Aggregations for downstream scoring ────────────────────────

    def aggregate_for_target(self, target_id: int,
                             *, namespace_id: Optional[str] = None
                             ) -> dict:
        """Counts by kind for one target.

        Returned dict keys are short labels suitable for the scoring
        modifier:
            endorsed, corrected, retracted, retrieval_up, retrieval_down,
            entity_merged, contradiction_resolved
        Plus 'last_event_ts' = most recent event timestamp (0 if none).
        """
        counts: Counter = Counter()
        last_ts = 0
        for ev in self.iter_events(target_id=target_id,
                                   namespace_id=namespace_id):
            counts[ev.kind] += 1
            if ev.timestamp > last_ts:
                last_ts = ev.timestamp
        return {
            "endorsed":            counts["fact_endorsed"],
            "corrected":           counts["fact_corrected"],
            "retracted":           counts["fact_retracted"],
            "retrieval_up":        counts["retrieval_thumbs_up"],
            "retrieval_down":      counts["retrieval_thumbs_down"],
            "entity_merged":       counts["entity_merged"],
            "contradiction_resolved": counts["contradiction_resolved"],
            "last_event_ts":       last_ts,
        }

    def per_target_counts(self, *,
                          namespace_id: Optional[str] = None,
                          kind: Optional[str] = None
                          ) -> dict[int, int]:
        """Map of target_id -> count, for a slice of the log."""
        out: dict[int, int] = defaultdict(int)
        for ev in self.iter_events(namespace_id=namespace_id, kind=kind):
            out[ev.target_id] += 1
        return dict(out)
