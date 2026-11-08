"""The time memory is read at.

A handle opened with ``DB.open(path, as_of=t)`` has its clock pinned to that
moment (``DB.now()``). Everything above the engine that ranks, decays or
expires memory reads the time here, so replaying an agent against the past
scores, decays and expires records the way they were scored at the time.
"""
from __future__ import annotations

import time


def now(db) -> float:
    """``db.now()`` when the handle has one (DB, RemoteDB), else the wall clock."""
    f = getattr(db, "now", None)
    return float(f()) if callable(f) else time.time()
