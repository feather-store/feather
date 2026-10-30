"""Dated facts at ingest, for LongMemEval: one extraction per session,
cached on disk, used to FIND sessions — the reader still gets the
sessions themselves (single-session-assistant punishes summary-only
memory).

The cache is keyed by extractor version + model + session date + session
content. LongMemEval_S haystacks reuse sessions across questions, so a
full run extracts far fewer sessions than it ingests.
"""
from __future__ import annotations
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Optional

from feather_db.extractors import Fact, SessionFactExtractor

from .datasets.longmemeval import LME_DIR


def _fact_to_dict(f: Fact) -> dict:
    return {"subject": f.subject, "predicate": f.predicate, "object": f.object,
            "valid_at": f.valid_at, "metadata": f.metadata}


def _fact_from_dict(d: dict) -> Fact:
    return Fact(subject=d["subject"], predicate=d["predicate"],
                object=d["object"], valid_at=d.get("valid_at"),
                extractor_name=SessionFactExtractor.name,
                extractor_version=SessionFactExtractor.version,
                metadata=dict(d.get("metadata") or {}))


class SessionFactCache:
    """SessionFactExtractor behind an append-only JSONL cache."""

    def __init__(self, provider, model_name: str,
                 path: Optional[Path] = None, **extractor_kw):
        self._extractor = SessionFactExtractor(provider, **extractor_kw)
        self._model = model_name or "unknown"
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in self._model)
        self._path = Path(path or LME_DIR / "facts" / f"{safe}.jsonl")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, list[dict]] = {}
        if self._path.exists():
            with open(self._path) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        self._mem[row["key"]] = row["facts"]
                    except (ValueError, KeyError):
                        continue    # a torn last line from a killed run
        self.calls = self.hits = self.failures = 0
        # Questions run on worker threads. The extraction itself runs
        # outside the lock; two workers may both extract a session the
        # first time it is seen, and the second write is harmless.
        self._lock = threading.Lock()

    def _key(self, session_date: str, turns: list[dict]) -> str:
        h = hashlib.sha1()
        h.update(f"{SessionFactExtractor.version}\0{self._model}\0"
                 f"{session_date}\0".encode())
        h.update(json.dumps([[t.get("role"), t.get("content")] for t in turns],
                            ensure_ascii=False).encode())
        return h.hexdigest()

    def facts(self, session_id: str, session_date: str,
              turns: list[dict]) -> list[Fact]:
        """Fresh Fact objects (callers mutate metadata via link_validity)."""
        key = self._key(session_date, turns)
        with self._lock:
            cached = self._mem.get(key)
            if cached is not None:
                self.hits += 1
            else:
                self.calls += 1
        if cached is None:
            errors = []
            got = self._extractor.extract_session(
                turns, session_date, context={"on_error": errors.append})
            if errors:
                with self._lock:
                    self.failures += 1
                return [self._stamp(f, session_id) for f in got]   # not cached
            cached = [_fact_to_dict(f) for f in got]
            with self._lock:
                self._mem[key] = cached
                with open(self._path, "a") as f:
                    f.write(json.dumps({"key": key, "facts": cached},
                                       ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
        return [self._stamp(_fact_from_dict(d), session_id) for d in cached]

    @staticmethod
    def _stamp(f: Fact, session_id: str) -> Fact:
        # A cached session can sit in another question's haystack under
        # the same content; the id is this question's.
        f.metadata["session_id"] = session_id
        return f

    def stats(self) -> dict:
        return {"fact_extraction_calls": self.calls,
                "fact_cache_hits": self.hits,
                "fact_extraction_failures": self.failures,
                "fact_cache_file": str(self._path)}
