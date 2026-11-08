"""Ingest: write memory now, index it later.

Extraction is on the write path: one LLM call per session (~$0.0008, seconds
each), and an index made of what it returns (facts as index, memory as
payload; see feather_db.pocket). Doing it inline makes every write wait on a
model and lose its index when the model fails. So:

    q = IngestQueue(store_or_pocket_or_db, SessionFactExtractor(provider))
    q.remember(pocket, "k", "Alice moved to Denver")      # written now
    mem.ingest_session(turns, date, queue=q)              # notes written now
    q.drain()                                             # (tests, shutdown)

What a write costs is the write. The memory is readable, and found by its own
words, at once; the index (and the supersession it carries) follows when a
worker has made it.

Jobs are records in the database (source `_pocket_job`, in the scope they
index), so a crash loses none: a queue that opens the file resumes them. One
job per memory or session (by key), so submitting again replaces a pending
job instead of queueing a second. A memory rewritten while its extraction ran
is not given an index of its old text (the index write is conditional on the
memory's version; it runs again).

Many workers, many processes. A worker CLAIMS a job before running it: a
lease (`lease_seconds`) written with compare-and-set, so of two queues that
reach one job (two processes over the Cloud API, which is one writer for
them all) one runs it. A claim names its process (host and pid): a queue
on the same host takes over a dead process's claims at once, and anyone
else's when the lease expires. A local file has one writing process, so
there the workers are threads.

A crash never pays the model twice for one input. What an extraction returns
is written to the job (its index entries, and the version of each memory it
read) before the index is; a job run again after a crash uses it if the
memories are unchanged. The job then holds extracted statements, so it links
to each memory it read (`derived_from`): erasing the memory takes the job.

Failures retry with exponential backoff (`backoff` x 2^n, capped, jittered).
An extractor that reports an error (SessionFactExtractor's `on_error`) counts
as a failure, not as "no facts". After `max_attempts` a job is `failed`, kept
with its error (`failed()`, `retry_failed()`).

The index is versioned derived data. Each memory records the extractor that
indexed it (`name@version+fingerprint`: the fingerprint covers the prompt,
limits and model) and the job that indexes it. When the extractor changes,
indexes made by the old one are re-made lazily, two ways. A read that returns
one (Pocket.recall / hot, FeatherStore search) puts its job at the front of
the queue: what is being used is re-made first, and the read itself neither
writes nor waits. And an idle worker re-queues the rest, `reindex_batch` at a
time. `reindex_stale()` does it all on demand.
"""
from __future__ import annotations

import datetime as _dt
import heapq
import importlib
import itertools
import json
import os
import random
import socket
import threading
import time
import uuid
from typing import Any, Callable, Iterable, Optional

import numpy as np

import feather_db
from feather_db import keys

# The module: `feather_db.pocket` is also the name of the factory function.
P = importlib.import_module("feather_db.pocket")

PENDING, FAILED = "pending", "failed"
JOB_STATE    = "_pocket_job_state"
JOB_KIND     = "_pocket_job_kind"
JOB_PAYLOAD  = "_pocket_job_payload"
JOB_ATTEMPTS = "_pocket_job_attempts"
JOB_NEXT_AT  = "_pocket_job_next_at"
JOB_ERROR    = "_pocket_job_error"
JOB_LEASE    = "_pocket_job_lease_until"   # a claim: no other worker runs it before then
JOB_OWNER    = "_pocket_job_owner"
JOB_RESULT   = "_pocket_job_result"        # the extraction, kept until the index is written
_SEP = P._KEY_SEP
MAX_CONFLICTS = 5          # a memory rewritten during extraction, this often, then an attempt


class ExtractionFailed(RuntimeError):
    """The extractor reported an error: retried, never read as 'no facts'."""


def extractor_id(extractor) -> str:
    """`name@version` plus the extractor's fingerprint if it has one."""
    name = getattr(extractor, "name", type(extractor).__name__)
    version = getattr(extractor, "version", "0")
    fp = extractor.fingerprint() if hasattr(extractor, "fingerprint") else ""
    return f"{name}@{version}" + (f"+{fp}" if fp else "")


def job_key(kind: str, ident: str) -> str:
    return f"{_SEP}job{_SEP}{kind}{_SEP}{ident}"


def _memory_text(meta) -> tuple[str, str]:
    """(role, text) of a memory: a FeatherStore item's value `text` (a note
    stored as "role: text" loses the prefix) or a Pocket memory's content."""
    raw = meta.get_attribute("_lg_value")
    if raw:
        try:
            value = json.loads(raw)
        except ValueError:
            value = {}
        text = str(value.get("text") or meta.content or "")
        role = str(value.get("role") or "")
        if role and text.startswith(role + ": "):
            text = text[len(role) + 2:]
        return role or "user", text
    return "user", meta.content


def _when(x) -> Optional[_dt.datetime]:
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)):
        return _dt.datetime.fromtimestamp(float(x))
    from feather_db.extractors.session_facts import _parse_day
    return _parse_day(x)


_HOST = socket.gethostname()


def _owner_dead(owner: str) -> bool:
    """True when the claim's process is known to be gone: on this host, its
    pid no longer runs. Anything else (another host, a pid reused) is left
    to its lease."""
    host, _, rest = owner.partition(":")
    pid = rest.partition(":")[0]
    if host != _HOST or not pid.isdigit() or int(pid) == os.getpid():
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except OSError:                     # exists, not ours to signal
        return False
    return False


def _entry_dict(e) -> dict:
    return {"text": e.text, "key": e.key, "value": e.value, "valid_at": e.valid_at}


class IngestQueue:
    """Indexes memory in the background (see the module docstring).

    `target`: a FeatherStore (its db and embedder), a Pocket (its db and
    embedder) or a DB / RemoteDB (`embed` for the index vectors, else a
    placeholder: keyword search only). `workers` threads; jobs due no sooner
    than `backoff` x 2^(attempts-1) seconds after a failure (at most
    `max_backoff`); `max_attempts` tries; a claim lasts `lease_seconds`."""

    def __init__(self, target, extractor, *, embed: Optional[Callable[[str], Any]] = None,
                 workers: int = 2, max_attempts: int = 5, backoff: float = 2.0,
                 max_backoff: float = 300.0, poll_seconds: float = 5.0,
                 reindex: bool = True, reindex_batch: int = 20,
                 lease_seconds: float = 300.0, start: bool = True):
        if hasattr(target, "index_vector"):                       # FeatherStore
            self.db, self._vec = target.db, target.index_vector
        elif hasattr(target, "_vec") and hasattr(target, "namespace"):   # Pocket
            self.db, self._vec = target.db, target._vec
        else:
            self.db = target
            dim = target.dim("text")
            self._vec = ((lambda t: np.asarray(embed(t), dtype=np.float32)) if embed
                         else (lambda t, d=dim: np.eye(1, d, dtype=np.float32)[0]))
        self.extractor = extractor
        self.extractor_id = extractor_id(extractor)
        self.owner = f"{_HOST}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.max_attempts, self.backoff, self.max_backoff = max_attempts, backoff, max_backoff
        self.poll_seconds, self.reindex, self.reindex_batch = poll_seconds, reindex, reindex_batch
        self.lease_seconds = lease_seconds
        self._heap: list = []
        self._seq = itertools.count()
        self._scheduled: dict = {}       # (ns, job key) -> when it is due here
        self._running: set = set()
        self._conflicts: dict = {}
        self._nudges: dict = {}          # (ns, job ref) -> job args, seen stale on a read
        self._nudged: set = set()
        self._cv = threading.Condition()
        self._stop = False
        self._threads: list = []
        self._workers = workers
        self._last_poll = 0.0
        self._last_sweep = 0.0
        self._sweep_lock = threading.Lock()
        self.counts = {"done": 0, "retried": 0, "failed": 0, "requeued_stale": 0,
                       "reused_extraction": 0, "lost_claims": 0, "nudged": 0}
        if start:
            self.start()

    # ── submitting ───────────────────────────────────────────────────────

    def submit_memory(self, namespace: str, key: str, *, user_name: Optional[str] = None) -> str:
        """Index the memory at (namespace, key) from what it says: one
        extractor call. Replaces a pending job for it. Returns the job key."""
        return self._submit(namespace, "memory", key, {"key": key, "user_name": user_name})

    def submit_session(self, namespace: str, note_keys: Iterable[str], session_date: Any, *,
                       session_id: str, user_name: Optional[str] = None) -> str:
        """Index a session's notes (one per turn, in order) from one
        extractor call over the session: each fact indexes the turn it
        cites. Replaces a pending job for the session."""
        if isinstance(session_date, _dt.datetime):
            session_date = session_date.isoformat()
        return self._submit(namespace, "session", str(session_id),
                            {"keys": list(note_keys), "session_date": session_date,
                             "session_id": str(session_id), "user_name": user_name})

    def remember(self, pocket, key: str, content: str, *, user_name: Optional[str] = None,
                 **kw) -> int:
        """Pocket.remember() now, its index later."""
        if "index" in kw:
            raise ValueError("an index given now needs no queue: use Pocket.remember(index=)")
        rid = pocket.remember(key, content, **kw)
        self.submit_memory(pocket.namespace, key, user_name=user_name)
        return rid

    def _submit(self, namespace: str, kind: str, ident: str, payload: dict,
                at: Optional[float] = None) -> str:
        jk = job_key(kind, ident)
        m = feather_db.Metadata()
        m.timestamp = int(time.time())
        m.namespace_id, m.key = namespace, jk
        m.source = P._JOB_KIND
        m.importance = 0.0
        m.set_attribute(JOB_STATE, PENDING)
        m.set_attribute(JOB_KIND, kind)
        m.set_attribute(JOB_PAYLOAD, json.dumps(payload))
        m.set_attribute(JOB_ATTEMPTS, "0")
        m.set_attribute(JOB_NEXT_AT, repr(time.time()))
        self.db.write(feather_db.WriteBatch().upsert(m))
        self._schedule(namespace, jk, time.time() if at is None else at)
        return jk

    # ── reads that find a stale index ────────────────────────────────────

    def stale_seen(self, meta) -> None:
        """Called by reads (pocket.note_stale) for each memory they return:
        one indexed by another extractor goes to the front of the queue. No
        I/O here: a worker submits it."""
        v = meta.get_attribute(P.INDEX_V)
        if not v or v == self.extractor_id:
            return
        ref, args = meta.get_attribute(P.INDEX_JOB), meta.get_attribute(P.INDEX_JOB_ARGS)
        if not ref or not args:
            return
        k = (meta.namespace_id, ref)
        with self._cv:
            if k in self._nudged:
                return
            self._nudged.add(k)
            self._nudges[k] = args
            self._cv.notify()

    def _take_nudges(self) -> None:
        with self._cv:
            nudges = dict(self._nudges)
        try:
            for ns, kind, ident, payload in self._unqueued(nudges.items()):
                self._submit(ns, kind, ident, payload, at=0.0)  # ahead of everything due
                self.counts["nudged"] += 1
        finally:
            with self._cv:              # held until submitted: drain() waits on them
                for k in nudges:
                    self._nudges.pop(k, None)

    def _unqueued(self, refs) -> list:
        """(namespace, kind, ident, payload) for each ((namespace, job ref),
        job args) with no job on file: one that is pending (perhaps running,
        perhaps waiting out a backoff) or failed is left as it is.
        Resubmitting it would clear its claim and its attempts, so a running
        job would run twice and a failed one forever. One lookup for all."""
        refs = list(refs)
        if not refs:
            return []
        held = self.db.ids_for_many([r for r, _ in refs])
        out = []
        for ((ns, _ref), args), rid in zip(refs, held):
            if rid is not None:
                continue
            payload = json.loads(args)
            kind = payload.pop("kind", "memory")
            out.append((ns, kind, payload["key"] if kind == "memory" else payload["session_id"],
                        payload))
        return out

    # ── the workers ──────────────────────────────────────────────────────

    def start(self) -> "IngestQueue":
        if not self._threads:
            self._stop = False
            P.register_indexer(self.db, self)
            self._poll_db()
            for i in range(self._workers):
                t = threading.Thread(target=self._loop, name=f"feather-ingest-{i}", daemon=True)
                t.start()
                self._threads.append(t)
        return self

    def close(self, wait: bool = True) -> None:
        """Stop the workers. Unfinished jobs stay pending in the database;
        a claim this queue holds expires and another queue takes it."""
        P.unregister_indexer(self.db, self)
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if wait:
            for t in self._threads:
                t.join()
        self._threads = []

    def __enter__(self) -> "IngestQueue":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _schedule(self, ns: str, jk: str, at: float) -> None:
        """Due at `at`, or sooner if it already is: a job resubmitted while
        it waits out a backoff runs now. The later heap entry is skipped."""
        with self._cv:
            cur = self._scheduled.get((ns, jk))
            if cur is None or at < cur:
                self._scheduled[(ns, jk)] = at
                heapq.heappush(self._heap, (at, next(self._seq), ns, jk))
            self._cv.notify()

    def _poll_db(self) -> None:
        """Pending jobs in the database (another process wrote them, or a
        crash left them): scheduled when due, and a claim another worker
        holds when it expires."""
        self._last_poll = time.time()
        now = time.time()
        ids = list(self.db.ids_with_attribute(JOB_STATE, PENDING))
        for m in self.db.get_metadata_many(ids) if ids else []:
            if m is None or m.source != P._JOB_KIND:
                continue
            due = float(m.get_attribute(JOB_NEXT_AT) or 0)
            if self._held(m, now):
                due = max(due, float(m.get_attribute(JOB_LEASE)))
            self._schedule(m.namespace_id, m.key, due)

    def _held(self, job, now: float) -> bool:
        """Another live worker's claim is on the job."""
        lease = float(job.get_attribute(JOB_LEASE) or 0)
        owner = job.get_attribute(JOB_OWNER)
        return lease > now and owner != self.owner and not _owner_dead(owner)

    def _loop(self) -> None:
        while True:
            got = self._next()
            if got is None:
                return
            ns, jk = got
            try:
                self._run(ns, jk)
            except Exception:              # never kill a worker; the job record says why
                pass
            finally:
                with self._cv:
                    self._running.discard((ns, jk))
                    self._cv.notify_all()

    def _next(self):
        """The next due job, marked running; None once stopped. Between
        jobs: submits what reads found stale, re-reads pending jobs from the
        database every poll_seconds, and when idle sweeps for stale indexes."""
        while True:
            with self._cv:
                if self._stop:
                    return None
                now = time.time()
                while self._heap:
                    at, _, ns, jk = self._heap[0]
                    if self._scheduled.get((ns, jk)) != at:
                        heapq.heappop(self._heap)  # superseded by an earlier entry
                        continue
                    if at > now:
                        break
                    heapq.heappop(self._heap)
                    del self._scheduled[(ns, jk)]
                    # Running already: its finish sees a newer submission (a
                    # version conflict) and schedules it again.
                    if (ns, jk) not in self._running:
                        self._running.add((ns, jk))
                        return ns, jk
                wait = (self._heap[0][0] - now) if self._heap else self.poll_seconds
                # Idle: nothing to do before the next poll. A job waiting out
                # a backoff, or another queue's claim, does not stop a sweep.
                idle = not self._running and (not self._heap or
                                              self._heap[0][0] > now + self.poll_seconds)
                nudged = bool(self._nudges)
            if nudged:
                self._take_nudges()
                continue
            if time.time() - self._last_poll >= self.poll_seconds:
                self._poll_db()
            if idle and self.reindex and time.time() - self._last_sweep >= self.poll_seconds:
                self._sweep()
            with self._cv:
                if not self._stop and not self._nudges and \
                        not (self._heap and self._heap[0][0] <= time.time()):
                    self._cv.wait(timeout=max(0.01, min(wait, self.poll_seconds)))

    # ── one job ──────────────────────────────────────────────────────────

    def _claim(self, ns: str, jk: str):
        """The job, claimed by this queue (a lease, compare-and-set), or None
        when it is gone, failed, or another worker holds a live claim."""
        job = self.db.get_by_key(jk, ns)
        if job is None or job.source != P._JOB_KIND or job.get_attribute(JOB_STATE) != PENDING:
            return None
        now = time.time()
        if self._held(job, now):
            self._schedule(ns, jk, float(job.get_attribute(JOB_LEASE)))   # look again then
            return None
        job.set_attribute(JOB_LEASE, repr(now + self.lease_seconds))
        job.set_attribute(JOB_OWNER, self.owner)
        try:
            self.db.write(feather_db.WriteBatch().update((ns, jk), job,
                                                         expected_version=job.version))
        except feather_db.VersionConflictError:
            self.counts["lost_claims"] += 1      # another worker, or a new submission
            self._schedule(ns, jk, time.time())
            return None
        return self.db.get_by_key(jk, ns)

    def _run(self, ns: str, jk: str) -> None:
        job = self._claim(ns, jk)
        if job is None:
            return
        try:
            outcome = self._index(ns, jk, job)
        except feather_db.VersionConflictError:
            n = self._conflicts.get((ns, jk), 0) + 1
            self._conflicts[(ns, jk)] = n
            if n < MAX_CONFLICTS:          # a memory changed while we read it: again, now
                self._release(ns, jk)
                return
            self._fail(ns, jk, "memory kept changing during extraction")
            return
        except Exception as e:
            self._fail(ns, jk, f"{type(e).__name__}: {e}")
            return
        self._conflicts.pop((ns, jk), None)
        self.counts["done"] += outcome is not None
        self._finish(ns, jk)

    def _index(self, ns: str, jk: str, job):
        """Extract (or reuse a saved extraction) and write the index. None
        if every memory is gone."""
        kind = job.get_attribute(JOB_KIND)
        payload = json.loads(job.get_attribute(JOB_PAYLOAD) or "{}")
        want = [payload["key"]] if kind == "memory" else list(payload.get("keys") or [])
        ids = keys.resolve_many(self.db, [(ns, k) for k in want])
        live = [(k, rid) for k, rid in zip(want, ids) if rid is not None]
        metas = self.db.get_metadata_many([rid for _, rid in live]) if live else []
        mem = [(k, m) for (k, _), m in zip(live, metas)
               if m is not None and m.source not in P._NOT_MEMORY]
        if not mem:
            return None
        by_key = self._saved(job, mem)
        if by_key is not None:
            self.counts["reused_extraction"] += 1
        else:
            by_key = self._extract(kind, payload, mem)
            self._checkpoint(ns, jk, by_key, mem)
        args = json.dumps(dict(payload, kind=kind))
        P.apply_index(self.db, [
            P.IndexWrite(ns, k, by_key[k], m, expected_version=m.version,
                         old_n=int(m.get_attribute(P.INDEX_N) or 0))
            for k, m in mem], self._vec,
            extractor_id=self.extractor_id, job_ref=jk, job_args=args)
        return sum(len(v) for v in by_key.values())

    def _extract(self, kind: str, payload: dict, mem: list) -> dict:
        """{memory key: [IndexEntry]} from one extractor call."""
        errors: list = []
        ns = mem[0][1].namespace_id
        ctx = {"known_keys": P.known_keys(self.db, ns), "user_name": payload.get("user_name"),
               "on_error": errors.append}
        if kind == "memory":
            k, m = mem[0]
            ctx["session_id"] = k
            ctx["session_date"] = _dt.datetime.fromtimestamp(m.valid_time)
            facts = self.extractor.extract(_memory_text(m)[1], context=ctx)
            by_key = {k: list(facts)}
        else:
            ctx["session_id"] = payload.get("session_id")
            turns = [dict(zip(("role", "content"), _memory_text(m))) for _, m in mem]
            facts = self.extractor.extract_session(turns, _when(payload.get("session_date")),
                                                   context=ctx)
            by_key = {k: [] for k, _ in mem}
            for f in facts:
                t = (f.metadata or {}).get("turn")
                at = t if isinstance(t, int) and 0 <= t < len(mem) else 0
                by_key[mem[at][0]].append(f)
        if errors:
            raise ExtractionFailed(str(errors[0]))
        return {k: [P.IndexEntry.of(f) for f in v] for k, v in by_key.items()}

    def _checkpoint(self, ns: str, jk: str, by_key: dict, mem: list) -> None:
        """Keep the extraction on the job, with the version of every memory it
        read, before the index is written: a crash after this never pays the
        model again for the same input. The job now holds what was extracted,
        so it links to each memory (`derived_from`): an erasure takes it."""
        job = self.db.get_by_key(jk, ns)
        if job is None:
            return
        job.set_attribute(JOB_RESULT, json.dumps({
            "extractor": self.extractor_id,
            "inputs": {k: m.version for k, m in mem},
            "entries": {k: [_entry_dict(e) for e in v] for k, v in by_key.items()}}))
        b = feather_db.WriteBatch().update((ns, jk), job, expected_version=job.version)
        for k, _ in mem:
            b.link((ns, jk), (ns, k), "derived_from")
        self.db.write(b)

    def _saved(self, job, mem: list) -> Optional[dict]:
        """The job's saved extraction, if it is this extractor's and every
        memory is still at the version it read."""
        raw = job.get_attribute(JOB_RESULT)
        if not raw:
            return None
        try:
            r = json.loads(raw)
        except ValueError:
            return None
        if r.get("extractor") != self.extractor_id or \
                {k: m.version for k, m in mem} != {k: int(v) for k, v in r["inputs"].items()}:
            return None
        return {k: [P.IndexEntry(**d) for d in r["entries"].get(k, [])] for k, _ in mem}

    def _release(self, ns: str, jk: str) -> None:
        """Give up the claim and run again now."""
        job = self.db.get_by_key(jk, ns)
        if job is not None and job.get_attribute(JOB_OWNER) == self.owner:
            job.set_attribute(JOB_LEASE, "0")
            try:
                self.db.write(feather_db.WriteBatch().update((ns, jk), job,
                                                             expected_version=job.version))
            except feather_db.VersionConflictError:
                pass
        self._schedule(ns, jk, time.time())

    def _finish(self, ns: str, jk: str) -> None:
        """Forget the job, only as this worker left it: a newer submission
        (a new version) stays and runs."""
        job = self.db.get_by_key(jk, ns)
        if job is None:
            return
        if job.get_attribute(JOB_OWNER) != self.owner:
            # Resubmitted while it ran (the upsert cleared the claim): run
            # the new submission. Taken over: _claim sees the live lease.
            self._schedule(ns, jk, time.time())
            return
        try:
            self.db.write(feather_db.WriteBatch().forget((ns, jk), expected_version=job.version))
        except feather_db.VersionConflictError:
            self._schedule(ns, jk, time.time())

    def _fail(self, ns: str, jk: str, error: str) -> None:
        job = self.db.get_by_key(jk, ns)
        if job is None:
            return
        attempts = int(job.get_attribute(JOB_ATTEMPTS) or 0) + 1
        job.set_attribute(JOB_ATTEMPTS, str(attempts))
        job.set_attribute(JOB_ERROR, error[:2000])
        job.set_attribute(JOB_LEASE, "0")
        if attempts >= self.max_attempts:
            job.set_attribute(JOB_STATE, FAILED)
            self.counts["failed"] += 1
        else:
            delay = min(self.max_backoff, self.backoff * 2 ** (attempts - 1))
            delay *= 0.75 + random.random() / 2
            job.set_attribute(JOB_NEXT_AT, repr(time.time() + delay))
            self.counts["retried"] += 1
        try:
            self.db.write(feather_db.WriteBatch().update((ns, jk), job,
                                                         expected_version=job.version))
        except feather_db.VersionConflictError:
            self._schedule(ns, jk, time.time())     # resubmitted meanwhile: run that
            return
        if attempts < self.max_attempts:
            self._schedule(ns, jk, float(job.get_attribute(JOB_NEXT_AT)))

    # ── stale indexes ────────────────────────────────────────────────────

    def reindex_stale(self, limit: Optional[int] = None) -> int:
        """Queue the jobs that re-make indexes another extractor made, at most
        `limit` (default all): one per session or memory, from the job each
        memory records. An extractor no memory is indexed by any more leaves
        the registry. Returns how many jobs it queued."""
        ids = list(self.db.ids_with_attribute(P.IXVER_ALL, "1"))
        regs = [(rid, m) for rid, m in zip(ids, self.db.get_metadata_many(ids) if ids else [])
                if m is not None and m.get_attribute(P.IXVER)
                and m.get_attribute(P.IXVER) != self.extractor_id]
        queued, seen = 0, set()
        for rid, reg in regs:
            ns, old = reg.namespace_id, reg.get_attribute(P.IXVER)
            mids = list(self.db.ids_with_attribute(P.INDEX_V, old))
            stale = [m for m in (self.db.get_metadata_many(mids) if mids else [])
                     if m is not None and m.namespace_id == ns and m.source not in P._NOT_MEMORY]
            if not stale:
                self.db.write(feather_db.WriteBatch().forget(rid))
                continue
            refs = {}
            for m in stale:
                ref, args = m.get_attribute(P.INDEX_JOB), m.get_attribute(P.INDEX_JOB_ARGS)
                if ref and args and (ns, ref) not in seen:
                    seen.add((ns, ref))
                    refs[(ns, ref)] = args
            for ns_, kind, ident, payload in self._unqueued(refs.items()):
                self._submit(ns_, kind, ident, payload)
                queued += 1
                if limit is not None and queued >= limit:
                    self.counts["requeued_stale"] += queued
                    return queued
        self.counts["requeued_stale"] += queued
        return queued

    def _sweep(self) -> None:
        if not self._sweep_lock.acquire(blocking=False):
            return
        try:
            self._last_sweep = time.time()
            self.reindex_stale(limit=self.reindex_batch)
        except Exception:
            pass
        finally:
            self._sweep_lock.release()

    # ── inspection ───────────────────────────────────────────────────────

    def _jobs(self, state: str) -> list:
        ids = list(self.db.ids_with_attribute(JOB_STATE, state))
        return [m for m in (self.db.get_metadata_many(ids) if ids else [])
                if m is not None and m.source == P._JOB_KIND]

    def failed(self) -> list[dict]:
        """Jobs out of attempts: where, what, how often, why."""
        return [{"namespace": m.namespace_id, "job": m.key,
                 "kind": m.get_attribute(JOB_KIND),
                 "attempts": int(m.get_attribute(JOB_ATTEMPTS) or 0),
                 "error": m.get_attribute(JOB_ERROR)} for m in self._jobs(FAILED)]

    def retry_failed(self) -> int:
        """Give every failed job its attempts back."""
        n = 0
        for m in self._jobs(FAILED):
            payload = json.loads(m.get_attribute(JOB_PAYLOAD) or "{}")
            kind = m.get_attribute(JOB_KIND)
            self._submit(m.namespace_id, kind,
                         payload["key"] if kind == "memory" else payload["session_id"], payload)
            n += 1
        return n

    def stats(self) -> dict:
        with self._cv:
            running, scheduled = len(self._running), len(self._scheduled)
        return {"pending": len(self._jobs(PENDING)), "failed": len(self._jobs(FAILED)),
                "running": running, "scheduled": scheduled, "extractor": self.extractor_id,
                "owner": self.owner, **self.counts}

    def drain(self, timeout: Optional[float] = None) -> bool:
        """Wait until no job is pending or running, here or in the database.
        True if that happened within `timeout`; failed jobs do not count."""
        end = None if timeout is None else time.time() + timeout
        while True:
            # The database says what is pending (here, in another process, or
            # waiting out a backoff); the heap may still name a job another
            # queue has finished.
            with self._cv:
                busy = bool(self._running) or bool(self._nudges)
            if not busy and not self._jobs(PENDING):
                return True
            if end is not None and time.time() >= end:
                return False
            with self._cv:
                self._cv.wait(timeout=0.02)
