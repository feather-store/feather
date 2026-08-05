"""Concurrency + keyword-index consistency regressions.

Covers three fixes:

1. Queries take the DB lock in SHARED mode and release the GIL, so N threads
   search in parallel. Previously every search held both the GIL and an
   exclusive mutex, capping the whole process at one core.
2. Salience counters are mutable atomics, so a query records its hit under the
   shared lock and the increment is visible immediately (no deferral).
3. forget()/purge() remove the record from the BM25 index. They previously left
   the postings behind, so a forgotten record stayed keyword-searchable and a
   purged one surfaced as a hit with empty metadata.
"""
import threading
import time

import numpy as np
import pytest

import feather_db
from feather_db import DB

from .conftest import EMBED


def _meta(content, ns="bench"):
    m = feather_db.Metadata()
    m.content = content
    m.namespace_id = ns
    m.timestamp = 1700000000
    return m


# ── 1. Parallel search ────────────────────────────────────────────────────

def test_concurrent_search_scales(tmp_path_feather):
    """8 threads must beat 1 thread on search throughput.

    The bound is deliberately loose (>1.5x) so this asserts "the lock and the
    GIL are released" without being a flaky performance test on a busy CI box.
    """
    db = DB.open(tmp_path_feather, dim=128)
    rng = np.random.default_rng(0)
    n = 4000
    db.add_batch(list(range(n)), rng.random((n, 128), dtype=np.float32),
                 [_meta(f"record {i} alpha beta") for i in range(n)])
    queries = rng.random((64, 128), dtype=np.float32)

    def bench(nthreads, per_thread=150):
        barrier = threading.Barrier(nthreads)

        def worker(t):
            barrier.wait()
            for i in range(per_thread):
                db.search(queries[(t * 7 + i) % len(queries)], k=10)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(nthreads)]
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        return nthreads * per_thread / (time.perf_counter() - t0)

    one = bench(1)
    many = bench(8)
    assert many > one * 1.5, f"no parallel speedup: 1t={one:.0f} qps, 8t={many:.0f} qps"


def test_concurrent_readers_and_writers_are_safe(tmp_path_feather):
    """Mixed read/write load must not crash, deadlock, or corrupt the DB."""
    db = DB.open(tmp_path_feather, dim=128)
    rng = np.random.default_rng(1)
    n = 2000
    db.add_batch(list(range(n)), rng.random((n, 128), dtype=np.float32),
                 [_meta(f"record {i} alpha beta") for i in range(n)])

    stop = threading.Event()
    errors = []

    def reader():
        try:
            while not stop.is_set():
                db.search(rng.random(128, dtype=np.float32), k=5)
                db.keyword_search("alpha beta", k=5)
                db.hybrid_search(rng.random(128, dtype=np.float32), "alpha", k=5)
                db.context_chain(rng.random(128, dtype=np.float32), k=3, hops=1)
        except Exception as e:  # noqa: BLE001
            errors.append(("reader", repr(e)))

    def writer(start):
        try:
            i = start
            while not stop.is_set():
                db.add(i, rng.random(128, dtype=np.float32), _meta(f"live {i} gamma"))
                db.link(i, i - 1, "related_to", 0.5)
                db.update_importance(i, 0.5)
                i += 100
        except Exception as e:  # noqa: BLE001
            errors.append(("writer", repr(e)))

    threads = [threading.Thread(target=reader) for _ in range(3)]
    threads += [threading.Thread(target=writer, args=(10_000 + j,)) for j in range(2)]
    for t in threads: t.start()
    time.sleep(1.5)
    stop.set()
    for t in threads: t.join(timeout=30)

    assert not errors, errors
    assert all(not t.is_alive() for t in threads), "a thread failed to join (deadlock?)"
    assert len(db.search(rng.random(128, dtype=np.float32), k=5)) == 5


# ── 2. Salience is immediate ──────────────────────────────────────────────

def test_search_touch_is_visible_immediately(db):
    """recall_count must be readable right after the search that caused it."""
    db.add(id=1, vec=EMBED("hello"), meta=_meta("hello world"))
    before = db.get_metadata(1).recall_count
    db.search(EMBED("hello"), k=1)
    assert db.get_metadata(1).recall_count == before + 1


def test_concurrent_touches_are_not_lost(tmp_path_feather):
    """Atomic counters: parallel searches over the same record must not drop
    increments the way a plain uint32 would."""
    db = DB.open(tmp_path_feather, dim=128)
    db.add(id=1, vec=EMBED("only"), meta=_meta("only record"))
    per_thread, nthreads = 200, 4

    def worker():
        for _ in range(per_thread):
            db.search(EMBED("only"), k=1)

    threads = [threading.Thread(target=worker) for _ in range(nthreads)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert db.get_metadata(1).recall_count == per_thread * nthreads


# ── 3. Keyword index tracks record lifecycle ──────────────────────────────

def test_forget_removes_record_from_keyword_index(db):
    db.add(id=1, vec=EMBED("a"), meta=_meta("quarterly roas report"))
    db.add(id=2, vec=EMBED("b"), meta=_meta("weekly roas summary"))
    assert {r.id for r in db.keyword_search("roas", k=10)} == {1, 2}

    db.forget(1)
    hits = db.keyword_search("roas", k=10)
    assert {r.id for r in hits} == {2}, "forgotten record still keyword-searchable"


def test_purge_removes_records_from_keyword_index(db):
    db.add(id=1, vec=EMBED("a"), meta=_meta("campaign alpha", ns="doomed"))
    db.add(id=2, vec=EMBED("b"), meta=_meta("campaign beta", ns="kept"))
    assert len(db.keyword_search("campaign", k=10)) == 2

    db.purge("doomed")
    hits = db.keyword_search("campaign", k=10)
    assert [r.id for r in hits] == [2]
    # A stale posting would come back as a hit with empty metadata.
    assert all(r.metadata.content for r in hits), "ghost hit with no metadata"


def test_hybrid_search_excludes_forgotten(db):
    db.add(id=1, vec=EMBED("a"), meta=_meta("retention cohort analysis"))
    db.add(id=2, vec=EMBED("b"), meta=_meta("retention funnel analysis"))
    db.forget(1)
    hits = db.hybrid_search(EMBED("a"), "retention analysis", k=10)
    assert 1 not in {r.id for r in hits}


# ── 4. Incremental BM25 statistics stay correct ───────────────────────────

def test_incremental_bm25_matches_rebuilt_index(tmp_path):
    """avg_dl is maintained as a running total instead of being re-summed on
    every insert. A drifting total would change BM25 scores, so an incrementally
    built index must score identically to one built fresh from the same content.
    """
    docs = {
        1: "short campaign note",
        2: "a considerably longer campaign note with many additional filler terms here",
        3: "campaign budget pacing update",
        4: "unrelated creative brief",
    }

    # Incremental: insert, then overwrite two docs with different lengths,
    # then forget one — exercising every path that adjusts the running total.
    inc = DB.open(str(tmp_path / "inc.feather"), dim=128)
    for i, text in docs.items():
        inc.add(id=i, vec=EMBED(str(i)), meta=_meta("placeholder text"))
    for i, text in docs.items():
        inc.add(id=i, vec=EMBED(str(i)), meta=_meta(text))
    inc.add(id=5, vec=EMBED("5"), meta=_meta("campaign temporary record"))
    inc.forget(5)

    # Fresh: the same final content, indexed once.
    fresh = DB.open(str(tmp_path / "fresh.feather"), dim=128)
    for i, text in docs.items():
        fresh.add(id=i, vec=EMBED(str(i)), meta=_meta(text))

    a = [(r.id, round(r.score, 5)) for r in inc.keyword_search("campaign note", k=10)]
    b = [(r.id, round(r.score, 5)) for r in fresh.keyword_search("campaign note", k=10)]
    assert a == b, f"incremental BM25 diverged from rebuilt index:\n{a}\n{b}"


def test_bm25_survives_save_reload(tmp_path):
    """Index rebuild on load must reproduce the in-memory scores."""
    path = str(tmp_path / "reload.feather")
    db = DB.open(path, dim=128)
    for i, text in enumerate(["alpha beta gamma", "beta gamma delta", "gamma only"], start=1):
        db.add(id=i, vec=EMBED(str(i)), meta=_meta(text))
    db.forget(3)
    before = [(r.id, round(r.score, 5)) for r in db.keyword_search("beta gamma", k=10)]
    db.save()

    reloaded = DB.open(path, dim=128)
    after = [(r.id, round(r.score, 5)) for r in reloaded.keyword_search("beta gamma", k=10)]
    assert before == after
