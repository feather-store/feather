"""Compaction without freezing the DB, slot reuse, and exclusive-section fixes (0.19).

* compact() builds the new indexes WITHOUT holding the data lock and replays
  writes made meanwhile before swapping; auto-compaction runs on a background
  thread instead of inside the forget() that crossed the threshold.
* Indexes reuse the slots of deleted vectors, so tombstones stop accumulating.
* update_metadata() retires reverse-index entries in O(edges of the record).
"""
import threading
import time

import numpy as np

import feather_db
from feather_db import DB


def _m(content="x", ns="n"):
    m = feather_db.Metadata()
    m.content = content
    m.namespace_id = ns
    return m


def _vecs(n, dim, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def test_writes_during_compaction_are_kept(tmp_path):
    db = DB.open(str(tmp_path / "c.feather"), dim=32)
    n = 6000
    V = _vecs(n, 32)
    db.add_batch(list(range(n)), V, [_m(f"doc {i}") for i in range(n)])
    for i in range(0, n, 3):
        db.forget(i)

    added, forgotten = [], []
    stop = threading.Event()

    def writer():
        k = 0
        W = _vecs(2000, 32, seed=7)
        while not stop.is_set() and k < 2000:
            rid = 100_000 + k
            db.add(rid, W[k], _m(f"new {k}"))
            added.append((rid, W[k]))
            victim = 1 + 3 * k           # live before compaction
            if victim < n:
                db.forget(victim)
                forgotten.append(victim)
            k += 1

    t = threading.Thread(target=writer)
    t.start()
    time.sleep(0.05)
    removed = db.compact()
    stop.set()
    t.join()
    assert removed >= n // 3 - 1
    for rid, v in added:
        hits = [r.id for r in db.search(v, k=1, record_salience=False)]
        assert hits == [rid], f"record {rid} added during compaction is not searchable"
    ids = set(db.get_all_ids())
    assert not ids & set(forgotten), "a record forgotten during compaction came back"
    db.close()


def test_reads_continue_during_compaction(tmp_path):
    db = DB.open(str(tmp_path / "r.feather"), dim=64)
    n = 20_000
    db.add_batch(list(range(n)), _vecs(n, 64), [])
    for i in range(0, n, 4):
        db.forget(i)
    q = _vecs(1, 64, seed=3)[0]
    lat = []
    done = threading.Event()

    def reader():
        while not done.is_set():
            t0 = time.perf_counter()
            db.search(q, k=10, record_salience=False)
            lat.append(time.perf_counter() - t0)

    r = threading.Thread(target=reader)
    r.start()
    t0 = time.perf_counter()
    db.compact()
    compact_s = time.perf_counter() - t0
    done.set()
    r.join()
    assert max(lat) < max(0.5 * compact_s, 0.05), (
        f"a read waited {max(lat):.3f}s during a {compact_s:.3f}s compaction")
    db.close()


def test_auto_compaction_does_not_block_forget(tmp_path):
    db = DB.open(str(tmp_path / "a.feather"), dim=64)
    n = 20_000
    db.add_batch(list(range(n)), _vecs(n, 64), [])
    db.set_auto_compact(0.10)
    worst = 0.0
    for i in range(0, int(n * 0.12)):
        t0 = time.perf_counter()
        db.forget(i)
        worst = max(worst, time.perf_counter() - t0)
    assert db.wait_for_compaction(timeout=120)
    assert worst < 0.5, f"a forget() blocked {worst:.2f}s (compaction ran inline?)"
    assert db.get_metadata(0) is None, "auto-compaction never ran"
    db.close()


def test_close_during_background_compaction(tmp_path):
    p = str(tmp_path / "cl.feather")
    db = DB.open(p, dim=32)
    n = 20_000
    db.add_batch(list(range(n)), _vecs(n, 32), [])
    db.set_auto_compact(0.05)
    for i in range(0, n // 10):
        db.forget(i)
    db.close()                       # joins the compactor
    db2 = DB.open(p, dim=32)
    assert db2.size() in (n - n // 10, n)   # compacted or not, never corrupt
    assert db2.get_metadata(n - 1) is not None
    db2.close()


def test_deleted_slots_are_reused(tmp_path):
    db = DB.open(str(tmp_path / "s.feather"), dim=16)
    n = 2000
    db.add_batch(list(range(n)), _vecs(n, 16), [])
    for i in range(n // 2):
        db.forget(i)
    V = _vecs(n // 2, 16, seed=9)
    for k in range(n // 2):
        db.add(10_000 + k, V[k], _m())
    # every new record took a freed slot: the index did not grow
    assert len(db.get_all_ids()) == n
    for k in (0, 17, n // 2 - 1):
        assert [r.id for r in db.search(V[k], k=1, record_salience=False)] == [10_000 + k]
    db.close()


def test_forgotten_id_can_be_re_added(tmp_path):
    db = DB.open(str(tmp_path / "re.feather"), dim=8)
    db.add(1, np.ones(8, dtype=np.float32), _m("old"))
    db.forget(1)
    v = np.arange(8, dtype=np.float32)
    db.add(1, v, _m("new"))
    assert db.get_metadata(1).content == "new"
    assert [r.id for r in db.search(v, k=1)] == [1]
    db.close()


def test_update_metadata_keeps_reverse_index_exact(tmp_path):
    db = DB.open(str(tmp_path / "u.feather"), dim=8)
    for i in (1, 2, 3, 4):
        db.add(i, np.full(8, i, dtype=np.float32), _m())
    db.link(1, 2, "rel")
    db.link(1, 3, "rel")
    m = db.get_metadata(1)
    m.edges = [feather_db.Edge(3, "rel", 1.0), feather_db.Edge(4, "rel", 1.0)]
    db.update_metadata(1, m)
    assert [e.source_id for e in db.get_incoming(2)] == []
    assert [e.source_id for e in db.get_incoming(3)] == [1]
    assert [e.source_id for e in db.get_incoming(4)] == [1]
    db.close()


def test_add_with_edges_maintains_reverse_index(tmp_path):
    db = DB.open(str(tmp_path / "ae.feather"), dim=8)
    db.add(2, np.ones(8, dtype=np.float32), _m())
    m = _m()
    m.edges = [feather_db.Edge(2, "cites", 1.0)]
    db.add(1, np.zeros(8, dtype=np.float32) + 0.5, m)
    assert [e.source_id for e in db.get_incoming(2)] == [1]
    db.close()


def test_add_batch_does_not_starve_readers(tmp_path):
    """add_batch used to hold the exclusive lock for the whole parallel graph
    build; it now releases it between chunks."""
    db = DB.open(str(tmp_path / "b.feather"), dim=64)
    db.add_batch(list(range(5000)), _vecs(5000, 64), [])
    q = _vecs(1, 64, seed=5)[0]
    lat = []
    done = threading.Event()

    def reader():
        while not done.is_set():
            t0 = time.perf_counter()
            db.search(q, k=10, record_salience=False)
            lat.append(time.perf_counter() - t0)

    r = threading.Thread(target=reader)
    r.start()
    t0 = time.perf_counter()
    db.add_batch(list(range(10_000, 30_000)), _vecs(20_000, 64, seed=6), [])
    batch_s = time.perf_counter() - t0
    done.set()
    r.join()
    assert len(lat) > 10, "reader made almost no progress during add_batch"
    assert max(lat) < max(batch_s / 4, 0.05), (
        f"a read waited {max(lat):.3f}s during a {batch_s:.3f}s add_batch")
    assert db.size() == 25_000
    db.close()
