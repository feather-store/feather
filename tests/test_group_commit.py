"""WAL group commit, non-blocking checkpoints and the new WAL paths (0.19).

* Writers append under the exclusive lock but fsync after releasing it, and one
  fsync covers every record appended before it started.
* save() rotates the WAL to <wal>.old at the snapshot point and makes the base
  file durable outside the lock; recovery replays <wal>.old then <wal>.
* purge() and auto_link() are WAL-logged (they used to survive only via save()).
"""
import os
import shutil
import threading

import numpy as np
import pytest

import feather_db
from feather_db import DB


def _m(content="x", ns="n"):
    m = feather_db.Metadata()
    m.content = content
    m.namespace_id = ns
    return m


def _v(dim=16, seed=None):
    return np.random.default_rng(seed).random(dim).astype(np.float32)


def test_concurrent_writers_share_fsyncs(tmp_path):
    db = DB.open(str(tmp_path / "gc.feather"), dim=16)
    before = db.wal_fsync_count()
    threads, per = 16, 40

    def writer(t):
        for i in range(per):
            db.add(t * 1000 + i, _v(), _m(f"t{t} i{i}"))

    ts = [threading.Thread(target=writer, args=(t,)) for t in range(threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    fsyncs = db.wal_fsync_count() - before
    assert db.size() == threads * per
    assert fsyncs < threads * per, f"{fsyncs} fsyncs for {threads * per} writes: no group commit"
    db.close()


def test_every_acknowledged_write_survives_a_crash(tmp_path):
    p = str(tmp_path / "ack.feather")
    db = DB.open(p, dim=16)
    acked = []
    lock = threading.Lock()

    def writer(t):
        for i in range(50):
            rid = t * 1000 + i
            db.add(rid, _v(), _m())
            with lock:
                acked.append(rid)

    ts = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
    saver_stop = threading.Event()

    def saver():                    # checkpoints racing the writers
        while not saver_stop.is_set():
            db.save()

    s = threading.Thread(target=saver)
    s.start()
    [t.start() for t in ts]
    [t.join() for t in ts]
    saver_stop.set()
    s.join()
    db.close(save=False)            # "crash": no final checkpoint
    db2 = DB.open(p, dim=16)
    missing = [r for r in acked if db2.get_metadata(r) is None]
    assert not missing, f"{len(missing)} acknowledged writes lost, e.g. {missing[:5]}"
    db2.close()


def test_save_leaves_no_rotated_wal_behind(tmp_path):
    p = str(tmp_path / "rot.feather")
    db = DB.open(p, dim=16)
    db.add(1, _v(), _m())
    db.save()
    assert not os.path.exists(p + ".wal.old")
    assert not os.path.exists(p + ".wal") or db.wal_size() == 0
    db.close()


def test_recovery_replays_old_then_current_wal(tmp_path):
    """Simulate a crash between WAL rotation and the base-file rename."""
    p = str(tmp_path / "old.feather")
    db = DB.open(p, dim=16)
    db.add(1, _v(), _m("first"))
    db.close(save=False)
    os.replace(p + ".wal", p + ".wal.old")          # rotated, base never published
    db = DB.open(p, dim=16)
    assert db.get_metadata(1) is not None           # replayed from .wal.old
    db.add(2, _v(), _m("second"))
    db.forget(1)
    db.close(save=False)                            # now both .wal.old and .wal exist
    assert os.path.exists(p + ".wal.old") and os.path.exists(p + ".wal")
    db = DB.open(p, dim=16)
    assert db.get_metadata(2) is not None
    assert db.get_metadata(1).source == "_forgotten", ".wal must replay AFTER .wal.old"
    db.save()                                       # synchronous path merges them away
    assert not os.path.exists(p + ".wal.old")
    db.close()


def test_purge_is_wal_logged(tmp_path):
    p = str(tmp_path / "purge.feather")
    db = DB.open(p, dim=16)
    for i in range(6):
        db.add(i, _v(seed=i), _m(f"doc {i}", ns="a" if i < 3 else "b"))
    db.save()
    assert db.purge("a") == 3
    db.close(save=False)                            # crash before any checkpoint
    db = DB.open(p, dim=16)
    assert sorted(db.all_ids()) == [3, 4, 5], "purge was lost on crash"
    assert not db.keyword_search("doc", k=10) or all(r.id >= 3 for r in db.keyword_search("doc", k=10))
    db.close()


def test_auto_link_is_wal_logged(tmp_path):
    p = str(tmp_path / "al.feather")
    db = DB.open(p, dim=4)
    base = np.array([1, 0, 0, 0], dtype=np.float32)
    for i in range(5):
        db.add(i, base + np.float32(i) * 0.001, _m())
    db.save()
    n = db.auto_link(threshold=0.5)
    assert n > 0
    db.close(save=False)
    db = DB.open(p, dim=4)
    assert sum(len(db.get_edges(i)) for i in range(5)) == n
    db.close()


def test_strict_mode_still_recovers(tmp_path):
    import subprocess, sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = str(tmp_path / "strict.feather")
    code = f"""
import sys; sys.path.insert(0, {repo!r})
import numpy as np, feather_db
db = feather_db.DB.open({p!r}, dim=8)
for i in range(20):
    db.add(i, np.ones(8, dtype=np.float32) * i, feather_db.Metadata())
db.close(save=False)
db = feather_db.DB.open({p!r}, dim=8); print("SIZE", db.size())
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "FEATHER_WAL_STRICT": "1"}, timeout=120)
    assert "SIZE 20" in out.stdout, out.stdout + out.stderr


def test_dimension_mismatch_is_rejected_before_logging(tmp_path):
    """A wrong-dim add used to be WAL-logged before the dim check, and replayed
    into the index (reading past the vector) on the next open."""
    p = str(tmp_path / "dim.feather")
    db = DB.open(p, dim=8)
    db.add(1, np.ones(8, dtype=np.float32), _m())
    with pytest.raises(RuntimeError, match="Dimension mismatch"):
        db.add(2, np.ones(5, dtype=np.float32), _m())
    db.close(save=False)
    db = DB.open(p, dim=8)
    assert db.get_metadata(2) is None and db.size() == 1
    db.close()
