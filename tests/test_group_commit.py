"""WAL group commit, non-blocking checkpoints and the new WAL paths (0.19).

* Writers append under the exclusive lock but fsync after releasing it, and one
  fsync covers every record appended before it started.
* save() rotates the WAL to <wal>.old at the snapshot point and makes the base
  file durable outside the lock; recovery replays <wal>.old then <wal>.
* purge() and auto_link() are WAL-logged (they used to survive only via save()).
"""
import os
import shutil
import sys
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


_GROUP_COMMIT = """
import sys, threading
sys.path.insert(0, {repo!r})
import numpy as np, feather_db
db = feather_db.DB.open({path!r}, dim=16)
threads, per, rounds = 16, 40, []
for r in range(3):
    before = db.wal_fsync_count()
    def writer(t):
        for i in range(per):
            m = feather_db.Metadata(); m.content = "x"
            db.add(r * 100_000 + t * 1000 + i, np.random.default_rng().random(16).astype(np.float32), m)
    ts = [threading.Thread(target=writer, args=(t,)) for t in range(threads)]
    [t.start() for t in ts]; [t.join() for t in ts]
    rounds.append(db.wal_fsync_count() - before)
    assert db.size() == (r + 1) * threads * per
    if rounds[-1] < threads * per:
        break
db.close()
print("ROUNDS", rounds)
"""


def test_concurrent_writers_share_fsyncs(tmp_path):
    """Without group commit every write gets its own fsync, every time, so one
    round that shares any proves it; whether writers overlap in a given round
    is up to the scheduler, so best of 3 with early exit — the same shape as
    test_concurrent_search_scales.

    It runs in a subprocess with FEATHER_WAL_FULLFSYNC=1 (read once per
    process). On macOS plain fsync() skips the drive cache and takes ~25us, so
    writers almost never overlap it: measured 611-640 fsyncs per 640 writes,
    and 640 in every round once other suite files had run. F_FULLFSYNC costs
    what fsync costs on Linux, where the variable is a no-op.
    """
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = _GROUP_COMMIT.format(repo=repo, path=str(tmp_path / "gc.feather"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "FEATHER_WAL_FULLFSYNC": "1"}, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    rounds = [int(x) for x in out.stdout.split("ROUNDS", 1)[1].strip(" []\n").split(",")]
    assert min(rounds) < 16 * 40, (
        f"fsyncs per round {rounds} for 640 writes each: one fsync per write in "
        f"every round means writers never share one — no group commit")


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
    assert db.get_metadata(1) is None, ".wal must replay AFTER .wal.old"
    assert db.get_metadata(1, include_dead=True).source == "_forgotten"
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
