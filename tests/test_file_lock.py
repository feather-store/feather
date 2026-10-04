"""Inter-process file lock + close(save=...).

Two PROCESSES on one file would each replay and append to the same WAL and each
overwrite the other's checkpoints. DB.open() takes an exclusive OS lock on
<path>.lock and refuses a second process; close() (or a `with` block) releases
it. The lock is reentrant within one process (v0.20.0): a second open() in the
same process is admitted. tests/test_file_locking.py covers the lock itself;
this file covers close(save=...) and the paths that must release the handle.
"""
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

import feather_db
from feather_db import DB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _v(dim=16, seed=0):
    return np.random.default_rng(seed).random(dim).astype(np.float32)


def _child(src, timeout=60):
    code = f"import sys; sys.path.insert(0, {REPO!r})\n" + src
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=timeout)


def test_second_open_in_same_process_is_reentrant(tmp_path):
    """Same-process double-open is admitted (the lock is per process); the file
    stays locked against OTHER processes until the last handle closes."""
    p = str(tmp_path / "a.feather")
    db = DB.open(p, dim=16)
    db2 = DB.open(p, dim=16)
    db2.close(save=False)
    proc = _child(f"""
from feather_db import DB
try:
    DB.open({p!r}, dim=16); print("OPENED")
except RuntimeError as e:
    print("REFUSED", e)
""")
    assert "REFUSED" in proc.stdout, proc.stdout + proc.stderr
    db.close()


def test_close_releases_lock_and_persists(tmp_path):
    p = str(tmp_path / "b.feather")
    db = DB.open(p, dim=16)
    db.add(1, _v(), feather_db.Metadata())
    db.close()
    assert db.closed
    db2 = DB.open(p, dim=16)
    assert db2.size() == 1
    db2.close()


def test_closed_db_rejects_writes(tmp_path):
    db = DB.open(str(tmp_path / "c.feather"), dim=16)
    db.close()
    with pytest.raises(Exception, match="closed"):
        db.add(1, _v(), feather_db.Metadata())
    db.close()   # idempotent


def test_context_manager_checkpoints_and_releases(tmp_path):
    p = str(tmp_path / "d.feather")
    with DB.open(p, dim=16) as db:
        db.add(7, _v(), feather_db.Metadata())
    assert not os.path.exists(p + ".wal"), "close() should checkpoint (WAL cleared)"
    with DB.open(p, dim=16) as db2:
        assert db2.get_metadata(7) is not None


def test_close_without_save_keeps_the_wal(tmp_path):
    """close(save=False) models a crash: no checkpoint, WAL left for replay."""
    p = str(tmp_path / "e.feather")
    db = DB.open(p, dim=16)
    db.add(3, _v(), feather_db.Metadata())
    db.close(save=False)
    assert os.path.getsize(p + ".wal") > 0
    db2 = DB.open(p, dim=16)
    assert db2.get_metadata(3) is not None
    db2.close()


def test_second_process_is_refused(tmp_path):
    p = str(tmp_path / "f.feather")
    db = DB.open(p, dim=16)
    proc = _child(f"""
from feather_db import DB
try:
    DB.open({p!r}, dim=16); print("OPENED")
except RuntimeError as e:
    print("REFUSED", e)
""")
    assert "REFUSED" in proc.stdout, proc.stdout + proc.stderr
    db.close()
    proc = _child(f"from feather_db import DB\nDB.open({p!r}, dim=16).close(); print('OPENED')")
    assert "OPENED" in proc.stdout, proc.stdout + proc.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL semantics")
def test_lock_is_released_when_the_owner_dies(tmp_path):
    p = str(tmp_path / "g.feather")
    proc = _child(f"""
import os, signal, numpy as np, feather_db
db = feather_db.DB.open({p!r}, dim=16)
db.add(1, np.ones(16, dtype=np.float32), feather_db.Metadata())
os.kill(os.getpid(), signal.SIGKILL)
""")
    assert proc.returncode == -signal.SIGKILL
    db = DB.open(p, dim=16)          # no stale lock after a crash
    assert db.get_metadata(1) is not None
    db.close()


def test_lock_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("FEATHER_LOCK", "0")
    p = str(tmp_path / "h.feather")
    a = DB.open(p, dim=16)
    b = DB.open(p, dim=16)           # allowed (and unsafe) when explicitly off
    a.close(save=False)
    b.close(save=False)


def test_merge_releases_the_source(tmp_path):
    src, dst = str(tmp_path / "src.feather"), str(tmp_path / "dst.feather")
    with DB.open(src, dim=16) as s:
        s.add(10, _v(seed=1), feather_db.Metadata())
    with DB.open(dst, dim=16) as d:
        feather_db.merge(d, src, dim=16)
        assert d.get_metadata(10) is not None
    DB.open(src, dim=16).close()     # source lock was released by merge()
