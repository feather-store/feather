"""A failed WAL write or fsync is fatal to the handle.

Before this, every one of them was ignored: wal_append() did not look at
fwrite/fflush, a WAL that could not be opened made the mutation go ahead with
no record at all, and fsync's return code was dropped — WAL and checkpoint
alike. Each was a write reported as saved that a crash, or simply the next
reopen, would not bring back.

Now the first failure poisons the handle, as PostgreSQL PANICs on an fsync
error: that call raises, and so does every later write and save until the file
is reopened. Retrying is the trap. A record appended after a torn one is never
replayed, and an fsync retried after a failure can succeed without the pages
that failed ("fsyncgate", 2018). The reopen replays what is actually on disk.
"""
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

import feather_db
from feather_db import DB
from feather_db import core

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 16


def _add(db, i):
    m = feather_db.Metadata(); m.content = f"rec {i}"
    db.add(i, np.random.rand(DIM).astype(np.float32), m)


def _ids(path):
    with DB.open(path, dim=DIM) as db:
        return {i for i in db.get_all_ids("text") if db.get_metadata(i) is not None}


@pytest.fixture
def fail_fsyncs():
    """core._fail_fsyncs_after(n), always switched off again afterwards."""
    yield core._fail_fsyncs_after
    core._fail_fsyncs_after(-1)


# ── fsync ─────────────────────────────────────────────────────────────────

def test_failed_wal_fsync_fails_the_write_and_every_later_one(tmp_path, fail_fsyncs):
    path = str(tmp_path / "f.feather")
    db = DB.open(path, dim=DIM)
    _add(db, 0)

    fail_fsyncs(0)
    with pytest.raises(RuntimeError, match="fsync failed"):
        _add(db, 1)
    fail_fsyncs(-1)              # the disk "recovers": the handle must not

    with pytest.raises(RuntimeError, match="stopped accepting writes"):
        _add(db, 2)
    with pytest.raises(RuntimeError, match="stopped accepting writes"):
        db.update_importance(0, 0.5)
    with pytest.raises(RuntimeError, match="stopped accepting writes"):
        db.save()
    assert "fsync failed" in db.write_error()
    assert db.get_metadata(0) is not None     # reads keep working
    db.close()                   # releases without a checkpoint, does not raise

    ids = _ids(path)
    assert 0 in ids and 2 not in ids
    with DB.open(path, dim=DIM) as db:        # reopened: writes accepted again
        assert db.write_error() is None
        _add(db, 3)
    assert 3 in _ids(path)


def test_failed_checkpoint_fsync_is_not_retried(tmp_path, fail_fsyncs):
    """The case fsyncgate is about: a second save() must not "succeed"."""
    path = str(tmp_path / "c.feather")
    db = DB.open(path, dim=DIM)
    for i in range(5):
        _add(db, i)
    db.save()
    for i in range(5, 10):
        _add(db, i)

    fail_fsyncs(1)               # the WAL rotation syncs; the data file does not
    with pytest.raises(RuntimeError, match="fsync failed"):
        db.save()
    fail_fsyncs(-1)
    with pytest.raises(RuntimeError, match="stopped accepting writes"):
        db.save()
    db.close()

    assert _ids(path) == set(range(10))       # 5-9 come back from <wal>.old


def test_every_writer_a_failed_group_fsync_covered_is_told(tmp_path, fail_fsyncs):
    path = str(tmp_path / "g.feather")
    db = DB.open(path, dim=DIM)
    _add(db, 0)
    fail_fsyncs(0)

    ok, failed = [], []
    def writer(base):
        for i in range(base, base + 20):
            try:
                _add(db, i)
                ok.append(i)
            except RuntimeError:
                failed.append(i)
    threads = [threading.Thread(target=writer, args=(1000 * t,)) for t in range(1, 9)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert ok == [], f"acknowledged without a successful fsync: {ok[:5]}"
    assert len(failed) == 160
    db.close()


def test_with_block_reraises_the_write_error_not_a_close_error(tmp_path, fail_fsyncs):
    path = str(tmp_path / "w.feather")
    with pytest.raises(RuntimeError, match="fsync failed"):
        with DB.open(path, dim=DIM) as db:
            _add(db, 0)
            fail_fsyncs(0)
            _add(db, 1)
    fail_fsyncs(-1)
    assert 0 in _ids(path)       # lock released, WAL kept and replayed


# ── write ─────────────────────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0,
                    reason="needs POSIX permissions that apply to this user")
def test_a_wal_that_cannot_be_opened_refuses_the_write(tmp_path):
    """Used to return LSN 0 and apply the mutation with no WAL record at all.
    Now the write is refused — but the handle is NOT poisoned: nothing reached
    the file, so once the open works again (EMFILE is often transient) writes
    are accepted without a reopen."""
    d = tmp_path / "ro"; d.mkdir()
    path = str(d / "o.feather")
    db = DB.open(path, dim=DIM)
    _add(db, 0)
    db.save()                    # the checkpoint closes and removes the WAL
    os.chmod(d, 0o500)
    try:
        with pytest.raises(RuntimeError, match="cannot open the WAL"):
            _add(db, 1)
        assert db.get_metadata(1) is None     # not applied in memory either
        assert db.write_error() is None
    finally:
        os.chmod(d, 0o700)
    _add(db, 2)
    db.close()
    assert _ids(path) == {0, 2}


@pytest.mark.parametrize("err", ["EINVAL", "ENOTSUP", "ENOSYS"])
def test_a_filesystem_that_cannot_fsync_keeps_working(tmp_path, fail_fsyncs, err):
    """Some FUSE / network mounts answer fsync with "not supported". That was
    always treated as synced; refusing every write there would take down a
    service that has run on such a mount until now."""
    import errno as errno_mod
    path = str(tmp_path / "u.feather")
    db = DB.open(path, dim=DIM)
    fail_fsyncs(0, getattr(errno_mod, err))
    _add(db, 0)
    db.save()
    _add(db, 1)
    assert db.write_error() is None
    fail_fsyncs(-1)
    db.close()
    assert _ids(path) == {0, 1}


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_FSIZE is POSIX")
def test_a_torn_wal_write_is_fatal_and_the_reopen_cuts_it(tmp_path):
    """A real short write: RLIMIT_FSIZE lets part of the record land, then
    EFBIG. Appending after those bytes would hide every later record from
    replay, so the handle stops; the reopen cuts the torn tail."""
    path = str(tmp_path / "t.feather")
    src = f'''
import os, resource, signal, sys
sys.path.insert(0, {REPO!r})
import numpy as np, feather_db
db = feather_db.DB.open({path!r}, dim={DIM})
def add(i):
    m = feather_db.Metadata(); m.content = "x" * 400
    db.add(i, np.random.rand({DIM}).astype(np.float32), m)
for i in range(5):
    add(i)
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
resource.setrlimit(resource.RLIMIT_FSIZE, (os.path.getsize({path!r} + ".wal") + 40, hard))
out = []
try:
    add(5); out.append("add5 ok")
except RuntimeError as e:
    out.append("add5 " + str(e))
resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
try:
    add(6); out.append("add6 ok")
except RuntimeError as e:
    out.append("add6 " + str(e))
db.close()
print("\\n".join(out))
'''
    env = dict(os.environ); env.pop("FEATHER_SIMD", None)
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-800:]
    add5, add6 = r.stdout.strip().splitlines()
    assert "WAL write failed" in add5, add5
    assert "stopped accepting writes" in add6, add6

    assert _ids(path) == set(range(5))
    with DB.open(path, dim=DIM) as db:
        _add(db, 7)
        db.close(save=False)     # 7 lives only in the WAL, after the cut
    assert _ids(path) == set(range(5)) | {7}
