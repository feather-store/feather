"""WAL crash-recovery regressions.

The WAL exists so writes survive a crash between save() checkpoints — the
throttled bulk-import path depends on exactly that guarantee, deferring save()
for up to FEATHER_IMPORT_SAVE_INTERVAL_S while relying on the WAL to hold the
data. These tests pin the guarantee down.
"""
import os
import struct
import subprocess
import sys

import pytest

import feather_db
from feather_db import DB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _crash_writer(path, n, save_first):
    """Write n records in a child process, then SIGKILL it so nothing unwinds:
    no destructor, no save(), no flush beyond what the WAL already did."""
    src = f'''
import sys, os, signal
sys.path.insert(0, {REPO!r})
import numpy as np, feather_db
db = feather_db.DB.open({path!r}, dim=16)
if {save_first!r}:
    seed = feather_db.Metadata(); seed.content = "seed"
    db.add(999999, np.random.rand(16).astype(np.float32), seed)
    db.save()
metas = []
for i in range({n}):
    m = feather_db.Metadata(); m.content = f"rec {{i}} alpha"; metas.append(m)
db.add_batch(list(range({n})), np.random.rand({n}, 16).astype(np.float32), metas)
os.kill(os.getpid(), signal.SIGKILL)
'''
    proc = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
    assert proc.returncode == -9, f"child did not die by SIGKILL: {proc.returncode} {proc.stderr}"


def test_wal_recovers_namespace_that_was_never_saved(tmp_path):
    """The regression that mattered: a brand-new namespace has no base .feather,
    and load_vectors() used to `return` on the missing file BEFORE reaching
    replay_wal() — silently discarding a complete WAL. This is precisely the
    state bulk import leaves a fresh namespace in between throttled saves.
    """
    path = str(tmp_path / "fresh.feather")
    _crash_writer(path, 300, save_first=False)

    assert not os.path.exists(path), "no base file should exist (never saved)"
    assert os.path.getsize(path + ".wal") > 0, "WAL should hold the writes"

    db = DB.open(path, dim=16)
    assert db.size() == 300, "WAL was ignored — every record was lost"


def test_wal_recovers_on_top_of_an_existing_base_file(tmp_path):
    """The already-working case must keep working: base file + later writes."""
    path = str(tmp_path / "seeded.feather")
    _crash_writer(path, 300, save_first=True)

    assert os.path.exists(path)
    db = DB.open(path, dim=16)
    assert db.size() == 301          # 300 batched + 1 seed


def test_wal_replay_reconstructs_searchable_state(tmp_path):
    """Recovery must restore the indexes, not just the record count."""
    path = str(tmp_path / "searchable.feather")
    _crash_writer(path, 300, save_first=False)

    db = DB.open(path, dim=16)
    import numpy as np
    assert len(db.search(np.random.rand(16).astype(np.float32), k=5)) == 5
    assert db.keyword_search("alpha", k=5), "BM25 index not rebuilt after replay"
    meta = db.get_metadata(7)
    assert meta is not None and meta.content == "rec 7 alpha"


def test_corrupt_wal_length_does_not_allocate_wildly(tmp_path):
    """A torn/corrupt length field must not be taken at face value.

    A 13-byte WAL declaring a 4.29 GB payload previously drove peak RSS to
    ~1.5 GB building the std::string before the read failed — enough to OOM a
    memory-capped container on startup, making the namespace unloadable. The
    base-file loader already guarded dim/element_count this way; the WAL did not.
    """
    path = str(tmp_path / "corrupt.feather")
    db = DB.open(path, dim=16)
    import numpy as np
    m = feather_db.Metadata(); m.content = "seed"
    db.add(1, np.random.rand(16).astype(np.float32), m)
    db.save()
    del db

    with open(path + ".wal", "wb") as fh:
        fh.write(struct.pack("<B", 0x01))         # WalOp::ADD
        fh.write(struct.pack("<Q", 9))            # id
        fh.write(struct.pack("<I", 0xFFFFFF00))   # ~4.29 GB payload (corrupt)

    src = f'''
import sys, resource
sys.path.insert(0, {REPO!r})
import feather_db
db = feather_db.DB.open({path!r}, dim=16)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(db.size(), rss)
'''
    proc = subprocess.run([sys.executable, "-c", src], capture_output=True,
                          text=True, timeout=180)
    assert proc.returncode == 0, f"load crashed on corrupt WAL: {proc.stderr[-400:]}"
    size, rss = proc.stdout.split()
    assert int(size) == 1, "the valid base record should still load"
    # ru_maxrss is bytes on macOS, KiB on Linux — normalise to MB and allow slack
    rss_mb = int(rss) / (1e6 if sys.platform == "darwin" else 1e3)
    assert rss_mb < 600, f"corrupt length still drove a huge allocation: {rss_mb:.0f} MB"


def test_truncated_wal_keeps_records_written_before_the_tear(tmp_path):
    """A half-written trailing record is normal after a crash. Replay should
    keep everything before it rather than discarding the whole WAL."""
    path = str(tmp_path / "torn.feather")
    _crash_writer(path, 50, save_first=False)

    wal = path + ".wal"
    full = os.path.getsize(wal)
    with open(wal, "r+b") as fh:          # lop off part of the final record
        fh.truncate(full - 40)

    db = DB.open(path, dim=16)
    assert 0 < db.size() <= 50, f"expected a partial recovery, got {db.size()}"


def test_save_checkpoints_and_clears_the_wal(tmp_path):
    path = str(tmp_path / "checkpoint.feather")
    db = DB.open(path, dim=16)
    import numpy as np
    m = feather_db.Metadata(); m.content = "one"
    db.add(1, np.random.rand(16).astype(np.float32), m)
    assert os.path.exists(path + ".wal")
    db.save()
    assert not os.path.exists(path + ".wal"), "save() should checkpoint the WAL away"
    del db
    assert DB.open(path, dim=16).size() == 1
