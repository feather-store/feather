"""A failed open() must not damage the file it failed to read.

DB::open() builds a unique_ptr<DB>. When load_vectors() threw, the pointer
unwound straight into ~DB(), which ran save_vectors() unconditionally — writing
a HALF-PARSED view of the file back over the file, then clearing the WAL that
could have rebuilt it. The failure path and the destroy-the-evidence path were
the same path.

Reproduced before the fix: a .feather truncated to 60% left a 16 KB
`<path>.tmp` behind, then SIGSEGV'd. The original survived only because the
crash landed before std::rename — a different corruption that failed cleanly
would have completed the rename.

Two invariants here:
  * a load that fails leaves the input byte-identical, and
  * it fails by raising, not by crashing or hanging.
"""
import hashlib
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

import feather_db
from feather_db import DB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _digest(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def _make(path, n=200, dim=32):
    db = DB.open(path, dim=dim)
    metas = []
    for i in range(n):
        m = feather_db.Metadata(); m.content = f"important record {i}"
        metas.append(m)
    db.add_batch(list(range(n)), np.random.rand(n, dim).astype(np.float32), metas)
    db.save()
    del db


def _open_in_child(path, dim=32, timeout=60):
    """Open in a subprocess: pre-fix failures were SIGSEGV, which pytest cannot
    catch, and an unbounded loop, which needs a timeout rather than an assert."""
    src = f'''
import sys
sys.path.insert(0, {REPO!r})
from feather_db import DB
try:
    db = DB.open({path!r}, dim={dim})
    print("OPENED", db.size())
except Exception as e:
    print("RAISED", type(e).__name__)
'''
    return subprocess.run([sys.executable, "-c", src], capture_output=True,
                          text=True, timeout=timeout)


# ── the file must survive ─────────────────────────────────────────────────

@pytest.mark.parametrize("frac", [0.3, 0.6, 0.9])
def test_failed_open_leaves_the_file_byte_identical(tmp_path, frac):
    path = str(tmp_path / f"t{int(frac*100)}.feather")
    _make(path)
    with open(path, "r+b") as fh:
        fh.truncate(int(os.path.getsize(path) * frac))

    before = _digest(path)
    proc = _open_in_child(path)

    assert _digest(path) == before, (
        f"a failed open modified the file it could not read (truncated to {frac:.0%})")
    assert proc.returncode == 0, (
        f"open crashed instead of raising (rc={proc.returncode})")
    assert "RAISED" in proc.stdout, f"expected a clean exception, got: {proc.stdout!r}"


def test_failed_open_leaves_no_tmp_file_behind(tmp_path):
    """The .tmp is the tell that the destructor began rewriting the file."""
    path = str(tmp_path / "tmp.feather")
    _make(path)
    with open(path, "r+b") as fh:
        fh.truncate(int(os.path.getsize(path) * 0.6))

    _open_in_child(path)
    leftovers = [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
    assert not leftovers, f"destructor started a checkpoint from a failed load: {leftovers}"


def test_failed_open_does_not_clear_the_wal(tmp_path):
    """The WAL is what could rebuild the data. A failed load must not delete it."""
    path = str(tmp_path / "wal.feather")
    _make(path, n=20)
    db = DB.open(path, dim=32)          # append records that live only in the WAL
    for i in range(500, 505):
        m = feather_db.Metadata(); m.content = f"unsaved {i}"
        db.add(i, np.random.rand(32).astype(np.float32), m)
    del db
    assert os.path.exists(path + ".wal")

    with open(path, "r+b") as fh:       # now damage the base file
        fh.truncate(int(os.path.getsize(path) * 0.5))

    _open_in_child(path)
    assert os.path.exists(path + ".wal"), "a failed load deleted the recovery log"


# ── it must fail fast, not spin ───────────────────────────────────────────

def test_absurd_metadata_count_fails_fast(tmp_path):
    """meta_count is a loop trip count. A forged 0xFFFFFF00 previously left the
    process spinning for over ten minutes; it must be rejected by arithmetic."""
    path = str(tmp_path / "count.feather")
    _make(path, n=1)

    raw = bytearray(open(path, "rb").read())
    struct.pack_into("<I", raw, 8, 0xFFFFFF00)
    open(path, "wb").write(bytes(raw))

    proc = _open_in_child(path, timeout=30)   # TimeoutExpired = regression
    assert proc.returncode == 0, f"crashed (rc={proc.returncode})"
    assert "RAISED" in proc.stdout, f"expected rejection, got {proc.stdout!r}"


# ── the healthy path is untouched ─────────────────────────────────────────

def test_a_good_file_still_opens_and_saves(tmp_path):
    path = str(tmp_path / "good.feather")
    _make(path, n=50)
    db = DB.open(path, dim=32)
    assert db.size() == 50
    m = feather_db.Metadata(); m.content = "added after reopen"
    db.add(9999, np.random.rand(32).astype(np.float32), m)
    db.save()
    del db
    assert DB.open(path, dim=32).size() == 51, "guard broke the normal save path"
