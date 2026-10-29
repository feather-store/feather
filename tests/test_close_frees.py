"""close() frees the handle's memory, and Python owns the DB object.

The binding held DB with py::nodelete, so nothing a handle allocated was ever
freed: measured ~50 MB per open/close of a 39 MB file (10 cycles: +493 MB),
which is how a long-running API or MCP server slowly ran out of memory.

Now close() drops the graphs, metadata and indexes at once, and the last
Python reference destroys the object — which, for a writer nobody closed,
checkpoints and releases the lock exactly as ~DB() always did in C++.
"""
import ctypes
import gc
import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

import feather_db
from feather_db import DB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 16


def build(path, n=200):
    rng = np.random.default_rng(0)
    db = DB.open(path, dim=DIM)
    metas = []
    for i in range(n):
        m = feather_db.Metadata(); m.content = f"record {i}"; metas.append(m)
    db.add_batch(list(range(n)), rng.random((n, DIM), dtype=np.float32), metas)
    return db


def child(code):
    return subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {REPO!r})\n" + code],
                          capture_output=True, text=True, timeout=120)


def test_every_call_after_close_raises(tmp_path):
    db = build(str(tmp_path / "a.feather"))
    db.close()
    q = np.zeros(DIM, dtype=np.float32)
    calls = [lambda: db.search(q, k=3), lambda: db.get_metadata(1), lambda: db.size(),
             lambda: db.get_vector(1), lambda: db.keyword_search("record"),
             lambda: db.all_ids(), lambda: db.storage_stats(), lambda: db.touch(1),
             lambda: db.add(500, q), lambda: db.save()]
    for call in calls:
        with pytest.raises(RuntimeError, match="closed"):
            call()
    db.close()                                      # still idempotent


def test_a_with_block_closes_and_frees(tmp_path):
    p = str(tmp_path / "w.feather")
    build(p).close()
    with DB.open(p, dim=DIM) as db:
        alias = db
        assert db.size() == 200
    assert alias.closed
    with pytest.raises(RuntimeError, match="closed"):
        alias.size()


def test_dropping_the_last_reference_checkpoints_and_unlocks(tmp_path):
    p = str(tmp_path / "drop.feather")
    db = build(p)
    db.add(999, np.ones(DIM, dtype=np.float32))
    del db
    gc.collect()
    # Another process can take the write lock, and sees the records in the
    # base file: the WAL was checkpointed, not merely left behind.
    out = child(f"import feather_db; db = feather_db.DB.open({p!r}, dim={DIM}); "
                f"print(db.size(), db.storage_stats()['format'])")
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["201", "10"]
    assert not os.path.exists(p + ".wal") or os.path.getsize(p + ".wal") <= 8


def test_a_temporary_handle_does_not_hold_the_lock(tmp_path):
    p = str(tmp_path / "tmp.feather")
    build(p).close()
    assert DB.open(p, dim=DIM).size() == 200       # never assigned
    gc.collect()
    out = child(f"import feather_db; print(feather_db.DB.open({p!r}, dim={DIM}).size())")
    assert out.returncode == 0 and out.stdout.strip() == "200", out.stderr


def test_dropping_a_reader_never_writes(tmp_path):
    p = str(tmp_path / "ro.feather")
    w = build(p)
    w.save()
    w.add(500, np.ones(DIM, dtype=np.float32))      # in the WAL only
    digest = lambda: (hashlib.md5(open(p, "rb").read()).hexdigest(),
                      hashlib.md5(open(p + ".wal", "rb").read()).hexdigest())
    before = digest()
    r = DB.open(p, dim=DIM, read_only=True)
    assert r.size() == 201
    del r
    gc.collect()
    assert digest() == before
    w.close()


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="needs a private-memory probe")
def test_open_close_cycles_do_not_accumulate_memory(tmp_path):
    p = str(tmp_path / "leak.feather")
    rng = np.random.default_rng(1)
    db = DB.open(p, dim=128)
    db.add_batch(list(range(20000)), rng.random((20000, 128), dtype=np.float32), None)
    db.close()
    file_mb = os.path.getsize(p) / 1e6
    out = child(f"""
import ctypes, gc, os, sys, numpy as np, feather_db
def private():
    if sys.platform == "darwin":
        b = (ctypes.c_uint8 * 512)()
        ctypes.CDLL("/usr/lib/libSystem.B.dylib").proc_pid_rusage(os.getpid(), 2, b)
        return int.from_bytes(bytes(b[72:80]), "little") / 1e6
    return sum(int(l.split()[1]) for l in open("/proc/self/smaps_rollup")
               if l.startswith(("Private_Clean", "Private_Dirty"))) / 1e3
q = np.zeros(128, dtype=np.float32)
def cycle():
    d = feather_db.DB.open({p!r}, dim=128); d.search(q, k=3); d.close(); del d; gc.collect()
cycle(); base = private()
for _ in range(15): cycle()
print(private() - base)
""")
    assert out.returncode == 0, out.stderr
    grew = float(out.stdout.strip())
    # Leaking, 15 cycles grow by ~15x the file; the allocator caching freed
    # pages for reuse is bounded by about one open.
    assert grew < 2 * file_mb, f"grew {grew:.0f} MB over 15 cycles of a {file_mb:.0f} MB file"
