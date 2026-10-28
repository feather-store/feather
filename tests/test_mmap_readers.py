"""Read-only handles map the vector image instead of copying it.

A full rewrite stores each modality's vectors as one VECTORS section aligned
for mmap. A read-only handle maps it copy-on-write: N reader processes share
one copy through the page cache, and anything a reader changes in memory (WAL
replay, later commits) becomes a private page — the file is never written.

What is pinned here:
  * readers map, writers do not, FEATHER_MMAP=0 turns it off
  * a mapped reader returns exactly what a heap-loaded one does
  * later commits and a WAL tail apply on top of the mapping, privately
  * a reader is unaffected by the writer appending and vacuuming under it
  * the image is kept fresh: appends past it trigger a rewrite
  * damage inside the image is caught by its checksum
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

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX mmap")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 32


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def meta(c):
    m = feather_db.Metadata()
    m.content = c
    return m


def build(path, n=3000, seed=0, vacuum=True):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, DIM)).astype(np.float32)
    db = DB.open(path, dim=DIM)
    db.add_batch(list(range(n)), V, [meta(f"r{i}") for i in range(n)])
    db.save()
    return db, V, rng


def answers(db, Q):
    return [[(r.id, r.score) for r in db.search(q, k=10)] for q in Q]


def vectors_offset(path):
    """File offset of the first VECTORS section's vector bytes."""
    raw = open(path, "rb").read()
    pos = 8192
    while pos + 24 <= len(raw):
        typ, _, ln = struct.unpack_from("<IIQ", raw, pos)
        if typ == 5:
            p = pos + 16
            nl = struct.unpack_from("<H", raw, p)[0]
            p += 2 + nl + 4 + 4
            pad = struct.unpack_from("<I", raw, p)[0]
            return p + 4 + pad
        pos += 24 + ln
    return None


def test_readers_map_writers_do_not(tmp_path, monkeypatch):
    p = str(tmp_path / "a.feather")
    db, V, _ = build(p)
    assert db.storage_stats()["mapped_bytes"] == 0           # a writer holds a heap copy
    db.close()
    r = DB.open(p, dim=DIM, read_only=True)
    assert r.storage_stats()["mapped_bytes"] == 3000 * DIM * 4
    r.close()
    monkeypatch.setenv("FEATHER_MMAP", "0")
    r = DB.open(p, dim=DIM, read_only=True)
    assert r.storage_stats()["mapped_bytes"] == 0
    r.close()


def test_the_image_is_aligned_for_any_page_size(tmp_path):
    p = str(tmp_path / "al.feather")
    db, _, _ = build(p, n=100)
    db.close()
    off = vectors_offset(p)
    assert off is not None and off % 65536 == 0


def test_a_mapped_reader_answers_exactly_like_a_heap_one(tmp_path, monkeypatch):
    p = str(tmp_path / "eq.feather")
    db, V, rng = build(p)
    db.close()
    Q = rng.standard_normal((50, DIM)).astype(np.float32)
    r = DB.open(p, dim=DIM, read_only=True)
    mapped = answers(r, Q), [r.get_vector(i).tobytes() for i in range(0, 3000, 97)]
    r.close()
    monkeypatch.setenv("FEATHER_MMAP", "0")
    r = DB.open(p, dim=DIM, read_only=True)
    heap = answers(r, Q), [r.get_vector(i).tobytes() for i in range(0, 3000, 97)]
    r.close()
    assert mapped == heap
    assert np.array_equal(np.frombuffer(mapped[1][1], np.float32), V[97])


def test_later_commits_and_a_wal_tail_apply_privately(tmp_path):
    """Updated vectors (copy-on-write pages of the image) and new ones
    (beyond it) are visible to the reader; the file is not touched."""
    p = str(tmp_path / "cow.feather")
    db, V, rng = build(p)
    W = rng.standard_normal((40, DIM)).astype(np.float32)
    for k in range(20):                         # committed after the image
        db.add(k, W[k], meta(f"upd{k}"))        # existing id: rewrites an image slot
        db.add(5000 + k, W[20 + k], meta(f"new{k}"))
    db.save()
    db.add(100, W[0] * 3, meta("wal-only"))     # WAL only
    db.close(save=False)
    before = md5(p)

    r = DB.open(p, dim=DIM, read_only=True)
    assert r.storage_stats()["mapped_bytes"] > 0
    assert r.size() == 3020
    for k in range(20):
        assert np.array_equal(r.get_vector(k), W[k])
        assert np.array_equal(r.get_vector(5000 + k), W[20 + k])
    assert np.array_equal(r.get_vector(100), W[0] * 3)
    assert np.array_equal(r.get_vector(2999), V[2999])         # untouched image slot
    hits = [x.id for x in r.search(W[3], k=1)]
    assert hits == [3]
    r.close()
    assert md5(p) == before, "a read-only mapping wrote to the file"


def test_a_reader_survives_the_writer_appending_and_vacuuming(tmp_path):
    p = str(tmp_path / "live.feather")
    w, V, rng = build(p)
    r = DB.open(p, dim=DIM, read_only=True)
    Q = rng.standard_normal((30, DIM)).astype(np.float32)
    expect = answers(r, Q)
    for round_ in range(3):
        for i in range(200):
            w.add(10_000 + round_ * 200 + i, rng.standard_normal(DIM).astype(np.float32))
        w.save()
        w.vacuum()                               # replaces the inode the reader maps
        assert answers(r, Q) == expect, "the reader's mapped snapshot changed"
    r.close()
    fresh = DB.open(p, dim=DIM, read_only=True)
    assert fresh.size() == 3600
    fresh.close()
    w.close()


def test_int8_ram_modality_maps_too(tmp_path):
    p = str(tmp_path / "i8.feather")
    rng = np.random.default_rng(2)
    db = DB.open(p, dim=DIM)
    db.set_int8_ram("text", 4.0)
    V = rng.standard_normal((500, DIM)).astype(np.float32)
    db.add_batch(list(range(500)), V, [meta(str(i)) for i in range(500)])
    q = V[7] + 0.01
    want = [x.id for x in db.search(q, k=5)]
    db.close()
    r = DB.open(p, dim=DIM, read_only=True)
    assert r.is_int8_ram("text") and r.storage_stats()["mapped_bytes"] == 500 * DIM
    assert [x.id for x in r.search(q, k=5)] == want
    r.close()


def test_on_disk_quantized_modality_is_read_not_mapped(tmp_path):
    p = str(tmp_path / "q.feather")
    rng = np.random.default_rng(3)
    db = DB.open(p, dim=DIM)
    db.set_quantized("text", True)
    V = rng.standard_normal((300, DIM)).astype(np.float32)
    db.add_batch(list(range(300)), V, [meta(str(i)) for i in range(300)])
    db.close()
    r = DB.open(p, dim=DIM, read_only=True)
    assert r.storage_stats()["mapped_bytes"] == 0              # int8 on disk, float in RAM
    assert np.allclose(r.get_vector(5), V[5], atol=np.abs(V[5]).max() / 100)
    r.close()


def test_appends_past_the_image_trigger_a_rewrite(tmp_path, monkeypatch):
    """Vectors outside the image are private to every reader, so once they
    outgrow it the next save rewrites the file with a fresh image."""
    monkeypatch.setenv("FEATHER_VACUUM_MIN_BYTES", "0")
    p = str(tmp_path / "grow.feather")
    db, V, rng = build(p, n=500)
    fulls = []
    for k in range(6):
        db.add_batch(list(range(1000 + k * 120, 1000 + (k + 1) * 120)),
                     rng.standard_normal((120, DIM)).astype(np.float32), None)
        db.save()
        fulls.append(db.storage_stats()["last_commit_full"])
    assert fulls[:4] == [False] * 4 and fulls[4], fulls        # 480 < 500 < 600
    db.close()
    r = DB.open(p, dim=DIM, read_only=True)
    assert r.storage_stats()["mapped_bytes"] == r.size() * DIM * 4 - 120 * DIM * 4
    r.close()


def test_damage_inside_the_image_is_caught(tmp_path):
    p = str(tmp_path / "rot.feather")
    db, _, _ = build(p)
    db.close()
    off = vectors_offset(p)
    raw = bytearray(open(p, "rb").read())
    raw[off + 1000] ^= 0xFF
    open(p, "wb").write(bytes(raw))
    for ro in (True, False):
        with pytest.raises(RuntimeError, match="corrupt"):
            DB.open(p, dim=DIM, read_only=ro)
    assert open(p, "rb").read() == bytes(raw)


def test_readers_in_separate_processes_share_the_image(tmp_path):
    """The point of it: each extra reader process adds its private memory,
    not another copy of the vectors. Measured as physical footprint on macOS;
    elsewhere only the mapping itself is checked."""
    p = str(tmp_path / "share.feather")
    rng = np.random.default_rng(9)
    n, dim = 20000, 384
    db = DB.open(p, dim=dim)
    db.add_batch(list(range(n)), rng.random((n, dim), dtype=np.float32), None)
    db.close()
    code = f"""
import ctypes, os, sys
sys.path.insert(0, {REPO!r})
import feather_db
def fp():
    if sys.platform != "darwin": return 0
    b = (ctypes.c_uint8 * 512)()
    ctypes.CDLL("/usr/lib/libSystem.B.dylib").proc_pid_rusage(os.getpid(), 2, b)
    return int.from_bytes(bytes(b[72:80]), "little")
f0 = fp()
db = feather_db.DB.open({p!r}, dim={dim}, read_only=True)
print(db.storage_stats()["mapped_bytes"], fp() - f0)
"""
    vec_bytes = n * dim * 4
    out = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={**os.environ, "FEATHER_MMAP": v}) for v in ("1", "0")]
    mapped, private_mapped = map(int, out[0].stdout.split())
    _, private_heap = map(int, out[1].stdout.split())
    assert mapped == vec_bytes
    if sys.platform == "darwin":
        assert private_heap - private_mapped > 0.8 * vec_bytes, (private_heap, private_mapped)
