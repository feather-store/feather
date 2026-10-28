"""The HNSW graph is saved even when records are deleted, delete marks included.

v9 used to save the graph only when it held exactly the live set, so a single
forget() sent every later open down the rebuild path: measured at 100k x 384,
0.15 s -> 53 s (writer) and 63 s (reader). v10 never had that restriction, and
the v9 writer (FEATHER_FORMAT=9) no longer does either.

Keeping deleted nodes is only safe if every dead record's node IS marked
deleted. A record killed with the `_deleted` attribute kept a live node: it
came back from unfiltered vector search, and once saved with the graph (v10)
it returned after a reload as a hit with no metadata. These tests pin both.
"""
import os
import struct

import numpy as np
import pytest

import feather_db
from feather_db import DB

DIM = 16


def meta(c, deleted=False):
    m = feather_db.Metadata()
    m.content = c
    if deleted:
        m.set_attribute("_deleted", "true")
    return m


def build(path, n=400, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, DIM)).astype(np.float32)
    db = DB.open(path, dim=DIM)
    db.add_batch(list(range(n)), V, [meta(f"r{i}") for i in range(n)])
    db.save()
    return db, V, rng


def top1(db, v):
    r = db.search(v, k=1)
    return (r[0].id, r[0].metadata.content) if r else None


@pytest.fixture(params=["10", "9"])
def fmt(request, monkeypatch):
    if request.param == "9":
        monkeypatch.setenv("FEATHER_FORMAT", "9")
    else:
        monkeypatch.delenv("FEATHER_FORMAT", raising=False)
    return request.param


def test_a_forget_keeps_the_graph_on_disk(tmp_path, fmt):
    """The graph — link lists included — is still in the file after a forget;
    the vectors-only fallback was ~140 bytes per node smaller."""
    p = str(tmp_path / "g.feather")
    db, V, rng = build(p)
    db.vacuum() if fmt == "10" else db.save()
    clean = os.path.getsize(p)
    db.forget(7)
    db.vacuum() if fmt == "10" else db.save()
    Q = rng.standard_normal((40, DIM)).astype(np.float32)
    before = [[(r.id, r.score) for r in db.search(q, k=8)] for q in Q]
    db.close()
    assert os.path.getsize(p) > clean - 400 * 100, "graph dropped from the file"

    db = DB.open(p, dim=DIM)
    after = [[(r.id, r.score) for r in db.search(q, k=8)] for q in Q]
    assert after == before, "graph was rebuilt rather than restored"
    assert top1(db, V[7])[0] != 7
    assert 7 not in db.get_all_ids()
    db.close()


@pytest.mark.parametrize("how", ["update_metadata", "add", "add_batch"])
def test_a_deleted_record_is_not_searchable_anywhere(tmp_path, fmt, how):
    p = str(tmp_path / f"{how}.feather")
    db, V, _ = build(p)
    if how == "update_metadata":
        db.update_metadata(5, meta("gone", deleted=True))
    elif how == "add":
        db.add(5, V[5], meta("gone", deleted=True))
    else:
        db.add_batch([5], V[5:6], [meta("gone", deleted=True)])
    assert top1(db, V[5])[0] != 5, "in memory"
    db.close()
    db = DB.open(p, dim=DIM)
    got = top1(db, V[5])
    assert got[0] != 5 and got[1] != "", f"after reload: {got}"
    db.close()


def test_a_deleted_record_stays_unsearchable_after_crash_replay(tmp_path):
    p = str(tmp_path / "wal.feather")
    db, V, _ = build(p)
    db.update_metadata(5, meta("gone", deleted=True))
    db.add(6, V[6], meta("gone too", deleted=True))
    db.close(save=False)
    db = DB.open(p, dim=DIM)
    assert top1(db, V[5])[0] != 5 and top1(db, V[6])[0] != 6
    db.close()


def test_reviving_a_deleted_record_with_a_vector_makes_it_searchable(tmp_path):
    p = str(tmp_path / "rev.feather")
    db, V, _ = build(p)
    db.update_metadata(5, meta("gone", deleted=True))
    db.add(5, V[5], meta("back"))
    assert top1(db, V[5]) == (5, "back")
    db.close()
    db = DB.open(p, dim=DIM)
    assert top1(db, V[5]) == (5, "back")
    db.close()


def test_a_ghost_node_in_an_existing_file_is_retired_on_open(tmp_path):
    """A v10 file written before this fix can hold a live node whose record
    was saved as a delete. Recreate one by appending exactly such a commit
    (a META delete for a record whose node is live) and check open repairs it."""
    xxhash = pytest.importorskip("xxhash")
    p = str(tmp_path / "ghost.feather")
    db, V, _ = build(p)
    db.close()
    raw = bytearray(open(p, "rb").read())
    payload = struct.pack("<BQI", 2, 5, 0)                     # META delete id 5
    hdr = struct.pack("<IIQ", 3, 0, len(payload))
    sec = hdr + payload + struct.pack("<Q", xxhash.xxh64_intdigest(
        payload, seed=xxhash.xxh64_intdigest(hdr)))
    gens = [struct.unpack_from("<Q", raw, s * 4096 + 16)[0] for s in (0, 1)]
    gen = max(gens) + 1
    raw += sec
    o = (gen % 2) * 4096
    struct.pack_into("<IIIIQQQQQ", raw, o, 0x46454154, 10, 0xFFFFFFFF, gen % 2, gen,
                     len(raw), 8192, 0, 0)
    struct.pack_into("<Q", raw, o + 56, xxhash.xxh64_intdigest(bytes(raw[o:o + 56])))
    open(p, "wb").write(bytes(raw))

    for ro in (True, False):
        db = DB.open(p, dim=DIM, read_only=ro)
        assert db.get_metadata(5) is None
        got = top1(db, V[5])
        assert got[0] != 5 and got[1] != "", f"ghost hit {got}"
        if not ro:
            db.save()                    # the repaired mark is committed
        db.close()
    db = DB.open(p, dim=DIM, read_only=True)
    assert top1(db, V[5])[0] != 5
    db.close()
