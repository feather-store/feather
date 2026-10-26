"""Regressions found reviewing the concurrency rewrite (several also on 0.21.0).

* A read-only handle shares the writer's WAL, so every mutation on it must
  refuse — purge() and auto_link() started WAL-logging and did not, and a
  reader's purge() deleted the writer's records on its next open.
* WAL replay must apply ADD/UPDATE with the live upsert rule (edges kept when
  the new metadata carries none), or a crash loses links and replaying
  <wal>.old twice is not idempotent.
* A torn WAL tail must be cut on open; otherwise later records are appended
  behind it, acknowledged, and unreachable by the next recovery.
* Two same-process handles on one path must release the flock once both close.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import feather_db
from feather_db import DB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _m(content="x", ns="ns"):
    m = feather_db.Metadata()
    m.content = content
    m.namespace_id = ns
    return m


def _v(i, dim=8):
    return np.random.default_rng(i).random(dim).astype(np.float32)


def test_read_only_handle_refuses_every_wal_writing_mutation(tmp_path):
    p = str(tmp_path / "ro.feather")
    w = DB.open(p, dim=8)
    for i in range(5):
        w.add(i, _v(i), _m())
    w.save()
    w.close()

    r = DB.open(p, dim=8, read_only=True)
    for call in (lambda: r.purge("ns"), lambda: r.auto_link("text", 0.1, "rel"),
                 lambda: r.update_importance(1, 0.5), lambda: r.forget_expired()):
        with pytest.raises(RuntimeError, match="read-only"):
            call()
    r.close()

    w = DB.open(p, dim=8)
    assert sorted(w.ids_in_namespace("ns")) == [0, 1, 2, 3, 4]
    w.close()


def test_replayed_re_add_keeps_existing_edges(tmp_path):
    p = str(tmp_path / "e.feather")
    db = DB.open(p, dim=8)
    db.add(1, _v(1), _m("one"))
    db.add(2, _v(2), _m("two"))
    db.link(1, 2, "rel", 0.5)
    db.save()
    db.add(1, _v(1), _m("one v2"))          # no edges in the new metadata
    assert [e.target_id for e in db.get_edges(1)] == [2]
    db.close(save=False)                     # crash: recovery replays the WAL

    db = DB.open(p, dim=8)
    assert [e.target_id for e in db.get_edges(1)] == [2], "link lost in WAL replay"
    assert db.get_metadata(1).content == "one v2"
    db.close()


def test_write_after_a_torn_wal_tail_survives_the_next_crash(tmp_path):
    p = str(tmp_path / "t.feather")
    db = DB.open(p, dim=8)
    db.add(1, _v(1), _m())
    db.add(2, _v(2), _m())
    db.close(save=False)
    with open(p + ".wal", "ab") as f:          # a record torn mid-write
        f.write(b"\x01\x07\x00")

    db = DB.open(p, dim=8)
    assert sorted(db.get_all_ids("text")) == [1, 2]
    db.add(3, _v(3), _m())                      # acknowledged (fsynced)
    db.close(save=False)

    db = DB.open(p, dim=8)
    assert sorted(db.get_all_ids("text")) == [1, 2, 3], "acknowledged write lost"
    db.close()


def test_read_only_open_never_rewrites_the_writers_wal(tmp_path):
    p = str(tmp_path / "rt.feather")
    db = DB.open(p, dim=8)
    db.add(1, _v(1), _m())
    db.close(save=False)
    with open(p + ".wal", "ab") as f:
        f.write(b"\x01\x07\x00")
    size = os.path.getsize(p + ".wal")
    r = DB.open(p, dim=8, read_only=True)
    assert sorted(r.get_all_ids("text")) == [1]
    r.close()
    assert os.path.getsize(p + ".wal") == size


def test_flock_released_when_two_same_process_handles_close(tmp_path):
    p = str(tmp_path / "l.feather")
    a = DB.open(p, dim=8)
    b = DB.open(p, dim=8)                        # reentrant: same process
    a.add(1, _v(1), _m())
    a.close()                                    # the handle that took the flock
    b.close()
    code = (f"import sys; sys.path.insert(0, {REPO!r}); from feather_db import DB; "
            f"DB.open({p!r}, dim=8).close(); print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.stdout.strip() == "ok", r.stderr[-400:]
