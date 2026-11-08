"""Time travel: DB.open(path, as_of=...) loads a past commit.

A v10 commit only appends, and every commit ends with a COMMIT marker, so the
file up to a marker is the file as that commit left it. A full rewrite starts
a new file; the old one is kept under <path>.history/ while the retention
window or a tag needs it. erase() deletes all history.

These tests hold the promises: every reachable commit loads exactly as it
was; history survives rewrites only when asked to, and says where it starts
when it does not; a replay scores with the clock of its moment and changes
nothing; erased data cannot come back through the past; remote mode matches.
"""
import datetime as dt
import glob
import os
import time

import numpy as np
import pytest

import feather_db
from feather_db import DB, Metadata

DIM = 8


def vec(i):
    return np.random.default_rng(i).random(DIM).astype(np.float32)


def put(db, rid, content, ns="app", ts=None):
    m = Metadata()
    m.namespace_id, m.content = ns, content
    m.timestamp = int(ts if ts is not None else time.time())
    db.add(rid, vec(rid), m)


def state(db, ns=""):
    """{id: content} of every live record."""
    ids = list(db.record_versions(ns))
    return {i: m.content for i, m in zip(ids, db.get_metadata_many(ids))}


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "t.feather")


@pytest.fixture(autouse=True)
def _no_env_retention(monkeypatch):
    monkeypatch.delenv("FEATHER_HISTORY_KEEP_S", raising=False)


def commits(db, n=6):
    """n saves, each adding one record and rewriting record 0. Returns
    [(time just after the save, expected state)]."""
    out, want = [], {}
    for i in range(1, n + 1):
        put(db, i, f"r{i}")
        put(db, 0, f"zero@{i}")
        want = {**want, i: f"r{i}", 0: f"zero@{i}"}
        db.save()
        out.append((time.time(), dict(want)))
        time.sleep(0.002)
    return out


# ── every reachable commit loads exactly ───────────────────────────────────

def test_every_commit_loads_as_it_was(path):
    db = DB.open(path, dim=DIM)
    snaps = commits(db)
    for t, want in snaps:
        with DB.open(path, dim=DIM, as_of=t) as past:
            assert state(past) == want
            assert past.is_read_only()
    # by generation, too
    hist = db.history()
    assert [h["generation"] for h in hist] == list(range(1, len(snaps) + 1))
    for h, (_, want) in zip(hist, snaps):
        with DB.open(path, dim=DIM, generation=h["generation"]) as past:
            assert state(past) == want
            assert past.as_of()["generation"] == h["generation"]
    db.close()


def test_as_of_accepts_a_datetime(path):
    db = DB.open(path, dim=DIM)
    snaps = commits(db, 3)
    when = dt.datetime.fromtimestamp(snaps[1][0], tz=dt.timezone.utc)
    with DB.open(path, as_of=when) as past:
        assert state(past) == snaps[1][1]
    db.close()


def test_as_of_ignores_unsaved_writes_and_the_wal(path):
    db = DB.open(path, dim=DIM)
    commits(db, 2)
    put(db, 99, "only in the WAL")
    with DB.open(path, as_of=time.time()) as past:
        assert 99 not in state(past)
    with DB.open(path, read_only=True) as live:           # an ordinary reader sees it
        assert 99 in state(live)
    db.close()


def test_as_of_is_read_only(path):
    db = DB.open(path, dim=DIM)
    commits(db, 2)
    with DB.open(path, as_of=time.time()) as past:
        with pytest.raises(Exception):
            put(past, 50, "x")
        with pytest.raises(Exception):
            past.tag("nope")
    db.close()


def test_before_the_history_says_where_it_starts(path):
    db = DB.open(path, dim=DIM)
    t0 = time.time() - 3600
    commits(db, 2)
    with pytest.raises(ValueError, match="history .* begins at generation 1"):
        DB.open(path, as_of=t0)
    with pytest.raises(ValueError, match="does not exist"):
        DB.open(path, generation=99)
    with pytest.raises(ValueError, match="no tag 'missing'"):
        DB.open(path, as_of="missing")
    with pytest.raises(ValueError, match="no such file"):
        DB.open(path + ".nope", as_of=time.time())
    db.close()


# ── rewrites: history kept only when asked ────────────────────────────────

def test_a_rewrite_without_retention_drops_history(path):
    db = DB.open(path, dim=DIM)
    snaps = commits(db, 3)
    db.vacuum()
    assert not os.path.exists(path + ".history")
    assert [h["generation"] for h in db.history()] == [4]
    with pytest.raises(ValueError, match="begins at generation 4"):
        DB.open(path, as_of=snaps[0][0])
    db.close()


@pytest.mark.parametrize("rewrite", ["vacuum", "compact", "purge", "auto"])
def test_retention_keeps_history_across_every_rewrite(path, rewrite, monkeypatch):
    db = DB.open(path, dim=DIM)
    db.set_history_retention(3600)
    snaps = commits(db, 3)
    if rewrite == "vacuum":
        db.vacuum()
    elif rewrite == "compact":
        db.forget(1); db.compact(); db.save()
    elif rewrite == "purge":
        put(db, 500, "other", ns="tmp"); db.purge("tmp"); db.save()
    else:
        monkeypatch.setenv("FEATHER_VACUUM_RATIO", "0.000001")
        monkeypatch.setenv("FEATHER_VACUUM_MIN_BYTES", "0")
        put(db, 501, "trigger"); db.save()
    assert db.storage_stats()["last_commit_full"]
    assert db.storage_stats()["history_files"] == 1
    for t, want in snaps:
        with DB.open(path, as_of=t) as past:
            assert state(past) == want
            assert past.as_of()["file"].startswith(path + ".history" + os.sep)
    assert [h["archived"] for h in db.history()][:3] == [True, True, True]
    db.close()


def test_retention_is_stored_in_the_file(path):
    db = DB.open(path, dim=DIM)
    db.set_history_retention(7 * 86400)
    db.close()
    db = DB.open(path, dim=DIM)
    assert db.history_retention() == 7 * 86400
    db.set_history_retention(float("inf"))
    db.close()
    db = DB.open(path, dim=DIM)
    assert db.history_retention() == float("inf")
    db.close()


def test_env_retention_applies_to_a_file_without_one(path, monkeypatch):
    monkeypatch.setenv("FEATHER_HISTORY_KEEP_S", "60")
    db = DB.open(path, dim=DIM)
    assert db.history_retention() == 60
    commits(db, 2)
    db.vacuum()
    assert db.storage_stats()["history_files"] == 1
    db.close()


def test_files_outside_the_window_are_pruned(path):
    db = DB.open(path, dim=DIM)
    db.set_history_retention(3600)
    commits(db, 2)
    db.vacuum()
    commits(db, 2)
    db.vacuum()
    assert db.storage_stats()["history_files"] == 2
    db.set_history_retention(0)              # nothing kept now
    assert db.storage_stats()["history_files"] == 0
    db.close()


# ── tags ──────────────────────────────────────────────────────────────────

def test_a_tag_names_a_commit_and_pins_it(path):
    db = DB.open(path, dim=DIM)
    commits(db, 2)
    db.tag("run-1")
    want = state(db)
    commits(db, 2)                           # more commits, then rewrites
    db.vacuum()
    db.vacuum()
    assert db.tags() == {"run-1": 3}
    with DB.open(path, as_of="run-1") as past:
        assert state(past) == want
    db.close()
    db = DB.open(path, dim=DIM)              # tags are in the file
    assert db.tags() == {"run-1": 3}
    db.untag("run-1")
    assert db.tags() == {}
    assert db.storage_stats()["history_files"] == 0
    with pytest.raises(ValueError):
        DB.open(path, as_of="run-1")
    db.close()


def test_a_tag_on_an_unchanged_db_still_names_its_state(path):
    db = DB.open(path, dim=DIM)
    commits(db, 1)
    db.tag("a")
    db.tag("b")
    with DB.open(path, as_of="a") as a, DB.open(path, as_of="b") as b:
        assert state(a) == state(b) == state(db)
    with pytest.raises(ValueError):
        db.tag("")
    with pytest.raises(ValueError):
        db.untag("never")
    db.close()


# ── replay: the clock of its moment, nothing changed ──────────────────────

def test_scoring_runs_at_the_as_of_time(path):
    db = DB.open(path, dim=DIM)
    put(db, 1, "x", ts=time.time())
    db.save()
    db.close()
    q = vec(1)
    cfg = feather_db.ScoringConfig(half_life=30.0, weight=0.5, min=0.0)
    with DB.open(path, read_only=True) as live:
        fresh = live.search(q, k=1, scoring=cfg)[0].score
    later = time.time() + 90 * 86400          # resolves to the newest commit, clock pinned to it
    with DB.open(path, as_of=later) as past:
        assert past.now() == pytest.approx(later, abs=1e-5)
        assert past.as_of()["clock"] == pytest.approx(later, abs=1e-5)
        aged = past.search(q, k=1, scoring=cfg)[0].score
    # 90 days at a 30-day half-life: recency 1/8, freshness 0.5 + 0.5 / 8.
    assert aged == pytest.approx(fresh * (0.5 + 0.5 / 8), rel=1e-3)


def test_a_replay_records_no_salience(path):
    db = DB.open(path, dim=DIM)
    commits(db, 2)
    db.close()
    with DB.open(path, as_of=time.time()) as past:
        before = past.get_metadata(1).recall_count
        for _ in range(5):
            past.search(vec(1), k=3)
        assert past.get_metadata(1).recall_count == before


def test_pocket_heat_uses_the_pinned_clock(path):
    from feather_db.pocket import Pocket
    db = DB.open(path, dim=DIM)
    pk = Pocket(db, ("org", "u"), budget_tokens=1000, embed=lambda s: vec(len(s)))
    pk.remember("tone", "lowercase")
    db.save()
    t = time.time()
    db.close()
    with DB.open(path, as_of=t) as past:
        p2 = Pocket(past, ("org", "u"), budget_tokens=1000, embed=lambda s: vec(len(s)))
        item = p2.hot()[0]
        assert past.now() == pytest.approx(t, abs=1e-5)
        assert item.heat == pytest.approx(p2.heat(past.get_metadata(item.id), now=t))


# ── diff ──────────────────────────────────────────────────────────────────

def test_diff_between_two_runs(path):
    db = DB.open(path, dim=DIM)
    put(db, 1, "a"); put(db, 2, "b"); put(db, 3, "c")
    db.tag("run-1")
    put(db, 2, "b2"); db.forget(3); put(db, 4, "d")
    db.search(vec(1), k=2)                    # a recall is not a change
    db.tag("run-2")
    with DB.open(path, as_of="run-1") as a, DB.open(path, as_of="run-2") as b:
        d = feather_db.diff(a, b)
    assert (d.added, d.removed, d.changed) == ([4], [3], [2])
    assert d.summary() == "+1 -1 ~1"
    assert not feather_db.diff(db, db)
    db.close()


# ── erase: the past cannot bring it back ─────────────────────────────────

def _bytes_on_disk(path, needle):
    hits = []
    for f in glob.glob(path + "*"):
        files = ([os.path.join(r, n) for r, _, ns in os.walk(f) for n in ns]
                 if os.path.isdir(f) else [f])
        hits += [x for x in files if needle in open(x, "rb").read()]
    return hits


def test_erase_deletes_all_history(path):
    db = DB.open(path, dim=DIM)
    db.set_history_retention(float("inf"))
    put(db, 7, "SECRET-alice-ssn")
    db.tag("with-alice")
    db.forget(7)
    db.vacuum()                              # the secret is now ONLY in history
    assert _bytes_on_disk(path, b"SECRET-alice-ssn")
    with DB.open(path, as_of="with-alice") as past:
        assert past.get_metadata(7).content == "SECRET-alice-ssn"
    r = db.erase(ids=[7])                    # matches nothing live, still cuts history
    assert r["verified"], r["problems"]
    assert r["history_files_removed"] >= 1
    assert _bytes_on_disk(path, b"SECRET-alice-ssn") == []
    assert db.tags() == {}
    assert len(db.history()) == 1
    with pytest.raises(ValueError):
        DB.open(path, as_of="with-alice")
    db.close()


def test_a_crashed_erase_still_cuts_history_on_reopen(path):
    from feather_db import core
    db = DB.open(path, dim=DIM)
    db.set_history_retention(float("inf"))
    put(db, 7, "SECRET-bob")
    db.save()
    db.forget(7)
    db.vacuum()
    core._halt_erase_after_log(True)
    try:
        with pytest.raises(RuntimeError, match="halted"):
            db.erase(ids=[7])
    finally:
        core._halt_erase_after_log(False)
    db.close(save=False)                     # the crash: WAL kept, nothing rewritten
    assert _bytes_on_disk(path, b"SECRET-bob")
    db = DB.open(path, dim=DIM)              # replays the ERASE, finishes it
    assert _bytes_on_disk(path, b"SECRET-bob") == []
    assert not os.path.exists(path + ".history")
    db.close()


# ── compatibility ─────────────────────────────────────────────────────────

def test_a_v9_file_has_no_history(path, monkeypatch):
    monkeypatch.setenv("FEATHER_FORMAT", "9")
    db = DB.open(path, dim=DIM)
    put(db, 1, "a")
    db.save()
    with pytest.raises(RuntimeError):
        db.tag("x")
    db.close()
    with pytest.raises(ValueError, match="not a v10 file"):
        DB.open(path, as_of=time.time())
    monkeypatch.delenv("FEATHER_FORMAT")
    db = DB.open(path, dim=DIM)              # the first save rewrites it as v10
    db.save()
    with DB.open(path, as_of=time.time()) as past:
        assert state(past) == {1: "a"}
    db.close()


def test_a_reader_opened_at_the_head_matches_a_plain_reader(path):
    db = DB.open(path, dim=DIM)
    commits(db, 3)
    db.close()
    with DB.open(path, read_only=True) as live, DB.open(path, as_of=time.time()) as past:
        assert state(live) == state(past)
        assert past.as_of()["generation"] == live.storage_stats()["generation"]


# ── remote: the same calls over HTTP ─────────────────────────────────────

@pytest.fixture
def remote(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import sys
    api = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "feather-api")
    monkeypatch.setenv("FEATHER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FEATHER_DEV_MODE", "1")
    monkeypatch.setenv("FEATHER_DB_DIM", str(DIM))
    if api not in sys.path:
        sys.path.insert(0, api)
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from fastapi.testclient import TestClient
    from app.main import app
    from feather_db.remote import RemoteDB
    with TestClient(app) as client:
        r = RemoteDB("http://testserver", "tt", session=client)
        r.ensure_namespace(dim=DIM)
        yield r, (lambda **kw: RemoteDB("http://testserver", "tt", session=client, **kw))


def test_remote_time_travel(remote):
    db, open_remote = remote
    put(db, 1, "a"); put(db, 2, "b")
    db.tag("run-1")
    put(db, 2, "b2"); put(db, 3, "c")
    db.save()
    t = time.time()
    db.set_history_retention(3600)
    db.vacuum()
    assert db.history_retention() == 3600
    gen = db.tags()["run-1"]                 # creating the namespace may have committed first
    assert gen in [h["generation"] for h in db.history()]

    past = open_remote(as_of="run-1")
    assert past.is_read_only()
    assert state(past) == {1: "a", 2: "b"}
    assert past.as_of()["generation"] == gen
    assert past.now() == pytest.approx(past.as_of()["clock"])
    with pytest.raises(ValueError, match="cannot write"):
        put(past, 9, "x")
    d = feather_db.diff(past, db)
    assert (d.added, d.changed) == ([3], [2])

    assert state(open_remote(as_of=t)) == {1: "a", 2: "b2", 3: "c"}
    with pytest.raises(ValueError, match="no tag"):
        state(open_remote(as_of="nope"))
    db.untag("run-1")
    assert db.tags() == {}
    db.set_history_retention(float("inf"))
    assert db.history_retention() == float("inf")
