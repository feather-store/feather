"""The (timestamp, id) time index behind newest(), and packet replay.

  * `DB.newest()` walks an ordered (timestamp, id) index from the end, so "the
    n most recent" costs O(log n + n) instead of reading and sorting the
    namespace. FeatherStore.newest() uses it.
  * A packet records each ref at the record's VERSION. It used to record
    recall_count: every search moved it and an edit of the rule did not, so
    replaying a stored manifest could neither trust an unchanged ref nor
    notice a changed one. `PacketBuilder.revalidate()` replays one.
"""
import time

import numpy as np
import pytest

import feather_db
from feather_db import DB, Metadata, PacketBuilder, RequiredRule
from feather_db.pocket import Pocket

DIM = 8


def meta(ns, ts, content=""):
    m = Metadata()
    m.namespace_id, m.timestamp, m.content = ns, ts, content
    return m


@pytest.fixture
def db(tmp_path):
    d = DB.open(str(tmp_path / "t.feather"), dim=DIM)
    yield d
    d.close()


def v(i=0):
    return np.random.default_rng(i).random(DIM, dtype=np.float32)


# ── the time index ────────────────────────────────────────────────────────

def test_newest_orders_by_timestamp_then_id(db):
    rng = np.random.default_rng(0)
    ts = {i: int(rng.integers(1000, 1050)) for i in range(300)}       # many ties
    for i, t in ts.items():
        db.add(i, v(i), meta("a", t))
    want = sorted(ts, key=lambda i: (ts[i], i), reverse=True)
    assert db.newest("a", 20) == want[:20]
    assert db.newest("a", 1000) == want
    assert db.newest("nobody", 5) == []


def test_paging_with_a_cursor_visits_everything_once(db):
    for i in range(250):
        db.add(i, v(i), meta("a", 1000 + i // 7))
    seen, before = [], None
    while True:
        page = db.newest("a", 40, before=before)
        seen += page
        if len(page) < 40:
            break
        before = (db.get_metadata(page[-1]).timestamp, page[-1])
    assert seen == sorted(range(250), key=lambda i: (1000 + i // 7, i), reverse=True)
    assert db.newest("a", 5, before=1003) == [20, 19, 18, 17, 16]     # a bare timestamp


def test_a_prefix_merges_its_namespaces(db):
    for i in range(60):
        db.add(i, v(i), meta(["org.u1", "org.u1.prefs", "org.u10", "other"][i % 4], 1000 + i))
    assert db.newest("org.u1", 100, prefix=True) == [i for i in reversed(range(60)) if i % 4 in (0, 1)]
    assert db.newest("org", 3, prefix=True) == [58, 57, 56]
    assert db.newest("", 2, prefix=True) == [59, 58]


def test_writes_keep_the_index_exact_and_a_reload_rebuilds_it(tmp_path):
    path = str(tmp_path / "w.feather")
    d = DB.open(path, dim=DIM)
    for i in range(10):
        d.add(i, v(i), meta("a", 1000 + i))
    d.update_metadata(2, meta("a", 5000))                 # newer now
    d.forget(9)                                           # gone
    d.update_metadata(8, meta("b", 6000))                 # moved out
    d.add(3, v(3), meta("a", 999))                        # older now
    want = [2, 7, 6, 5, 4, 1, 0, 3]
    assert d.newest("a", 20) == want and d.newest("b", 5) == [8]
    d.close(save=False)                                   # crash: replay
    d = DB.open(path, dim=DIM)
    assert d.newest("a", 20) == want
    d.close()
    d = DB.open(path, dim=DIM)                            # from the file
    assert d.newest("a", 20) == want
    d.purge("a")
    assert d.newest("a", 5) == []
    d.close()


def test_featherstore_newest_walks_the_index(tmp_path):
    pytest.importorskip("langgraph")
    from feather_db.integrations.langgraph_store import FeatherStore
    store = FeatherStore(str(tmp_path / "s.feather"), dim=DIM, embed=None, auto_save=False)
    ns = ("org", "u1", "notes")
    for i in range(400):
        store.put(ns, f"k{i}", {"n": i})
    store.put(ns, "short", {"n": -1}, ttl=0.0001)         # expires: skipped
    time.sleep(1.2)
    calls = []
    real = store.db

    class Spy:
        def __getattr__(self, name):
            attr = getattr(real, name)
            if name in ("get_metadata", "ids_in_namespace"):
                return lambda *a, **kw: (calls.append(name), attr(*a, **kw))[1]
            return attr
    store.db = Spy()
    got = store.newest(ns, 5)
    store.db = real
    assert [i.value["n"] for i in got] == [399, 398, 397, 396, 395]
    assert "ids_in_namespace" not in calls and len(calls) <= 8, calls
    # the scan path (a custom key) agrees
    scan = store.newest(ns, 5, key=lambda m: (m.timestamp, store.id_of(ns, m.key)))
    assert [i.value["n"] for i in scan] == [399, 398, 397, 396, 395]
    store.close()


# ── packet replay ─────────────────────────────────────────────────────────

@pytest.fixture
def pkt(db):
    p = Pocket(db, ("org", "brand"))
    p.remember("compliance", "never claim a cure", pinned=True)
    p.remember("tone", "warm, plain words")
    return p


def test_a_ref_records_the_version_and_reads_do_not_move_it(pkt, db):
    b = PacketBuilder(pkt, budget_tokens=500)
    packet = b.build(required=["compliance"])
    rid = db.id_for("compliance", "org.brand")
    ref = packet.refs[0]
    assert ref.version == db.get_metadata(rid).version
    for _ in range(5):
        pkt.recall("cure")                       # recall_count moves; the version must not
    assert db.get_metadata(rid).recall_count >= 5
    again = b.build(required=["compliance"])
    assert again.refs[0].version == ref.version
    assert b.revalidate(packet.manifest()) == {
        "current": True, "may_mutate": True, "stale": [], "unverified": []}


def test_an_edited_rule_makes_the_packet_stale(pkt):
    b = PacketBuilder(pkt, budget_tokens=500)
    manifest = b.build(required=["compliance"]).manifest()
    pkt.remember("compliance", "never claim a cure or a prevention", pinned=True)
    r = b.revalidate(manifest)
    assert not r["current"] and not r["may_mutate"]
    assert r["stale"][0]["reason"] == "changed" and r["stale"][0]["required"]
    assert r["stale"][0]["now"] > r["stale"][0]["was"]


def test_a_forgotten_memory_is_missing_and_a_context_change_does_not_block(pkt):
    b = PacketBuilder(pkt, budget_tokens=500)
    manifest = b.build(required=["compliance"]).manifest()
    assert any(not r["required"] and r["key"] == "tone" for r in manifest["refs"])
    pkt.forget("tone")
    r = b.revalidate(manifest)
    assert [s["reason"] for s in r["stale"]] == ["missing"]
    assert r["may_mutate"], "a stale non-required ref must not block a mutation"


def test_a_resolvers_versions_are_what_is_replayed(pkt):
    rules = {"compliance": RequiredRule("never claim a cure", version=7, source="mongo"),
             "legacy": "no version given"}
    b = PacketBuilder(pkt, budget_tokens=500, resolve_required=rules.get)
    manifest = b.build(required=["compliance", "legacy"]).manifest()
    first = b.revalidate(manifest)
    assert first["current"] and first["unverified"] == ["external/legacy"]   # no version to compare
    rules["compliance"] = RequiredRule("never claim a cure", version=8, source="mongo")
    r = b.revalidate(manifest)
    assert [(s["reason"], s["was"], s["now"]) for s in r["stale"]] == [("changed", 7, 8)]
    assert not r["may_mutate"]
