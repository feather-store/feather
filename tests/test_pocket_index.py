"""Facts as index, memories as payload, and supersession on raw memory
(Pocket.remember(index=...), recall, group_superseded, PacketBuilder).

Keyword search throughout (no embedder): deterministic."""
import sys
import time

import pytest

import feather_db
from feather_db import keys
from feather_db.extractors import Fact
from feather_db.packet import PacketBuilder

P = sys.modules["feather_db.pocket"]          # feather_db.pocket is also a function
DAY = 86400


@pytest.fixture
def db(tmp_path):
    d = feather_db.DB.open(str(tmp_path / "ix.feather"), dim=8)
    yield d
    d.close()


def _pkt(db, scope=("org", "alice"), **kw):
    return P.Pocket(db, scope, **kw)


def test_an_index_hit_returns_the_memory_never_the_index(db):
    pkt = _pkt(db)
    pkt.remember("t1", "Caroline: Gonna go see the doc about the knee tomorrow",
                 index=["Caroline has a knee injury and sees an orthopedist"])
    pkt.remember("t2", "Melanie: The kids loved the pottery class")
    got = pkt.recall("orthopedist injury", k=3)
    assert [i.key for i in got] == ["t1"]
    assert got[0].content.startswith("Caroline: Gonna go see the doc")
    # never carried, never counted as memory
    assert {i.key for i in pkt.hot()} == {"t1", "t2"}
    assert pkt.stats()["index_items"] == 1 and pkt.stats()["total_items"] == 2


def test_a_memory_hit_twice_is_returned_once_at_its_better_score(db):
    pkt = _pkt(db)
    pkt.remember("t1", "pottery class with the kids", index=["pottery class kids"])
    got = pkt.recall("pottery class kids", k=5)
    assert [i.key for i in got] == ["t1"]


def _city(pkt, key, text, value, days_ago, now, origin="user"):
    pkt.remember(key, text, valid_at=now - days_ago * DAY, origin=origin,
                 index=[P.IndexEntry(text, key="user|home city", value=value)])


def test_a_chain_comes_back_once_newest_first_with_its_history(db):
    pkt, now = _pkt(db), time.time()
    _city(pkt, "n1", "Alice lives in Bergen", "Bergen", 300, now)
    _city(pkt, "n2", "Alice moved to Osaka", "Osaka", 120, now)
    _city(pkt, "n3", "Alice's home city is Leeds", "Leeds", 10, now)
    pkt.remember("d1", "Alice's sister lives in Austin", valid_at=now - 5 * DAY)

    got = pkt.recall("where does Alice live", k=5)
    top = {i.key: i for i in got}
    assert "n1" not in top and "n2" not in top            # only as history
    assert [e.key for e in top["n3"].earlier] == ["n2", "n1"]   # newest first

    # A query that matches only an old statement brings the current one in.
    got = pkt.recall("Bergen", k=1)
    assert [i.key for i in got] == ["n3"] and got[0].earlier[-1].key == "n1"

    text = PacketBuilder(pkt, budget_tokens=500, history="inline").build(
        query="Bergen", k=3).text
    lines = text.splitlines()
    i = next(n for n, line in enumerate(lines) if "Leeds" in line)
    assert lines[i].startswith("- [") and lines[i + 1].startswith("    earlier: [")
    assert "Osaka" in lines[i + 1] and "Bergen" in lines[i + 2]
    assert sum("Bergen" in line for line in lines) == 1
    # by default the history is a section of its own, after the context
    text = PacketBuilder(pkt, budget_tokens=500).build(query="Bergen", k=3).text
    assert "earlier:" not in text
    assert text.index("Leeds") < text.index("EARLIER VALUES") < text.index("Osaka")


def test_valid_time_decides_not_write_order(db):
    pkt, now = _pkt(db), time.time()
    _city(pkt, "n3", "Alice's home city is Leeds", "Leeds", 10, now)     # newest first
    _city(pkt, "n1", "Alice lives in Bergen", "Bergen", 300, now)
    got = pkt.recall("Alice Bergen Leeds", k=5)
    assert [i.key for i in got] == ["n3"] and [e.key for e in got[0].earlier] == ["n1"]


def test_untrusted_memory_never_makes_a_trusted_one_history(db):
    pkt, now = _pkt(db), time.time()
    _city(pkt, "n1", "Alice lives in Bergen", "Bergen", 300, now)
    _city(pkt, "x1", "Alice moved to Mars", "Mars", 1, now, origin="external")
    got = {i.key: i for i in pkt.recall("Alice Bergen Mars", k=5)}
    assert "n1" in got and not got["n1"].earlier
    assert "x1" in got                                   # still returned, quoted later
    # but a trusted newer value does make an untrusted older one history
    _city(pkt, "n2", "Alice moved to Oslo", "Oslo", 0.5, now)
    got = {i.key: i for i in pkt.recall("Alice Bergen Mars Oslo", k=5)}
    assert {e.key for e in got["n2"].earlier} == {"n1", "x1"}


def test_additive_facts_and_repeats_are_not_superseded(db):
    pkt, now = _pkt(db), time.time()
    hobby = Fact(subject="user", predicate="hobby", object="hiking",
                 valid_at=int(now - 50 * DAY),
                 metadata={"statement": "Alice likes hiking", "replaces": False})
    hobby2 = Fact(subject="user", predicate="hobby", object="chess",
                  valid_at=int(now - 5 * DAY),
                  metadata={"statement": "Alice likes chess", "replaces": False})
    assert P.IndexEntry.of(hobby).key is None
    pkt.remember("h1", "Alice: I love hiking", index=[hobby])
    pkt.remember("h2", "Alice: chess is my new thing", index=[hobby2])
    got = {i.key: i for i in pkt.recall("Alice hiking chess", k=5)}
    assert set(got) == {"h1", "h2"} and not got["h1"].earlier and not got["h2"].earlier

    _city(pkt, "c1", "Alice lives in Leeds", "Leeds", 40, now)
    _city(pkt, "c2", "Alice is still in Leeds", "leeds", 4, now)        # same value
    got = {i.key: i for i in pkt.recall("Alice Leeds", k=5)}
    assert {"c1", "c2"} <= set(got) and not got["c2"].earlier


def test_a_fact_converts_to_an_entry_with_its_key(db):
    f = Fact(subject="User", predicate="Home City", object="Denver", valid_at=1700000000,
             metadata={"statement": "On 2023/11/14 Alice moved to Denver.", "replaces": True})
    e = P.IndexEntry.of(f)
    assert (e.text, e.key, e.value, e.valid_at) == (
        "On 2023/11/14 Alice moved to Denver.", "user|home city", "Denver", 1700000000.0)
    pkt = _pkt(db)
    pkt.remember("n1", "Alice: we moved to Denver!", index=[f])
    assert pkt.known_keys() == ["user|home city"]
    assert _pkt(db, ("org", "bob")).known_keys() == []


def test_rewrite_replaces_the_index_and_omitting_it_keeps_it(db):
    pkt = _pkt(db)
    pkt.remember("n1", "first", index=["alpha zebra", "beta yak"])
    assert pkt.stats()["index_items"] == 2
    pkt.remember("n1", "second", index=["gamma"])
    assert pkt.stats()["index_items"] == 1
    assert pkt.recall("zebra", k=3) == []
    pkt.remember("n1", "third")                          # index kept
    assert [i.content for i in pkt.recall("gamma", k=3)] == ["third"]


def test_forget_and_erase_take_the_index_with_the_memory(db):
    pkt = _pkt(db)
    pkt.remember("n1", "Alice: knee", index=["Alice orthopedist appointment"])
    pkt.forget("n1")
    assert pkt.recall("orthopedist", k=3) == [] and pkt.stats()["index_items"] == 0
    pkt.remember("n2", "Alice: allergy", index=["Alice is allergic to penicillin"])
    pkt.erase("n2")
    assert pkt.recall("penicillin", k=3) == []
    ns = pkt.namespace
    assert all(m is None or m.source == "_forgotten" for m in
               db.get_metadata_many(list(db.ids_in_namespace(ns))))


def test_hot_text_groups_without_changing_hot(db):
    pkt, now = _pkt(db), time.time()
    _city(pkt, "n1", "Alice lives in Bergen", "Bergen", 3, now)
    _city(pkt, "n2", "Alice's home city is Leeds", "Leeds", 1, now)
    hot = pkt.hot()
    text = pkt.hot_text()
    assert text.index("Leeds") < text.index("EARLIER VALUES") < text.index("Bergen")
    assert "    earlier: [" in pkt.hot_text(history="inline")
    assert "Bergen" not in pkt.hot_text(history="none")
    assert all(not i.earlier for i in pkt.hot()) and pkt.hot() == hot


def test_memories_without_index_behave_as_before(db):
    pkt = _pkt(db)
    pkt.remember("a", "plain note about tea")
    items = pkt.recall("tea", k=3)
    assert pkt.group_superseded(items) == items and items[0].supkeys == ()
    _rid, m = keys.get(db, pkt.namespace, "a")
    assert not m.get_attribute(P.INDEX_N) and not m.get_attribute(P.SUPKEYS)
