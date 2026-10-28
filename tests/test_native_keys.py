"""Native string keys.

Pocket and FeatherStore used to place a record at a 53-bit SHA-1 of
(namespace, key): two keys that collided shared a record, and the second write
silently overwrote the first. A key is now a field of the record, unique per
namespace_id, and the id is allocated, so:
  * two keys never share a record, whatever their hashes;
  * an id is never handed out twice, not even after a delete and a reload;
  * BM25 indexes the key, so a search for the name a memory was stored under
    finds it;
  * the key persists everywhere a record does (crash, save, vacuum, compact,
    a FEATHER_FORMAT=9 file);
  * records written the old way are found, adopted under the same id, and
    become searchable by key.
"""
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

import feather_db
from feather_db import DB, Metadata, WriteBatch, DuplicateKeyError, keys
from feather_db.pocket import Pocket

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 8


def meta(ns, key, content="", **kw):
    m = Metadata()
    m.namespace_id, m.key, m.content = ns, key, content
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def vec(seed=0):
    return np.random.default_rng(seed).random(DIM, dtype=np.float32)


@pytest.fixture
def db(tmp_path):
    d = DB.open(str(tmp_path / "k.feather"), dim=DIM)
    yield d
    d.close()


# ── the engine ────────────────────────────────────────────────────────────

class TestEngine:
    def test_upsert_addresses_a_record_by_key(self, db):
        i1, v1 = db.upsert(meta("u1", "comms", "prefers async"), vec())
        i2, v2 = db.upsert(meta("u1", "comms", "prefers written async"), vec())
        i3, _ = db.upsert(meta("u2", "comms", "phone"), vec())
        assert i1 == i2 and v2 > v1 and i3 != i1
        assert db.id_for("comms", "u1") == i1 and db.id_for("comms", "u2") == i3
        assert db.get_by_key("comms", "u1").content == "prefers written async"
        assert db.get_by_key("comms", "u1").key == "comms"
        assert db.id_for("missing", "u1") is None and db.get_by_key("missing", "u1") is None
        assert i1 < 2 ** 53, "ids must survive JSON"

    def test_a_key_belongs_to_one_live_record(self, db):
        rid, _ = db.upsert(meta("ns", "k", "first"), vec())
        with pytest.raises(DuplicateKeyError):
            db.add(5, vec(), meta("ns", "k", "explicit id, same key"))
        db.add(6, vec(), meta("ns", "other"))
        with pytest.raises(DuplicateKeyError):
            db.update_metadata(6, meta("ns", "k"))
        with pytest.raises(DuplicateKeyError):
            db.add_batch([7, 8], np.stack([vec(), vec()]), [meta("ns", "x"), meta("ns", "x")])
        with pytest.raises(DuplicateKeyError):
            db.write(WriteBatch().upsert(meta("ns", "y"), vec()).put(9, vec(), meta("ns", "y")))
        assert db.get_by_key("k", "ns").content == "first"
        assert [db.version(i) for i in (5, 7, 8, 9)] == [0, 0, 0, 0]

    def test_a_batch_sees_the_keys_the_ops_before_it_bind_and_free(self, db):
        a, _ = db.upsert(meta("ns", "old"), vec())
        r = db.write(WriteBatch()
                     .update(("ns", "old"), meta("ns", "renamed"))      # frees "old"
                     .upsert(meta("ns", "old", "a new record"), vec())  # takes it
                     .link(("ns", "old"), ("ns", "renamed")))
        assert r.ids[0] == a and r.ids[1] != a and r.ids[2] == r.ids[1]
        assert db.id_for("renamed", "ns") == a and db.id_for("old", "ns") == r.ids[1]
        assert [e.target_id for e in db.get_edges(r.ids[1])] == [a]

    def test_a_forgotten_key_is_free_and_ids_are_never_reused(self, tmp_path):
        path = str(tmp_path / "r.feather")
        d = DB.open(path, dim=DIM)
        old, _ = d.upsert(meta("ns", "k"), vec())
        d.forget(old)
        assert d.id_for("k", "ns") is None
        d.close()                                   # the save drops the record
        d = DB.open(path, dim=DIM)
        new, _ = d.upsert(meta("ns", "k"), vec())
        assert new != old and new > old
        d.close()

    def test_bm25_matches_a_record_by_its_key(self, db):
        rid, _ = db.upsert(meta("u1", "shipping_address", "12 Elm Street"), vec())
        assert [h.id for h in db.keyword_search("shipping", k=5)] == [rid]
        db.upsert(meta("u1", "shipping_address", "4 Oak Road"), vec())      # re-indexed
        assert [h.id for h in db.keyword_search("oak", k=5)] == [rid]
        assert not db.keyword_search("elm", k=5)
        db.forget(rid)
        assert not db.keyword_search("shipping", k=5)

    def test_keys_persist_everywhere(self, tmp_path, monkeypatch):
        path = str(tmp_path / "p.feather")
        src = f"""
import sys, os, signal, json
sys.path.insert(0, {REPO!r})
import numpy as np, feather_db
db = feather_db.DB.open({path!r}, dim={DIM})
out = {{}}
for i in range(5):
    m = feather_db.Metadata(); m.namespace_id = "ns"; m.key = f"k{{i}}"; m.content = f"c{{i}}"
    out[m.key] = db.upsert(m, np.ones({DIM}, np.float32))[0]
print(json.dumps(out), flush=True)
os.kill(os.getpid(), signal.SIGKILL)
"""
        p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
        assert p.returncode == -9, p.stderr
        want = __import__("json").loads(p.stdout.splitlines()[-1])

        def check(d, stage):
            assert {k: d.id_for(k, "ns") for k in want} == want, stage
            assert d.get_by_key("k3", "ns").attributes == {}, "the key leaked into attributes"

        d = DB.open(path, dim=DIM); check(d, "crash replay"); d.close()
        d = DB.open(path, dim=DIM); check(d, "save"); d.vacuum(); d.compact(); check(d, "compact"); d.close()
        d = DB.open(path, dim=DIM); check(d, "vacuum"); d.close()
        monkeypatch.setenv("FEATHER_FORMAT", "9")
        d = DB.open(path, dim=DIM); d.update_importance(want["k0"], 0.5); d.close()
        monkeypatch.delenv("FEATHER_FORMAT")
        d = DB.open(path, dim=DIM); check(d, "FEATHER_FORMAT=9"); d.close()

    def test_concurrent_upserts_of_a_new_key_make_one_record(self, db):
        ids = []
        def go(i):
            ids.append(db.upsert(meta("ns", "shared", f"writer {i}"), vec(i))[0])
        ts = [threading.Thread(target=go, args=(i,)) for i in range(8)]
        for t in ts: t.start()
        for t in ts: t.join()
        assert len(set(ids)) == 1 and len(db.ids_in_namespace("ns")) == 1


# ── collisions: what the hashed ids did ───────────────────────────────────

def test_keys_whose_hashes_collide_stay_two_records(db, monkeypatch):
    """Force every key to the same legacy hash. The old scheme put both
    memories in one record and returned the second for both keys."""
    monkeypatch.setattr(keys, "legacy_id", lambda ns, key: 4242)
    p = Pocket(db, "agent")
    a = p.remember("diet", "vegetarian")
    b = p.remember("allergy", "shellfish")
    assert a != b
    assert db.get_by_key("diet", "agent").content == "vegetarian"
    assert db.get_by_key("allergy", "agent").content == "shellfish"


# ── records written before native keys ────────────────────────────────────

def _legacy_record(db, ns, key, content, source):
    m = Metadata()
    m.namespace_id, m.entity_id, m.content, m.source = ns, key, content, source
    rid = keys.legacy_id(ns, key)
    db.add(rid, vec(), m)
    return rid


class TestLegacy:
    def test_a_pocket_adopts_its_legacy_records_under_the_same_id(self, db):
        rid = _legacy_record(db, "org.agent", "tone", "captions stay lowercase", "pocket")
        db.add(1, vec(1), Metadata())
        db.link(1, rid)                               # an edge into the old record
        p = Pocket(db, ("org", "agent"))
        assert db.id_for("tone", "org.agent") == rid, "not adopted under its own id"
        assert [h.id for h in db.keyword_search("tone", k=5)] == [rid]
        assert p.remember("tone", "captions stay lowercase, always") == rid
        assert len(db.ids_in_namespace("org.agent")) == 1
        assert [e.target_id for e in db.get_edges(1)] == [rid]

    def test_a_featherstore_adopts_its_legacy_items(self, tmp_path):
        pytest.importorskip("langgraph")
        from feather_db.integrations.langgraph_store import FeatherStore
        path = str(tmp_path / "s.feather")
        d = DB.open(path, dim=DIM)
        rid = _legacy_record(d, "acme.u1.preferences", "comms", "comms: async", "langgraph")
        d.close()
        store = FeatherStore(path, dim=DIM, embed=None)
        assert store.id_of(("acme", "u1", "preferences"), "comms") == rid
        store.put(("acme", "u1", "preferences"), "comms", {"text": "written"})
        assert store.id_of(("acme", "u1", "preferences"), "comms") == rid
        assert store.get(("acme", "u1", "preferences"), "comms").value == {"text": "written"}
        store.close()

    def test_a_legacy_id_holding_another_key_is_not_this_key(self, db):
        """Under a collision the hashed id holds ANOTHER key's record. The old
        code returned it; resolve() checks the record says it is this key."""
        stranger = Metadata()
        stranger.namespace_id, stranger.entity_id, stranger.source = "ns", "b", "pocket"
        stranger.content = "b's memory"
        db.add(keys.legacy_id("ns", "a"), vec(), stranger)
        assert keys.resolve(db, "ns", "a") is None
        rid = keys.write(db, "ns", "a", Metadata(), vec())
        assert rid != keys.legacy_id("ns", "a")
        assert db.get_metadata(keys.legacy_id("ns", "a")).content == "b's memory"

    def test_a_required_rule_resolves_by_key(self, db):
        from feather_db import PacketBuilder
        monkey = Pocket(db, ("org", "brand"))
        monkey.remember("compliance", "never claim a cure", pinned=True)
        child = Pocket(db, ("org", "brand", "creative"))
        packet = PacketBuilder(child, budget_tokens=500).build(required=["compliance"])
        assert "never claim a cure" in packet.text
