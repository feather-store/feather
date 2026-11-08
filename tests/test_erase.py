"""erase(): nothing of an erased record remains, anywhere Feather writes.

forget() is not erasure: the record's bytes stay in the file until a vacuum,
its vector stays in the graph as a deleted node (a vacuum writes those too),
the WAL holds what was written since the last save, and a summary or
observation made from it survives. These tests write personal data into every
one of those places, erase it, and then read every file Feather has on disk,
byte by byte, for any trace of it.
"""
import glob
import json
import os
import struct
import zlib

import numpy as np
import pytest

import feather_db
from feather_db import DB, Metadata, core

DIM = 16


def vec(seed):
    return np.random.default_rng(seed).random(DIM).astype(np.float32)


def put(db, rid, content, ns="app", entity="", modality="text", seed=None):
    m = Metadata()
    m.namespace_id, m.entity_id, m.content = ns, entity, content
    db.add(rid, vec(rid if seed is None else seed), m, modality=modality)


def on_disk(path, *needles):
    """{needle: [files holding it]} over every file Feather keeps for `path`
    (.feather, .wal, .wal.old, .tmp, .lock, and the kept files under .history/)."""
    found = {}
    files = []
    for f in sorted(glob.glob(path + "*")):
        if os.path.isdir(f):
            files += sorted(os.path.join(r, n) for r, _, ns in os.walk(f) for n in ns)
        else:
            files.append(f)
    for f in files:
        data = open(f, "rb").read()
        for n in needles:
            if (n.encode() if isinstance(n, str) else n) in data:
                found.setdefault(n if isinstance(n, str) else "vector", []).append(
                    os.path.basename(f))
    return found


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "e.feather")


@pytest.fixture(autouse=True)
def _no_halt():
    yield
    core._halt_erase_after_log(False)


# ── the bug, and the fix ──────────────────────────────────────────────────

def test_forget_leaves_the_bytes_and_erase_does_not(path):
    db = DB.open(path, dim=DIM)
    put(db, 1, "ALICE-SECRET 12 Baker Street", entity="alice")
    db.save()
    db.forget(1)
    db.save()
    assert on_disk(path, "ALICE-SECRET", vec(1).tobytes()), "forget() erased the bytes now?"
    r = db.erase(entity="alice")
    assert r["verified"]
    assert on_disk(path, "ALICE-SECRET", vec(1).tobytes()) == {}
    db.close()


def test_every_copy_is_gone_from_every_file(path):
    db = DB.open(path, dim=DIM)
    # A second modality first: an add without metadata blanks the record's.
    db.add(1, vec(101), modality="visual")
    put(db, 1, "ALICE-SECRET old phone 555-0101", entity="alice")
    put(db, 2, "ALICE-SECRET diary", entity="alice")
    put(db, 3, "BOB-KEEP likes tea", entity="bob")
    db.save()
    m = db.get_metadata(1)
    m.content = "ALICE-SECRET new phone 555-0202"          # the old version is superseded
    db.update_metadata(1, m)
    db.save()
    db.forget(2)                                            # a forgotten record
    put(db, 4, "ALICE-SECRET written after the last save", entity="alice")   # WAL only
    needles = ("ALICE-SECRET", "555-0101", vec(1).tobytes(), vec(101).tobytes(),
               vec(2).tobytes(), vec(4).tobytes())
    # Every copy is really there first: superseded, forgotten, WAL-only.
    assert on_disk(path, "555-0101") == {"555-0101": ["e.feather"]}
    assert "e.feather" in on_disk(path, vec(2).tobytes())["vector"]
    assert "e.feather" in on_disk(path, vec(101).tobytes())["vector"]
    assert on_disk(path, vec(4).tobytes()) == {"vector": ["e.feather.wal"]}

    r = db.erase(entity="alice")
    assert on_disk(path, *needles) == {}
    assert on_disk(path, "BOB-KEEP") == {"BOB-KEEP": ["e.feather"]}
    assert sorted(db.get_all_ids("text")) == [3] and db.get_all_ids("visual") == []
    assert r["wal_bytes_after"] == 0 and r["full_rewrite"] and r["verified"]
    db.close()
    assert on_disk(path, *needles) == {}


def test_int8_originals_are_erased_with_the_record(path):
    """An in-RAM int8 modality keeps float originals in the file for refine."""
    db = DB.open(path, dim=DIM)
    db.set_int8_ram("text", 1.0)
    for i in range(1, 40):
        put(db, i, f"note {i}", entity="alice" if i < 5 else "bob")
    db.save()
    assert on_disk(path, vec(2).tobytes())
    db.erase(entity="alice")
    assert all(on_disk(path, vec(i).tobytes()) == {} for i in range(1, 5))
    assert on_disk(path, vec(10).tobytes())                 # bob's are kept
    db.close()


# ── what is selected ──────────────────────────────────────────────────────

def test_a_namespace_prefix_stops_at_the_dot(path):
    db = DB.open(path, dim=DIM)
    for rid, ns in enumerate(["org.u1", "org.u1.prefs", "org.u10", "org"], start=1):
        put(db, rid, f"content of {ns}", ns=ns)
    assert db.erase(namespace="org.u1", prefix=True)["matched"] == [1, 2]
    assert sorted(db.get_all_ids("text")) == [3, 4]
    assert db.erase(namespace="org", dry_run=True)["matched"] == [4]   # exact without prefix
    db.close()


def test_namespace_and_entity_intersect_and_ids_add(path):
    db = DB.open(path, dim=DIM)
    put(db, 1, "a", ns="app", entity="alice")
    put(db, 2, "b", ns="other", entity="alice")
    put(db, 3, "c", ns="app", entity="bob")
    r = db.erase(namespace="app", entity="alice", ids=[3, 99], dry_run=True)
    assert r["matched"] == [1, 3] and r["not_found"] == [99]
    db.close()


def test_an_erase_without_a_selector_is_refused(path):
    db = DB.open(path, dim=DIM)
    put(db, 1, "x")
    with pytest.raises(ValueError):
        db.erase()
    assert db.get_metadata(1) is not None
    db.close()


# ── derived records ───────────────────────────────────────────────────────

def test_records_derived_from_or_citing_it_go_too(path):
    db = DB.open(path, dim=DIM)
    put(db, 1, "ALICE-SECRET source", entity="alice")
    put(db, 2, "SUMMARY-SECRET of alice")          # 1 --consolidated_into--> 2
    put(db, 3, "FACT-SECRET extracted")            # 3 --extracted_from--> 1
    put(db, 4, "OBS-SECRET drawn from the fact")   # 4 --cites--> 3 (transitive)
    put(db, 5, "RELATED-KEEP similar")             # 5 --related_to--> 1: not derived
    put(db, 6, "CONTRA-KEEP")                      # 6 --contradicts--> 2: not derived
    db.link(1, 2, "consolidated_into")
    db.link(3, 1, "extracted_from")
    db.link(4, 3, "cites")
    db.link(5, 1, "related_to")
    db.link(6, 2, "contradicts")
    db.save()

    plan = db.erase(entity="alice", dry_run=True)
    assert {(d["id"], d["via"], d["rel"]) for d in plan["derived"]} == {
        (2, 1, "consolidated_into"), (3, 1, "extracted_from"), (4, 3, "cites")}
    assert plan["pruned_edges"] == [{"from": 5, "to": 1, "rel": "related_to"},
                                    {"from": 6, "to": 2, "rel": "contradicts"}]
    assert db.get_metadata(2) is not None, "a dry run changed something"

    r = db.erase(entity="alice")
    assert r["erased"] == 4
    assert sorted(db.get_all_ids("text")) == [5, 6]
    assert db.get_edges(5) == [] and db.get_edges(6) == []
    assert on_disk(path, "SECRET") == {}
    assert set(on_disk(path, "RELATED-KEEP", "CONTRA-KEEP")) == {"RELATED-KEEP", "CONTRA-KEEP"}
    db.close()


def test_the_edge_kinds_followed_can_be_chosen(path):
    db = DB.open(path, dim=DIM)
    put(db, 1, "source", entity="alice")
    put(db, 2, "derived")
    db.link(2, 1, "derived_from")
    assert db.erase(entity="alice", cites=[], dry_run=True)["derived"] == []
    assert db.erase(entity="alice", dry_run=True)["derived"][0]["id"] == 2
    assert list(core.ERASE_CITES)[:2] == ["derived_from", "extracted_from"]
    db.close()


# ── the receipt ───────────────────────────────────────────────────────────

def test_the_receipt(path):
    db = DB.open(path, dim=DIM)
    put(db, 1, "x", entity="alice")
    put(db, 2, "y", entity="bob")
    db.save()
    r = db.erase(entity="alice")
    assert r["receipt_id"].startswith("er_") and r["completed_at"] > 0
    assert r["matched"] == [1] and r["erased"] == 1 and not r["dry_run"]
    assert r["request"]["entity"] == "alice" and "derived_from" in r["request"]["cites"]
    assert r["graph_nodes_removed"] >= 1 and r["generation"] > 0
    assert r["verified"] and r["problems"] == [] and r["residuals"]
    json.dumps(r)                                   # loggable as it is
    db.close()


def test_a_dry_run_changes_nothing(path):
    db = DB.open(path, dim=DIM)
    put(db, 1, "x", entity="alice")
    db.save()
    before = open(path, "rb").read()
    r = db.erase(entity="alice", dry_run=True)
    assert r["dry_run"] and r["matched"] == [1] and not r["verified"]
    assert db.get_metadata(1) is not None and open(path, "rb").read() == before
    db.close()


def test_a_read_only_handle_may_plan_but_not_erase(path):
    db = DB.open(path, dim=DIM)
    put(db, 1, "x", entity="alice")
    db.close()
    with DB.open(path, dim=DIM, read_only=True) as ro:
        assert ro.erase(entity="alice", dry_run=True)["matched"] == [1]
        with pytest.raises(Exception):
            ro.erase(entity="alice")


# ── interrupted erasures finish ───────────────────────────────────────────

def _halted(path):
    db = DB.open(path, dim=DIM)
    put(db, 1, "ALICE-SECRET one", entity="alice")
    put(db, 2, "BOB-KEEP", entity="bob")
    db.save()
    core._halt_erase_after_log(True)
    with pytest.raises(RuntimeError, match="halted"):
        db.erase(entity="alice")
    core._halt_erase_after_log(False)
    assert db.get_metadata(1) is None               # applied in memory
    assert on_disk(path, "ALICE-SECRET")            # ...not yet on disk
    return db


def test_a_crash_after_logging_is_finished_by_the_next_open(path):
    db = _halted(path)
    db.close(save=False)                            # the crash
    assert on_disk(path, "ALICE-SECRET")
    for _ in range(2):                              # replay is idempotent
        with DB.open(path, dim=DIM, read_only=True) as ro:
            assert ro.get_metadata(1) is None and ro.get_metadata(2) is not None
    assert on_disk(path, "ALICE-SECRET")            # readers never write
    db = DB.open(path, dim=DIM)
    assert on_disk(path, "ALICE-SECRET", vec(1).tobytes()) == {}
    assert sorted(db.get_all_ids("text")) == [2]
    db.close()


@pytest.mark.parametrize("how", ["close", "save"])
def test_a_halted_erasure_is_finished_before_any_checkpoint(path, how):
    """A checkpoint rotates the WAL away. If it ran while the erased vectors
    were still graph nodes, nothing would be left to finish the erasure."""
    db = _halted(path)
    getattr(db, how)()
    assert on_disk(path, "ALICE-SECRET", vec(1).tobytes()) == {}
    if how == "save":
        db.close()


# ── corrupt input ─────────────────────────────────────────────────────────

def test_an_erase_record_naming_more_ids_than_it_holds_is_refused(path):
    db = DB.open(path, dim=DIM)
    put(db, 1, "x")
    db.close(save=False)
    payload = struct.pack("<I", 1 << 30) + struct.pack("<Q", 1)   # claims 2^30 ids
    rec = bytearray(struct.pack("<BQI", 0x0B, 0, len(payload)) + payload + b"\0\0\0\0")
    struct.pack_into("<I", rec, len(rec) - 4, zlib.crc32(bytes(rec[:-4])))
    with open(path + ".wal", "ab") as f:
        f.write(rec)
    before = open(path + ".wal", "rb").read()
    with pytest.raises(Exception, match="more ids than it holds"):
        DB.open(path, dim=DIM)
    assert open(path + ".wal", "rb").read() == before


# ── the layers above ──────────────────────────────────────────────────────

def test_a_pocket_does_not_serve_erased_text_from_its_cache(path):
    from feather_db.pocket import pocket
    db = DB.open(path, dim=DIM)
    p = pocket(db, ("org", "u1"))
    p.remember("address", "ALICE-SECRET 12 Baker Street")
    assert "ALICE-SECRET" in p.hot_text()            # memoised now
    db.erase(namespace="org.u1")
    assert "ALICE-SECRET" not in p.hot_text()
    db.close()


@pytest.fixture
def mem(tmp_path):
    pytest.importorskip("langgraph", reason="needs langgraph")
    from feather_db.integrations.langgraph_store import FeatherStore
    from feather_db.integrations.agent_memory import AgentMemory
    store = FeatherStore(str(tmp_path / "m.feather"), dim=DIM)
    yield (AgentMemory(store, org="hawky", user="u1", agent="creative", brand="nike"),
           AgentMemory(store, org="hawky", user="u2", agent="creative", brand="nike"),
           str(tmp_path / "m.feather"))
    store.close()


def test_erase_subject_takes_everything_about_one_person(mem):
    u1, u2, path = mem
    u1.preferences.set("comms", "U1-SECRET prefers phone")
    u1.episodes.record("U1-SECRET missed the deadline")
    u1.observations.note("style", "U1-SECRET seems rushed", evidence=["comms"])
    u1.store.put(("hawky", "entities"), "u1", {"text": "U1-SECRET entity"})
    u2.preferences.set("comms", "U2-KEEP prefers email")
    r = u1.erase_subject()
    assert r["verified"] and r["erased"] == 4
    assert on_disk(path, "U1-SECRET") == {}
    assert u2.preferences.get("comms")["text"] == "U2-KEEP prefers email"
    assert u1.preferences.get("comms") is None


def test_erasing_one_item_takes_the_observations_citing_it(mem):
    u1, _, path = mem
    u1.preferences.set("comms", "U1-SECRET prefers phone")
    u1.preferences.set("tz", "U1-KEEP lives in IST")
    u1.observations.note("style", "OBS-SECRET calls a lot", evidence=["comms"])
    u1.observations.note("other", "OBS-KEEP is punctual", evidence=["tz"])
    # Written before observations cited by edge: the key in the value only.
    u1.store.put(u1.observations.namespace, "legacy",
                 {"kind": "observations", "text": "LEGACY-SECRET", "evidence": ["comms"]})
    r = u1.erase("preferences", "comms")
    assert r["verified"]
    assert on_disk(path, "U1-SECRET", "OBS-SECRET", "LEGACY-SECRET") == {}
    assert u1.observations.get("other") and u1.preferences.get("tz")


def test_erasing_a_deleted_store_item_finishes_the_job(mem):
    u1, _, path = mem
    store = u1.store
    store.put(("hawky", "notes"), "n1", {"text": "NOTE-SECRET"})
    store.delete(("hawky", "notes"), "n1")          # a forget: the bytes remain
    assert on_disk(path, "NOTE-SECRET")
    store.erase(("hawky", "notes"), "n1")
    assert on_disk(path, "NOTE-SECRET") == {}


# ── the graph after an erasure: scrubbed, not rebuilt ─────────────────────

def test_search_stays_exact_after_most_of_the_graph_is_erased(path):
    """Erased nodes are cut out with their neighbours' links repaired (no
    rebuild). Erasing 90% of a graph, entry point and all, must leave every
    survivor findable, before and after a reload."""
    db = DB.open(path, dim=DIM)
    for i in range(1, 401):
        put(db, i, f"r{i}", entity="gone" if i % 10 else "kept")
    r = db.erase(entity="gone")
    assert r["erased"] == 360 and r["graph_nodes_removed"] == 360 and r["verified"]
    db.set_ef(50)
    for reopened in (False, True):
        if reopened:
            db.close()
            db = DB.open(path, dim=DIM)
            db.set_ef(50)
        for i in range(10, 401, 10):
            assert db.search(vec(i), k=1, record_salience=False)[0].id == i
    db.compact()                                   # the scrubbed slots go here
    assert db.search(vec(200), k=1, record_salience=False)[0].id == 200
    db.close()


def test_an_erasure_also_removes_the_vectors_of_earlier_forgets(path):
    db = DB.open(path, dim=DIM)
    put(db, 1, "forgotten long ago", entity="carol")
    put(db, 2, "alice", entity="alice")
    db.save()
    db.forget(1)
    db.save()
    assert on_disk(path, vec(1).tobytes())          # a forget leaves it
    db.erase(entity="alice")
    assert on_disk(path, vec(1).tobytes(), vec(2).tobytes()) == {}
    db.close()


# ── nothing else in the process or on the machine keeps serving it ────────

def test_a_reader_opened_before_an_erasure_refuses_after_it(path):
    import time
    w = DB.open(path, dim=DIM)
    put(w, 1, "ALICE-SECRET", entity="alice")
    put(w, 2, "BOB-KEEP", entity="bob")
    w.save()
    r = DB.open(path, dim=DIM, read_only=True)
    assert r.get_metadata(1).content == "ALICE-SECRET"
    w.erase(entity="alice")
    time.sleep(0.06)                               # readers check every 50 ms
    with pytest.raises(feather_db.ClosedError, match="erased"):
        r.get_metadata(2)
    r.close()
    with DB.open(path, dim=DIM, read_only=True) as r:
        assert r.get_metadata(1) is None and r.get_metadata(2).content == "BOB-KEEP"
    w.close()


def test_an_erasure_empties_the_embedding_caches(path):
    from feather_db.integrations import embedders
    calls = []
    embed = embedders._cached(lambda t: calls.append(t) or [1.0] * DIM, 64)
    embed("ALICE-SECRET")
    embed("ALICE-SECRET")
    assert calls == ["ALICE-SECRET"]               # the second was a hit
    db = DB.open(path, dim=DIM)
    put(db, 1, "x", entity="alice")
    db.erase(entity="alice")
    embed("ALICE-SECRET")
    assert calls == ["ALICE-SECRET"] * 2           # the cached vector was dropped
    db.close()


def test_a_pocket_erases_one_key_or_its_whole_scope(path):
    from feather_db.pocket import pocket
    db = DB.open(path, dim=DIM)
    pocket(db, ("org",)).remember("rule", "ORG-RULE-KEEP")
    p = pocket(db, ("org", "u1"))
    p.remember("a", "U1-SECRET-A")
    p.remember("b", "U1-SECRET-B")
    with p.session() as s:
        s.note("U1-SCRATCH")
        assert p.erase("a")["erased"] == 1
        assert on_disk(path, "U1-SECRET-A") == {} and on_disk(path, "U1-SECRET-B")
        p.erase_all()                              # own scope and its scratch
    assert on_disk(path, "U1-SECRET", "U1-SCRATCH") == {}
    assert on_disk(path, "ORG-RULE-KEEP")          # inherited: not this scope's
    db.close()


# ── over MCP: forget erases, forgetting everything needs the user ─────────

def _mcp(tmp_path):
    pytest.importorskip("mcp", reason="needs the MCP SDK")
    from feather_db.integrations.mcp_agent import build
    from feather_db.pocket import pocket
    p = str(tmp_path / "agent.feather")
    d = DB.open(p, dim=DIM)
    pocket(d, ("org",)).remember("policy", "ORG-POLICY-KEEP")
    d.close()
    return p, build(p, "org.u1", dim=DIM, budget_tokens=400)


def _session(server, user, fn):
    import asyncio
    from tests.mcp_harness import connect

    async def go():
        async with connect(server, user=user) as mcp:
            return await fn(mcp)
    return asyncio.run(go())


def test_mcp_forget_erases_the_memory(tmp_path):
    p, server = _mcp(tmp_path)

    async def go(mcp):
        await mcp.call("remember", key="addr", content="ALICE-SECRET 12 Baker Street")
        assert "erased 'addr'" in await mcp.call("forget", key="addr")
        assert "nothing was stored" in await mcp.call("forget", key="addr")
    _session(server, "confirms", go)
    assert on_disk(p, "ALICE-SECRET") == {}


@pytest.mark.parametrize("user, erased", [("confirms", True), ("declines", False), (None, False)])
def test_mcp_forget_everything_needs_the_users_confirmation(tmp_path, user, erased):
    p, server = _mcp(tmp_path)

    async def go(mcp):
        await mcp.call("remember", key="a", content="U1-SECRET-A")
        return await mcp.call("forget_everything")
    out = _session(server, user, go)
    assert ("erased 1 memories" in out) == erased
    assert bool(on_disk(p, "U1-SECRET-A")) != erased
    assert on_disk(p, "ORG-POLICY-KEEP")            # inherited rules stay
