"""The write path: memory written now, indexed later by a durable queue with
retries; index records versioned and re-made lazily; known_keys() from a key
registry; index writes batched (one round trip remotely). A fake extractor
stands in for the model: one fact per "<Name> lives in <City>" sentence."""
import json
import re
import time

import numpy as np
import pytest

import feather_db
from feather_db import DB, keys
import importlib
# The module: `feather_db.pocket` is also the name of the factory function.
P = importlib.import_module("feather_db.pocket")
from feather_db.extractors.base import Fact
from feather_db.ingest import IngestQueue, JOB_STATE, PENDING, extractor_id

LIVES = re.compile(r"(\w+) lives in (\w+)")


class FakeExtractor:
    name = "fake"

    def __init__(self, version="1", fail=0, tag="", on_extract=None):
        self.version, self.fail, self.tag = version, fail, tag
        self.on_extract = on_extract
        self.calls = 0
        self.sessions = 0

    def _facts(self, text, turn=None):
        return [Fact(subject=m.group(1), predicate="home city", object=m.group(2),
                     metadata={"statement": f"{self.tag}{m.group(1)} lives in {m.group(2)}",
                               "replaces": True, "turn": turn})
                for m in LIVES.finditer(text)]

    def _call(self, context):
        self.calls += 1
        if self.on_extract:
            self.on_extract(self.calls)
        if self.calls <= self.fail:
            context["on_error"]("model timed out")      # as SessionFactExtractor reports
            return False
        return True

    def extract(self, text, *, context=None):
        return self._facts(text) if self._call(context) else []

    def extract_session(self, turns, session_date, *, context=None):
        self.sessions += 1
        if not self._call(context):
            return []
        return [f for i, t in enumerate(turns) for f in self._facts(t["content"], i)]


@pytest.fixture
def db(tmp_path):
    d = DB.open(str(tmp_path / "i.feather"), dim=8)
    yield d
    if not d.closed:
        d.close()


def _queue(target, ex, **kw):
    kw.setdefault("backoff", 0.01)
    kw.setdefault("poll_seconds", 0.05)
    return IngestQueue(target, ex, **kw)


def _index_texts(db, ns, key):
    ids = list(db.ids_with_attribute(P.INDEX_FOR, key))
    return sorted(m.content for m in db.get_metadata_many(ids)
                  if m is not None and m.source == P._INDEX_KIND and m.namespace_id == ns)


# ── written now, indexed later ────────────────────────────────────────────

def test_a_memory_is_written_at_once_and_indexed_by_the_queue(db):
    pkt = P.pocket(db, ("org", "u1"))
    ex = FakeExtractor()
    q = _queue(pkt, ex, start=False)
    q.remember(pkt, "m1", "Alice lives in Denver")
    assert [i.key for i in pkt.recall("Denver", k=3)] == ["m1"]    # found by its own words
    assert ex.calls == 0 and P.known_keys(db, pkt.namespace) == []
    assert q.stats()["pending"] == 1
    q.start()
    assert q.drain(timeout=10)
    q.close()
    assert ex.calls == 1
    _, m = keys.get(db, pkt.namespace, "m1")
    assert m.get_attribute(P.INDEX_N) == "1"
    assert m.get_attribute(P.INDEX_V) == extractor_id(ex) == "fake@1"
    assert P.known_keys(db, pkt.namespace) == ["alice|home city"]
    assert q.stats()["pending"] == 0


def test_a_failing_extraction_is_retried_not_read_as_no_facts(db):
    pkt = P.pocket(db, ("org", "u1"))
    ex = FakeExtractor(fail=2)
    with _queue(pkt, ex) as q:
        q.remember(pkt, "m1", "Alice lives in Denver")
        assert q.drain(timeout=10)
        assert q.counts["retried"] == 2 and q.counts["done"] == 1
    assert _index_texts(db, pkt.namespace, "m1") == ["Alice lives in Denver"]


def test_a_job_out_of_attempts_is_kept_with_its_error(db):
    pkt = P.pocket(db, ("org", "u1"))
    with _queue(pkt, FakeExtractor(fail=99), max_attempts=3) as q:
        q.remember(pkt, "m1", "Alice lives in Denver")
        assert q.drain(timeout=10)
        [f] = q.failed()
        assert f["attempts"] == 3 and "timed out" in f["error"] and f["kind"] == "memory"
    with _queue(pkt, FakeExtractor()) as q:                  # the model is back
        assert q.retry_failed() == 1
        assert q.drain(timeout=10) and q.failed() == []
    assert _index_texts(db, pkt.namespace, "m1") == ["Alice lives in Denver"]


def test_a_crash_loses_no_job(tmp_path):
    path = str(tmp_path / "c.feather")
    d = DB.open(path, dim=8)
    pkt = P.pocket(d, ("org", "u1"))
    _queue(pkt, FakeExtractor(), start=False).remember(pkt, "m1", "Alice lives in Denver")
    d.close(save=False)                                       # the crash
    d = DB.open(path, dim=8)
    pkt = P.pocket(d, ("org", "u1"))
    with _queue(pkt, FakeExtractor()) as q:                   # a new process resumes it
        assert q.drain(timeout=10)
    assert _index_texts(d, pkt.namespace, "m1") == ["Alice lives in Denver"]
    d.close()


def test_submitting_again_replaces_the_pending_job(db):
    pkt = P.pocket(db, ("org", "u1"))
    q = _queue(pkt, FakeExtractor(), start=False)
    q.remember(pkt, "m1", "Alice lives in Denver")
    q.remember(pkt, "m1", "Alice lives in Austin")
    assert len(list(db.ids_with_attribute(JOB_STATE, PENDING))) == 1


def test_a_memory_rewritten_during_extraction_is_indexed_by_its_new_text(db):
    pkt = P.pocket(db, ("org", "u1"))

    def rewrite(call):
        if call == 1:                                          # while the model "runs"
            pkt.remember("m1", "Alice lives in Austin")
    with _queue(pkt, FakeExtractor(on_extract=rewrite)) as q:
        q.remember(pkt, "m1", "Alice lives in Denver")
        assert q.drain(timeout=10)
    assert _index_texts(db, pkt.namespace, "m1") == ["Alice lives in Austin"]


def test_an_erased_memory_leaves_its_job_with_nothing_to_do(db):
    pkt = P.pocket(db, ("org", "u1"))
    ex = FakeExtractor()
    q = _queue(pkt, ex, start=False)
    q.remember(pkt, "m1", "Alice lives in Denver")
    pkt.erase("m1")
    q.start()
    assert q.drain(timeout=10)
    q.close()
    assert ex.calls == 0 and q.stats()["pending"] == 0


# ── the index is versioned derived data ──────────────────────────────────

def test_a_new_extractor_reindexes_lazily(db):
    pkt = P.pocket(db, ("org", "u1"))
    with _queue(pkt, FakeExtractor("1")) as q:
        q.remember(pkt, "m1", "Alice lives in Denver")
        q.remember(pkt, "m2", "Bob lives in Lima")
        assert q.drain(timeout=10)
    v2 = FakeExtractor("2", tag="v2: ")
    with _queue(pkt, v2, reindex_batch=10) as q:              # an idle worker sweeps
        deadline = time.time() + 10
        while v2.calls < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert q.drain(timeout=10)
    assert v2.calls == 2
    for key, text in (("m1", "v2: Alice lives in Denver"), ("m2", "v2: Bob lives in Lima")):
        assert _index_texts(db, pkt.namespace, key) == [text]
        assert keys.get(db, pkt.namespace, key)[1].get_attribute(P.INDEX_V) == "fake@2"
    with _queue(pkt, v2, start=False) as q:                   # nothing stale left
        assert q.reindex_stale() == 0


def test_the_extractor_id_covers_prompt_limits_and_model():
    from feather_db.extractors.session_facts import SessionFactExtractor

    class Provider:
        def __init__(self, model):
            self._model = model
    a = extractor_id(SessionFactExtractor(Provider("m1")))
    assert a.startswith("session_fact_extractor@") and "+" in a
    assert a == extractor_id(SessionFactExtractor(Provider("m1")))
    assert a != extractor_id(SessionFactExtractor(Provider("m2")))
    assert a != extractor_id(SessionFactExtractor(Provider("m1"), max_facts=5))


# ── the key registry ──────────────────────────────────────────────────────

def test_known_keys_reads_the_registry_not_the_scope(db, monkeypatch):
    pkt = P.pocket(db, ("org", "u1"))
    for i in range(30):
        pkt.remember(f"n{i}", f"note {i}")
    pkt.remember("m1", "Alice moved", index=[P.IndexEntry("Alice lives in Denver",
                                                          key="alice|home city", value="Denver")])
    calls = []
    real = db.get_metadata_many
    monkeypatch.setattr(type(db), "get_metadata_many",
                        lambda self, ids, *a: calls.append(len(ids)) or real(ids, *a))
    assert pkt.known_keys() == ["alice|home city"]
    assert calls == [1]                                       # the registry entry only


def test_the_registry_is_backfilled_for_an_index_written_before_it(tmp_path):
    path = str(tmp_path / "b.feather")
    d = DB.open(path, dim=8)
    pkt = P.pocket(d, ("org", "u1"))
    pkt.remember("m1", "Alice moved", index=[P.IndexEntry("x", key="alice|home city",
                                                          value="Denver")])
    reg = list(d.ids_with_attribute(P.KEYREG_NS, pkt.namespace))
    marker = d.id_for(keys.UPKEEP_KEY, "")                   # nor had the upkeep pass run
    assert marker is not None
    reg.append(marker)
    b = feather_db.WriteBatch()
    for rid in reg:
        b.forget(rid)
    d.write(b)                                                # as an index from before 0.22
    assert P.known_keys(d, pkt.namespace) == []
    d.close()
    P._MIGRATED.clear()                                       # a new process
    d = DB.open(path, dim=8)
    assert P.pocket(d, ("org", "u1")).known_keys() == ["alice|home city"]
    d.close()


def test_erasing_the_data_takes_its_keys_name(db):
    pkt = P.pocket(db, ("org", "u1"))
    pkt.remember("m1", "Alice moved", index=[P.IndexEntry("x", key="alice|home city",
                                                          value="Denver")])
    pkt.remember("m2", "Bob moved", index=[P.IndexEntry("y", key="bob|home city",
                                                        value="Lima")])
    pkt.erase("m1")
    assert pkt.known_keys() == ["bob|home city"]


# ── batched index writes ─────────────────────────────────────────────────

def test_an_index_is_one_write(db, monkeypatch):
    pkt = P.pocket(db, ("org", "u1"))
    writes = []
    real = db.write
    monkeypatch.setattr(type(db), "write", lambda self, b, *a, **k:
                        writes.append(len(b)) or real(b, *a, **k))
    entries = [P.IndexEntry(f"fact {i}", key=f"k{i}|attr", value=str(i)) for i in range(10)]
    pkt.remember("m1", "a note", index=entries)
    assert len(writes) == 1 and writes[0] >= 21               # memory + 10 + 10 registry
    pkt.remember("m1", "a note", index=entries[:3])           # shrinks: stale ones forgotten
    assert len(writes) == 2
    assert len(_index_texts(db, pkt.namespace, "m1")) == 3


def test_ids_for_many_resolves_many_keys_at_once(db):
    pkt = P.pocket(db, ("org", "u1"))
    a = pkt.remember("a", "x")
    b = pkt.remember("b", "y")
    assert db.ids_for_many([("org.u1", "a"), ("org.u1", "nope"), ("org.u1", "b")]) == [a, None, b]
    assert keys.resolve_many(db, [("org.u1", "b"), ("x", "y")]) == [b, None]


# ── AgentMemory sessions ──────────────────────────────────────────────────

@pytest.fixture
def mem(tmp_path):
    pytest.importorskip("langgraph", reason="needs langgraph")
    from feather_db.integrations.langgraph_store import FeatherStore
    from feather_db.integrations.agent_memory import AgentMemory
    store = FeatherStore(str(tmp_path / "m.feather"), dim=8)
    yield AgentMemory(store, org="o", user="u1")
    store.close()


def test_a_queued_session_is_written_at_once_and_indexed_per_turn(mem):
    ex = FakeExtractor()
    q = _queue(mem.store, ex, start=False)
    turns = [{"role": "user", "content": "Alice lives in Denver"},
             {"role": "assistant", "content": "noted"},
             {"role": "user", "content": "Bob lives in Lima"}]
    r = mem.ingest_session(turns, "2024/03/01", queue=q)
    assert r["turns"] == 3 and ex.sessions == 0
    ns = ".".join(mem.notes.namespace)
    assert mem.notes.get(f"{r['queued'].split(P._KEY_SEP)[-1]}#0")["text"] == \
        "user: Alice lives in Denver"
    q.start()
    assert q.drain(timeout=10)
    q.close()
    assert ex.sessions == 1                                   # one call for the session
    sid = r["queued"].split(P._KEY_SEP)[-1]
    assert _index_texts(mem.store.db, ns, f"{sid}#0") == ["Alice lives in Denver"]
    assert _index_texts(mem.store.db, ns, f"{sid}#2") == ["Bob lives in Lima"]
    assert _index_texts(mem.store.db, ns, f"{sid}#1") == []
    assert mem.notes.known_keys() == ["alice|home city", "bob|home city"]
    # A new extractor re-runs the SESSION, once, not each turn.
    v2 = FakeExtractor("2", tag="v2: ")
    with _queue(mem.store, v2, start=False) as q2:
        assert q2.reindex_stale() == 1
        q2.start()
        assert q2.drain(timeout=10)
    assert v2.sessions == 1
    assert _index_texts(mem.store.db, ns, f"{sid}#2") == ["v2: Bob lives in Lima"]


# ── over the Cloud API: round trips do not grow with the index ───────────

def test_a_remote_index_write_is_a_few_requests_whatever_its_size(tmp_path, monkeypatch):
    from tests.test_contracts import _remote
    with _remote(tmp_path, monkeypatch) as rdb:
        sent = []
        real = rdb._session.request
        rdb._session.request = lambda method, url, **kw: sent.append((method, url)) or \
            real(method, url, **kw)
        pkt = P.Pocket(rdb, ("org", "u1"))
        pkt.remember("warm", "x")                 # the dimension, once per pocket
        cost = {}
        for n in (2, 20):
            sent.clear()
            entries = [P.IndexEntry(f"fact {i}", key=f"k{i}|attr", value=str(i))
                       for i in range(n)]
            pkt.remember(f"m{n}", "a note", index=entries)
            cost[n] = len(sent)
        # Read the memory, write it with its index: 2 requests, whatever the
        # index's size (it was 2-3 per index entry, plus one per vector).
        assert cost[20] == cost[2] == 2, cost
        sent.clear()
        assert len(pkt.known_keys()) == 20 and len(sent) == 2     # k0..k19
        sent.clear()
        assert rdb.ids_for_many([("org.u1", "m2"), ("org.u1", "x")])[1] is None
        assert len(sent) == 1


# ── many workers, many processes: claims ─────────────────────────────────

from feather_db.ingest import JOB_LEASE, JOB_OWNER, JOB_RESULT, job_key  # noqa: E402


class SlowExtractor(FakeExtractor):
    def _call(self, context):
        time.sleep(0.02)                         # long enough for claims to collide
        return super()._call(context)


def test_two_queues_on_one_db_run_each_job_once(db):
    pkt = P.pocket(db, ("org", "u1"))
    a, b = SlowExtractor(), SlowExtractor()
    qa = _queue(pkt, a, start=False)
    for i in range(12):
        qa.remember(pkt, f"m{i}", f"P{i} lives in City{i}")
    qb = _queue(pkt, b, start=False)               # another process over the same API
    qa.start(), qb.start()
    assert qa.drain(timeout=20) and qb.drain(timeout=20)
    qa.close(), qb.close()
    assert a.calls + b.calls == 12, (a.calls, b.calls)     # never twice
    assert a.calls and b.calls                              # and shared
    assert qa.counts["done"] + qb.counts["done"] == 12
    for i in range(12):
        assert _index_texts(db, pkt.namespace, f"m{i}") == [f"P{i} lives in City{i}"]


def _claim_as(db, ns, jk, owner, seconds):
    job = db.get_by_key(jk, ns)
    job.set_attribute(JOB_LEASE, repr(time.time() + seconds))
    job.set_attribute(JOB_OWNER, owner)
    db.write(feather_db.WriteBatch().update((ns, jk), job, expected_version=job.version))


def test_a_claim_is_respected_until_its_lease_expires(db):
    pkt = P.pocket(db, ("org", "u1"))
    ex = FakeExtractor()
    _queue(pkt, ex, start=False).remember(pkt, "m1", "Alice lives in Denver")
    _claim_as(db, pkt.namespace, job_key("memory", "m1"), "elsewhere:1:x", 0.4)
    t = time.time()
    with _queue(pkt, ex) as q:
        time.sleep(0.15)
        assert ex.calls == 0                                # another host's live claim
        assert q.drain(timeout=10)
    assert ex.calls == 1 and time.time() - t >= 0.35


def test_a_dead_process_claim_is_taken_over_at_once(db):
    import socket
    import subprocess
    import sys
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()                                                # a pid that no longer runs
    pkt = P.pocket(db, ("org", "u1"))
    ex = FakeExtractor()
    _queue(pkt, ex, start=False).remember(pkt, "m1", "Alice lives in Denver")
    _claim_as(db, pkt.namespace, job_key("memory", "m1"),
              f"{socket.gethostname()}:{p.pid}:dead", 300)
    with _queue(pkt, ex) as q:
        assert q.drain(timeout=5)                           # not 300 s
    assert ex.calls == 1


# ── a crash never pays the model twice ────────────────────────────────────

def test_a_crash_after_extraction_reuses_it(tmp_path, monkeypatch):
    path = str(tmp_path / "x.feather")
    d = DB.open(path, dim=8)
    pkt = P.pocket(d, ("org", "u1"))
    ex1 = FakeExtractor()
    real = P.apply_index

    def die(*a, **k):                    # the process dies between the two writes
        d.close(save=False)
        raise RuntimeError("killed")
    monkeypatch.setattr(P, "apply_index", die)
    q1 = _queue(pkt, ex1, workers=1, lease_seconds=0.2)
    q1.remember(pkt, "m1", "Alice lives in Denver")
    deadline = time.time() + 10
    while not d.closed and time.time() < deadline:
        time.sleep(0.01)
    q1.close()
    assert ex1.calls == 1
    monkeypatch.setattr(P, "apply_index", real)

    d = DB.open(path, dim=8)
    pkt = P.pocket(d, ("org", "u1"))
    assert d.get_by_key(job_key("memory", "m1"), pkt.namespace).get_attribute(JOB_RESULT)
    ex2 = FakeExtractor()
    with _queue(pkt, ex2) as q:
        assert q.drain(timeout=10)
        assert q.counts["reused_extraction"] == 1
    assert ex2.calls == 0                                   # no second model call
    assert _index_texts(d, pkt.namespace, "m1") == ["Alice lives in Denver"]
    assert d.get_by_key(job_key("memory", "m1"), pkt.namespace) is None
    d.close()


def test_a_saved_extraction_of_old_text_is_not_reused(db, monkeypatch):
    pkt = P.pocket(db, ("org", "u1"))
    real, failed = P.apply_index, []

    def fail_once(*a, **k):
        if not failed:
            failed.append(1)
            raise RuntimeError("disk full")
        return real(*a, **k)
    monkeypatch.setattr(P, "apply_index", fail_once)
    ex = FakeExtractor()
    q = _queue(pkt, ex, backoff=60)                          # held back after the failure
    q.remember(pkt, "m1", "Alice lives in Denver")
    deadline = time.time() + 10
    while not failed and time.time() < deadline:
        time.sleep(0.01)
    pkt.remember("m1", "Alice lives in Austin")              # rewritten: the saved one is stale
    q.close()
    with _queue(pkt, ex) as q2:
        q2.submit_memory(pkt.namespace, "m1")                # due now, not after the backoff
        assert q2.drain(timeout=10)
    assert ex.calls == 2
    assert _index_texts(db, pkt.namespace, "m1") == ["Alice lives in Austin"]


def test_erasing_a_memory_takes_its_saved_extraction(db, monkeypatch):
    pkt = P.pocket(db, ("org", "u1"))
    monkeypatch.setattr(P, "apply_index", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("disk full")))
    q = _queue(pkt, FakeExtractor(), backoff=60)
    q.remember(pkt, "m1", "Alice lives in Denver")
    jk = job_key("memory", "m1")
    deadline = time.time() + 10
    while not (db.get_by_key(jk, pkt.namespace).get_attribute(JOB_RESULT)) \
            and time.time() < deadline:
        time.sleep(0.01)
    q.close()
    r = pkt.erase("m1")
    assert db.get_by_key(jk, pkt.namespace) is None          # it held "Alice lives in Denver"
    assert r["erased"] >= 2


# ── a read re-makes what it uses first ───────────────────────────────────

def test_a_read_puts_its_stale_index_first(db):
    pkt = P.pocket(db, ("org", "u1"))
    with _queue(pkt, FakeExtractor("1")) as q:
        q.remember(pkt, "m1", "Alice lives in Denver")
        q.remember(pkt, "m2", "Bob lives in Lima")
        assert q.drain(timeout=10)
    v2 = FakeExtractor("2", tag="v2: ")
    with _queue(pkt, v2, reindex=False) as q:                # no idle sweep: reads only
        assert [i.key for i in pkt.recall("Denver", k=1)] == ["m1"]
        assert q.drain(timeout=10)
        assert q.counts["nudged"] == 1
    assert v2.calls == 1
    assert keys.get(db, pkt.namespace, "m1")[1].get_attribute(P.INDEX_V) == "fake@2"
    assert keys.get(db, pkt.namespace, "m2")[1].get_attribute(P.INDEX_V) == "fake@1"
    assert not P._INDEXERS                                   # closed: reads cost nothing


def test_a_failed_reindex_is_not_resubmitted_by_every_sweep(db):
    pkt = P.pocket(db, ("org", "u1"))
    with _queue(pkt, FakeExtractor("1")) as q:
        q.remember(pkt, "m1", "Alice lives in Denver")
        assert q.drain(timeout=10)
    bad = FakeExtractor("2", fail=99)
    with _queue(pkt, bad, max_attempts=2) as q:
        deadline = time.time() + 10
        while not q.failed() and time.time() < deadline:
            time.sleep(0.02)
        pkt.recall("Denver", k=1)                            # a read nudges too
        time.sleep(0.4)                                      # several sweeps
        assert bad.calls == 2 and q.failed()[0]["attempts"] == 2


# ── FeatherStore over the Cloud API ───────────────────────────────────────

def test_a_remote_store_writes_a_run_of_puts_in_a_few_requests(tmp_path, monkeypatch):
    pytest.importorskip("langgraph", reason="needs langgraph")
    from langgraph.store.base import PutOp
    from feather_db.integrations.langgraph_store import FeatherStore
    from tests.test_contracts import _remote
    with _remote(tmp_path, monkeypatch) as rdb:
        store = FeatherStore(db=rdb, embed=None)
        assert store.dim == rdb.dim() and not store._auto_save
        sent = []
        real = rdb._session.request
        rdb._session.request = lambda method, url, **kw: sent.append((method, url)) or \
            real(method, url, **kw)
        cost = {}
        for n in (2, 30):
            sent.clear()
            store.batch([PutOp(("o", "u1"), f"k{n}-{i}", {"text": f"item {i}"})
                         for i in range(n)])
            cost[n] = len(sent)
        assert cost[30] == cost[2] <= 3, cost
        sent.clear()
        store.batch([PutOp(("o", "u1"), f"k30-{i}", {"text": f"item {i}!"}) for i in range(30)])
        assert len(sent) <= 3, sent                           # rewrites too
        assert store.get(("o", "u1"), "k30-7").value == {"text": "item 7!"}
        store.batch([PutOp(("o", "u1"), "k30-7", None)])
        assert store.get(("o", "u1"), "k30-7") is None
        store.close()
        assert not rdb.closed                                 # the caller's handle


def test_a_run_of_puts_keeps_the_last_of_each_key(tmp_path):
    pytest.importorskip("langgraph", reason="needs langgraph")
    from langgraph.store.base import GetOp, PutOp
    from feather_db.integrations.langgraph_store import FeatherStore
    store = FeatherStore(str(tmp_path / "s.feather"), dim=8, embed=None)
    out = store.batch([PutOp(("o",), "a", {"v": 1}), PutOp(("o",), "a", {"v": 2}),
                       GetOp(("o",), "a"), PutOp(("o",), "a", None), GetOp(("o",), "a")])
    assert out[2].value == {"v": 2} and out[4] is None
    store.close()


# ── the MCP server indexes through the queue ──────────────────────────────

def test_the_mcp_server_indexes_memories_in_the_background(tmp_path):
    pytest.importorskip("mcp", reason="needs the MCP SDK")
    import asyncio
    from feather_db.integrations.mcp_agent import build
    from tests.mcp_harness import connect
    ex = FakeExtractor()
    server = build(str(tmp_path / "a.feather"), "org.agent", dim=8, extractor=ex)
    q = server.ingest_queue
    q.backoff, q.poll_seconds = 0.01, 0.05

    async def go():
        async with connect(server) as mcp:
            out = await mcp.call("remember", key="home", content="Alice lives in Denver")
            assert "indexing queued" in out
            assert await asyncio.to_thread(q.drain, 10)
            stats = json.loads(await mcp.call("memory_stats"))
            assert stats["indexing"]["done"] == 1 and stats["indexing"]["pending"] == 0
            assert "Denver" in await mcp.call("recall", query="where does Alice live")
    asyncio.run(go())
    q.close()
    assert ex.calls == 1
    assert P.known_keys(q.db, "org.agent") == ["alice|home city"]
    q.db.close()


def test_a_new_handle_runs_upkeep_whatever_id_it_gets(tmp_path):
    """Handles are remembered weakly: one opened after another was collected
    can get its id(), and keyed by id() it skipped the pass (CI, flakily)."""
    import gc
    gc.collect()
    before = len(P._MIGRATED)
    for i in range(20):                       # ids are reused within a few rounds
        d = DB.open(str(tmp_path / f"h{i}.feather"), dim=8)
        P.pocket(d, ("org", "u"))
        assert d.id_for(keys.UPKEEP_KEY, "") is not None
        d.close()
        del d
        gc.collect()
    assert len(P._MIGRATED) == before          # none of them is still remembered
