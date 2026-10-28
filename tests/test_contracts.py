"""The contracts in CONTRACTS.md, tested.

Every test that takes `db` runs twice: against embedded Feather (`DB.open`)
and against the Cloud API through `RemoteDB` — the same calls, the same
assertions. A feature or a behaviour that exists in one mode and not the other
fails here. (The file-level contracts — one writer, consistent read-only
snapshots — are about a file and run embedded only.)
"""
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest

import feather_db
from feather_db import (DB, Metadata, FilterBuilder, WriteBatch, ContextType,
                        VersionConflictError, DuplicateKeyError)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO, "feather-api")
DIM = 16


@contextlib.contextmanager
def _remote(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("FEATHER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FEATHER_DB_DIM", str(DIM))
    monkeypatch.delenv("FEATHER_API_KEY", raising=False)
    monkeypatch.setenv("FEATHER_DEV_MODE", "1")
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from app.main import app
    from feather_db.remote import RemoteDB
    with TestClient(app) as client:
        assert client.post("/v1/namespaces", json={"name": "contracts"}).status_code < 300
        yield RemoteDB("http://testserver", "contracts", session=client)


@pytest.fixture(params=["embedded", "remote"])
def db(request, tmp_path, monkeypatch):
    if request.param == "embedded":
        d = DB.open(str(tmp_path / "contracts.feather"), dim=DIM)
        yield d
        d.close()
    else:
        with _remote(tmp_path, monkeypatch) as d:
            yield d


def meta(ns="", key="", content="", ts=None, **attrs):
    m = Metadata()
    m.namespace_id, m.key, m.content = ns, key, content
    m.timestamp = int(time.time()) if ts is None else ts
    for k, v in attrs.items():
        m.set_attribute(k, str(v))
    return m


def clustered(n, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((20, DIM)).astype(np.float32)
    x = c[rng.integers(0, 20, n)] + rng.standard_normal((n, DIM)).astype(np.float32) * 0.5
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def ids(results):
    return [r.id for r in results]


# ══ Exact results by default ══════════════════════════════════════════════

def test_filtered_search_is_exact(db):
    X = clustered(1500)
    db.add_batch(list(range(1500)), X,
                 [meta(f"org.t{i % 3}", tier=("gold" if i % 5 == 0 else "std")) for i in range(1500)])
    def brute(pool, q):
        pool = np.array(pool)
        return pool[np.argsort(((X[pool] - q) ** 2).sum(1), kind="stable")[:10]].tolist()
    for q in X[:15]:
        t1 = [i for i in range(1500) if i % 3 == 1]
        got = db.search(q, k=10, filter=FilterBuilder().namespace("org.t1").build(), record_salience=False)
        assert ids(got) == brute(t1, q)
        gold = [i for i in range(1500) if i % 5 == 0]
        got = db.search(q, k=10, filter=FilterBuilder().attribute("tier", "gold").build(), record_salience=False)
        assert ids(got) == brute(gold, q)
        got = db.search(q, k=10, filter=FilterBuilder().namespace_prefix("org").build(), record_salience=False)
        assert sorted(ids(got)) == sorted(brute(list(range(1500)), q))


def test_keyword_search_and_newest_are_exact(db):
    for i in range(40):
        db.add(i, clustered(1, i)[0], meta("a" if i % 2 else "b", content=f"note {i} alpha" if i % 4 else f"note {i}",
                                            ts=1000 + i // 3))
    hits = db.keyword_search("alpha", k=100, filter=FilterBuilder().namespace("a").build())
    assert sorted(ids(hits)) == [i for i in range(40) if i % 2 and i % 4]
    want = sorted([i for i in range(40) if i % 2], key=lambda i: (1000 + i // 3, i), reverse=True)
    assert db.newest("a", 100) == want
    assert db.newest("", 5, prefix=True) == sorted(range(40), key=lambda i: (1000 + i // 3, i), reverse=True)[:5]


def test_namespace_graphs_are_opt_in(db):
    assert db.namespace_ann_info()["min_records"] == 0


# ══ Records ═══════════════════════════════════════════════════════════════

def test_what_is_written_is_what_is_read(db):
    m = meta("org.u1", content="prefers async", ts=0, channel="email")   # 0 and "" kept as given
    m.importance, m.confidence, m.ttl, m.source = 0.25, 0.5, 3600, "agent"
    m.type = ContextType.PREFERENCE
    m.tags_json = '["a"]'
    m.entity_id = "u1"
    db.add(7, clustered(1)[0], m)
    got = db.get_metadata(7)
    assert (got.namespace_id, got.content, got.timestamp, got.importance, got.confidence, got.ttl,
            got.source, int(got.type), got.tags_json, got.entity_id, got.get_attribute("channel")) == \
           ("org.u1", "prefers async", 0, 0.25, 0.5, 3600, "agent", 1, '["a"]', "u1", "email")
    assert got.version > 0 and db.version(7) == got.version
    assert np.allclose(db.get_vector(7), clustered(1)[0])
    blank = Metadata()
    db.add(8, clustered(1, 3)[0], blank)
    assert db.get_metadata(8).namespace_id == "" and db.get_metadata(8).timestamp == 0
    assert db.get_metadata(999) is None and db.version(999) == 0


def test_update_replaces_and_importance_updates(db):
    db.add(1, clustered(1)[0], meta("n", content="a"))
    db.add(2, clustered(1, 1)[0], meta("n", content="b"))
    assert db.link(1, 2, "rel", 0.5)
    v = db.version(1)
    db.update_metadata(1, meta("n", content="replaced"))          # no edges given: none kept
    assert db.get_metadata(1).content == "replaced"
    assert list(db.get_edges(1)) == []
    assert db.version(1) > v
    db.update_importance(1, 0.1)
    assert abs(db.get_metadata(1).importance - 0.1) < 1e-6


def test_a_forgotten_record_is_gone(db):
    db.add(1, clustered(1)[0], meta("n", key="k", content="secret words"))
    db.forget(1)
    assert db.get_metadata(1) is None and db.version(1) == 0
    assert db.id_for("k", "n") is None and db.get_by_key("k", "n") is None
    assert db.keyword_search("secret", k=5) == []
    assert 1 not in ids(db.search(clustered(1)[0], k=5, record_salience=False))


# ══ Keys, versions, batches ═══════════════════════════════════════════════

def test_keys(db):
    i1, v1 = db.upsert(meta("u1", key="comms", content="async"), clustered(1)[0])
    i2, v2 = db.upsert(meta("u1", key="comms", content="written"), clustered(1)[0])
    assert i1 == i2 and v2 > v1
    assert db.get_by_key("comms", "u1").content == "written" and db.id_for("comms", "u1") == i1
    assert ids(db.keyword_search("comms", k=5)) == [i1]                  # BM25 finds the key
    with pytest.raises(DuplicateKeyError):
        db.add(5, clustered(1)[0], meta("u1", key="comms"))


def test_compare_and_set(db):
    db.add(1, clustered(1)[0], meta("n", content="a"))
    v = db.version(1)
    v2 = db.put_if(1, meta("n", content="b"), v)
    assert v2 > v
    with pytest.raises(VersionConflictError):
        db.put_if(1, meta("n", content="stale"), v)
    assert db.get_metadata(1).content == "b"
    with pytest.raises(VersionConflictError):
        db.put_if(1, meta("n", content="create-only"), 0)


def test_a_batch_is_all_or_nothing(db):
    b = WriteBatch()
    b.upsert(meta("kb", key="acme", content="entity"), clustered(1)[0])
    b.upsert(meta("kb", key="fact", content="raised a series B"), clustered(1, 1)[0])
    b.link(("kb", "fact"), ("kb", "acme"), "about")
    r = db.write(b)
    assert len(r.ids) == 3 and r.ids[1] == r.ids[2]
    assert [e.target_id for e in db.get_edges(db.id_for("fact", "kb"))] == [db.id_for("acme", "kb")]
    bad = WriteBatch().upsert(meta("kb", key="half"), clustered(1)[0]).link(("kb", "half"), 424242)
    with pytest.raises(ValueError):
        db.write(bad)
    assert db.id_for("half", "kb") is None


def test_an_idempotent_write_is_applied_once(db):
    db.add(1, clustered(1)[0], meta("n", content="original"))
    b = WriteBatch().update(1, meta("n", content="first attempt"))
    first = db.write(b, idempotency_key="req-1")
    db.update_metadata(1, meta("n", content="later write"))
    again = db.write(b, idempotency_key="req-1")
    assert again.replayed and again.versions == first.versions
    assert db.get_metadata(1).content == "later write"


# ══ Graph ═════════════════════════════════════════════════════════════════

def test_link_rules(db):
    db.add(1, clustered(1)[0], meta("n"))
    assert db.link(1, 77, "forward") is True                          # a forward reference
    assert db.link(99, 1) is False                                     # no source: nothing
    assert [(e.target_id, e.rel_type) for e in db.get_edges(1)] == [(77, "forward")]
    db.add(2, clustered(1, 2)[0], meta("n"))
    db.link(2, 1, "back", 0.25)
    assert [(e.source_id, e.rel_type) for e in db.get_incoming(1)] == [(2, "back")]


def test_context_chain_and_graph_export(db):
    X = clustered(6)
    for i in range(6):
        db.add(i, X[i], meta("g", content=f"n{i}"))
    db.link(0, 5, "rel")
    r = db.context_chain(X[0], k=1, hops=1)
    assert {n.id for n in r.nodes} >= {0, 5}
    assert any(e.source == 0 and e.target == 5 for e in r.edges)
    g = json.loads(db.export_graph_json("g", ""))
    assert {n["id"] for n in g["nodes"]} == set(range(6))


# ══ Search options ════════════════════════════════════════════════════════

def test_search_filters_and_options(db):
    X = clustered(30)
    for i in range(30):
        m = meta("s", content=f"doc marker{i}", ts=1000 + i)
        m.type = ContextType.EVENT if i % 2 else ContextType.FACT
        m.tags_json = '["red"]' if i % 3 == 0 else '["blue"]'
        db.add(i, X[i], m)
    events = db.search(X[0], k=30, filter=FilterBuilder().types([ContextType.EVENT]).build(),
                       record_salience=False)
    assert sorted(ids(events)) == [i for i in range(30) if i % 2]
    red = db.search(X[0], k=30, filter=FilterBuilder().contains_tags("red").build(), record_salience=False)
    assert sorted(ids(red)) == [i for i in range(30) if i % 3 == 0]
    late = db.search(X[0], k=30, filter=FilterBuilder().after(1020).build(), record_salience=False)
    assert sorted(ids(late)) == list(range(20, 30))
    top = db.search(X[0], k=3, record_salience=False, with_cosine=True)
    assert top[0].id == 0 and top[0].cosine is not None and abs(top[0].cosine - 1.0) < 1e-4
    hyb = db.hybrid_search(X[3], "marker3", k=3)
    assert hyb[0].id == 3


def test_salience(db):
    db.add(1, clustered(1)[0], meta("n"))
    db.touch(1)
    db.search(clustered(1)[0], k=1)                                   # record_salience default
    db.search(clustered(1)[0], k=1, record_salience=False)
    assert db.get_metadata(1).recall_count == 2


# ══ Listings and namespaces ═══════════════════════════════════════════════

def test_listings(db):
    for i in range(9):
        db.add(i, clustered(1, i)[0], meta(f"ns{i % 3}", tier="x" if i < 4 else "y"))
    assert sorted(db.ids_in_namespace("ns1")) == [1, 4, 7]
    assert sorted(db.ids_with_attribute("tier", "x")) == [0, 1, 2, 3]
    assert sorted(db.all_ids()) == list(range(9)) and sorted(db.get_all_ids("text")) == list(range(9))
    assert sorted(db.list_namespaces()) == ["ns0", "ns1", "ns2"] and db.namespace_size("ns2") == 3
    assert db.purge("ns2") == 3 and db.ids_in_namespace("ns2") == []


def test_expiry(db):
    m = meta("n", ts=int(time.time()) - 100)
    m.ttl = 10
    db.add(1, clustered(1)[0], m)
    db.add(2, clustered(1, 1)[0], meta("n"))
    assert db.forget_expired() == 1 and db.get_metadata(1) is None and db.get_metadata(2) is not None


# ══ Maintenance and settings ══════════════════════════════════════════════

def test_settings_and_maintenance(db):
    X = clustered(50)
    db.add_batch(list(range(50)), X, [meta("m") for _ in range(50)])
    db.set_ef(80)
    assert db.get_ef("text") == 80
    db.set_auto_compact(0.3)
    assert abs(db.get_auto_compact() - 0.3) < 1e-6
    db.set_namespace_ann_min(4096)
    assert db.namespace_ann_info()["min_records"] == 4096
    db.set_namespace_ann_min(0)
    db.set_quantized("text", True)
    assert db.is_quantized("text")
    db.set_quantized("text", False)
    for i in range(10):
        db.forget(i)
    assert db.compact() == 10
    db.save()
    db.vacuum()
    assert db.storage_stats()["format"] == 10 and db.size() == 40
    assert db.dim() == DIM and "text" in db.modality_names()
    db.set_int8_ram("text", 1.0, refine=2)
    assert db.is_int8_ram("text") and db.int8_info("text")["refine"] == 2
    assert ids(db.search(X[20], k=1, record_salience=False)) == [20]


# ══ One writer, many readers (a file contract: embedded) ═════════════════

def test_one_writer_many_readers(tmp_path):
    path = str(tmp_path / "w.feather")
    w = DB.open(path, dim=DIM)
    w.add(1, clustered(1)[0], meta("n"))
    w.save()
    code = f"import sys; sys.path.insert(0, {REPO!r})\nimport feather_db\n"
    second = subprocess.run([sys.executable, "-c", code + f"feather_db.DB.open({path!r}, dim={DIM})"],
                            capture_output=True, text=True)
    assert second.returncode != 0 and "lock" in second.stderr.lower(), "a second writer was let in"
    readers = [subprocess.run([sys.executable, "-c", code +
                               f"r = feather_db.DB.open({path!r}, dim={DIM}, read_only=True)\n"
                               f"print(sorted(r.all_ids()))"], capture_output=True, text=True)
               for _ in range(3)]
    assert all(r.returncode == 0 and r.stdout.strip() == "[1]" for r in readers), readers
    r = DB.open(path, dim=DIM, read_only=True)
    with pytest.raises(Exception):
        r.add(2, clustered(1)[0], meta("n"))
    r.close()
    w.close()


# ══ Read-only means a consistent snapshot ════════════════════════════════

_WRITER = """
import sys, time, numpy as np
sys.path.insert(0, {repo!r})
import feather_db
db = feather_db.DB.open({path!r}, dim=4)
i, end = 0, time.time() + {secs}
while time.time() < end:
    b = feather_db.WriteBatch()                      # three records, one atomic write
    for j in range(3):
        m = feather_db.Metadata(); m.content = str(i + j)
        b.put(i + j, np.full(4, i + j, np.float32), m)
    db.write(b); i += 3
    if i % 21 == 0: db.save()
    if i % 210 == 0: db.vacuum()
db.close()
"""


def test_a_read_only_open_is_a_consistent_snapshot(tmp_path):
    """Records go in as 0, 1, 2, ... three per atomic batch, with commits and
    full rewrites racing the readers. A consistent snapshot is a prefix:
    no gap, no partial batch, every record's content its own id. Before the
    WALGEN chain check, 187 of 7395 opens were torn."""
    path = str(tmp_path / "s.feather")
    secs = 6
    w = subprocess.Popen([sys.executable, "-c", _WRITER.format(repo=REPO, path=path, secs=secs)])
    time.sleep(0.5)
    opens = torn = 0
    end = time.time() + secs - 1
    while time.time() < end:
        try:
            r = DB.open(path, dim=4, read_only=True)
        except Exception:
            continue
        got = sorted(r.all_ids())
        bad = (got != list(range(len(got))) or len(got) % 3 or
               any(r.get_metadata(i).content != str(i) for i in got[-6:]))
        r.close()
        opens += 1
        torn += bool(bad)
    w.wait()
    assert opens > 50, f"only {opens} opens: the test measured nothing"
    assert torn == 0, f"{torn} of {opens} read-only opens saw a torn view"
