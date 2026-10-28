"""Multi-tenant search: a namespace-prefix index and per-namespace ANN.

  * `namespace_prefix` is an indexed filter: a search ranks INSIDE the
    namespaces under the prefix. FeatherStore used to rank everything and keep
    what was under the prefix afterwards, so with enough other tenants a
    user's own search returned [].
  * A namespace with at least `set_namespace_ann_min()` records gets its own
    HNSW graph (links only, over the modality's slots), built in the
    background when a search first needs it and kept current by every write.
    A filtered search inside it walks the graph instead of scanning every
    record; below the threshold it stays an exact scan.
"""
import os
import threading

import numpy as np
import pytest

import feather_db
from feather_db import DB, Metadata, FilterBuilder

DIM = 32


def clustered(n, seed=0, centers=None):
    rng = np.random.default_rng(seed)
    c = centers if centers is not None else rng.standard_normal((40, DIM)).astype(np.float32)
    x = c[rng.integers(0, len(c), n)] + rng.standard_normal((n, DIM)).astype(np.float32) * 0.4
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32), c


def meta(ns, **attrs):
    m = Metadata()
    m.namespace_id = ns
    for k, v in attrs.items():
        m.set_attribute(k, str(v))
    return m


@pytest.fixture
def db(tmp_path):
    d = DB.open(str(tmp_path / "ns.feather"), dim=DIM)
    yield d
    d.close()


def exact(X, ids, q, k=10):
    ids = np.asarray(ids)
    d = ((X[ids] - q) ** 2).sum(1)
    return set(ids[np.argsort(d)[:k]].tolist())


def ids_of(results):
    return {r.id for r in results}


# ── the prefix filter ─────────────────────────────────────────────────────

def test_prefix_matches_the_namespace_and_below_only(db):
    X, _ = clustered(8)
    for i, ns in enumerate(["acme.u1", "acme.u1.prefs", "acme.u1.prefs.deep", "acme.u10",
                            "acme.u10.prefs", "acme", "other.u1", "acmeX"]):
        m = meta(ns)
        m.content = "quarterly note"
        db.add(i, X[i], m)
    got = lambda p: ids_of(db.search(X[0], k=20, filter=FilterBuilder().namespace_prefix(p).build()))
    assert got("acme.u1") == {0, 1, 2}
    assert got("acme.u10") == {3, 4}
    assert got("acme") == {0, 1, 2, 3, 4, 5}
    assert got("acme.u1.prefs") == {1, 2}
    assert got("nobody") == set()
    # BM25 filters per posting through SearchFilter::matches(): the same rule.
    kw = db.keyword_search("note", k=20, filter=FilterBuilder().namespace_prefix("acme.u1").build())
    assert ids_of(kw) == {0, 1, 2}


def test_a_tenants_search_is_ranked_inside_its_namespaces(db):
    """The bug: 300 tenants, a global top-k, then the prefix kept nothing."""
    X, C = clustered(300 * 20, seed=1)
    ids = list(range(len(X)))
    db.add_batch(ids, X, [meta(f"org.t{i % 300}.notes") for i in ids])
    q = X[7]                                 # a record of tenant t7
    mine = [i for i in ids if i % 300 == 7]
    got = db.search(q, k=10, filter=FilterBuilder().namespace_prefix("org.t7").build())
    assert len(got) == 10
    assert all(r.metadata.namespace_id == "org.t7.notes" for r in got)
    assert ids_of(got) == exact(X, mine, q)


def test_featherstore_search_returns_the_users_items_among_many_tenants(tmp_path):
    pytest.importorskip("langgraph")
    from feather_db.integrations.langgraph_store import FeatherStore
    rng = np.random.default_rng(3)
    vecs = {}

    def embed(text):
        if text not in vecs:
            vecs[text] = rng.standard_normal(DIM).astype(np.float32)
        return vecs[text]

    store = FeatherStore(str(tmp_path / "s.feather"), dim=DIM, embed=embed, auto_save=False)
    for t in range(200):
        for j in range(5):
            store.put(("org", f"user{t}", "notes"), f"n{j}", {"text": f"tenant {t} note {j}"})
    calls = []

    class Spy:                               # count the engine searches
        def __init__(self, db):
            self._db = db

        def __getattr__(self, name):
            attr = getattr(self._db, name)
            if name == "search":
                def search(*a, **kw):
                    calls.append(kw.get("k"))
                    return attr(*a, **kw)
                return search
            return attr

    real = store.db
    store.db = Spy(real)
    hits = store.search(("org", "user42"), query="tenant 7 note 3", limit=5)
    store.db = real
    assert len(hits) == 5, "a user's search came back short among many tenants"
    assert all(h.namespace[:2] == ("org", "user42") for h in hits)
    assert len(calls) == 1, f"ranked globally and over-fetched: {len(calls)} searches, k={calls}"
    store.close()


# ── per-namespace graphs ──────────────────────────────────────────────────

def build_tenants(db, n_big=3000, n_small=200, seed=0):
    X, C = clustered(n_big + n_small * 5, seed=seed)
    ns = ["big"] * n_big + [f"small{i % 5}" for i in range(n_small * 5)]
    ids = list(range(len(X)))
    db.add_batch(ids, X, [meta(n) for n in ns])
    return X, C, ns


def graph_search(db, q, ns, k=10, **extra):
    fb = FilterBuilder().namespace(ns)
    for key, v in extra.items():
        fb = fb.attribute(key, v)
    return db.search(q, k=k, filter=fb.build(), record_salience=False)


def warm(db, ns, q):
    graph_search(db, q, ns)                  # first search queues the build
    assert db.wait_for_namespace_graphs(60)


def test_a_big_namespace_gets_a_graph_and_keeps_its_results(db):
    db.set_namespace_ann_min(1000)
    X, C, ns = build_tenants(db)
    big = [i for i, n in enumerate(ns) if n == "big"]
    Q, _ = clustered(30, seed=9, centers=C)
    before = [ids_of(graph_search(db, q, "big")) for q in Q]       # exact scans
    assert before == [exact(X, big, q) for q in Q]
    warm(db, "big", Q[0])
    info = db.namespace_ann_info()
    assert info["graphs"] == 1 and info["nodes"] == len(big) and info["pending"] == 0
    recall = np.mean([len(ids_of(graph_search(db, q, "big")) & exact(X, big, q)) / 10 for q in Q])
    assert recall >= 0.97, recall
    # small namespaces: no graph, still exact
    small = [i for i, n in enumerate(ns) if n == "small2"]
    assert ids_of(graph_search(db, Q[0], "small2")) == exact(X, small, Q[0])
    assert db.namespace_ann_info()["graphs"] == 1


def test_writes_keep_the_graph_current(db):
    db.set_namespace_ann_min(1000)
    X, C, ns = build_tenants(db)
    warm(db, "big", X[0])
    probe = np.asarray(C[0] / np.linalg.norm(C[0]), dtype=np.float32)

    db.add(10_000, probe, meta("big"))                               # added after the build
    assert 10_000 in ids_of(graph_search(db, probe, "big", k=1))
    db.forget(10_000)                                                # deleted
    assert 10_000 not in ids_of(graph_search(db, probe, "big", k=10))
    db.add(10_001, probe, meta("small1"))                            # moved in
    m = db.get_metadata(10_001); m.namespace_id = "big"; db.update_metadata(10_001, m)
    assert 10_001 in ids_of(graph_search(db, probe, "big", k=1))
    m.namespace_id = "small1"; db.update_metadata(10_001, m)         # and out again
    assert 10_001 not in ids_of(graph_search(db, probe, "big", k=10))
    far = -probe
    db.add(5, far, meta("big"))                                      # vector replaced
    assert ids_of(graph_search(db, far, "big", k=1)) == {5}
    db.update_metadata(5, meta("big", note="metadata only"))         # unmarked, not rebuilt
    assert ids_of(graph_search(db, far, "big", k=1)) == {5}


def test_other_predicates_filter_the_walk(db):
    db.set_namespace_ann_min(1000)
    X, C = clustered(3000, seed=4)
    ids = list(range(len(X)))
    db.add_batch(ids, X, [meta("big", parity=i % 2) for i in ids])
    warm(db, "big", X[0])
    odd = [i for i in ids if i % 2]
    for q in X[:10]:
        got = graph_search(db, q, "big", parity="1")
        assert all(r.id % 2 for r in got) and len(got) == 10
        assert len(ids_of(got) & exact(X, odd, q)) >= 9


def test_a_prefix_merges_walked_and_scanned_namespaces(db):
    db.set_namespace_ann_min(1000)
    X, C = clustered(3000 + 300, seed=5)
    ns = ["org.big"] * 3000 + ["org.small"] * 300
    ids = list(range(len(X)))
    db.add_batch(ids, X, [meta(n) for n in ns])
    pf = FilterBuilder().namespace_prefix("org").build()
    db.search(X[0], k=10, filter=pf)
    assert db.wait_for_namespace_graphs(60)
    assert db.namespace_ann_info()["graphs"] == 1
    recall = np.mean([len(ids_of(db.search(q, k=10, filter=pf)) & exact(X, ids, q)) / 10 for q in X[:20]])
    assert recall >= 0.97, recall


def test_compaction_and_int8_drop_graphs_that_would_point_at_old_slots(db):
    db.set_namespace_ann_min(1000)
    X, C, ns = build_tenants(db)
    big = [i for i, n in enumerate(ns) if n == "big"]
    warm(db, "big", X[0])
    for i in big[:500]:
        db.forget(i)
    db.compact()
    assert db.namespace_ann_info()["graphs"] == 0
    live = big[500:]
    assert ids_of(graph_search(db, X[600], "big")) == exact(X, live, X[600])   # scans meanwhile
    assert db.wait_for_namespace_graphs(60) and db.namespace_ann_info()["graphs"] == 1
    assert len(ids_of(graph_search(db, X[600], "big")) & exact(X, live, X[600])) >= 9
    db.set_int8_ram("text", 1.0)
    assert db.namespace_ann_info()["graphs"] == 0


def test_graphs_off_is_exact(db):
    db.set_namespace_ann_min(0)
    X, C, ns = build_tenants(db)
    big = [i for i, n in enumerate(ns) if n == "big"]
    for q in X[:5]:
        assert ids_of(graph_search(db, q, "big")) == exact(X, big, q)
    assert db.namespace_ann_info()["graphs"] == 0 and db.namespace_ann_info()["pending"] == 0


def test_writes_during_the_build_are_in_the_graph(db):
    db.set_namespace_ann_min(1000)
    X, C, ns = build_tenants(db, n_big=4000)
    extra, _ = clustered(400, seed=11, centers=C)
    stop = threading.Event()

    def writer():
        for i, v in enumerate(extra):
            db.add(20_000 + i, v, meta("big"))
        stop.set()

    t = threading.Thread(target=writer)
    graph_search(db, X[0], "big")            # queue the build, then write over it
    t.start()
    t.join()
    assert db.wait_for_namespace_graphs(60)
    assert db.namespace_ann_info()["nodes"] == 4000 + len(extra)
    for i in range(0, len(extra), 50):
        assert 20_000 + i in ids_of(graph_search(db, extra[i], "big", k=1))


def test_a_reader_builds_its_own_graphs_and_close_stops_the_builder(tmp_path):
    path = str(tmp_path / "r.feather")
    w = DB.open(path, dim=DIM)
    X, C, ns = build_tenants(w)
    w.close()
    r = DB.open(path, dim=DIM, read_only=True)
    r.set_namespace_ann_min(1000)
    graph_search(r, X[0], "big")
    r.close()                                 # must not hang on, or crash in, the builder
