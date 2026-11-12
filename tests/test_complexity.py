"""Every agent-facing read costs O(result), not O(scope), and it is asserted.

Two reads did return a little by reading the whole scope: a FeatherStore
search with no query read every id in the database (get_all_ids) and every
record under the prefix, and count() downloaded every record of a namespace
to count it. Both looked fine at 100 items. So cost is measured here the way
it is paid, as requests and response bytes over the Cloud API (RemoteDB
against the app, in-process), at two scope sizes. Neither may grow with the
scope. Where time is the cost (erase), its growth is bounded instead.
"""
import contextlib
import json
import os
import sys
import time

import numpy as np
import pytest

import feather_db

pytest.importorskip("langgraph")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from langgraph.store.base import PutOp  # noqa: E402

from feather_db.integrations import AgentMemory  # noqa: E402
from feather_db.integrations.langgraph_store import FeatherStore  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO, "feather-api")
DIM = 16
SMALL, LARGE = 1_000, 10_000
ORG, USER = "acme", "user_1"


def _populate(path, n):
    """n items under one user (spread over kinds), and as many under another
    user: scope a read must not touch."""
    store = FeatherStore(path, dim=DIM, auto_save=False)
    ops = []
    for who in (USER, "user_2"):
        for i in range(n):
            kind = ("notes", "facts", "episodes", "preferences")[i % 4]
            ops.append(PutOp((ORG, who, kind), f"item-{i:06d}", {"kind": kind, "text": f"item {i}"}))
    for at in range(0, len(ops), 2000):
        store.batch(ops[at: at + 2000])
    store.close()


@contextlib.contextmanager
def _remote_store(data_dir, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("FEATHER_DATA_DIR", str(data_dir))
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
        remote = RemoteDB("http://testserver", "mem", session=client)
        meter = {"calls": 0, "bytes": 0}
        call = remote._call

        def counted(method, path, **kw):
            r = call(method, path, **kw)
            meter["calls"] += 1
            meter["bytes"] += len(r.content) if kw.get("raw") else len(json.dumps(r))
            return r
        remote._call = counted
        yield FeatherStore(db=remote, dim=DIM), meter


# (name, the read, most requests it may take)
READS = [
    ("count(namespace)", lambda s: s.count((ORG, USER, "notes")), 1),
    ("count(prefix)", lambda s: s.count((ORG, USER), prefix=True), 1),
    ("search(prefix), no query", lambda s: s.search((ORG, USER), limit=10), 3),
    ("search(), no namespace", lambda s: s.search((), limit=10), 3),
    ("newest(namespace)", lambda s: s.newest((ORG, USER, "notes"), 10), 3),
    ("about()", lambda s: AgentMemory(s, org=ORG, user=USER).about(), 3),
]


@pytest.fixture(scope="module")
def data_dirs(tmp_path_factory):
    out = {}
    for n in (SMALL, LARGE):
        d = tmp_path_factory.mktemp(f"scope{n}")
        _populate(str(d / "mem.feather"), n)
        out[n] = d
    return out


def _cost(data_dir, monkeypatch, read):
    monkeypatch.delenv("FEATHER_EMBED_PROVIDER", raising=False)
    with _remote_store(data_dir, monkeypatch) as (store, meter):
        read(store)                                # warm: first use opens the namespace
        c, b = meter["calls"], meter["bytes"]
        result = read(store)
        return meter["calls"] - c, meter["bytes"] - b, result


@pytest.mark.parametrize("name,read,max_calls", READS, ids=[r[0] for r in READS])
def test_reads_cost_the_result_not_the_scope(data_dirs, monkeypatch, name, read, max_calls):
    small = _cost(data_dirs[SMALL], monkeypatch, read)
    large = _cost(data_dirs[LARGE], monkeypatch, read)
    assert large[0] <= max_calls, f"{name}: {large[0]} requests at {LARGE} items (at most {max_calls})"
    assert large[0] == small[0], f"{name}: requests grew with the scope: {small[0]} -> {large[0]}"
    # Same result size at both scopes, so the same bytes give or take a few
    # digits; reading the scope would be ~10x.
    assert large[1] <= 1.25 * small[1] + 512, (
        f"{name}: response bytes grew with the scope: {small[1]} -> {large[1]}")


def test_count_agrees_with_a_full_scan(data_dirs, monkeypatch):
    """The engine's count is the count of what scan() returns: deleted items
    and index records left out."""
    monkeypatch.delenv("FEATHER_EMBED_PROVIDER", raising=False)
    with _remote_store(data_dirs[SMALL], monkeypatch) as (store, _):
        ns = (ORG, USER, "facts")
        store.delete(ns, "item-000001")
        store.index(ns, "item-000005", ["an index entry", "another"])
        assert store.count(ns) == len(store.scan(ns)) == SMALL // 4 - 1
        assert store.count((ORG, USER), prefix=True) == len(store.scan((ORG, USER), prefix=True))


def _erase_seconds(path, n):
    rng = np.random.default_rng(0)
    db = feather_db.DB.open(path, dim=DIM)
    ids = list(range(1, n + 1))
    metas = []
    for i in ids:
        m = feather_db.Metadata()
        m.content, m.namespace_id = f"r{i}", "ns"
        m.entity_id = "subject" if i % 10 == 0 else f"e{i % 50}"
        metas.append(m)
    db.add_batch(ids, rng.standard_normal((n, DIM)).astype(np.float32), metas)
    db.save()
    t = time.perf_counter()
    db.erase(entity="subject")
    dt = time.perf_counter() - t
    db.close()
    return dt


def test_erase_time_is_linear_in_n(tmp_path):
    """10% of the records, at n and 4n: linear is ~4.5x (measured), quadratic
    ~16x. It was quadratic once, past ~3% removed (79 s at 20k x 768)."""
    def best(n):
        return min(_erase_seconds(str(tmp_path / f"e{n}-{k}.feather"), n) for k in range(3))
    small, large = best(2_000), best(8_000)
    assert large / small < 8, f"erase: {small * 1000:.1f} ms at 2k, {large * 1000:.1f} ms at 8k"
