"""Cloud API: namespaces open when used and close when idle.

A shard is limited by memory, and only open namespaces hold memory. So:
nothing opens at startup; a namespace opens on its first request; one idle
past FEATHER_IDLE_CLOSE_S closes (checkpointed, lock released), and past
FEATHER_MAX_OPEN the least recently used close first. A request that raced
the close — it took the handle just before — is run again on a fresh one.
"""
import contextlib
import os
import sys
import threading
import time

import numpy as np
import pytest

import feather_db

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO, "feather-api")
DIM = 8


def _dbm():
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    import app.db_manager as dbm  # noqa: WPS433
    return dbm


def _files(tmp_path, names, n=20):
    for ns in names:
        with feather_db.DB.open(str(tmp_path / f"{ns}.feather"), dim=DIM) as db:
            db.add_batch(list(range(n)), np.random.default_rng(1).random((n, DIM), dtype=np.float32))


def test_nothing_opens_until_used(tmp_path):
    _files(tmp_path, ["a", "b", "c"])
    mgr = _dbm().DBManager(data_dir=str(tmp_path), default_dim=DIM, idle_close_s=0, max_open=0)
    assert mgr.open_namespaces() == [] and mgr.list_namespaces() == ["a", "b", "c"]
    assert mgr.get("b", create=False).size() == 20
    assert mgr.open_namespaces() == ["b"]
    with pytest.raises(KeyError):
        mgr.get("missing", create=False)
    mgr.close_all()


def test_idle_and_lru_closes_keep_the_data(tmp_path):
    _files(tmp_path, ["a", "b", "c"])
    mgr = _dbm().DBManager(data_dir=str(tmp_path), default_dim=DIM, idle_close_s=0.2, max_open=0)
    for ns in ("a", "b"):
        db = mgr.get(ns)
        with mgr.lock(ns):
            db.add(100, np.ones(DIM, np.float32))           # in the WAL only
    time.sleep(0.3)
    assert sorted(mgr.sweep()) == ["a", "b"] and mgr.open_namespaces() == []
    assert mgr.stats_of("a")["records"] == 21
    assert mgr.get("a").get_metadata(100) is not None       # reopened, checkpointed write kept
    mgr.close_all()

    mgr = _dbm().DBManager(data_dir=str(tmp_path), default_dim=DIM, idle_close_s=0, max_open=2)
    for ns in ("a", "b", "c"):
        mgr.get(ns)
        time.sleep(1.1)                                      # LRU skips anything used in the last second
    assert mgr.sweep() == ["a"] and sorted(mgr.open_namespaces()) == ["b", "c"]
    mgr.close_all()


def test_a_namespace_in_use_is_not_closed(tmp_path):
    _files(tmp_path, ["a"])
    mgr = _dbm().DBManager(data_dir=str(tmp_path), default_dim=DIM, idle_close_s=0.01, max_open=0)
    mgr.get("a")
    time.sleep(0.05)
    with mgr.lock("a"):                                      # a write in progress
        assert mgr.sweep() == []
    mgr.get("a")                                             # just used
    assert mgr.close_namespace("a", idle_s=10) is False
    mgr.close_all()


@contextlib.contextmanager
def _client(tmp_path, monkeypatch, **env):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("FEATHER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FEATHER_DB_DIM", str(DIM))
    monkeypatch.delenv("FEATHER_API_KEY", raising=False)
    monkeypatch.setenv("FEATHER_DEV_MODE", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    _dbm()
    from app import main  # noqa: WPS433
    with TestClient(main.app) as c:
        yield c, main


def test_requests_racing_closes_all_succeed(tmp_path, monkeypatch):
    """A closer runs flat out while writers and readers hammer three
    namespaces. Every request succeeds, every write is there afterwards."""
    with _client(tmp_path, monkeypatch, FEATHER_IDLE_CLOSE_S="0.001") as (c, main):
        stop = threading.Event()
        closes = []

        failures, written = [], {ns: [] for ns in ("t1", "t2", "t3")}

        def closer():
            while not stop.is_set():
                try:
                    closes.extend(main.manager.sweep())
                except Exception as e:  # noqa: BLE001 — a failing close is a failure
                    failures.append(("closer", repr(e)))

        def worker(ns):
            try:
                work(ns)
            except Exception as e:  # noqa: BLE001 — TestClient re-raises server errors
                failures.append((ns, "raised", repr(e)[:500]))

        def work(ns):
            for i in range(120):
                r = c.post(f"/v1/{ns}/vectors", json={"id": i, "vector": [float(i) + 1] * DIM})
                if r.status_code != 201:
                    failures.append((ns, i, r.status_code, r.text[:200]))
                    continue
                written[ns].append(i)
                s = c.post(f"/v1/{ns}/search", json={"vector": [float(i) + 1] * DIM, "k": 1})
                if s.status_code != 200:
                    failures.append((ns, "search", s.status_code, s.text[:200]))

        t = threading.Thread(target=closer)
        t.start()
        ws = [threading.Thread(target=worker, args=(ns,)) for ns in written]
        for w in ws:
            w.start()
        for w in ws:
            w.join()
        stop.set()
        t.join()
        assert not failures, failures[:3]
        assert len(closes) > 10, "the closer never closed anything: the test proves nothing"
        for ns, ids in written.items():
            assert sorted(c.get(f"/v1/{ns}/ids").json()["ids"]) == ids


def test_overview_counts_closed_namespaces_without_opening_them(tmp_path, monkeypatch):
    _files(tmp_path, ["a", "b"], n=7)
    with _client(tmp_path, monkeypatch, FEATHER_IDLE_CLOSE_S="0.01") as (c, main):
        c.get("/v1/a/records/1")                            # opens a
        time.sleep(0.05)
        main.manager.sweep()                                # closes a: stats recorded
        ov = c.get("/v1/admin/overview").json()
        assert ov["nsCount"] == 2 and main.manager.open_namespaces() == []
        a = next(x for x in ov["topNamespaces"] if x["name"] == "a")
        assert a["records"] == 7 and a["open"] is False
        h = c.get("/health").json()
        assert h["namespaces_loaded"] == 0 and h["namespaces_known"] == 2


def test_a_request_given_a_handle_closed_under_it_is_rerun(tmp_path, monkeypatch):
    """The race above, made certain: the route's first get() returns a handle
    the idle closer has just closed. The engine refuses it (ClosedError,
    before doing anything) and the route runs again on a fresh one."""
    with _client(tmp_path, monkeypatch) as (c, main):
        assert c.post("/v1/ns/vectors", json={"id": 1, "vector": [1.0] * DIM}).status_code == 201
        mgr = main.manager
        for route in ("write", "search"):
            stale = mgr.get("ns")
            assert mgr.close_namespace("ns", blocking=True)
            real, handed = mgr.get, []

            def get_once(*a, **kw):
                if not handed:
                    handed.append(1)
                    return stale
                return real(*a, **kw)
            monkeypatch.setattr(mgr, "get", get_once)
            if route == "write":
                r = c.post("/v1/ns/vectors", json={"id": 2, "vector": [2.0] * DIM})
                assert r.status_code == 201, r.text
            else:
                r = c.post("/v1/ns/search", json={"vector": [2.0] * DIM, "k": 1})
                assert r.status_code == 200 and r.json()["results"][0]["id"] == 2, r.text
            assert handed, "the stale handle was never used: the test proves nothing"
            monkeypatch.setattr(mgr, "get", real)
