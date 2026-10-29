"""DBManager: open on first use, one lock per namespace, LRU eviction.

* startup lists files and opens none, so time-to-ready doesn't grow with data
* a slow open of one namespace never blocks requests to another
* past FEATHER_MEMORY_LIMIT_MB, idle namespaces close LRU-first, and a
  namespace with a request in flight or a write in progress is never closed
* DB.close() releases the dataset (py::nodelete means nothing else does)
"""
import os
import sys
import threading
import time

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="Cloud API tests need fastapi")
pytest.importorskip("httpx", reason="fastapi TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

import feather_db  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO, "feather-api")
DIM = 16


def _fresh_app_modules():
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]


def _db_manager():
    _fresh_app_modules()
    from app import db_manager  # noqa: WPS433
    return db_manager


def _make_ns(data_dir, name, n, dim=DIM, seed=0):
    rng = np.random.default_rng(seed)
    db = feather_db.DB.open(os.path.join(str(data_dir), f"{name}.feather"), dim=dim)
    metas = []
    for i in range(n):
        m = feather_db.Metadata()
        m.content = f"{name} doc {i}"
        metas.append(m)
    db.add_batch(list(range(1, n + 1)), rng.random((n, dim)).astype(np.float32), metas)
    db.close()


def _wait_until(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


# ── startup ────────────────────────────────────────────────────────────────

def test_startup_opens_nothing_and_does_not_scale_with_data(tmp_path):
    dbm = _db_manager()
    small, large = tmp_path / "small", tmp_path / "large"
    small.mkdir(), large.mkdir()
    _make_ns(small, "tiny", 10)
    _make_ns(large, "big", 40_000, dim=64)
    assert os.path.getsize(large / "big.feather") > 50 * os.path.getsize(small / "tiny.feather")

    def startup(d):
        t0 = time.perf_counter()
        m = dbm.DBManager(data_dir=str(d), default_dim=DIM)
        return m, time.perf_counter() - t0

    m_small, t_small = startup(small)
    m_large, t_large = startup(large)
    try:
        # Nothing was opened, so there is nothing for the file size to cost.
        assert m_large.loaded_namespaces() == []
        assert m_large.list_namespaces() == ["big"]
        assert t_large < 0.25, f"startup took {t_large:.3f}s with a large namespace"
        assert t_large < t_small + 0.1

        # First use opens it, with the data intact.
        assert m_large.get("big", create=False).size() == 40_000
        assert m_large.loaded_namespaces() == ["big"]
    finally:
        m_small.close_all()
        m_large.close_all()


def test_startup_never_touches_a_corrupt_file(tmp_path):
    """Startup used to parse every file; one unloadable file is now a 503 on
    that namespace only, and the parse is not repeated per request."""
    dbm = _db_manager()
    _make_ns(tmp_path, "good", 5)
    _make_ns(tmp_path, "bad", 500)
    size = os.path.getsize(tmp_path / "bad.feather")
    with open(tmp_path / "bad.feather", "r+b") as fh:
        fh.truncate(int(size * 0.6))          # a half-written file

    m = dbm.DBManager(data_dir=str(tmp_path), default_dim=DIM)
    calls = []
    real = m._opener
    m._opener = lambda path, dim: (calls.append(path), real(path, dim=dim))[1]
    try:
        assert m.list_namespaces() == ["bad", "good"] and m.loaded_namespaces() == []
        for _ in range(3):
            with pytest.raises(dbm.NamespaceUnavailable):
                m.get("bad", create=False)
        assert len(calls) == 1, "a failed open must be cached until the file changes"
        assert m.get("good", create=False).size() == 5
    finally:
        m.close_all()


# ── per-namespace locking ──────────────────────────────────────────────────

def test_slow_open_of_one_namespace_does_not_block_another(tmp_path):
    dbm = _db_manager()
    _make_ns(tmp_path, "competitor-big", 50)
    _make_ns(tmp_path, "other", 50)
    m = dbm.DBManager(data_dir=str(tmp_path), default_dim=DIM)

    gate, entered = threading.Event(), threading.Event()
    real = m._opener

    def opener(path, dim):
        if "competitor-big" in path:
            entered.set()
            assert gate.wait(30), "test gate never released"
        return real(path, dim=dim)

    m._opener = opener
    slow = threading.Thread(target=m.get, args=("competitor-big",), kwargs={"create": False})
    slow.start()
    try:
        assert entered.wait(10)
        t0 = time.perf_counter()
        db = m.get("other", create=False)
        assert db.size() == 50
        assert len(db.search(np.zeros(DIM, dtype=np.float32), k=3)) == 3
        assert time.perf_counter() - t0 < 1.0, "a slow open blocked another namespace"
        assert slow.is_alive(), "the slow open should still be in progress"
        # Listing and peeking never wait on an open either.
        assert "competitor-big" in m.list_namespaces()
        assert m.peek("competitor-big") is None
    finally:
        gate.set()
        slow.join(30)
        m.close_all()
    assert not slow.is_alive()


def test_concurrent_first_use_opens_once(tmp_path):
    dbm = _db_manager()
    _make_ns(tmp_path, "shared", 20)
    m = dbm.DBManager(data_dir=str(tmp_path), default_dim=DIM)
    calls, real = [], m._opener

    def opener(path, dim):
        calls.append(path)
        time.sleep(0.2)
        return real(path, dim=dim)

    m._opener = opener
    got = []
    ts = [threading.Thread(target=lambda: got.append(m.get("shared", create=False)))
          for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    try:
        assert len(calls) == 1
        assert len({id(d) for d in got}) == 1
    finally:
        m.close_all()


def test_api_serves_other_namespaces_during_a_slow_open(tmp_path, monkeypatch):
    _make_ns(tmp_path, "slow", 30)
    _make_ns(tmp_path, "fast", 30)
    monkeypatch.setenv("FEATHER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FEATHER_DB_DIM", str(DIM))
    monkeypatch.delenv("FEATHER_API_KEY", raising=False)
    monkeypatch.setenv("FEATHER_DEV_MODE", "1")
    monkeypatch.setenv("FEATHER_EMBED_PROVIDER", "mock")
    monkeypatch.setenv("FEATHER_EMBED_DIM", str(DIM))
    _fresh_app_modules()
    from app import main  # noqa: WPS433

    with TestClient(main.app) as client:
        h = client.get("/health").json()
        assert h["namespaces_loaded"] == 0 and h["namespaces_total"] == 2

        gate, entered = threading.Event(), threading.Event()
        real = main.manager._opener

        def opener(path, dim):
            if "slow" in path:
                entered.set()
                assert gate.wait(30)
            return real(path, dim=dim)

        main.manager._opener = opener
        res = {}
        t = threading.Thread(target=lambda: res.setdefault(
            "slow", client.get("/v1/namespaces/slow/stats")))
        t.start()
        try:
            assert entered.wait(10)
            t0 = time.perf_counter()
            r = client.get("/v1/namespaces/fast/stats")
            assert r.status_code == 200 and r.json()["records"] == 30
            assert time.perf_counter() - t0 < 2.0
            # Dashboard listings open nothing.
            assert client.get("/v1/admin/overview").status_code == 200
            detail = client.get("/v1/namespaces?detail=true").json()
            assert {d["name"]: d["loaded"] for d in detail["details"]} == \
                {"fast": True, "slow": False}
        finally:
            gate.set()
            t.join(30)
        assert res["slow"].status_code == 200 and res["slow"].json()["records"] == 30
        assert client.get("/health").json()["namespaces_loaded"] == 2


# ── eviction ───────────────────────────────────────────────────────────────

def _evicting_manager(dbm, tmp_path, names, keep):
    """A manager whose limit fits the namespaces in `keep` and no more (the
    slack is half the smallest namespace, so sizes needn't match exactly)."""
    sizes = {n: os.path.getsize(tmp_path / f"{n}.feather") for n in names}
    limit = sum(sizes[n] for n in keep) + min(sizes.values()) // 2
    return dbm.DBManager(data_dir=str(tmp_path), default_dim=DIM,
                         memory_limit_bytes=limit, evict_idle_s=0)


def test_eviction_closes_least_recently_used_and_reopens(tmp_path):
    dbm = _db_manager()
    for i, n in enumerate(("a", "b", "c")):
        _make_ns(tmp_path, n, 200, seed=i)
    m = _evicting_manager(dbm, tmp_path, ["a", "b", "c"], keep=["b", "c"])
    try:
        a = m.get("a", create=False)
        m.get("b", create=False)
        m.get("c", create=False)       # a is now least recently used
        assert _wait_until(lambda: "a" not in m.loaded_namespaces() and a.is_closed())
        assert sorted(m.loaded_namespaces()) == ["b", "c"]
        assert a.is_closed() and a.size() == 0, "close() must release the dataset"
        assert "a" in m.list_namespaces()

        a2 = m.get("a", create=False)  # reopened on demand, data intact
        assert a2 is not a and a2.size() == 200
        assert _wait_until(lambda: "b" not in m.loaded_namespaces())
    finally:
        m.close_all()


def test_eviction_skips_pinned_and_busy_namespaces(tmp_path):
    dbm = _db_manager()
    for i, n in enumerate(("a", "b")):
        _make_ns(tmp_path, n, 200, seed=i)
    m = _evicting_manager(dbm, tmp_path, ["a", "b"], keep=[])   # nothing fits
    try:
        m.pin("a")                      # a request in flight on a
        m.pin("b")                      # (pinned until its write lock is held)
        a = m.get("a", create=False)
        b = m.get("b", create=False)
        with m.lock("b"):               # a write in progress on b
            m.unpin("b")
            assert m.evict() == []
            assert not a.is_closed() and not b.is_closed()
        m.unpin("a")
        assert _wait_until(lambda: (m.evict(), m.loaded_namespaces())[1] == [])
        assert a.is_closed() and b.is_closed()
    finally:
        m.close_all()


def test_eviction_checkpoints_unsaved_writes(tmp_path):
    dbm = _db_manager()
    _make_ns(tmp_path, "a", 10)
    _make_ns(tmp_path, "b", 10)
    m = _evicting_manager(dbm, tmp_path, ["a", "b"], keep=[])
    try:
        m.pin("a")                      # as the API does for the request's lifetime
        a = m.get("a", create=False)
        a.add(id=999, vec=np.ones(DIM, dtype=np.float32))   # only in the WAL
        assert a.wal_size() > 0 and not a.is_closed()
        m.unpin("a")
        assert _wait_until(lambda: (m.evict(), a.is_closed())[1])
        # Checkpointed on the way out, not left for WAL replay.
        assert not os.path.exists(tmp_path / "a.feather.wal") or \
            os.path.getsize(tmp_path / "a.feather.wal") <= 8
        m.pin("a")                      # else this fresh handle is evictable at once
        a2 = m.get("a", create=False)
        assert a2.size() == 11 and a2.get_metadata(999) is not None
        m.unpin("a")
    finally:
        m.close_all()


def test_no_limit_means_no_eviction(tmp_path):
    dbm = _db_manager()
    _make_ns(tmp_path, "a", 10)
    m = dbm.DBManager(data_dir=str(tmp_path), default_dim=DIM,
                      memory_limit_bytes=0, evict_idle_s=0)
    try:
        m.get("a", create=False)
        assert m.evict() == [] and m.loaded_namespaces() == ["a"]
    finally:
        m.close_all()
