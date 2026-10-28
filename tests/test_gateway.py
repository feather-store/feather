"""feather-gateway: namespace-sharded routing over real API shards."""
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("uvicorn")
import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO, "feather-api")
GW_DIR = os.path.join(REPO, "feather-gateway")
DIM = 8


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fresh_modules():
    sys.path.insert(0, API_DIR)
    sys.path.insert(0, GW_DIR)
    for mod in [m for m in sys.modules if m in ("gateway",) or m == "app" or m.startswith("app.")]:
        del sys.modules[mod]


def test_owner_rule_matches_the_api():
    _fresh_modules()
    from app import placement                              # noqa: WPS433
    import gateway                                          # noqa: WPS433
    for ids in (["0"], ["0", "1"], ["a", "b", "c"], [str(k) for k in range(7)]):
        for i in range(500):
            ns = f"tenant-{i}_x"
            assert placement.rendezvous_owner(ns, ids) == gateway.rendezvous_owner(ns, ids)
            assert placement.modulo_owner(ns, ids) == gateway.modulo_owner(ns, ids)


def test_adding_a_shard_moves_about_one_in_n_plus_one():
    """crc32 % N moved ~N/(N+1) of the namespaces when N changed: 4 -> 5
    shards moved 80%. Rendezvous moves only what the new shard wins: ~20%,
    and every namespace that moves, moves TO the new shard."""
    _fresh_modules()
    from app import placement                              # noqa: WPS433
    names = [f"tenant-{i}" for i in range(20_000)]
    four, five = ["s0", "s1", "s2", "s3"], ["s0", "s1", "s2", "s3", "s4"]
    moved = [n for n in names if placement.rendezvous_owner(n, four) != placement.rendezvous_owner(n, five)]
    assert 0.18 < len(moved) / len(names) < 0.22, len(moved) / len(names)
    assert all(placement.rendezvous_owner(n, five) == "s4" for n in moved)
    mod = sum(placement.modulo_owner(n, four) != placement.modulo_owner(n, five) for n in names)
    assert mod / len(names) > 0.75
    load = [sum(placement.rendezvous_owner(n, five) == s for n in names) for s in five]
    assert max(load) / min(load) < 1.1, load                 # and stays balanced


@pytest.fixture
def shards(tmp_path):
    procs, urls = [], []
    for idx in range(2):
        port = _free_port()
        env = dict(os.environ, FEATHER_DATA_DIR=str(tmp_path), FEATHER_DEV_MODE="1",
                   FEATHER_DB_DIM=str(DIM), FEATHER_SHARD_COUNT="2", FEATHER_SHARD_INDEX=str(idx),
                   PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
        env.pop("FEATHER_API_KEY", None)
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"],
            cwd=API_DIR, env=env))
        urls.append(f"http://127.0.0.1:{port}")
    for u in urls:
        for _ in range(150):
            try:
                if httpx.get(u + "/health", timeout=1).status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        else:
            pytest.fail(f"shard {u} did not start")
    yield urls, tmp_path
    for p in procs:
        p.terminate()
        p.wait(20)


def test_gateway_routes_and_merges(shards, monkeypatch):
    urls, data_dir = shards
    monkeypatch.setenv("FEATHER_SHARDS", ",".join(urls))
    sys.path.insert(0, GW_DIR)
    sys.modules.pop("gateway", None)
    from gateway import app, shard_owner  # noqa: WPS433

    names = [f"brand{i}" for i in range(12)]
    rng = np.random.default_rng(0)
    with TestClient(app) as gw:
        for ns in names:
            v = rng.random(DIM).astype(np.float32).tolist()
            r = gw.post(f"/v1/{ns}/vectors", json={"id": 1, "vector": v, "metadata": {"content": ns}})
            assert r.status_code == 201, r.text
            assert r.headers["X-Feather-Shard"] == str(shard_owner(ns, 2))
            s = gw.post(f"/v1/{ns}/search", json={"vector": v, "k": 1, "track": False}).json()
            assert s["results"][0]["id"] == 1

        listed = gw.get("/v1/namespaces").json()["namespaces"]
        assert sorted(listed) == sorted(names)
        ov = gw.get("/v1/admin/overview").json()
        assert ov["nsCount"] == 12 and ov["totalRecords"] == 12 and ov["shardsAnswered"] == 2
        assert gw.get("/health").json()["status"] == "ok"

    # each namespace lives on exactly the shard that owns it
    for ns in names:
        owner = shard_owner(ns, 2)
        other = urls[1 - owner]
        r = httpx.get(f"{other}/v1/{ns}/records/1")
        assert r.status_code == 421, (ns, r.status_code)
        assert (data_dir / f"{ns}.feather").exists() or (data_dir / f"{ns}.feather.wal").exists()


def test_namespace_named_like_an_api_route_is_routed_to_its_owner(monkeypatch):
    """`admin` and `namespaces` are legal namespace names; only the API's own
    /v1/admin/<route> and /v1/namespaces[/{ns}[/stats]] are not namespace routes."""
    monkeypatch.setenv("FEATHER_SHARDS", "http://a,http://b")
    sys.path.insert(0, GW_DIR)
    sys.modules.pop("gateway", None)
    from gateway import _namespace_of  # noqa: WPS433
    assert _namespace_of("/v1/admin/search", "POST") == "admin"
    assert _namespace_of("/v1/admin/records/3", "GET") == "admin"
    assert _namespace_of("/v1/admin/overview") is None
    assert _namespace_of("/v1/admin/embedding_config", "PUT") is None
    assert _namespace_of("/v1/namespaces/search", "POST") == "namespaces"
    assert _namespace_of("/v1/namespaces/records", "GET") == "namespaces"
    assert _namespace_of("/v1/namespaces/brand1", "DELETE") == "brand1"
    assert _namespace_of("/v1/namespaces/brand1/stats") == "brand1"
    assert _namespace_of("/v1/namespaces") is None
    assert _namespace_of("/v1/brand1/search", "POST") == "brand1"


def test_gateway_follows_a_421_to_the_owner(shards, monkeypatch):
    """The gateway's placement is behind the shards' (a move it did not make):
    the shard it picks answers 421 naming the owner, and the gateway resends
    there — safe for a POST, since a 421 is refused before any work."""
    urls, _ = shards
    monkeypatch.setenv("FEATHER_SHARDS", ",".join(urls))
    sys.path.insert(0, GW_DIR)
    sys.modules.pop("gateway", None)
    from gateway import app, shard_owner  # noqa: WPS433
    ns = next(f"moved{i}" for i in range(1000) if shard_owner(f"moved{i}", 2) == 0)
    with TestClient(app) as gw:                    # loads placement (epoch 0) now
        for u in urls:                             # then the shards move ns behind its back
            assert httpx.put(f"{u}/v1/admin/placement",
                             json={"epoch": 1, "overrides": {ns: "1"}}).status_code == 200
        r = gw.post(f"/v1/{ns}/vectors", json={"id": 1, "vector": [0.1] * DIM})
        assert r.status_code == 201, r.text
        assert r.headers["X-Feather-Shard"] == "1"
    assert httpx.get(f"{urls[0]}/v1/{ns}/records/1").json()["error"] == "wrong_shard"


def test_a_hot_namespace_moves_with_no_restart_and_no_lost_write(shards, monkeypatch):
    """Writes keep arriving through the gateway while one namespace moves to
    the other shard. Every write lands exactly once; none is refused except
    retryably; afterwards the target serves it and the source answers 421."""
    import threading
    urls, _ = shards
    monkeypatch.setenv("FEATHER_SHARDS", ",".join(urls))
    sys.path.insert(0, GW_DIR)
    sys.modules.pop("gateway", None)
    from gateway import app, shard_owner  # noqa: WPS433
    ns = next(f"hot{i}" for i in range(1000) if shard_owner(f"hot{i}", 2) == 0)
    rng = np.random.default_rng(0)
    with TestClient(app) as gw:
        for i in range(200):
            assert gw.post(f"/v1/{ns}/vectors", json={"id": i, "vector": rng.random(DIM).tolist()}).status_code == 201
        stop, errors, written = threading.Event(), [], []

        def writer():
            i = 1000
            while not stop.is_set():
                body = {"id": i, "vector": [float(i % 7)] * DIM}
                for _ in range(50):                # retryable answers are retried, keyed
                    r = gw.post(f"/v1/{ns}/vectors", json=body, headers={"Idempotency-Key": f"w{i}"})
                    if r.status_code == 201:
                        written.append(i)
                        break
                    if not r.json().get("retryable"):
                        errors.append((i, r.status_code, r.text[:200]))
                        break
                    time.sleep(0.01)
                i += 1

        t = threading.Thread(target=writer)
        t.start()
        time.sleep(0.3)
        mv = gw.post("/v1/admin/placement/move", json={"namespace": ns, "shard": "1"})
        time.sleep(0.3)
        stop.set()
        t.join()
        assert mv.status_code == 200 and mv.json()["moved"] and mv.json()["released"], mv.text
        assert not errors, errors[:3]
        assert len(written) > 10
        r = gw.get(f"/v1/{ns}/records/{written[-1]}")
        assert r.status_code == 200 and r.headers["X-Feather-Shard"] == "1"
        ids = gw.get(f"/v1/{ns}/ids").json()["ids"]
        assert set(range(200)) | set(written) == set(ids)
        assert gw.get("/v1/admin/placement").json()["overrides"] == {ns: "1"}
        back = gw.post("/v1/admin/placement/move", json={"namespace": ns, "shard": "0"})
        assert back.status_code == 200 and back.json()["overrides"] == {}     # override removed
        assert gw.get(f"/v1/{ns}/records/5").headers["X-Feather-Shard"] == "0"
    assert httpx.get(f"{urls[1]}/v1/{ns}/records/5").json()["error"] == "wrong_shard"


def test_gateway_types_an_unreachable_shard(monkeypatch):
    monkeypatch.setenv("FEATHER_SHARDS", f"http://127.0.0.1:{_free_port()}")
    sys.path.insert(0, GW_DIR)
    sys.modules.pop("gateway", None)
    from gateway import app  # noqa: WPS433
    with TestClient(app) as gw:
        r = gw.post("/v1/ns/vectors", json={"id": 1, "vector": [0.1] * DIM})
    assert r.status_code == 503 and r.headers["Retry-After"]
    assert r.json()["error"] == "shard_unreachable" and r.json()["retryable"] is True
