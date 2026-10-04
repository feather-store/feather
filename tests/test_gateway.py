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


def test_owner_rule_matches_the_api():
    sys.path.insert(0, API_DIR)
    sys.path.insert(0, GW_DIR)
    for mod in [m for m in sys.modules if m in ("gateway",) or m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from app.db_manager import shard_owner as api_owner  # noqa: WPS433
    from gateway import shard_owner as gw_owner           # noqa: WPS433
    for n in (1, 2, 3, 4, 7, 16):
        for i in range(500):
            ns = f"tenant-{i}_x"
            assert api_owner(ns, n) == gw_owner(ns, n)


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
