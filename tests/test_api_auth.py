"""Cloud API authentication defaults.

These exist because of what shipping an *official* container image means. The
image sets `ENV FEATHER_API_KEY=""` and `verify_api_key()` used to read an empty
key as "dev mode — let everyone in". Anyone following the README's one-command
deploy therefore got a vector database on a public port with read *and delete*
access to every namespace, and nothing anywhere said so. That had already
happened once on a live host before this guard existed.

The invariant: **no auth requires an explicit opt-in.** An unset key is a
startup error, never a silent downgrade to open access.
"""
import importlib
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO, "feather-api")

pytest.importorskip("fastapi", reason="Cloud API tests need fastapi")
pytest.importorskip("httpx", reason="fastapi TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

NS = "authns"
DIM = 8


def _fresh_app(monkeypatch, tmp_path, **env):
    """Import app.main with a clean module cache so module-level env is re-read.

    The guard runs at import time, so every case here needs a genuinely fresh
    import rather than a re-used module object.
    """
    monkeypatch.setenv("FEATHER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FEATHER_DB_DIM", str(DIM))
    monkeypatch.delenv("FEATHER_API_KEY", raising=False)
    monkeypatch.delenv("FEATHER_DEV_MODE", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    return importlib.import_module("app.main")


# ── The guard ─────────────────────────────────────────────────────────────

def test_missing_api_key_refuses_to_start(monkeypatch, tmp_path):
    """The whole point: no key, no boot. Not 'no key, no auth'."""
    with pytest.raises(RuntimeError, match="FEATHER_API_KEY"):
        _fresh_app(monkeypatch, tmp_path)


def test_empty_api_key_is_not_a_dev_mode_sentinel(monkeypatch, tmp_path):
    """The container image ships FEATHER_API_KEY="" — an empty value must not
    be what decides that auth is off, or the image is open by default."""
    with pytest.raises(RuntimeError, match="FEATHER_API_KEY"):
        _fresh_app(monkeypatch, tmp_path, FEATHER_API_KEY="")


def test_dev_mode_opt_in_starts_without_auth(monkeypatch, tmp_path):
    """Local dev still works — but only when someone asked for it by name."""
    main = _fresh_app(monkeypatch, tmp_path, FEATHER_DEV_MODE="1")
    with TestClient(main.app) as c:
        assert c.get("/v1/namespaces").status_code == 200


def test_key_set_means_auth_is_enforced(monkeypatch, tmp_path):
    main = _fresh_app(monkeypatch, tmp_path, FEATHER_API_KEY="s3cret")
    with TestClient(main.app) as c:
        assert c.get("/v1/namespaces").status_code == 401
        assert c.get("/v1/namespaces", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/v1/namespaces", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_a_real_key_beats_dev_mode(monkeypatch, tmp_path):
    """DEV_MODE must not be able to *disable* a configured key — otherwise a
    stray env var in a deploy silently opens a production host."""
    main = _fresh_app(monkeypatch, tmp_path,
                      FEATHER_API_KEY="s3cret", FEATHER_DEV_MODE="1")
    with TestClient(main.app) as c:
        assert c.get("/v1/namespaces").status_code == 401


# ── The endpoints that actually lose data ─────────────────────────────────

def test_destructive_endpoints_are_not_reachable_without_the_key(monkeypatch, tmp_path):
    """Read access leaking is bad; delete access leaking is the incident.
    Every mutating route must sit behind the same check."""
    main = _fresh_app(monkeypatch, tmp_path, FEATHER_API_KEY="s3cret")
    auth = {"X-API-Key": "s3cret"}
    with TestClient(main.app) as c:
        body = {"id": 1, "vector": np.random.rand(DIM).astype(np.float32).tolist(),
                "metadata": {"content": "seed"}}
        assert c.post(f"/v1/{NS}/vectors", json=body, headers=auth).status_code in (200, 201)

        # Same calls, no key.
        assert c.post(f"/v1/{NS}/vectors", json=body).status_code == 401
        assert c.delete(f"/v1/{NS}/records/1").status_code == 401
        assert c.post(f"/v1/{NS}/records/batch_delete", json={"ids": [1]}).status_code == 401
        assert c.delete(f"/v1/namespaces/{NS}").status_code == 401

        # The record survived every unauthenticated attempt.
        assert c.get(f"/v1/{NS}/records/1", headers=auth).status_code == 200
