"""The Cloud API after a WAL write or fsync failure.

The engine refuses every write on a handle whose WAL write or fsync failed
until the file is reopened (tests/test_write_failures.py). The API holds
handles for its whole lifetime, so without help a namespace stayed read-only
until the server restarted, and save_all() stopped at the first namespace that
raised. Now:

  * the failed write is a 503 with Retry-After — retryable, and clearly NOT
    acknowledged — instead of an opaque 500;
  * DBManager.get() (and the checkpointer) reopen the namespace, which replays
    the WAL that reached the disk;
  * a reopen that fails keeps the old handle: reads still work, writes still
    refuse, and the retry is throttled instead of reloading per request.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="Cloud API tests need fastapi")
pytest.importorskip("httpx", reason="fastapi TestClient needs httpx")

import feather_db
from feather_db import DB, core

from tests.test_api_concurrency import API_DIR, DIM, _client

NS = "wf"


@pytest.fixture
def fail_fsyncs():
    yield core._fail_fsyncs_after
    core._fail_fsyncs_after(-1)


@pytest.fixture
def client(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, FEATHER_EMBED_PROVIDER="mock",
                 FEATHER_EMBED_DIM=DIM) as c:
        yield c


def _manager_module():
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    from app import db_manager  # noqa: WPS433
    return db_manager


def _vec(i):
    return np.random.default_rng(i).random(DIM).astype(np.float32)


def _post(client, i):
    return client.post(f"/v1/{NS}/vectors",
                       json={"id": i, "vector": _vec(i).tolist(),
                             "metadata": {"content": f"rec {i}"}})


def test_failed_write_is_a_503_and_the_namespace_recovers(client, fail_fsyncs):
    assert _post(client, 1).status_code == 201

    fail_fsyncs(0)
    r = _post(client, 2)
    fail_fsyncs(-1)
    assert r.status_code == 503, r.text
    assert r.json()["retryable"] is True
    assert "write not saved" in r.json()["detail"]
    assert int(r.headers["Retry-After"]) >= 1

    # The next request reopens the namespace instead of refusing forever.
    assert _post(client, 3).status_code == 201
    assert client.get(f"/v1/{NS}/records/1").status_code == 200
    assert client.get(f"/v1/{NS}/records/3").status_code == 200


def test_a_failed_reopen_keeps_serving_reads(tmp_path, monkeypatch, fail_fsyncs):
    dbm = _manager_module()
    monkeypatch.setattr(dbm, "RECOVER_RETRY_S", 3600)
    mgr = dbm.DBManager(data_dir=str(tmp_path), default_dim=DIM)
    db = mgr.get(NS)
    db.add(1, _vec(1))
    fail_fsyncs(0)
    with pytest.raises(feather_db.WriteFailedError):
        db.add(2, _vec(2))
    fail_fsyncs(-1)

    real_open = mgr._open_namespace
    calls = []
    def broken_open(ns, dim=None):
        calls.append(ns)
        raise RuntimeError("disk still gone")
    monkeypatch.setattr(mgr, "_open_namespace", broken_open)

    assert mgr.get(NS) is db                  # old handle kept
    assert mgr.get(NS) is db                  # ...and the retry is throttled
    assert calls == [NS]
    assert db.get_metadata(1) is not None     # reads still served
    with pytest.raises(feather_db.WriteFailedError):
        db.add(3, _vec(3))

    monkeypatch.setattr(mgr, "_open_namespace", real_open)
    monkeypatch.setattr(dbm, "RECOVER_RETRY_S", 0)
    mgr._recover_after.clear()
    fresh = mgr.get(NS)
    assert fresh is not db and fresh.write_error() is None
    fresh.add(4, _vec(4))
    assert fresh.get_metadata(1) is not None
    mgr.close_all()


def test_save_all_does_not_stop_at_a_failed_namespace(tmp_path, fail_fsyncs):
    dbm = _manager_module()
    mgr = dbm.DBManager(data_dir=str(tmp_path), default_dim=DIM)
    a, b = mgr.get("a"), mgr.get("b")
    a.add(1, _vec(1))
    fail_fsyncs(0)
    with pytest.raises(feather_db.WriteFailedError):
        a.add(2, _vec(2))
    fail_fsyncs(-1)
    b.add(1, _vec(1))

    failed = mgr.save_all()
    assert set(failed) == {"a"}
    assert b.wal_size() == 0                  # b was checkpointed regardless
    mgr.close_all()


def test_a_v10_file_can_be_uploaded(client, tmp_path):
    pytest.importorskip("multipart", reason="uploads need python-multipart")
    src = str(tmp_path / "built" / "local.feather")
    os.makedirs(os.path.dirname(src))
    with DB.open(src, dim=DIM) as db:
        db.add(1, _vec(1))
    with open(src, "rb") as fh:
        assert fh.read(8)[4:] == (10).to_bytes(4, "little")
        fh.seek(0)
        r = client.post("/v1/admin/upload", data={"namespace": "up"},
                        files={"file": ("local.feather", fh, "application/octet-stream")})
    assert r.status_code == 200, r.text
    assert r.json()["records"] == 1


def test_a_request_still_holding_the_old_handle_keeps_reading(tmp_path, fail_fsyncs):
    """Recovery swaps the handle while other requests may be mid-search on the
    old one. It is retired, not closed, so they finish normally."""
    dbm = _manager_module()
    mgr = dbm.DBManager(data_dir=str(tmp_path), default_dim=DIM)
    old = mgr.get(NS)
    old.add(1, _vec(1))
    fail_fsyncs(0)
    with pytest.raises(feather_db.WriteFailedError):
        old.add(2, _vec(2))
    fail_fsyncs(-1)

    fresh = mgr.get(NS)
    assert fresh is not old
    assert not old.is_closed()
    assert [r.id for r in old.search(_vec(1), k=1)] == [1]
    fresh.add(3, _vec(3))
    mgr.close_all()


def test_api_runs_on_an_engine_without_the_write_failure_api(tmp_path, monkeypatch):
    """A deployment can pair this app with an older feather-db wheel: it must
    start, serve, and treat every handle as healthy."""
    monkeypatch.delattr(feather_db, "WriteFailedError")
    with _client(tmp_path, monkeypatch, FEATHER_EMBED_PROVIDER="mock",
                 FEATHER_EMBED_DIM=DIM) as c:
        assert _post(c, 1).status_code == 201
    dbm = _manager_module()
    assert dbm.write_error(object()) is None
