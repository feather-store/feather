"""Cloud API namespace names are validated, never rewritten.

DBManager used to strip every character but letters, digits, '-' and '_' to
build the file name, while keying its handles by the name as given. So `ni.ke`
and `nike` were two namespaces — two handles — on ONE file. Each checkpoint
wrote its own handle's view over the other's: a write acknowledged with a 201
on one was gone after the other saved. And DELETE /v1/namespaces/ni.ke deleted
nike's data.

Now such a name is a 400 before anything is opened. Every name the old
sanitiser left unchanged — so every file it ever created — is still valid. A
valid name the filesystem folds onto another open namespace's file (case on
macOS/Windows) is a 409.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="Cloud API tests need fastapi")
pytest.importorskip("httpx", reason="fastapi TestClient needs httpx")

from feather_db import DB

from tests.test_api_concurrency import API_DIR, DIM, _client


@pytest.fixture
def client(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, FEATHER_EMBED_PROVIDER="mock",
                 FEATHER_EMBED_DIM=DIM) as c:
        yield c


def _dbm():
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    from app import db_manager  # noqa: WPS433
    return db_manager


def _vec(i):
    return np.random.default_rng(i).random(DIM).astype(np.float32).tolist()


def _post(client, ns, i):
    return client.post(f"/v1/{ns}/vectors",
                       json={"id": i, "vector": _vec(i), "metadata": {"content": f"rec {i}"}})


def _ids_on_disk(path):
    with DB.open(path, dim=DIM, read_only=True) as db:
        return {i for i in db.get_all_ids("text") if db.get_metadata(i) is not None}


# ── the data loss ─────────────────────────────────────────────────────────

def test_a_stripped_alias_no_longer_shares_a_file(client, tmp_path):
    assert _post(client, "nike", 1).status_code == 201

    r = _post(client, "ni.ke", 2)
    assert r.status_code == 400, r.text
    assert "'.'" in r.json()["detail"]
    assert "ni.ke" not in client.get("/v1/namespaces").text

    assert _post(client, "nike", 3).status_code == 201
    from app.main import manager  # the app's own (re-imported per client)
    manager.save_all()
    assert _ids_on_disk(str(tmp_path / "nike.feather")) == {1, 3}


def test_deleting_an_alias_cannot_delete_another_namespace(client, tmp_path):
    assert _post(client, "nike", 1).status_code == 201
    r = client.delete("/v1/namespaces/ni.ke")
    assert r.status_code in (400, 404), r.text
    assert client.get("/v1/nike/records/1").status_code == 200

    from app.main import manager
    with pytest.raises(_dbm().InvalidNamespaceError):
        manager.delete("ni.ke")
    assert client.get("/v1/nike/records/1").status_code == 200


@pytest.mark.parametrize("name", ["ni.ke", "a/b", "..", "x y", "café́",
                                  "tab\t", "nul\x00", "semi;colon"])
def test_names_that_used_to_be_rewritten_are_refused(client, tmp_path, name):
    from urllib.parse import quote
    r = client.post(f"/v1/{quote(name, safe='')}/vectors",
                    json={"id": 1, "vector": _vec(1), "metadata": {"content": "x"}})
    assert r.status_code in (400, 404), (name, r.status_code, r.text)
    assert [f for f in os.listdir(tmp_path) if f.endswith(".feather")] == []


def test_an_empty_or_stripped_to_nothing_name_is_a_400_not_a_500(tmp_path):
    dbm = _dbm()
    mgr = dbm.DBManager(data_dir=str(tmp_path), default_dim=DIM)
    for bad in ("", "...", "!!"):
        with pytest.raises(dbm.InvalidNamespaceError):
            mgr.get(bad)
    mgr.close_all()


# ── nothing that worked before breaks ─────────────────────────────────────

@pytest.mark.parametrize("name", ["nike", "Nike_2024", "brand-1", "_x_", "a",
                                  "café", "日本", "0123456789"])
def test_every_name_the_old_sanitiser_kept_still_works(client, tmp_path, name):
    from urllib.parse import quote
    assert _post(client, quote(name, safe=""), 1).status_code == 201
    assert os.path.exists(tmp_path / f"{name}.feather.lock")


def test_existing_files_still_load_at_startup(tmp_path, capsys):
    for ns in ("nike", "brand-1", "café"):
        with DB.open(str(tmp_path / f"{ns}.feather"), dim=DIM) as db:
            db.add(1, np.asarray(_vec(1), dtype=np.float32))
    # Written by hand, never by the API: it used to load as namespace
    # "my.brand" while serving the file "mybrand.feather".
    with DB.open(str(tmp_path / "my.brand.feather"), dim=DIM) as db:
        db.add(1, np.asarray(_vec(1), dtype=np.float32))

    mgr = _dbm().DBManager(data_dir=str(tmp_path), default_dim=DIM)
    assert sorted(mgr.list_namespaces()) == ["brand-1", "café", "nike"]
    assert "skipping namespace 'my.brand'" in capsys.readouterr().err
    assert not os.path.exists(tmp_path / "mybrand.feather")
    mgr.close_all()


def test_length_limit_leaves_room_for_the_wal(tmp_path):
    dbm = _dbm()
    mgr = dbm.DBManager(data_dir=str(tmp_path), default_dim=DIM)
    longest = 255 - len(".feather.wal.old")
    db = mgr.get("n" * longest)
    db.add(1, np.asarray(_vec(1), dtype=np.float32))
    db.save()
    with pytest.raises(dbm.InvalidNamespaceError, match="bytes"):
        mgr.get("n" * (longest + 1))
    with pytest.raises(dbm.InvalidNamespaceError):
        mgr.get("é" * (longest // 2 + 1))       # bytes, not characters
    mgr.close_all()


# ── filesystem aliasing ───────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform == "win32", reason="hardlinks need POSIX here")
def test_two_names_on_one_file_are_a_conflict(client, tmp_path):
    """Case folding (macOS/Windows) or Unicode normalisation make two valid
    names one file. A hardlink reproduces that on any filesystem."""
    assert _post(client, "nike", 1).status_code == 201
    os.link(tmp_path / "nike.feather.lock", tmp_path / "swoosh.feather.lock")
    r = _post(client, "swoosh", 2)
    assert r.status_code == 409, r.text
    assert "'nike'" in r.json()["detail"]
    assert _post(client, "nike", 3).status_code == 201


def _case_insensitive(d):
    (d / "probe").write_text("x")
    return os.path.exists(d / "PROBE")


def test_case_variants_conflict_where_the_filesystem_folds_case(client, tmp_path):
    """macOS / Windows defaults: `NIKE` would open nike's file — before this,
    as a second handle whose saves overwrote nike's."""
    if not _case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem: the names are separate files")
    assert _post(client, "nike", 1).status_code == 201
    assert _post(client, "NIKE", 2).status_code == 409
    from app.main import manager
    manager.save_all()
    assert _ids_on_disk(str(tmp_path / "nike.feather")) == {1}


def test_case_variants_stay_separate_where_the_filesystem_keeps_them_apart(client, tmp_path):
    if _case_insensitive(tmp_path):
        pytest.skip("case-insensitive filesystem: covered by the conflict test")
    assert _post(client, "nike", 1).status_code == 201
    assert _post(client, "Nike", 2).status_code == 201
    from app.main import manager
    manager.save_all()
    assert _ids_on_disk(str(tmp_path / "nike.feather")) == {1}
    assert _ids_on_disk(str(tmp_path / "Nike.feather")) == {2}


def test_a_recovery_reopen_is_not_a_conflict_with_itself(tmp_path):
    dbm = _dbm()
    mgr = dbm.DBManager(data_dir=str(tmp_path), default_dim=DIM)
    mgr.get("nike")
    mgr._open_namespace("nike")                  # what _recover_locked does
    mgr.close_all()


# ── upload ────────────────────────────────────────────────────────────────

def test_upload_with_an_invalid_name_is_refused_and_cleaned_up(client, tmp_path):
    pytest.importorskip("multipart", reason="uploads need python-multipart")
    src = tmp_path / "built" / "local.feather"
    os.makedirs(src.parent)
    with DB.open(str(src), dim=DIM) as db:
        db.add(1, np.asarray(_vec(1), dtype=np.float32))
    with open(src, "rb") as fh:
        r = client.post("/v1/admin/upload", data={"namespace": "ni.ke"},
                        files={"file": ("local.feather", fh, "application/octet-stream")})
    assert r.status_code == 400, r.text
    assert [f for f in os.listdir(tmp_path) if f.endswith(".upload")] == []
    assert not os.path.exists(tmp_path / "nike.feather")
