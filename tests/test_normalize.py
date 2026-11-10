"""Vectors at unit length: search ranks by L2 distance, which is cosine order
only for vectors of one length, and scored search is exact only for unit
vectors. A Matryoshka-cut embedding (Gemini at 768-d) is ~0.59 long."""
import os
import sys

import numpy as np
import pytest

import feather_db
from feather_db import DB, keys
from feather_db.pocket import pocket

DIM = 16


def _meta(text, ns="n"):
    m = feather_db.Metadata()
    m.content, m.namespace_id = text, ns
    return m


def _norms(db, ids, modality="text"):
    return [float(np.linalg.norm(db.get_vector(i, modality=modality))) for i in ids]


def test_normalize_vectors_rescales_once_and_survives_a_reload(tmp_path):
    path = str(tmp_path / "n.feather")
    db = DB.open(path, dim=DIM)
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((50, DIM)).astype(np.float32) * rng.uniform(0.3, 2.0, (50, 1)).astype(np.float32)
    for i, v in enumerate(vecs):
        db.add(i + 1, v, _meta(f"r{i}"))
    q = rng.standard_normal(DIM).astype(np.float32)
    assert db.normalize_vectors("text") == 50
    assert db.normalize_vectors("text") == 0                 # idempotent
    assert np.allclose(_norms(db, range(1, 51)), 1.0, atol=1e-5)
    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    by_cosine = list(np.argsort(-(unit @ (q / np.linalg.norm(q))))[:10] + 1)
    assert [r.id for r in db.search(q, k=10, record_salience=False)] == by_cosine
    db.save()
    db.close()
    db = DB.open(path, dim=DIM)
    assert np.allclose(_norms(db, range(1, 51)), 1.0, atol=1e-5)
    assert db.normalize_vectors("missing") == 0
    db.close()


def test_an_int8_modality_is_refused(tmp_path):
    db = DB.open(str(tmp_path / "i.feather"), dim=DIM)
    db.set_int8_ram("text", 1.0)
    db.add(1, np.full(DIM, 0.2, dtype=np.float32), _meta("x"))
    with pytest.raises(ValueError, match="int8"):
        db.normalize_vectors("text")
    db.close()


def test_the_agent_layer_stores_and_searches_unit_vectors(tmp_path):
    db = DB.open(str(tmp_path / "p.feather"), dim=DIM)
    long_embed = lambda text: np.arange(1, DIM + 1, dtype=np.float32) * (len(text) + 1)  # noqa: E731
    pkt = pocket(db, ("org", "u"), embed=long_embed)
    pkt.remember("k", "a note")
    rid = keys.get(db, "org.u", "k")[0]
    assert abs(_norms(db, [rid])[0] - 1.0) < 1e-5
    assert pkt.recall("note", k=1)[0].key == "k"
    db.close()


def test_upkeep_rescales_an_old_file_once(tmp_path):
    path = str(tmp_path / "old.feather")
    db = DB.open(path, dim=DIM)
    for i in range(5):
        db.add(i + 1, np.full(DIM, 0.59 / DIM ** 0.5, dtype=np.float32), _meta(f"r{i}", "org.u"))
    db.save()
    db.close()
    db = DB.open(path, dim=DIM)
    pocket(db, ("org", "u"))                          # the once-per-file pass
    assert keys.upkeep_done(db)
    assert np.allclose(_norms(db, range(1, 6)), 1.0, atol=1e-5)
    db.close(save=False)                               # the pass saved what it rescaled
    db = DB.open(path, dim=DIM)
    assert np.allclose(_norms(db, range(1, 6)), 1.0, atol=1e-5)
    db.close()


def test_embedders_renormalise_a_cut_vector():
    from feather_db.integrations import embedders
    assert embedders._fit([3.0, 4.0, 12.0], 2) == [0.6, 0.8]
    assert embedders._fit([3.0, 4.0], 4) == [3.0, 4.0, 0.0, 0.0]     # padding keeps length


def test_cloud_api_renormalises_gemini_prefixes(monkeypatch):
    api = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "feather-api")
    if api not in sys.path:
        sys.path.insert(0, api)
    from app.embedding import _Cfg, _build, _check_dims
    monkeypatch.delenv("FEATHER_EMBED_STRICT_DIM", raising=False)
    cfg = lambda dim: _Cfg(provider="gemini", model="gemini-embedding-001", key="k",  # noqa: E731
                           base_url="", deployment="", api_version="", dim=dim)
    url, _, body, parse = _build(cfg(768), ["a"])
    assert body["requests"][0]["outputDimensionality"] == 768
    (v,) = parse({"embeddings": [{"values": [0.3, 0.4] + [0.0] * 766}]})   # ~0.59-long prefix
    assert np.isclose(np.linalg.norm(v), 1.0)
    (w,) = _check_dims(cfg(4), [[3.0, 4.0, 0.0, 0.0, 9.0]])               # cut here: renormalised
    assert np.isclose(np.linalg.norm(w), 1.0)


def test_new_agent_files_keep_int8_vectors_reranked(tmp_path, monkeypatch):
    from feather_db.pocket import open_agent_db
    db = open_agent_db(str(tmp_path / "new.feather"), 16)
    assert db.is_int8_ram("text")
    db.close()
    old = DB.open(str(tmp_path / "old.feather"), dim=16)
    old.add(1, np.ones(16, dtype=np.float32) / 4, _meta("x"))
    old.close()
    db = open_agent_db(str(tmp_path / "old.feather"), 16)    # an existing file keeps float
    assert not db.is_int8_ram("text")
    db.close()
    monkeypatch.setenv("FEATHER_INT8", "0")
    db = open_agent_db(str(tmp_path / "opt_out.feather"), 16)
    assert not db.is_int8_ram("text")
    db.close()
