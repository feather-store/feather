"""
Core DB tests: add, search, metadata, persistence, multimodal.
"""
import numpy as np
import pytest
import feather_db
from feather_db import DB, FilterBuilder
from .conftest import EMBED, make_embedder


class TestAddAndSearch:
    def test_add_single_vector(self, db):
        meta = feather_db.Metadata()
        meta.content = "hello world"
        db.add(id=1, vec=EMBED("hello world"), meta=meta)
        results = db.search(EMBED("hello world"), k=1)
        assert len(results) == 1
        assert results[0].id == 1

    def test_search_returns_k_results(self, populated_db):
        results = populated_db.search(EMBED("feature request"), k=4)
        assert len(results) == 4

    def test_search_score_descending(self, populated_db):
        results = populated_db.search(EMBED("deploy"), k=6)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_result_is_most_similar(self, populated_db):
        # Use exact content text as query so hash-based embedder gives cosine ~1.0
        query = "Onboarding drop-off rate is 42% at step 3"
        results = populated_db.search(EMBED(query), k=6)
        assert results[0].id == 1  # exact content match

    def test_search_returns_metadata(self, populated_db):
        results = populated_db.search(EMBED("onboarding"), k=1)
        assert results[0].metadata.content != ""

    def test_search_respects_k_larger_than_index(self, db):
        db.add(id=1, vec=EMBED("a"), meta=feather_db.Metadata())
        results = db.search(EMBED("a"), k=100)
        assert len(results) == 1  # only 1 record exists


class TestMultimodal:
    def test_add_multiple_modalities(self, populated_db):
        text_ids   = populated_db.get_all_ids(modality="text")
        visual_ids = populated_db.get_all_ids(modality="visual")
        assert set(text_ids) == set(visual_ids)

    def test_modalities_are_independent(self, populated_db):
        text_results   = populated_db.search(EMBED("latency"), k=1, modality="text")
        visual_results = populated_db.search(make_embedder(64)("latency"), k=1, modality="visual")
        assert len(text_results) == 1
        assert len(visual_results) == 1

    def test_get_vector_text(self, populated_db):
        vec = populated_db.get_vector(id=1, modality="text")
        assert vec is not None
        assert vec.shape == (128,)

    def test_get_vector_visual(self, populated_db):
        vec = populated_db.get_vector(id=1, modality="visual")
        assert vec.shape == (64,)


class TestMetadata:
    def test_get_metadata(self, populated_db):
        meta = populated_db.get_metadata(1)
        # content/namespace may be cleared by second modality add — check non-None
        assert meta is not None
        assert meta.namespace_id == "product"
        assert meta.entity_id == "user_retention"

    def test_update_metadata(self, db):
        meta = feather_db.Metadata()
        meta.content = "original"
        db.add(id=1, vec=EMBED("original"), meta=meta)

        new_meta = feather_db.Metadata()
        new_meta.content = "updated"
        db.update_metadata(id=1, meta=new_meta)
        assert db.get_metadata(1).content == "updated"

    def test_update_importance(self, db):
        meta = feather_db.Metadata()
        meta.importance = 0.5
        db.add(id=1, vec=EMBED("x"), meta=meta)
        db.update_importance(id=1, importance=0.99)
        assert db.get_metadata(1).importance == pytest.approx(0.99, abs=0.01)

    def test_set_get_attribute(self, db):
        meta = feather_db.Metadata()
        meta.set_attribute("channel", "instagram")
        meta.set_attribute("ctr", "0.045")
        db.add(id=1, vec=EMBED("ad"), meta=meta)
        loaded = db.get_metadata(1)
        assert loaded.get_attribute("channel") == "instagram"
        assert loaded.get_attribute("ctr") == "0.045"

    def test_attribute_dict_mutation_is_noop(self, db):
        """Confirm pybind11 gotcha: meta.attributes['k'] = v silently does nothing."""
        meta = feather_db.Metadata()
        meta.attributes["should_not_work"] = "value"
        db.add(id=1, vec=EMBED("x"), meta=meta)
        loaded = db.get_metadata(1)
        assert loaded.get_attribute("should_not_work") == ""


class TestPersistence:
    def test_save_and_reload(self, populated_db, tmp_path_feather):
        populated_db.save()
        db2 = DB.open(tmp_path_feather, dim=128)
        assert set(db2.get_all_ids()) == {1, 2, 3, 4, 5, 6}

    def test_metadata_survives_reload(self, populated_db, tmp_path_feather):
        populated_db.save()
        db2 = DB.open(tmp_path_feather, dim=128)
        meta = db2.get_metadata(3)
        assert meta is not None
        assert meta.namespace_id == "product"

    def test_attributes_survive_reload(self, db, tmp_path_feather):
        meta = feather_db.Metadata()
        meta.set_attribute("tier", "enterprise")
        db.add(id=99, vec=EMBED("client"), meta=meta)
        db.save()
        db2 = DB.open(tmp_path_feather, dim=128)
        assert db2.get_metadata(99).get_attribute("tier") == "enterprise"

    def test_multimodal_survives_reload(self, populated_db, tmp_path_feather):
        populated_db.save()
        db2 = DB.open(tmp_path_feather, dim=128)
        visual_ids = db2.get_all_ids(modality="visual")
        assert len(visual_ids) == 6


class TestScoringDecay:
    def test_scoring_returns_results(self, populated_db):
        cfg = feather_db.ScoringConfig(half_life=30.0, weight=0.3, min=0.0)
        results = populated_db.search(EMBED("onboarding"), k=5, scoring=cfg)
        assert len(results) == 5

    def test_touch_increments_recall_count(self, db):
        meta = feather_db.Metadata()
        db.add(id=1, vec=EMBED("x"), meta=meta)
        initial = db.get_metadata(1).recall_count
        db.touch(id=1)
        db.touch(id=1)
        assert db.get_metadata(1).recall_count == initial + 2

    def test_search_auto_touches(self, db):
        meta = feather_db.Metadata()
        db.add(id=1, vec=EMBED("hello"), meta=meta)
        before = db.get_metadata(1).recall_count
        db.search(EMBED("hello"), k=1)
        after = db.get_metadata(1).recall_count
        assert after > before
