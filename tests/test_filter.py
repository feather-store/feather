"""
FilterBuilder and SearchFilter tests.
"""
import time
import pytest
import feather_db
from feather_db import DB, FilterBuilder
from .conftest import EMBED


def _db_with_diverse_records(path):
    db = DB.open(path, dim=128)
    records = [
        (1, "product", "user_123", "instagram", "Dark mode preference",      0.9, feather_db.ContextType.PREFERENCE),
        (2, "product", "user_123", "twitter",   "Likes the onboarding flow", 0.6, feather_db.ContextType.EVENT),
        (3, "product", "user_456", "instagram", "Wants faster search",        0.8, feather_db.ContextType.PREFERENCE),
        (4, "infra",   "svc_api",  "internal",  "High latency under load",    0.7, feather_db.ContextType.FACT),
        (5, "infra",   "svc_db",   "internal",  "DB connection pool exhausted",0.5,feather_db.ContextType.EVENT),
    ]
    for rid, ns, eid, channel, content, imp, ctype in records:
        meta = feather_db.Metadata()
        meta.content = content
        meta.namespace_id = ns
        meta.entity_id = eid
        meta.importance = imp
        meta.type = ctype
        meta.set_attribute("channel", channel)
        db.add(id=rid, vec=EMBED(content), meta=meta)
    return db


@pytest.fixture
def diverse_db(tmp_path_feather):
    return _db_with_diverse_records(tmp_path_feather)


class TestNamespaceFilter:
    def test_namespace_filters_correctly(self, diverse_db):
        f = FilterBuilder().namespace("infra").build()
        results = diverse_db.search(EMBED("latency"), k=10, filter=f)
        assert all(r.metadata.namespace_id == "infra" for r in results)
        assert len(results) == 2

    def test_unknown_namespace_returns_empty(self, diverse_db):
        f = FilterBuilder().namespace("nonexistent").build()
        results = diverse_db.search(EMBED("anything"), k=10, filter=f)
        assert len(results) == 0


class TestEntityFilter:
    def test_entity_filters_correctly(self, diverse_db):
        f = FilterBuilder().entity("user_123").build()
        results = diverse_db.search(EMBED("user"), k=10, filter=f)
        assert all(r.metadata.entity_id == "user_123" for r in results)
        assert len(results) == 2

    def test_namespace_and_entity_combined(self, diverse_db):
        f = FilterBuilder().namespace("product").entity("user_456").build()
        results = diverse_db.search(EMBED("search"), k=10, filter=f)
        assert len(results) == 1
        assert results[0].id == 3


class TestAttributeFilter:
    def test_attribute_filter(self, diverse_db):
        f = FilterBuilder().attribute("channel", "instagram").build()
        results = diverse_db.search(EMBED("preference"), k=10, filter=f)
        ids = [r.id for r in results]
        assert 1 in ids
        assert 3 in ids
        assert 2 not in ids  # twitter channel

    def test_namespace_plus_attribute(self, diverse_db):
        f = FilterBuilder().namespace("product").attribute("channel", "twitter").build()
        results = diverse_db.search(EMBED("user"), k=10, filter=f)
        assert len(results) == 1
        assert results[0].id == 2


class TestImportanceFilter:
    def test_importance_gte(self, diverse_db):
        f = FilterBuilder().min_importance(0.8).build()
        results = diverse_db.search(EMBED("preference"), k=10, filter=f)
        assert all(r.metadata.importance >= 0.8 for r in results)
        assert len(results) == 2  # ids 1 and 3


class TestTypeFilter:
    def test_type_filter_preference(self, diverse_db):
        f = FilterBuilder().types([feather_db.ContextType.PREFERENCE]).build()
        results = diverse_db.search(EMBED("user"), k=10, filter=f)
        assert all(r.metadata.type == feather_db.ContextType.PREFERENCE for r in results)
        assert len(results) == 2

    def test_type_filter_multiple(self, diverse_db):
        f = FilterBuilder().types([feather_db.ContextType.PREFERENCE, feather_db.ContextType.FACT]).build()
        results = diverse_db.search(EMBED("user"), k=10, filter=f)
        ids = {r.id for r in results}
        assert 1 in ids  # PREFERENCE
        assert 3 in ids  # PREFERENCE
        assert 4 in ids  # FACT
        assert 2 not in ids  # EVENT — excluded


class TestTimestampFilter:
    def test_timestamp_after(self, tmp_path_feather):
        db = DB.open(tmp_path_feather, dim=128)
        old = feather_db.Metadata()
        old.timestamp = 1000000
        db.add(id=1, vec=EMBED("old"), meta=old)
        new = feather_db.Metadata()
        new.timestamp = int(time.time())
        db.add(id=2, vec=EMBED("new"), meta=new)

        cutoff = int(time.time()) - 3600
        f = FilterBuilder().after(cutoff).build()
        results = db.search(EMBED("record"), k=10, filter=f)
        ids = [r.id for r in results]
        assert 2 in ids
        assert 1 not in ids
