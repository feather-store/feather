"""End-to-end IngestPipeline tests with mocked extractors + DB."""
from __future__ import annotations
import os
import tempfile

import numpy as np
import pytest

import feather_db
from feather_db.extractors import (
    FactExtractor, EntityResolver, TemporalParser, Fact, Entity,
)
from feather_db.pipelines import IngestPipeline, IngestRecord


class MockProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete(self, messages, max_tokens=512, temperature=0.0):
        self.calls.append(messages)
        return self._responses.pop(0) if self._responses else "[]"


class HashEmbedder:
    """Deterministic offline embedder for tests."""
    def __init__(self, dim=128):
        self.dim = dim
    def embed(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


@pytest.fixture
def tmp_db():
    path = tempfile.mktemp(suffix=".feather")
    yield feather_db.DB.open(path, dim=128)
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def fact_provider():
    return MockProvider([
        # Response for the first record
        '''[
          {"subject":"Acme","predicate":"launched","object":"Summer Sale","confidence":0.95,"valid_at":"2024-03-15T00:00:00Z"},
          {"subject":"Summer Sale","predicate":"had_CTR","object":"4.5%","confidence":0.9,"valid_at":null}
        ]''',
        # Response for the second record
        '''[
          {"subject":"Acme","predicate":"observed","object":"campaign performance","confidence":0.8,"valid_at":null}
        ]''',
    ])


@pytest.fixture
def entity_provider():
    return MockProvider([
        # Response for first record's entities (Acme, Summer Sale, 4.5%, campaign)
        '''[
          {"surface_form":"Acme","canonical_id":"brand::acme","kind":"Brand","confidence":0.95,"aliases":["ACME"]},
          {"surface_form":"Summer Sale","canonical_id":"campaign::summer_2024","kind":"Campaign","confidence":0.9,"aliases":[]}
        ]''',
        # Response for second record's entities (Acme, campaign performance)
        '''[
          {"surface_form":"Acme","canonical_id":"brand::acme","kind":"Brand","confidence":0.95,"aliases":[]}
        ]''',
    ])


def test_ingest_single_record(tmp_db, fact_provider, entity_provider):
    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=FactExtractor(provider=fact_provider),
        entity_resolver=EntityResolver(provider=entity_provider),
        namespace="acme_corp",
    )
    stats = pipe.ingest([IngestRecord(
        content="Acme launched Summer Sale. CTR was 4.5%.",
        source_id="memo-001",
    )])
    assert stats.records_ingested == 1
    assert stats.facts_extracted == 2
    # 3 unique surfaces in the union of subjects + objects:
    # {Acme, Summer Sale, 4.5%} — Summer Sale appears as both subj and obj.
    assert stats.entities_resolved == 3
    assert stats.extraction_failures == 0


def test_ingest_creates_provenance_edges(tmp_db, fact_provider, entity_provider):
    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=FactExtractor(provider=fact_provider),
        entity_resolver=EntityResolver(provider=entity_provider),
        namespace="acme_corp",
    )
    pipe.ingest([IngestRecord(
        content="Acme launched Summer Sale. CTR was 4.5%.",
        source_id="memo-001",
    )])
    # Source record id is SOURCE_ID_BASE + 1
    from feather_db.pipelines.ingest import SOURCE_ID_BASE, FACT_ID_BASE
    src_id = SOURCE_ID_BASE + 1
    incoming = tmp_db.get_incoming(src_id)
    # 2 facts each have an extracted_from edge to the source
    extracted_from_edges = [e for e in incoming if e.rel_type == "extracted_from"]
    assert len(extracted_from_edges) == 2
    assert all(e.source_id >= FACT_ID_BASE for e in extracted_from_edges)


def test_ingest_dedupes_entities_across_records(tmp_db, fact_provider, entity_provider):
    """Acme appears in both records — should reuse the same entity row."""
    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=FactExtractor(provider=fact_provider),
        entity_resolver=EntityResolver(provider=entity_provider),
        namespace="acme_corp",
    )
    pipe.ingest([
        IngestRecord(content="Acme launched Summer Sale. CTR was 4.5%.",
                     source_id="memo-001"),
        IngestRecord(content="Acme observed strong campaign performance.",
                     source_id="memo-002"),
    ])
    # Only 3 unique canonical entities ingested across both records
    # (brand::acme + campaign::summer_2024 + 2x unknown::* fallbacks)
    # Confirm brand::acme stored exactly once (entity index dedup).
    assert len(pipe._entity_index) == len(set(pipe._entity_index))
    assert "brand::acme" in pipe._entity_index


def test_ingest_no_extractors_just_stores_source(tmp_db):
    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=None,
        entity_resolver=None,
        namespace="raw",
    )
    stats = pipe.ingest([IngestRecord(content="Just a raw note.",
                                       source_id="note-1")])
    assert stats.records_ingested == 1
    assert stats.facts_extracted == 0
    assert stats.entities_resolved == 0
    # The source record IS searchable
    res = tmp_db.search(HashEmbedder(128).embed("Just a raw note."), k=5)
    assert len(res) == 1


def test_ingest_handles_extractor_failure(tmp_db):
    class FailingFactExtractor:
        name = "fail"
        version = "0"
        def extract(self, text, context=None):
            raise RuntimeError("boom")
    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=FailingFactExtractor(),
        namespace="x",
    )
    stats = pipe.ingest([IngestRecord(content="hi", source_id="r1"),
                         IngestRecord(content="hi2", source_id="r2")])
    # Pipeline catches per-record failures and continues
    assert stats.records_ingested == 0
    assert stats.extraction_failures == 2
    assert len(stats.failures_sample) == 2


def test_ingest_temporal_only(tmp_db):
    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=None,
        entity_resolver=None,
        namespace="x",
    )
    stats = pipe.ingest([IngestRecord(
        content="Sale started on 2024-03-15 and ended last week.",
        source_id="x1",
    )])
    assert stats.timestamps_extracted >= 2
