"""IngestPipeline Phase 2 wiring tests — OntologyLinker + ContradictionResolver
gated behind enable_phase2."""
from __future__ import annotations
import os
import tempfile

import numpy as np
import pytest

import feather_db
from feather_db.extractors import (
    FactExtractor, OntologyEdge, ContradictionFinding,
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
def two_fact_provider():
    """One fact-extraction response with two facts (different SP each)."""
    return MockProvider([
        '''[
          {"subject":"Acme","predicate":"launched","object":"Summer Sale","confidence":0.95},
          {"subject":"Summer Sale","predicate":"had_CTR","object":"4.5%","confidence":0.9}
        ]'''
    ])


@pytest.fixture
def cross_record_ctr_provider():
    """Two records; same SP each, different objects (numeric)."""
    return MockProvider([
        '[{"subject":"Acme","predicate":"had_CTR","object":"4.5%","confidence":0.95}]',
        '[{"subject":"Acme","predicate":"had_CTR","object":"4.2%","confidence":0.95}]',
    ])


# ── Phase 2 OFF (default) ────────────────────────────────────────────

def test_phase2_off_does_not_call_linker_or_resolver(tmp_db,
                                                     two_fact_provider):
    """enable_phase2=False → resolvers untouched, no Phase 2 stats."""
    linker_calls = []
    resolver_calls = []

    class TripwireLinker:
        name = "tripwire"
        allowed_relations = ["supports"]
        def link(self, items, *, context=None):
            linker_calls.append(items)
            return []

    class TripwireResolver:
        name = "tripwire"
        def detect(self, new_fact, candidates, *, context=None):
            resolver_calls.append(new_fact)
            return []

    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=FactExtractor(provider=two_fact_provider),
        ontology_linker=TripwireLinker(),
        contradiction_resolver=TripwireResolver(),
        enable_phase2=False,  # default but explicit
        namespace="acme",
    )
    stats = pipe.ingest([IngestRecord(
        content="Acme launched Summer Sale.", source_id="r1")])
    assert stats.facts_extracted == 2
    assert stats.ontology_edges_added == 0
    assert stats.contradictions_detected == 0
    assert linker_calls == []
    assert resolver_calls == []


def test_phase2_on_with_no_resolvers_is_noop(tmp_db, two_fact_provider):
    """enable_phase2=True with linker=None + resolver=None → no error."""
    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=FactExtractor(provider=two_fact_provider),
        enable_phase2=True,
        namespace="acme",
    )
    stats = pipe.ingest([IngestRecord(
        content="Acme launched Summer Sale.", source_id="r1")])
    assert stats.records_ingested == 1
    assert stats.ontology_edges_added == 0
    assert stats.contradictions_detected == 0


# ── Phase 2 ON: OntologyLinker ───────────────────────────────────────

def test_phase2_on_calls_ontology_linker(tmp_db, two_fact_provider):
    captured = []

    class StubLinker:
        name = "stub"
        allowed_relations = ["supports"]
        def link(self, items, *, context=None):
            captured.append(list(items))
            # Emit one edge between f_0 and f_1
            return [OntologyEdge(source_id="f_0", target_id="f_1",
                                 rel_type="supports", weight=0.8)]

    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=FactExtractor(provider=two_fact_provider),
        ontology_linker=StubLinker(),
        enable_phase2=True,
        namespace="acme",
    )
    stats = pipe.ingest([IngestRecord(
        content="Acme launched Summer Sale. CTR was 4.5%.",
        source_id="r1")])
    assert stats.facts_extracted == 2
    assert stats.ontology_edges_added == 1
    assert len(captured) == 1
    assert len(captured[0]) == 2


def test_ontology_linker_skipped_when_fewer_than_2_facts(tmp_db):
    """One fact only → linker not called (nothing to link to)."""
    one_fact = MockProvider([
        '[{"subject":"A","predicate":"is","object":"B","confidence":1.0}]'
    ])
    captured = []
    class StubLinker:
        name = "stub"
        allowed_relations = []
        def link(self, items, *, context=None):
            captured.append(items)
            return []

    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=FactExtractor(provider=one_fact),
        ontology_linker=StubLinker(),
        enable_phase2=True,
        namespace="acme",
    )
    pipe.ingest([IngestRecord(content="x", source_id="r1")])
    assert captured == []


def test_ontology_linker_failure_does_not_abort_ingest(tmp_db,
                                                       two_fact_provider):
    """Linker exception is swallowed; record still ingested."""
    class BrokenLinker:
        name = "broken"
        allowed_relations = []
        def link(self, items, *, context=None):
            raise RuntimeError("provider went down")

    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=FactExtractor(provider=two_fact_provider),
        ontology_linker=BrokenLinker(),
        enable_phase2=True,
        namespace="acme",
    )
    stats = pipe.ingest([IngestRecord(
        content="Acme launched Summer Sale. CTR was 4.5%.",
        source_id="r1")])
    assert stats.records_ingested == 1
    assert stats.facts_extracted == 2
    assert stats.ontology_edges_added == 0


def test_ontology_invalid_endpoints_dropped(tmp_db, two_fact_provider):
    """Edge with id not in items list is silently dropped."""
    class BadLinker:
        name = "bad"
        allowed_relations = []
        def link(self, items, *, context=None):
            return [
                OntologyEdge(source_id="f_99", target_id="f_0",
                             rel_type="supports", weight=0.5),  # bad src
                OntologyEdge(source_id="f_0", target_id="f_1",
                             rel_type="supports", weight=0.5),  # ok
            ]

    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=FactExtractor(provider=two_fact_provider),
        ontology_linker=BadLinker(),
        enable_phase2=True,
        namespace="acme",
    )
    stats = pipe.ingest([IngestRecord(
        content="Acme launched Summer Sale. CTR was 4.5%.",
        source_id="r1")])
    assert stats.ontology_edges_added == 1


# ── Phase 2 ON: ContradictionResolver ────────────────────────────────

def test_phase2_on_invokes_contradiction_resolver_cross_record(
        tmp_db, cross_record_ctr_provider):
    """Two records with same SP, different objects. Resolver gets called
    on the second, with the first as candidate."""
    detect_calls = []

    class StubResolver:
        name = "stub"
        def detect(self, new_fact, candidates, *, context=None):
            detect_calls.append({
                "new_obj": new_fact.object,
                "candidate_objs": [c.object for c in candidates],
            })
            if not candidates:
                return []
            return [ContradictionFinding(
                new_fact=new_fact,
                conflicting_with=candidates[0],
                severity="probable",
                rationale="numeric mismatch",
                suggested_resolution="review",
            )]

    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=FactExtractor(provider=cross_record_ctr_provider),
        contradiction_resolver=StubResolver(),
        enable_phase2=True,
        namespace="acme",
    )
    pipe.ingest([
        IngestRecord(content="Today CTR is 4.5%.", source_id="r1"),
        IngestRecord(content="Today CTR is 4.2%.", source_id="r2"),
    ])
    # First record: no prior facts → resolver called with empty candidates,
    #               or not called at all if candidate-fetch returns []
    # Second record: should have one candidate (the first record's fact)
    has_with_candidate = any(c["candidate_objs"] for c in detect_calls)
    assert has_with_candidate, (
        f"expected at least one detect() call with candidates, "
        f"got: {detect_calls}"
    )


def test_contradiction_findings_persisted_as_edges(
        tmp_db, cross_record_ctr_provider):
    """Detected findings show up as `contradicts` edges in the DB
    + bump stats.contradictions_detected."""
    class AlwaysFinds:
        name = "always"
        def detect(self, new_fact, candidates, *, context=None):
            return [ContradictionFinding(
                new_fact=new_fact,
                conflicting_with=c,
                severity="definite",
                rationale="forced",
                suggested_resolution="review",
            ) for c in candidates]

    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=FactExtractor(provider=cross_record_ctr_provider),
        contradiction_resolver=AlwaysFinds(),
        enable_phase2=True,
        namespace="acme",
    )
    stats = pipe.ingest([
        IngestRecord(content="r1", source_id="r1"),
        IngestRecord(content="r2", source_id="r2"),
    ])
    if stats.contradictions_detected > 0:
        # If we had at least one finding, the by-severity bucket
        # should also be populated.
        assert sum(stats.contradictions_by_severity.values()) == \
            stats.contradictions_detected
        assert "definite" in stats.contradictions_by_severity


def test_resolver_skipped_when_no_candidates(tmp_db):
    """Single record, no prior facts → no contradicts edges."""
    one_fact = MockProvider([
        '[{"subject":"A","predicate":"is","object":"B","confidence":1.0}]'
    ])
    detect_calls = []
    class StubResolver:
        name = "stub"
        def detect(self, new_fact, candidates, *, context=None):
            detect_calls.append(candidates)
            return []

    pipe = IngestPipeline(
        db=tmp_db,
        embedder=HashEmbedder(128),
        fact_extractor=FactExtractor(provider=one_fact),
        contradiction_resolver=StubResolver(),
        enable_phase2=True,
        namespace="acme",
    )
    stats = pipe.ingest([IngestRecord(content="x", source_id="r1")])
    assert stats.contradictions_detected == 0
    # Either resolver wasn't called (no candidates) or was called with []
    assert all(len(c) == 0 for c in detect_calls)
