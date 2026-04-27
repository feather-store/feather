"""IngestPipeline — wires extractors → Feather DB with provenance.

Reads raw text records, runs the configured extractors (facts,
entities, temporal), and persists the structured intel into a
feather_db.DB with edges back to the source. Storage layout:

    Source records          (id range: SOURCE_ID_BASE..)
        ↑ extracted_from
    Atomic facts            (id range: FACT_ID_BASE..)
        ↑ refers_to (optional, when entities mention them)
    Canonical entities      (id range: ENTITY_ID_BASE..)

All entries share the same `namespace_id` per-pipeline-run. Edges
follow Feather's typed-edge convention.

Status: Phase 9.1 Week 2. OntologyLinker + ContradictionResolver wiring
arrives in Week 3.
"""
from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import feather_db
from ..extractors import (
    FactExtractor,
    EntityResolver,
    TemporalParser,
    Fact,
    Entity,
)


# ID-range partitioning so source/fact/entity IDs never collide within a
# single namespace. Each tier gets a 1B-id space, well below uint64 max.
SOURCE_ID_BASE = 1_000_000_000
FACT_ID_BASE   = 2_000_000_000
ENTITY_ID_BASE = 3_000_000_000


@dataclass
class IngestRecord:
    """One unit of raw text to ingest. The pipeline embeds + extracts."""
    content: str
    source_id: str                              # caller-stable identifier
    timestamp: int = 0                          # unix; 0 = use now
    metadata: dict = field(default_factory=dict)


@dataclass
class IngestStats:
    records_ingested: int = 0
    facts_extracted: int = 0
    entities_resolved: int = 0
    timestamps_extracted: int = 0
    extraction_failures: int = 0
    elapsed_seconds: float = 0.0
    failures_sample: list[dict] = field(default_factory=list)


class IngestPipeline:
    """Stage 1–4 of the Phase 9 pipeline (ingest → infer → convert →
    store). Stage 5 (query-time reasoner) lives in feather_db.reason.

    Args:
        db:           a feather_db.DB instance (already opened).
        embedder:     anything with .embed(text) -> np.ndarray.
        fact_extractor:    FactExtractor instance, or None to skip.
        entity_resolver:   EntityResolver instance, or None to skip.
        temporal_parser:   TemporalParser instance, or None to skip
                           (defaults to a fresh one).
        namespace:    namespace_id applied to every stored record.
        modality:     which Feather modality to use (default 'text').
        on_failure:   optional callback(record_source_id, exception).

    Example:
        >>> from feather_db import DB
        >>> from feather_db.providers import ClaudeProvider
        >>> from feather_db.extractors import FactExtractor, EntityResolver
        >>> from feather_db.pipelines import IngestPipeline
        >>>
        >>> db = DB.open("memory.feather", dim=1536)
        >>> llm = ClaudeProvider(model="claude-haiku-4-5-20251001")
        >>> pipe = IngestPipeline(
        ...     db=db,
        ...     embedder=embed_fn,
        ...     fact_extractor=FactExtractor(provider=llm),
        ...     entity_resolver=EntityResolver(provider=llm),
        ...     namespace="acme_corp",
        ... )
        >>> stats = pipe.ingest([
        ...     IngestRecord(content="Acme launched Summer Sale on March 15.",
        ...                  source_id="memo-001"),
        ... ])
        >>> # facts have 'extracted_from' edges back to the source memory id.
    """

    def __init__(self, *,
                 db,
                 embedder,
                 fact_extractor: Optional[FactExtractor] = None,
                 entity_resolver: Optional[EntityResolver] = None,
                 temporal_parser: Optional[TemporalParser] = None,
                 namespace: str = "default",
                 modality: str = "text",
                 on_failure: Optional[Callable] = None):
        self._db = db
        self._embedder = embedder
        self._facts = fact_extractor
        self._entities = entity_resolver
        self._temporal = temporal_parser or TemporalParser()
        self._namespace = namespace
        self._modality = modality
        self._on_failure = on_failure
        # Per-run id counters — pipelines are typically single-shot per
        # ingest call; reusing the same pipeline across multiple ingest()
        # calls keeps id ranges monotonic.
        self._source_seq = 0
        self._fact_seq = 0
        self._entity_seq = 0
        # Canonical-id -> entity feather row id, for reuse across records.
        self._entity_index: dict[str, int] = {}

    # ── Public API ────────────────────────────────────────────────────

    def ingest(self, records: Iterable[IngestRecord]) -> IngestStats:
        """Ingest a batch of records. Returns stats."""
        stats = IngestStats()
        t0 = time.perf_counter()
        for rec in records:
            try:
                self._ingest_one(rec, stats)
                stats.records_ingested += 1
            except Exception as e:
                stats.extraction_failures += 1
                if len(stats.failures_sample) < 10:
                    stats.failures_sample.append({
                        "source_id": rec.source_id,
                        "error": f"{type(e).__name__}: {str(e)[:200]}",
                    })
                if self._on_failure:
                    self._on_failure(rec.source_id, e)
        stats.elapsed_seconds = time.perf_counter() - t0
        return stats

    # ── Internals ─────────────────────────────────────────────────────

    def _ingest_one(self, rec: IngestRecord, stats: IngestStats) -> None:
        # 1. Store the raw source record
        source_id_int = self._next_source_id()
        meta = feather_db.Metadata()
        meta.content = rec.content
        meta.timestamp = rec.timestamp or int(time.time())
        meta.namespace_id = self._namespace
        meta.entity_id = rec.source_id
        meta.source = "ingest_pipeline"
        meta.set_attribute("kind", "source_record")
        meta.set_attribute("ingest_source_id", rec.source_id)
        for k, v in rec.metadata.items():
            try:
                meta.set_attribute(str(k), str(v))
            except Exception:
                pass

        vec = self._embedder.embed(rec.content)
        self._db.add(id=source_id_int, vec=vec,
                     meta=meta, modality=self._modality)

        # 2. Extract facts (LLM-backed, optional)
        facts: list[Fact] = []
        if self._facts is not None:
            facts = self._facts.extract(rec.content, context={
                "source_id": rec.source_id,
                "namespace": self._namespace,
            })
            stats.facts_extracted += len(facts)

        # 3. Extract entities (over the union of fact subjects/objects)
        canonical_by_surface: dict[str, Entity] = {}
        if self._entities is not None and facts:
            surfaces = list({f.subject for f in facts}
                            | {f.object for f in facts})
            entities = self._entities.resolve(surfaces, context={
                "namespace": self._namespace,
            })
            stats.entities_resolved += len(entities)
            for e in entities:
                canonical_by_surface[e.surface_form] = e

        # 4. Extract temporal expressions in the source text
        temporals = self._temporal.extract(rec.content)
        stats.timestamps_extracted += len(temporals)

        # 5. Persist facts with edges back to source + entities
        for f in facts:
            fact_row = self._next_fact_id()
            fmeta = feather_db.Metadata()
            fmeta.content = f"{f.subject} {f.predicate} {f.object}"
            fmeta.timestamp = f.valid_at or meta.timestamp
            fmeta.namespace_id = self._namespace
            fmeta.entity_id = f.subject
            fmeta.source = self._facts.name if self._facts else "fact"
            fmeta.importance = float(f.confidence)
            fmeta.set_attribute("kind", "fact")
            fmeta.set_attribute("subject", f.subject)
            fmeta.set_attribute("predicate", f.predicate)
            fmeta.set_attribute("object", f.object)
            fmeta.set_attribute("confidence", f"{f.confidence:.3f}")
            if f.valid_at:
                fmeta.set_attribute("valid_at_unix", str(f.valid_at))

            fvec = self._embedder.embed(fmeta.content)
            self._db.add(id=fact_row, vec=fvec, meta=fmeta,
                         modality=self._modality)

            # provenance edge: fact extracted_from source
            self._db.link(from_id=fact_row, to_id=source_id_int,
                          rel_type="extracted_from", weight=1.0)

            # entity refs (when both endpoints have a canonical entity)
            for surface in (f.subject, f.object):
                ent = canonical_by_surface.get(surface)
                if not ent:
                    continue
                ent_row = self._ensure_entity_stored(ent)
                self._db.link(from_id=fact_row, to_id=ent_row,
                              rel_type="refers_to",
                              weight=float(ent.confidence))

    def _ensure_entity_stored(self, ent: Entity) -> int:
        """Idempotently store an Entity by canonical_id. Returns row id."""
        if ent.canonical_id in self._entity_index:
            return self._entity_index[ent.canonical_id]
        row = self._next_entity_id()
        em = feather_db.Metadata()
        em.content = f"{ent.kind}: {ent.surface_form}"
        em.timestamp = int(time.time())
        em.namespace_id = self._namespace
        em.entity_id = ent.canonical_id
        em.source = "entity_resolver"
        em.importance = float(ent.confidence)
        em.set_attribute("kind", "entity")
        em.set_attribute("entity_kind", ent.kind)
        em.set_attribute("canonical_id", ent.canonical_id)
        if ent.aliases:
            em.set_attribute("aliases", ", ".join(ent.aliases[:10]))
        evec = self._embedder.embed(em.content)
        self._db.add(id=row, vec=evec, meta=em, modality=self._modality)
        self._entity_index[ent.canonical_id] = row
        return row

    # ── ID allocation ──────────────────────────────────────────────────

    def _next_source_id(self) -> int:
        self._source_seq += 1
        return SOURCE_ID_BASE + self._source_seq

    def _next_fact_id(self) -> int:
        self._fact_seq += 1
        return FACT_ID_BASE + self._fact_seq

    def _next_entity_id(self) -> int:
        self._entity_seq += 1
        return ENTITY_ID_BASE + self._entity_seq


def hash_source_id(content: str) -> str:
    """Convenience: deterministic source_id from content sha256."""
    return "src::" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
