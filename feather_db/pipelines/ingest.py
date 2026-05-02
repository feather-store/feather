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

Phase 9.1 Week 3 adds an `enable_phase2` flag (off in OSS by default,
on in Cloud) that runs OntologyLinker + ContradictionResolver after
fact extraction:

  * OntologyLinker batches the freshly-extracted facts and emits typed
    edges between them (caused_by, supports, supersedes, etc.).
  * ContradictionResolver compares each new fact against same-(subject,
    predicate) facts already in the DB; conflicts are persisted as
    `contradicts` edges with severity-derived weight. Detect-only —
    audit-trail-friendly, never silently merges.
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
    OntologyLinker,
    ContradictionResolver,
    Fact,
    Entity,
    ContradictionFinding,
)
from ..filter import FilterBuilder


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
    # Phase 2 metrics — populated when enable_phase2=True.
    ontology_edges_added: int = 0
    contradictions_detected: int = 0
    contradictions_by_severity: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    failures_sample: list[dict] = field(default_factory=list)


# Maps ContradictionFinding.severity → outgoing edge weight on the
# `contradicts` link. Tuned conservatively; downstream scoring can apply
# additional policy.
_SEVERITY_WEIGHT = {
    "definite": 1.0,
    "probable": 0.7,
    "possible": 0.4,
}


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
                 ontology_linker: Optional[OntologyLinker] = None,
                 contradiction_resolver: Optional[ContradictionResolver] = None,
                 enable_phase2: bool = False,
                 candidate_search_k: int = 20,
                 namespace: str = "default",
                 modality: str = "text",
                 on_failure: Optional[Callable] = None):
        self._db = db
        self._embedder = embedder
        self._facts = fact_extractor
        self._entities = entity_resolver
        self._temporal = temporal_parser or TemporalParser()
        self._ontology = ontology_linker
        self._contradictions = contradiction_resolver
        # Phase 2 = OntologyLinker + ContradictionResolver. Off in OSS
        # default; Cloud verticals turn it on. The flag gates
        # *invocation*; the resolvers themselves can be left None.
        self._enable_phase2 = enable_phase2
        self._candidate_k = candidate_search_k
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
        # Track new fact rows in input order so OntologyLinker edge
        # endpoints (f_0, f_1, …) can be mapped back to row ids.
        new_fact_rows: list[tuple[int, Fact]] = []
        for f in facts:
            # 5a. Phase 2: gather contradiction candidates BEFORE
            #     persisting (so the new fact can't appear in its own
            #     candidate list).
            candidate_pairs: list[tuple[int, Fact]] = []
            if self._enable_phase2 and self._contradictions is not None:
                candidate_pairs = self._fetch_contradiction_candidates(f)

            # 5b. Persist the fact.
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
            new_fact_rows.append((fact_row, f))

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

            # 5c. Phase 2: contradiction findings → `contradicts` edges.
            if (self._enable_phase2 and self._contradictions is not None
                    and candidate_pairs):
                cand_facts = [cf for _, cf in candidate_pairs]
                cand_rows = {id(cf): row for row, cf in candidate_pairs}
                findings = self._contradictions.detect(f, cand_facts)
                for finding in findings:
                    cand_row = cand_rows.get(id(finding.conflicting_with))
                    if cand_row is None:
                        continue
                    weight = _SEVERITY_WEIGHT.get(finding.severity, 0.5)
                    self._db.link(from_id=fact_row, to_id=cand_row,
                                  rel_type="contradicts", weight=weight)
                    stats.contradictions_detected += 1
                    sev = finding.severity
                    stats.contradictions_by_severity[sev] = (
                        stats.contradictions_by_severity.get(sev, 0) + 1
                    )

        # 6. Phase 2: batch-link the freshly persisted facts via
        #    OntologyLinker (caused_by, supports, supersedes, etc.).
        if (self._enable_phase2 and self._ontology is not None
                and len(new_fact_rows) >= 2):
            self._apply_ontology_links(new_fact_rows, stats)

    # ── Phase 2 helpers ───────────────────────────────────────────────

    def _fetch_contradiction_candidates(self, f: Fact
                                        ) -> list[tuple[int, Fact]]:
        """Find prior facts in the DB with the same (subject, predicate).

        Returns (row_id, reconstructed_Fact) pairs. Uses the new fact's
        embedding to drive a filtered top-K vector search; the
        attribute filter (kind=fact + subject + predicate) does the
        real selection. Empty list if the search backend can't satisfy
        the filter or no candidates exist.
        """
        try:
            qvec = self._embedder.embed(
                f"{f.subject} {f.predicate} {f.object}"
            )
            flt = (FilterBuilder()
                   .namespace(self._namespace)
                   .attribute("kind", "fact")
                   .attribute("subject", f.subject)
                   .attribute("predicate", f.predicate)
                   .build())
            results = self._db.search(qvec, k=self._candidate_k,
                                       modality=self._modality,
                                       filter=flt)
        except Exception:
            return []

        out: list[tuple[int, Fact]] = []
        for r in results:
            try:
                meta = r.metadata
                if meta is None:
                    continue
                cand_obj = meta.get_attribute("object", "")
                if not cand_obj:
                    continue
                cand_conf = meta.get_attribute("confidence", "1.0")
                cand_valid = meta.get_attribute("valid_at_unix", "")
                cand_fact = Fact(
                    subject=meta.get_attribute("subject", f.subject),
                    predicate=meta.get_attribute("predicate", f.predicate),
                    object=cand_obj,
                    confidence=_safe_float(cand_conf, 1.0),
                    valid_at=(int(cand_valid) if cand_valid.isdigit()
                              else None),
                )
                out.append((r.id, cand_fact))
            except Exception:
                continue
        return out

    def _apply_ontology_links(self,
                              new_fact_rows: list[tuple[int, Fact]],
                              stats: IngestStats) -> None:
        """Run OntologyLinker over freshly persisted facts and persist
        the resulting typed edges via db.link()."""
        # Index id ('f_0', 'f_1', …) → row id, mirroring the linker's
        # rendering order (Facts come first, Entities second).
        id_to_row: dict[str, int] = {}
        items: list[Fact] = []
        for i, (row, f) in enumerate(new_fact_rows):
            id_to_row[f"f_{i}"] = row
            items.append(f)

        try:
            edges = self._ontology.link(items, context={
                "namespace": self._namespace,
            })
        except Exception:
            return

        for edge in edges:
            from_row = id_to_row.get(edge.source_id)
            to_row = id_to_row.get(edge.target_id)
            if from_row is None or to_row is None:
                continue
            if from_row == to_row:
                continue
            try:
                self._db.link(from_id=from_row, to_id=to_row,
                              rel_type=edge.rel_type,
                              weight=float(edge.weight))
                stats.ontology_edges_added += 1
            except Exception:
                continue

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


def _safe_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default
