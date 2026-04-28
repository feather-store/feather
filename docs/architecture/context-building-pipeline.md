---
id: context-building-pipeline
title: "Phase 9 Context-Building Pipeline at Scale"
status: design
audience: internal + claude-cowork
last_updated: 2026-04-28
parent: docs/architecture/phase9-plan.md
---

# How Feather Builds Context at Scale

> **Question being answered:** when a brand pushes us 10,000+ data points per day, what's the order of operations? Do we (a) extract context then search-and-map to existing intel, (b) search existing intel first then extract with hints, or (c) something else?

> **Answer:** Two-phase hybrid. Phase 1 is stateless extraction (parallelizable, cacheable). Phase 2 is per-record linking against existing brand context (entity linking, fact dedup, edge formation, contradiction detection, hierarchical attribution).

> Production systems (Mem0, Supermemory, knowledge-graph literature) all converge on this two-phase shape because pure single-pass approaches lose either parallelism or canonical alignment.

---

## 1. The two phases

### Phase 1 — Extraction

```
Input:    raw data point (chat turn, ad row, doc chunk, ...)
Process:  FactExtractor   (LLM, no brand-context hints, deterministic)
          EntityResolver  (LLM, no known-entity seed)
          TemporalParser  (rule-based)
Output:   raw_facts.jsonl + raw_entities.jsonl
Storage:  content-addressed cache keyed by sha256(text + extractor_version)
```

**Properties:**
- **Stateless** — no DB lookups, no awareness of existing brand context.
- **Parallelizable** — 10K data points run as 50–200 concurrent LLM calls.
- **Cacheable** — same input + same extractor version → bit-identical output. Re-running on the same source is a no-op.
- **Idempotent** — safe to retry, replay, reprocess after prompt updates.

### Phase 2 — Linking (the "search and map")

```
Input:    raw_facts + raw_entities from Phase 1
Process (per fact, sequential within a brand):
  1. Embed the fact's content
  2. Vector + BM25 search the brand's existing canonical entities (top-k=20)
  3. EntityResolver runs again WITH known_entities seed → canonical IDs
  4. Vector search for similar existing facts (top-k=10)
  5. ContradictionResolver — LLM judge over the peer set
  6. OntologyLinker — emit typed edges
  7. Hierarchical attribution from source data structure (e.g. ad→ad_set→campaign)
  8. Persist fact + entity rows + edges; update canonical-entity index
Output:   structured intel in feather_db.DB with full provenance
```

**Properties:**
- **Context-aware** — every linking decision sees the brand's existing knowledge.
- **Sequential within brand** — avoids race conditions on canonical IDs.
- **Lightweight LLM calls** — each judge call is ~200 tokens in / 50 out, cheap.
- **Self-learning telemetry** — every step logs entity-resolution confidence, fallback rate, contradiction count, edge types emitted.

---

## 2. Why this beats single-pass

| Property | One-pass (search-then-extract) | Two-phase hybrid |
|---|---|---|
| Parallelizable extraction | ❌ depends on retrieval | ✅ 10K in parallel |
| Cache hit on extraction | ❌ varies with brand state | ✅ same text → same facts |
| Reprocessable on prompt change | ❌ retroactive re-search needed | ✅ replay from cache |
| Canonical alignment quality | ✅ extractor sees context | ✅ linker sees context |
| Cost per data point | Higher | Lower |
| Cold start | Degenerate | Same as warm |

The single-pass approach trades operational ergonomics for a tiny gain in extractor alignment that the linker recovers anyway.

---

## 3. The "search and map" is 5 sub-tasks

Phase 2 isn't one search — it's a coordinated set of 5 lookups, each with a different optimal strategy:

| Sub-task | Question | Strategy |
|---|---|---|
| **Entity linking** | Surface form "Acme" → canonical entity? | vector + alias dict + LLM judge |
| **Fact deduplication** | Is this fact already in memory? | high-sim threshold (>0.95) + exact triple match |
| **Edge formation** | Which existing facts should this connect to? | vector top-k + LLM judge picks edge type |
| **Contradiction detection** | Does this contradict anything? | filter by same subject + LLM judge on conflicts |
| **Hierarchical attribution** | Where in the hierarchy does this fit? | **schema-aware** (derive from source data structure) |

The fifth is critical for verticals like Marketing: hierarchy is **structural** (Meta Ads API tells you `ad_id` belongs to `ad_set_id` belongs to `campaign_id`), not inferred. We don't ask an LLM to figure out "is creative_55 part of campaign_42" — the source data already says so. The other four are LLM-augmented searches.

---

## 4. Volume math at 10,000 data points/day per brand

```
Phase 1 — Extraction (batched, nightly)
  10,000 raw data points × 1 LLM call (FactExtractor)
  ~5 facts per data point → 50,000 facts/day
  System LLM = Claude Haiku at ~$0.001/call → $10/day per brand
  Wall time: parallel @ 50 concurrent → ~3 minutes

Phase 2 — Linking (per fact)
  50,000 facts × (1 vector search + 1 small LLM judge)
  Vector searches ~0.2ms each = ~10 seconds total
  LLM judges: ~250 tokens × 50K = 12.5M tokens × $0.30/1M = $3.75/day
  Wall time: parallel-batched in chunks of 100 → ~5 minutes

Per brand / day total:
  Compute cost   ~$14
  Wall time      ~10 minutes
  Storage growth ~50K facts × ~500 bytes ≈ 25 MB / day

At 100 brands:                         At 1,000 brands:
  ~$1,400/day · ~$510K/year              ~$14,000/day · ~$5.1M/year
  17 hours wall total                    170 hours (heavily parallelizable)
  2.5 GB storage / day                   25 GB / day
```

This is sustainable on the inference economics from the Phase 9 research. At 100 brands paying $200–500/month each = $240–600K/year revenue against ~$510K cost. Margin gets healthier at scale (LLM rates drop, batching gets denser).

---

## 5. Batch vs streaming — per vertical

The pipeline supports both: `IngestPipeline` takes a list, batch = list-of-N, streaming = list-of-1. Cloud's vertical config picks the cadence.

| Vertical | Cadence | Why |
|---|---|---|
| **Marketing** (Meta Ads, Google Ads, GA4) | Batch nightly | Data drops daily; no need for real-time |
| **DevTools** (GitHub, Linear) | Batch nightly | Same |
| **Customer Support** (Slack, email, Zendesk) | **Streaming** | Queries need same-day context |
| **Analytics** (Mixpanel, Amplitude) | Batch nightly | Same |
| **Sales/CRM** (HubSpot, Salesforce) | Hybrid | Nightly batch + streaming on key events (deal closed, opp won) |
| **Compliance / regulated** | Batch with audit | Ingestion gated by review |

---

## 6. Self-learning lives in Phase 2 telemetry

Every linking step emits a telemetry record:

```jsonc
{
  "tenant_id":           "acme_corp",
  "fact_id":             "fact_abc",
  "extractor_version":   "0.1.0",
  "linker_version":      "0.1.0",
  "entity_resolved":     "brand::acme",
  "resolution_conf":     0.97,
  "fallback_used":       false,
  "contradictions_found": 1,
  "edges_formed":        3,
  "edge_types":          ["refers_to", "extracted_from", "part_of"],
  "took_ms":             180
}
```

Aggregated daily/weekly per tenant, this drives:

| Signal | Interpretation | Auto-action |
|---|---|---|
| Unknown-rate trending up | New entity class emerging | Propose ontology extension to vertical |
| Same `unknown::xxx` ID appears N times | Implicit canonical entity | Promote to `<kind>::xxx` after threshold |
| Contradiction rate spike | Data quality issue OR prompt drift | Alert + sample for human review |
| Edge type distribution shifting | Ontology evolution | Recommend new edge types |
| Confidence drop on existing entities | Brand changed something | Flag for human review |
| Repeated retrieve-without-pin | Memory not actually useful | Demote in decay |

This is the **Cloud-only** self-learning loop (per the Phase 9 plan). OSS has the primitives; Cloud aggregates the cross-tenant signal.

---

## 7. The cold-start problem (and the Brand Bootstrap)

For a brand-new tenant on day 1, there's no existing context to search against. Phase 2 degenerates to: every entity is a fallback, no contradictions detected, no edges formed.

Solution: the **Brand Bootstrap** — Cloud runs a one-time vertical-aware initialization:

1. Customer connects their primary data sources (Meta Ads + GA4 + Slack).
2. Cloud pulls the last 30 days as a single batch.
3. Phase 1 + Phase 2 run end-to-end with **all 30 days** in scope.
4. Within the batch, canonical entities consolidate (Acme appearing 1,000 times across 30 days resolves to one `brand::acme`).
5. Hierarchical structure (campaigns, ad sets, ads) is built from source schema.
6. Result: a pre-warmed `.feather` file ready for queries on day 1.

This is what we sell: "Connect your sources today; query your full marketing memory tomorrow morning."

---

## 8. Implementation status

| Component | Status | Where |
|---|---|---|
| Phase 1 extractors | ✅ shipping (Week 2) | `feather_db/extractors/*.py` |
| Phase 1 → Phase 2 wiring | ⚠️ partial — current `IngestPipeline` does both inline | `feather_db/pipelines/ingest.py` |
| Phase 2 entity linking against existing canonical store | ⏳ Week 3 | needs `_search_existing_canonical()` step |
| Phase 2 ContradictionResolver | ⏳ Week 3 | `feather_db/consolidation/contradictions.py` |
| Phase 2 OntologyLinker | ⏳ Week 3 | `feather_db/extractors/ontology.py` |
| Hierarchical attribution from source schema | ⏳ Week 4 | per-vertical adapter logic |
| Self-learning telemetry log | ⏳ Cloud (Week 5+) | `feather_cloud/evolve/telemetry.py` |
| Brand Bootstrap | ⏳ Cloud (Week 6+) | `feather_cloud/verticals/marketing/bootstrap.py` |

Current `IngestPipeline.ingest()` does Phase 1 + a *partial* Phase 2 (entity dedup within the batch via `_entity_index`, but no search against existing brand store). Week 3 introduces:

```python
class IngestPipeline:
    def __init__(self, ..., enable_phase2: bool = False, ...): ...

    def _ingest_one(self, rec, stats):
        # ... existing Phase 1 ...
        if self._enable_phase2:
            self._link_against_existing(facts, entities, stats)
```

The split keeps OSS users on the simple path (current behavior) and Cloud / power users on the full pipeline.

---

## 9. Open questions for Week 3

1. **What's the entity-linking similarity threshold?** Below 0.85 cosine → fallback; above 0.95 → automatic canonical match; 0.85–0.95 → ask LLM judge. Tunable.
2. **How big is the "peer set" for contradiction detection?** Top-5 most-similar existing facts seems right; revisit after 50-question bench.
3. **When does Phase 2 batch ingestion run sequentially vs concurrent?** Within-brand sequential (canonical-ID race), across-brand concurrent.
4. **Is there a rebuild step?** When a vertical's ontology changes, do we re-link all existing facts? Probably yes, gated on a manual flag — Cloud admin action.
5. **Does Phase 2 run at query-time too?** Probably no — too expensive. But the QueryPlanner (Week 5+) can decide *to consult* the canonical entity index as a search axis.

---

## 10. What this means for v0.9.0

OSS v0.9.0 ships:
- Phase 1 (already done)
- Phase 2 with **explicit `enable_phase2` flag, default off** (so existing OSS users get a no-surprise upgrade path)
- Hierarchical attribution helpers (`feather_db.hierarchy`) — utility, not magic
- Telemetry hooks (no aggregation; Cloud aggregates)

Cloud v0.9.0:
- Phase 2 default-on, fully tuned
- Brand Bootstrap connectors (Marketing vertical: Meta Ads + Google Ads + GA4)
- Self-learning aggregation + auto-promotion
- Per-tenant tuning of thresholds

---

## 11. Decisions needed before Week 3

1. **Confirm two-phase architecture** (vs single-pass) — does this match your mental model?
2. **OSS gets Phase 2 with `enable_phase2` flag, off by default?** — keeps OSS simple while making the full pipeline replicable.
3. **First production target: Marketing vertical with batch nightly cadence?** — this is what we've discussed; locks the Brand Bootstrap design.

If yes to all three, I'll:
- Implement the Phase 2 search-against-existing in `IngestPipeline`
- Implement `OntologyLinker` + `ContradictionResolver`
- Add a small `feather_db.hierarchy` helper for schema-aware attribution
- Run a 10-question Phase-9 bench at the end of Week 3 to validate the lift
