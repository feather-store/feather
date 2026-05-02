---
title: Feather DB — State of the Product
version: v0.8.0
status: canonical reference
last_updated: 2026-04-30
revision: 2 (added §4.5 honest gap-to-Supermemory + roadmap reprioritization)
audience: internal (Hawky team), prospective hires, design partners under NDA
---

# Feather DB — State of the Product

> Single canonical reference covering positioning, current technical state,
> architecture, benchmarks, competitive landscape, and roadmap. If a piece of
> documentation contradicts this file, this file wins until updated.

---

## 0. Executive summary

**What Feather is, in one paragraph.** Feather DB is a lightweight embedded
vector database and living context engine written in C++17 with Python
(pybind11) and Rust (CLI) bindings. It ships sub-millisecond ANN via HNSW,
typed-and-weighted graph edges with a reverse index, adaptive temporal decay,
multimodal pockets (independent indices per modality), BM25 + vector hybrid
retrieval via reciprocal-rank-fusion, and single-file persistence. The current
shipped version is **v0.8.0** on PyPI and Crates.io.

**What Feather is for, in one paragraph.** Feather is the open-source
technical foundation under **Hawky.ai's marketing context engine** ("Glean for
marketing teams"). The OSS core wins on benchmarks and hiring; the Cloud
build-out — vertical ontology, brand-bootstrap onboarding, ad-platform
connectors, multi-tenant infra — is proprietary inside Hawky and powers an
enterprise product for performance-marketing teams.

**The two-product strategy.**

| Surface | Audience | Optimized for | Status |
|---|---|---|---|
| **Feather core (OSS)** | AI infra engineers, researchers | Citability, clean tech, benchmark wins | v0.8.0 shipped; Phase 9.1 in flight |
| **Feather Cloud / Hawky** | Performance-marketing teams at brands | Capture moat, vertical depth, time-to-wow | Phase 9.1 inline ingestion in progress; design-partner rollout next |

**The thesis.** When LLMs commoditize the agent layer (10M-token contexts,
agent-spawned tools, native memory), the durable position is **customer-
specific data and structure that doesn't get commoditized by bigger
models** — i.e. the captured conversations, extracted facts, and learned
brand ontology. Feather Cloud holds that layer for marketing teams. Feather
core gets the technical credibility that makes Cloud trustworthy.

---

## 1. Positioning

### 1.1 Feather core (OSS) — positioning

**Tagline.** "The embedded vector + context graph database for AI agents."

**Audience.**
- AI infra engineers building memory layers, agent frameworks, RAG systems
- Researchers running long-context / memory benchmarks
- Teams that want a single-file, embedded DB without standing up Pinecone/Weaviate

**Optimization targets.**
- Citability — every benchmark we publish should be reproducible and showable
  on a CV (LongMemEval, SIFT1M, BEIR/MS MARCO, LOCOMO).
- Clean tech — keep the surface area small and the API obvious.
- Hiring magnet — the OSS posture and research papers attract the senior
  engineers Hawky needs.

**Non-goals for OSS.**
- Don't ship the marketing ontology
- Don't ship the brand-bootstrap onboarding flow
- Don't ship the ad-platform connector library
- Don't ship multi-tenant infra
- These are Hawky's moat, not Feather's

### 1.2 Feather Cloud (Hawky's marketing context engine) — positioning

**Tagline.** "The context layer for marketing teams. Your brand's living
memory across Claude, ChatGPT, and every tool the team uses."

**The Glean analogy.** Glean became a $2B+ company by being the vertical
context engine for engineering teams (Slack + GitHub + Jira + Drive + Notion).
Hawky is the equivalent for marketing teams (Meta Ads + Google Ads + TikTok +
GA4 + Klaviyo + Slack + Notion + Drive + Claude conversations + creative
assets).

**The wedge.** Performance-marketing teams already use Claude / ChatGPT for
campaign analysis, creative briefs, and reporting — but every conversation
starts with re-explaining the brand. Hawky + Feather captures the brand's
living context once and injects the relevant slice into every Claude or
ChatGPT session via MCP.

**The lock-in.** Six months of captured conversations + an extracted fact
graph + a learned per-brand ontology + an audit trail of corrections.
Replicating this from scratch on a competitor takes a quarter; on a 10M-token
context-window LLM, it never happens because the data is the moat.

### 1.3 What is NOT the moat

- HNSW speed — copyable in a week.
- Adaptive decay formula — clever but copyable.
- Two-phase ingestion design — Mem0 and Zep do extraction; we do it better,
  but it's not unique.
- Single-file format — nice DX, not a moat.

### 1.4 What IS the moat (Hawky side)

1. **Captured customer conversations** indexed and fact-extracted across 6+
   months of usage — only Hawky has this for any given brand.
2. **Marketing-vertical ontology** with attribution-chain reasoning
   (creative → ad → ad_set → campaign → channel → brand) and KPI semantics.
3. **Brand-bootstrap layer** — per-tenant overlay configured during
   onboarding (channels, KPIs, naming conventions, attribution windows).
4. **Connector breadth** — Meta Ads / Google Ads / TikTok / Klaviyo / GA4
   plus knowledge-worker tools.
5. **Trust surface** — provenance UI, contradiction-with-audit-trail, human
   approval flow on ontology extension.

---

## 2. Current technical state (v0.8.0)

### 2.1 What's shipped

**Core DB.**
- C++17 core (`include/feather.h`, `include/metadata.h`, `include/scoring.h`,
  `include/filter.h`, `include/hnswalg.h`).
- Python bindings via pybind11 (`bindings/feather.cpp`).
- Rust CLI (`feather-cli/`, crate `feather-db-cli` v0.8.0 on crates.io).
- File format v5 (v3/v4 backward-compat load).

**Retrieval primitives.**
- HNSW per-modality (M=16, ef_construction=200, max_elements=1M).
- BM25 lexical scoring + reciprocal-rank-fusion (RRF) for hybrid search.
- Adaptive temporal decay (`stickiness = 1 + log(1 + recall_count)`,
  `effective_age = age / stickiness`, `recency = 0.5^(eff_age / half_life)`).
- Filter struct: namespace, entity, attributes, types, source, time-range,
  importance, tags.
- Multimodal pockets: each named modality (text, visual, audio) gets its own
  HNSW index sharing one Metadata store keyed by ID.

**Graph layer.**
- Typed weighted edges (`Edge { target_id, rel_type, weight }`).
- Reverse edge index built in-memory on load (incoming edges per node).
- `auto_link` by similarity threshold.
- `context_chain` — vector search + n-hop BFS expansion (Living Context).
- `export_graph_json` for D3/Cytoscape.
- `feather_db.graph.visualize()` — self-contained D3 force-graph HTML.

**Phase 9.1 extraction (Weeks 1–2 shipped, Week 3 in progress).**
- `feather_db/extractors/`:
  - `base.py` — `Extractor`, `Resolver`, `Linker` protocols + `Fact`,
    `Entity`, `ExtractedTimestamp`, `OntologyEdge` dataclasses.
  - `temporal.py` — `TemporalParser` (rule-based, no LLM): ISO dates,
    numeric dates, month+year, quarters, "N days ago", "last/next
    week|month|year|quarter", yesterday/today/tomorrow.
  - `facts.py` — `FactExtractor` (LLM-backed): atomic SPO triples with
    confidence and `valid_at`.
  - `entities.py` — `EntityResolver` (LLM-backed): canonical IDs with
    fallback synthesis (`unknown::<slug>_<sha1[:8]>`).
  - `ontology.py` — scaffolding for OntologyLinker (Week 3 work).
  - `_jsonparse.py` — robust JSON extraction from LLM output (handles
    fenced blocks, prose-wrapped JSON, trailing commas).
- `feather_db/pipelines/ingest.py` — `IngestPipeline` wires extractors → DB
  with provenance:
  - `SOURCE_ID_BASE = 1B`, `FACT_ID_BASE = 2B`, `ENTITY_ID_BASE = 3B`
  - Edges: `fact -- extracted_from --> source`, `fact -- refers_to -->
    entity` (when canonical match)
  - Idempotent entity dedup by canonical_id within a pipeline run.

**Test coverage.** 117 passing / 5 skipped (was 80 at the v0.5.0 baseline).
- `tests/test_extractors_temporal.py` — 17 tests
- `tests/test_extractors_facts_entities.py` — 14 tests with MockProvider
- `tests/test_pipelines_ingest.py` — 6 end-to-end tests

**Benchmark harness.** `bench/` directory with reproducible scenarios:
- `vector_ann` (synthetic, brute-force GT)
- `vector_ann_real` (SIFT1M, siftsmall — pre-computed GT)
- `longmemeval` (oracle / S / M variants)
- `longmemeval_phase9` (Phase 9 ingestion + retrieval)
- Embedders: deterministic (offline), Azure OpenAI, OpenAI direct.
- Judges: substring (free), LLM (Gemini, Claude, GPT-4o, Azure).

### 2.2 Public benchmark results

**SIFT1M (ANN correctness).**
- recall@10 = 0.972 at ef=50
- p50 latency = 0.19 ms (Apple M1, single-threaded)
- recall@100 = 0.985

**LongMemEval (memory benchmark, ICLR 2025).**

| Variant | Configuration | Overall | Notes |
|---|---|---|---|
| Oracle | Feather + Gemini-Flash, no decay | 0.656 | 0/500 failures, 38 min, ~$0.15 |
| Oracle | Feather + Gemini-Flash, decay on | 0.670 | +1.4pp (limited noise on oracle) |
| S | Feather + Gemini-Flash, decay on | **0.657** | Beats Zep+GPT-4o-mini (0.638) |
| S | Feather + GPT-4o, decay on | **0.693** | +3.6pp from stronger answerer |

**Reference points on LongMemEval_S:**
- LongMemEval paper full-context GPT-4o ceiling: 0.640
- Zep + GPT-4o-mini: 0.638
- Zep + GPT-4o: 0.712
- Mem0 + GPT-4o: ~0.68 (published)
- Supermemory + GPT-4o (production): 0.816
- Supermemory experimental swarm (8-prompt-variant any-correct): 0.986

**Diagnostic — where the gap to Supermemory comes from:**
The 12pp gap (Feather 0.693 vs Supermemory production 0.816) is concentrated
in three reasoning axes:
- knowledge-update: −17pp
- multi-session: −11pp
- temporal-reasoning: −29pp

**Closing this gap is structural, not model-class.** Stronger answerers
don't help; the bottleneck is that Feather currently retrieves raw turns,
not extracted facts with `valid_at` timestamps. **Phase 9.1 fixes this** by
extracting atomic SPO triples at ingest time, which is exactly Supermemory's
"ontology-aware edges" advertising.

### 2.3 What's NOT shipped yet

**Phase 9.1 Week 3 (in flight, ~2 weeks):**
- **OntologyLinker** — typed-edge inference between facts.
- **ContradictionResolver (detection-only)** — flag when new facts conflict
  with stored facts; surface to user; explicitly NOT auto-resolve.
- **Feedback / correction logging** — log user corrections for future
  calibration; data flywheel for later self-alignment.
- **Phase 2 wiring** — `enable_phase2` flag in `IngestPipeline` (off in
  OSS, on in Cloud).
- **Hierarchy helper** — `feather_db.hierarchy` for schema-aware attribution
  (ad → ad_set → campaign).
- **Reasoner skeleton** — `QueryPlanner` + `PlanExecutor` stubs.

**Phase 9.2 — closes critical Supermemory gaps (see §4.5):**
- **MCP server** — `feather.ingest` + `feather.recall` + `feather.context`
  tools with OAuth + API-key auth (gap #2, #4).
- **Hawky connector wiring** — Meta Ads / GA4 / Google Ads / TikTok /
  Klaviyo into Feather namespace (gap #3).
- **Image OCR ingestion** — for marketing creatives (gap #5).
- **Claude Cowork conversation capture** — browser extension or MCP-resident.
- **Multi-tenant Cloud infra (minimal)** — one process per brand on a
  single VPS, S3-backed (gap #1 minimally).

**Phase 10+ — Hawky-proprietary (NOT in OSS):**
- **Marketing ontology** — vertical entity types, predicates, attribution.
- **Brand-bootstrap UI** — onboarding overlay configuration.
- **Connector library** — already in Hawky, route to Feather namespace.
- **Provenance + audit trail UI** — corrections workflow.

**Phase 11+ — competitive parity (deferred):**
- **Reranker** — context-aware learned reranker (gap #6).
- **Auto-generated brand profile** — synthesized from memory stream (gap #7).
- **Multi-project per user** — beyond per-tenant namespaces (gap #8).

---

## 3. Architecture

### 3.1 Repository layout (current)

```
feather/
├── include/                 # C++ headers (core logic)
│   ├── feather.h            # MAIN DB class
│   ├── metadata.h           # Metadata + Edge + ContextType
│   ├── scoring.h            # Adaptive decay
│   ├── filter.h             # SearchFilter
│   ├── hnswalg.h            # HNSW (hnswlib fork)
│   ├── space_l2.h           # L2 distance (SIMD)
│   ├── space_ip.h           # Inner product
│   └── bruteforce.h         # Fallback brute force
├── src/
│   ├── feather_core.cpp     # extern "C" wrappers (Rust FFI)
│   ├── metadata.cpp         # Serialize/deserialize
│   ├── filter.cpp           # Filter logic
│   └── scoring.cpp          # Scorer thin wrapper
├── bindings/
│   └── feather.cpp          # pybind11 bridge
├── feather_db/              # Python package
│   ├── __init__.py          # Public exports
│   ├── filter.py            # FilterBuilder fluent API
│   ├── domain_profiles.py   # MarketingProfile (legacy)
│   ├── graph.py             # visualize(), export_graph(), RelType
│   ├── extractors/          # Phase 9.1
│   │   ├── base.py
│   │   ├── temporal.py
│   │   ├── facts.py
│   │   ├── entities.py
│   │   ├── ontology.py
│   │   └── _jsonparse.py
│   └── pipelines/
│       └── ingest.py        # IngestPipeline
├── feather-cli/             # Rust CLI crate
│   ├── src/main.rs
│   ├── src/lib.rs
│   ├── build.rs
│   └── Cargo.toml
├── feather-api/             # FastAPI cloud wrapper (legacy v0.4.b)
├── bench/                   # Benchmark harness
│   ├── __main__.py
│   ├── runner.py
│   ├── report.py
│   ├── datasets/
│   ├── scenarios/
│   ├── embedders.py
│   ├── embedders_openai.py
│   ├── judges.py
│   ├── judges_llm.py
│   └── providers_azure.py
├── tests/                   # 117 tests
├── examples/
├── docs/                    # this directory
├── setup.py
└── pyproject.toml
```

### 3.2 Core DB — data model

```
Metadata (keyed by uint64_t id; ONE per logical entity, shared across modalities)
├── timestamp:        int64_t
├── importance:       float [0.0–1.0]
├── type:             FACT | PREFERENCE | EVENT | CONVERSATION
├── source:           string (origin identifier)
├── content:          string (human-readable)
├── tags_json:        string (JSON array)
├── edges:            vector<Edge>  // outgoing typed edges
├── recall_count:     uint32_t      // adaptive decay signal
├── last_recalled_at: uint64_t
├── namespace_id:     string        // partition key (per-tenant in Cloud)
├── entity_id:        string        // subject key
└── attributes:       map<string,string>  // domain KV

Edge { target_id, rel_type, weight }
IncomingEdge { source_id, rel_type, weight }
```

### 3.3 Storage layout (.feather binary v5)

```
[magic 4B = 0x46454154 "FEAT"] [version 4B = 5]
─── Metadata Section ─────────────────
[meta_count 4B]
  per record: id, timestamp, importance, type, source, content,
              tags_json, edges, recall_count, last_recalled_at,
              namespace_id, entity_id, attributes
─── Modality Indices Section ─────────
[modal_count 4B]
  per modality: name, dim, element_count, [ id + float32[dim] ]*
```

Backward compat: v3 and v4 files load via `if (is.read(...))` guards in
`metadata.cpp`. Missing fields default to empty.

### 3.4 Phase 9.1 ingestion pipeline

```
                    ┌─────────────────────────────────────────────┐
                    │              IngestRecord                    │
                    │   { content, source_id, timestamp, meta }   │
                    └─────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │        Stage 1: Source persistence           │
                    │   embed(content) → DB.add(SOURCE_ID range)   │
                    └─────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │         Stage 2: FactExtractor (LLM)         │
                    │      content → list<Fact>                    │
                    │      Fact { subj, pred, obj, conf, valid_at }│
                    └─────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │       Stage 3: EntityResolver (LLM)          │
                    │   union(subj, obj) → list<Entity>            │
                    │   Entity { surface, canonical_id, kind, … }  │
                    └─────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │      Stage 4: TemporalParser (rule-based)    │
                    │   content → list<ExtractedTimestamp>         │
                    └─────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │  Stage 5: Persist facts + entities + edges   │
                    │   facts → DB.add (FACT_ID range)             │
                    │   entities → DB.add (ENTITY_ID range, dedup) │
                    │   fact -- extracted_from --> source          │
                    │   fact -- refers_to     --> entity (each end)│
                    └─────────────────────────────────────────────┘

                                   ▼
                    ┌─────────────────────────────────────────────┐
                    │  [Week 3] OntologyLinker + Contradiction     │
                    │           detection (planned)                │
                    └─────────────────────────────────────────────┘
```

ID partitioning prevents collision: source IDs in [1B, 2B), facts in [2B,
3B), entities in [3B, 4B). Each tier has 1B addressable rows per
namespace.

### 3.5 Cloud architecture (planned)

**This section describes the target state for the Hawky multi-tenant
deployment, not what's currently shipped.**

```
                          ┌─────────────────────────┐
                          │   Marketer in Claude     │
                          │   Desktop / Cursor /     │
                          │   ChatGPT / Hawky UI     │
                          └─────────────────────────┘
                                      │
                                      │ MCP, REST, SDK
                                      ▼
                          ┌─────────────────────────┐
                          │   Hawky API Gateway      │
                          │   (auth, routing, rate)  │
                          └─────────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
                ┌──────────────────┐   ┌──────────────────┐
                │  Capture surface │   │  Recall surface  │
                │  - feather.ingest│   │  - feather.recall│
                │  - connectors    │   │  - context_chain │
                │  - Cowork capture│   │  - hybrid search │
                └──────────────────┘   └──────────────────┘
                          │                       │
                          └───────────┬───────────┘
                                      ▼
                          ┌─────────────────────────┐
                          │  Per-tenant Feather DB   │
                          │  (one .feather per brand)│
                          │  - namespace = brand_id  │
                          │  - vertical ontology     │
                          │  - extraction at ingest  │
                          └─────────────────────────┘
                                      │
                                      ▼
                          ┌─────────────────────────┐
                          │  S3-backed persistence   │
                          │  (warm cache on VPS,     │
                          │   cold tier in S3)       │
                          └─────────────────────────┘
```

**Deployment shape (initial):**
- One process per brand for the first 2–5 design partners (single-tenant
  isolation, simple ops).
- Move to multi-tenant per-process at ~20 brands, each brand a `namespace_id`
  in a shared Feather instance.
- S3-backed cold tier; warm DB resident in process memory.

**Not yet built:**
- Multi-tenant routing layer
- Per-tenant resource isolation
- Cloud authentication & RBAC
- Audit-trail UI
- Brand-bootstrap onboarding

### 3.6 LLM integration patterns

Two directions of LLM communication: pull (LLM asks Feather) and push
(Feather tells LLM).

**Pull surfaces (5).**
1. **MCP** (primary) — `feather.ingest`, `feather.recall`, `feather.context`
   tools. Works with Claude Desktop, Cursor, Cline, any MCP client.
2. **Anthropic tool-use** — JSON schemas for direct API integration.
3. **OpenAI tool-use** — strict-mode JSON schemas.
4. **REST** — for custom integrations.
5. **Embedded SDK** — for in-process Hawky agents.

**Push patterns (3).**
- **B1 — Ambient MCP resource (default):** Feather exposes
  `feather://context/morning-brief` as a resource that MCP clients
  auto-refresh on session start. The marketer opens Claude in the morning
  and the brand context is already in the conversation without anyone
  asking for it.
- **B2 — Tool-result enrichment:** every tool response carries a `_context`
  sidecar with relevant memory facts.
- **B3 — Proactive push:** depends on MCP streamable-HTTP server-events
  landing in 2026 spec; deferred until then.

---

## 4. Competitive landscape

### 4.1 Supermemory (the closest comparable)

| Dimension | Supermemory | Feather + Hawky |
|---|---|---|
| **Storage** | Cloudflare Vectorize + R2 + Durable Objects | Single-file .feather, embeddable + Cloud |
| **Ingest** | Multi-format (PDF/image/video/code), atomic-fact extraction, contradiction handling | Same primitives; vertical-ontology layer in Cloud |
| **Retrieval (production)** | Hybrid + reranker, sub-300ms | Hybrid (RRF) + decay; comparable latency |
| **Retrieval (experimental)** | Multi-agent swarm, 98.6% any-correct on LongMemEval_S | Single retrieval pipeline; 0.693 with GPT-4o |
| **MCP** | `memory` + `recall` tools, OAuth + API keys | Planned: `feather.ingest` + `feather.recall` |
| **Ontology** | Generic / horizontal | Marketing-vertical (Hawky-side, proprietary) |
| **Connectors** | Slack, Notion, Drive, Gmail, S3, OneDrive | Same + Meta Ads, Google Ads, TikTok, Klaviyo, GA4 |
| **Hierarchy / attribution** | Not modeled | Native (creative → ad → ad_set → campaign) |
| **Embedded / on-prem** | Cloudflare-native only | Single-file embed; on-prem viable |
| **Audit trail / corrections** | Auto-resolve, no audit | Provenance UI + human-in-loop (planned) |
| **Pricing** | Free tier + paid API + enterprise | Hawky pricing (TBD, enterprise-tier) |
| **Funding** | $2.6M seed (Cloudflare + Google execs) | Hawky existing customer base |
| **Public benchmark headline** | 0.816 production (LongMemEval_S, GPT-4o); 0.986 experimental | 0.693 production (LongMemEval_S, GPT-4o); Phase 9.1 closes most of the gap |

**Where Supermemory is strong:**
- Cloudflare-native serverless infra (perf + cost at their tier)
- MCP-first universal positioning
- Connector breadth in knowledge-worker tier
- Public benchmark dominance + research blog cadence

**Where Supermemory is thin (our exploit surface):**
- No vertical depth — generic ontology, no marketing entities
- No ad-platform connectors (Meta / Google / TikTok / Klaviyo / GA4)
- No hierarchy / attribution model
- Cloudflare lock-in — blocker for AWS/GCP/Azure-only enterprises
- Auto-resolve contradictions without audit trail (trust gap for marketing
  teams)
- No on-prem / VPC option
- Their 99% number is a swarm-of-judges any-correct trick; production is 85%

**How we win:**
1. **Vertical depth** — marketing ontology built in, attribution-aware
2. **Connector breadth in our domain** — ad platforms, not just productivity
3. **Embedded option** — single-file `.feather` deployable in customer VPC
4. **Provenance + audit trail** — required for enterprise marketing trust
5. **Hawky distribution** — already in market with brand customers
6. **Author the benchmark we win on** — marketing-vertical eval, not pure
   LongMemEval_S (where they own the leaderboard)

### 4.2 Mem0

- Lighter, simpler atomic-fact extraction.
- Strong at personal memory, weak at temporal reasoning.
- We beat Mem0 on multi-session and temporal axes once Phase 9.1 lands.
- They don't have hybrid retrieval at our quality; no graph; no marketing
  vertical.

### 4.3 Zep / Graphiti

- Closest technically: they ship a knowledge graph + temporal layer.
- Heavier deployment; PostgreSQL + Neo4j + Python runtime.
- Comparable on LongMemEval_S to Feather + GPT-4o on simple axes; we beat
  them on cost and embedding option.
- They don't have a marketing vertical; they're horizontal infra.

### 4.4 OpenAI Memory / Claude memory tool / Gemini context caching

- These are **free with the model** and work for personal memory.
- They do NOT solve: multi-tenant brand context, vertical ontology, audit
  trail, provenance, ad-platform integration, cross-LLM portability.
- The bet is that vertical context engines coexist with horizontal LLM
  memory the way Glean coexists with Slack search.

### 4.5 What Supermemory has that we don't (honest gap)

This section is uncomfortable on purpose. We're vertical-strong and
distribution-strong; they're infra-shipped and mindshare-strong. Knowing
exactly which gaps to close (vs. ignore) drives the roadmap.

**Critical gaps — MUST close to be a credible Cloud product:**

| # | Capability | Their state | Our state | Phase |
|---|---|---|---|---|
| 1 | Multi-tenant Cloud, live with paying customers | live, billions of tokens/mo | local lib + planned Cloud | 10 |
| 2 | MCP server live in production | `mcp.supermemory.ai/mcp` with OAuth | planned | 9.2 |
| 3 | Connector library wired to memory layer | 6+ connectors live (Slack/Notion/Drive/Gmail/OneDrive/S3) | Hawky has marketing connectors but not wired into Feather | 9.2 |
| 4 | OAuth at MCP/API surface | shipped | none | 9.2 |
| 5 | Image OCR at ingest (marketing creatives, screenshot briefs) | shipped | none | 10 |

**Important — close in Q3/Q4 to stay competitive:**

| # | Capability | Their state | Our state | Phase |
|---|---|---|---|---|
| 6 | Context-aware reranker on retrieval | learned reranker | RRF only | 11 |
| 7 | Auto-generated user/brand profile from memory stream | shipped | none (Hawky-side equivalent later) | 11 |
| 8 | Multi-project per user / account | `x-sm-project` header | `namespace_id` per-tenant only | 11 |
| 9 | Public research blog cadence | 2–3 posts/month | one (LongMemEval) | start now (monthly) |
| 10 | Funded credibility signal ($) | $2.6M seed, Cloudflare CTO + Google execs | Hawky revenue (different leverage) | out of our control |
| 11 | MCP brand recognition in dev community | "the" MCP memory layer | invisible | earned via OSS publishing + research cadence |

**Out of scope — they have it, we don't want it (or don't need it):**

| Capability | Their state | Why we skip |
|---|---|---|
| Cloudflare-native infra | locked | blocker for enterprise VPC; embedded option beats this |
| Auto-resolve contradictions | shipped | we explicitly chose detect-only with audit trail (marketing teams need provenance) |
| Generic horizontal positioning | yes | we are vertical-first by design |
| Auto-forgetting / TTL deletion | shipped | decay scoring is more recoverable than deletion |
| Audio / video transcription | shipped | not on the marketing critical path |
| AST-aware code chunking | shipped | out of scope (engineering memory, not marketing) |

**Things we have they don't (worth stating explicitly so the team
remembers):**

- Single-file embedded option — their architecture cannot replicate this
  without leaving Cloudflare.
- Marketing-vertical ontology with hierarchy/attribution semantics.
- Audit-trail / provenance UI (planned).
- On-prem / VPC deployment path (blocker for them).
- Hawky's existing brand customer base (they sell to devs; we sell to
  brands).

**Net read.** Supermemory is ~6 months ahead on *infrastructure surface*
(Cloud, MCP, connectors, multi-format extraction). We are ahead on
*vertical strategy* and *distribution* (Hawky customers + marketing
ontology). The race is whether we close the infra gap before they build
vertical depth. They have shown zero signal of going vertical, so the
window is open — but it's not infinite. **Phase 9.2 (capture surface) and
Phase 10 (multi-tenant Cloud) are now the highest-leverage work on the
roadmap.**

---

## 5. Roadmap

### 5.1 Phase 9.1 Week 3 (current sprint, ~2 weeks)

1. **OntologyLinker (detect-only)** — typed-edge inference between facts.
2. **ContradictionResolver (detect-only)** — flag conflicts, surface to
   user, do NOT auto-resolve.
3. **Feedback / correction logging** — schema for user corrections; data
   feeds future self-alignment but no auto-action yet.
4. **`enable_phase2` flag in `IngestPipeline`** — off by default in OSS, on
   in Cloud.
5. **`feather_db.hierarchy` helper** — schema-aware attribution chain.
6. **Reasoner skeleton** — `QueryPlanner` stub + `PlanExecutor` scaffold.
7. **Live bench** — 10 questions on LongMemEval_oracle through Phase 9
   pipeline to validate non-zero facts extracted (~$0.50, ~5 min).

### 5.2 Phase 9.2 — capture surface + parity-with-Supermemory infra (~4 weeks after Week 3)

This phase is reprioritized in revision 2. The order below reflects the
gap analysis in §4.5 — close the infra gap that blocks credibility, then
build the wedge features.

1. **MCP server with OAuth** — `feather.ingest` + `feather.recall` +
   `feather.context` tools. OAuth (default) and API-key (`fdb_…`)
   auth. Installs in Claude Desktop / Cursor / Cline / VS Code in one
   click. **Closes gap #2 + #4.**
2. **Wire Hawky's existing connectors to Feather namespace** — Meta Ads,
   Google Ads, TikTok, Klaviyo, GA4 already exist in Hawky; route into
   per-brand Feather namespaces. **Closes gap #3.**
3. **Image OCR at ingest** — Tesseract or hosted OCR for ad creatives,
   screenshot briefs, PDF reports with embedded images. **Closes gap #5.**
4. **Claude Cowork conversation capture** — browser extension (or
   MCP-resident) auto-ingests Claude conversations into the brand's
   Feather namespace. **Wedge feature — beyond what Supermemory does
   for marketing.**
5. **Persistent context injection** — when a Claude session starts via
   Hawky's MCP, auto-inject relevant brand memory. **Wedge.**
6. **Multi-tenant Feather Cloud (minimal)** — one process per brand on a
   single VPS; S3-backed persistence; basic auth. **Closes gap #1
   minimally.**

### 5.3 Phase 10 — design-partner deployment (~Q3 2026)

1. **Pick 2 Hawky customers** as design partners (different verticals if
   possible).
2. **Hardcode marketing ontology** for design partners (no Brand Bootstrap
   UI yet).
3. **Ship inline Phase 9 extraction** in Hawky behind a flag.
4. **Instrument** for "saved 20 minutes" moments — concrete ROI capture.
5. **Iterate ontology** based on real brand data; collect correction
   signals for future self-alignment.

### 5.4 Phase 11 — hardening + scale-out (~Q4 2026)

1. WAL / crash recovery (currently #5 on backlog).
2. Format backward-compat fixtures (#6).
3. Thread-safety stress test (#7).
4. BEIR / MS MARCO BM25 benchmark (#9).
5. LOCOMO benchmark (#11).
6. Brand-bootstrap UI.
7. Provenance UI in Hawky.
8. Move to multi-tenant per-process at ~20 brands.

### 5.5 Continuous — public research / OSS cadence (start now)

Closes gap #9 and #11 from §4.5. Without monthly publishing, Feather core
fails its hiring + credibility mandate.

**Cadence target: one public artifact per month.** Each is 2–4 days of
work, mostly leveraging existing benchmark + Phase 9 work.

- **May 2026** — Phase 9 contradiction-detection write-up (research blog +
  arXiv workshop submission). Title: "Contradiction-aware retrieval in
  living memory systems."
- **June 2026** — BEIR / MS MARCO BM25 + hybrid benchmark report (closes
  pending task #9).
- **July 2026** — LOCOMO benchmark run (closes pending task #11).
- **August 2026** — Embedded-vs-Cloud architecture comparison
  (single-file `.feather` running in customer VPC vs. serverless memory
  layer). Differentiates against Supermemory's Cloudflare lock-in.
- **September 2026** — Marketing-vertical eval we author. Beats anyone
  else on the leaderboard we control. Headline metric for Hawky GTM.

### 5.6 Dated month-by-month timeline (2026)

Today is 2026-04-30. All dates below are end-of-week targets.

**May 2026 — Close Phase 9.1, design MME**
- W1 (May 4–10): OntologyLinker + ContradictionResolver detect-only
- W2 (May 11–17): Feedback logging + `enable_phase2` flag + reasoner stub
- W3 (May 18–24): Phase 9 live bench → full LongMemEval_S re-run (Phase 9 on)
- W4 (May 25–31): MME design doc + contradiction-aware retrieval blog

**June 2026 — MCP + connectors live**
- W1–2 (Jun 1–14): Production MCP server with OAuth at `mcp.feather-db.com`
- W3 (Jun 15–21): Wire Hawky connectors (Meta Ads + GA4 first)
- W4 (Jun 22–28): BEIR/MS MARCO benchmark blog + start MME dataset synthesis

**July 2026 — Cloud minimal + first design partner**
- W1–2 (Jul 1–14): Multi-tenant Feather Cloud (minimal — VPS, one process/brand)
- W3 (Jul 15–21): Image OCR ingestion (closes gap #5)
- W4 (Jul 22–31): Onboard design partner #1 + LOCOMO blog

**August 2026 — Capture wedge + MME public**
- W1 (Aug 3–9): Claude Cowork conversation capture
- W2 (Aug 10–16): Persistent context injection on session start
- W3 (Aug 17–23): MME dataset complete + baseline runs (Feather, Mem0, Zep, Supermemory)
- W4 (Aug 24–31): Publish MME public — leaderboard, blog, GTM headline

**September 2026 — Scale to 5 design partners**
- W1: Onboard design partner #2
- W2: Auto-brand-profile generator (closes gap #7)
- W3: Reranker prototype (closes gap #6)
- W4: Onboard design partners #3–5; embedded-vs-Cloud blog

**October 2026 — Hardening + Phase 11**
- W1: WAL / crash recovery
- W2: Multi-project per user (closes gap #8)
- W3: Format backward-compat fixtures + thread-safety stress
- W4: Provenance UI in Hawky

**November 2026 — Self-alignment evaluation**
- W1–2: 6 months of correction data analyzed; ontology-extension review
- W3: Brand-bootstrap UI (Hawky-side onboarding)
- W4: November blog: "Six months of correction signals"

**December 2026 — Year-end consolidation**
- W1–2: Multi-tenant per-process at scale
- W3: Hawky.ai context engine launch — public GA
- W4: Year-end retrospective + 2027 plan

**End-of-2026 targets:**
- 20 brand customers on Feather Cloud
- MME established as the marketing context-engine benchmark
- LongMemEval_S parity with Supermemory production
- Phase 9 fully shipped (extraction + reasoning + reranker)
- 8 monthly research artifacts published

**Critical path:** MCP + Hawky connectors in June. Every downstream
milestone (design partners, MME data flow, year-end GA) blocks on this.

### 5.7 Explicitly deferred (do NOT build yet)

- **Auto-evolving ontology** — too risky without 6+ months of correction
  data; revisit at 20+ brands.
- **Public Feather Cloud SKU** — Cloud = Hawky's multi-tenant infra, not a
  separate product.
- **Self-alignment LLM-judges-LLM loops** — deferred until calibration data
  exists.
- **Bundled inference** — customers can BYOL via standard providers; we
  don't host LLMs.
- **Multi-modal (image / video) extractors** — wait until first design
  partner needs it.

---

## 6. Marketing Memory Eval (MME) — design and rollout

The benchmark we author and own. Without it we're permanently chasing
Supermemory's leaderboard on a generic personal-memory dataset.

### 6.1 Why MME

LongMemEval_S is the public personal-memory benchmark. Supermemory leads
it (production 0.816, experimental swarm 0.986 with the any-correct trick).
Closing that gap is necessary for credibility but not sufficient for
positioning — the benchmark doesn't measure what marketing teams actually
need.

MME measures the primitives that drive marketing-team value: hierarchical
attribution, multi-channel synthesis, KPI temporal validity, brand voice,
audit trail, image grounding. Generic memory layers like Supermemory have
none of these as native primitives. We win by definition.

### 6.2 Structure

- **~300 questions** across 5 fictional brands
- 5 verticals: e-commerce, B2B SaaS, DTC consumer, agency-of-record, retail
- 6 months of synthesized data per brand:
  - Meta Ads + Google Ads + GA4 performance dumps
  - Slack threads (creative team + perf marketing team)
  - Claude conversations (mock briefs, analyses, approvals)
  - Creative briefs + approval logs
  - Image creatives (with embedded text → OCR test)

### 6.3 Ten evaluation axes

| # | Axis | Why competitors lose |
|---|---|---|
| 1 | Attribution-chain reasoning | needs hierarchy creative→ad→adset→campaign→channel |
| 2 | Multi-channel synthesis | needs canonical cross-channel entities + KPI normalization |
| 3 | Temporal performance reasoning | needs `valid_at` + attribution-window awareness |
| 4 | Brand voice consistency | needs brand-profile memory |
| 5 | Audit / approval recall | needs provenance + audit trail |
| 6 | Cross-team handoff / decision tracking | needs cross-conversation memory + decision extraction |
| 7 | KPI calculation with semantics | needs KPI formula awareness (CTR/ROAS/CAC/LTV) |
| 8 | Image-grounded recall | needs OCR + image-to-perf attribution |
| 9 | Compliance / brand safety | needs rule memory + violation history |
| 10 | Anomaly explanation | needs temporal pattern memory + cause attribution |

### 6.4 Scoring

- LLM-judge with marketing-domain prompt (Gemini-Flash for cost; GPT-4o
  for high-stakes runs)
- Per-axis recall + overall accuracy
- Cost tracking per system run

### 6.5 Public artifacts

- Dataset on HuggingFace (`Hawky-ai/marketing-memory-eval`)
- Leaderboard at `marketing-memory-eval.com` (or under `feather-db.com/mme`)
- Results blog post + GTM headline
- Paper submission: NeurIPS 2026 Datasets & Benchmarks track (or workshop)

### 6.6 Effort + timeline

- **May W4:** design doc finalized
- **June W4 – Aug W3:** dataset synthesis (~3 weeks active)
- **Aug W3:** competitor baseline runs (Feather, Mem0, Zep, Supermemory)
- **Aug W4:** public launch
- Total effort: ~4 person-weeks spread over Q2/Q3

---

## 7. The bet — in one paragraph

Feather core is the open-source vector + context graph database that wins
on benchmarks and earns the right to recruit senior infra engineers and
publish research. Hawky is "Glean for marketing teams" — the vertical
context engine that captures every brand's living memory across Claude,
ChatGPT, ad platforms, and creative tools, and serves it back via MCP.
The OSS gives us credibility; the Cloud gives us revenue; the captured
data and learned per-brand ontology become unreplaceable as soon as
customers have six months of usage. When LLMs commoditize the agent layer
in 12–18 months, the customers who depend on Hawky's brand memory don't
move, because the memory is the moat.

---

## 8. Open decisions (lock these next)

1. **Self-alignment scope this quarter:** confirm
   contradiction-detection-only + correction-logging (build), defer
   auto-evolving ontology (don't build).
2. **Capture surface order:** MCP-only first vs MCP + browser extension
   simultaneously.
3. **OSS scope:** confirm DB + base extractors + MCP server + LongMemEval
   harness ship as OSS; marketing ontology + Brand Bootstrap + connector
   library stay proprietary inside Hawky.
4. **Cloud storage backend:** single-VPS-per-brand for first 5 design
   partners, OR invest in proper multi-tenant from day one.
5. **Design partners:** which 2 Hawky brands sign up first, by when.

---

## 9. Document maintenance

- This is the canonical reference. Update it whenever a phase closes.
- Versioned by Feather DB version: each minor bump (v0.8 → v0.9 → v1.0)
  triggers a section refresh.
- If a contributor's local mental model diverges from this doc, this doc
  wins until updated.
- Source of truth for: positioning, roadmap, competitive framing, current
  shipped state, current in-flight work.
- NOT the source of truth for: API reference (see `feather_db/__init__.py`
  + pybind docstrings), benchmark numbers (see `docs/benchmarks/`), phase
  plans (see `docs/architecture/phase9-plan.md` for detail).
