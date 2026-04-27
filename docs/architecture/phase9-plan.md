---
id: phase9-plan
title: "Phase 9 — Agentic Context Engine: Architecture & GTM"
status: draft
audience: internal
last_updated: 2026-04-27
---

# Phase 9 — Agentic Context Engine

> **What Phase 9 turns Feather DB into:** a living context layer that ingests raw data of any kind, builds *structured intelligence* from it (facts, entities, time-anchored claims, contradictions, ontology edges), stores it in the existing Feather core, and **guides the LLM** at query time through a planned multi-step retrieval — instead of returning top-k vectors and hoping.

> **What it doesn't turn Feather into:** an LLM-as-a-service, a generic agent framework, or a replacement for the answerer model. We sit between data and the LLM, not on top of it.

---

## 1. The recommendation up front

**Open source AND cloud — a clean hybrid cut, not pick-one.**

| Layer | Where it ships | Why |
|---|---|---|
| **Engine** (existing C++ core: HNSW, BM25, decay, graph, WAL) | OSS (`feather-db`) | Already there. Stays MIT. The "free engine" promise stays intact. |
| **Extractors module** (`feather_db.extractors`) | OSS | Generic primitives — fact extraction, contradiction resolver, ontology linker, query reasoner. Pluggable on any LLMProvider. The "we publish what works" credibility play. |
| **Vertical agents** (marketing, devtools, support, finance, HR) | **Cloud only** | Each vertical = curated ontology + brand-context bootstrap + tuned reasoning workflows. This is where the *applied* IP lives. Not replicable from OSS alone. |
| **Multi-tenant management** (auth, isolation, billing, observability, compliance) | Cloud only | Standard SaaS cut. Nobody expects this in OSS. |
| **Brand-context loader + auto-tuning** | Cloud only | Mediated by the vertical agents above; depends on telemetry the cloud collects. |

**Why this cut, not "all OSS" or "Cloud only"**:

- **All OSS** kills monetization. The verticals are where customers will pay; if we ship them free, we have nothing to sell.
- **Cloud only** kills the credibility flywheel. Our entire OSS positioning is "the full engine is free." If Phase 9 ships closed, we look like we did a bait-and-switch — the OSS becomes a teaser. We lose the indie/research/edge audience that today's 0.693 buys us.
- **Hybrid** keeps both: OSS users get a meaningfully better engine (extractors lift LongMemEval ~10–15pp); Cloud users get the *applied* layer (verticals + ops) they would have to build themselves.

Mental model: **OSS Feather is to Phase 9 what PostgreSQL is to Heroku** — the engine is free, the productized + operated + opinionated stack on top is paid.

---

## 2. Pipeline architecture

The shape the user described — *"data pipeline → inference and intelligence context conversion → storage → guided retrieval"* — is exactly five stages:

```
                        ┌──────────────────────────────┐
                        │   1. INGEST + NORMALIZE      │
                        │   adapters: CSV / JSON / API │
                        │   Slack / email / docs / web │
                        │   chunking (token-aware)     │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │   2. INFERENCE LAYER  (LLM)  │
                        │   FactExtractor  → triples   │
                        │   EntityResolver → canonical │
                        │   TemporalParser → ISO dates │
                        │   IntentClassifier (optional)│
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │   3. CONTEXT CONVERSION      │
                        │   OntologyLinker (typed edges)│
                        │   ContradictionResolver       │
                        │   Confidence + provenance     │
                        │   Source-back-edge ('extracted_from')│
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │   4. STORAGE  (Feather core, │
                        │      already shipped)        │
                        │   - vector + BM25 + edges    │
                        │   - decay, recall_count      │
                        │   - WAL, atomic save         │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │   5. QUERY-TIME REASONER  ★  │
                        │   QueryPlanner: which axes?  │
                        │   plan = [vector(k), bm25(k),│
                        │           graph_hops(2),     │
                        │           filter(brand=X)]   │
                        │   Executor runs each step,   │
                        │   Synthesizer ranks + returns│
                        │   with provenance.            │
                        └──────────────────────────────┘
```

**The ★ (Step 5)** is the inversion the user is asking for. Today: `db.hybrid_search(vec, query, k)` is a single primitive. Tomorrow: `db.reason(query, ctx)` returns a *plan*, executes it, and hands the LLM a curated, traced answer. The LLM no longer drives the search — Feather does.

---

## 3. OSS module — `feather_db.extractors`

**Goal:** every piece a serious memory pipeline needs, except the closed verticals/ontologies. Generic enough that a Mem0/Zep-style pipeline can be assembled in 50 lines.

### 3.1 Module layout

```
feather_db/
├── extractors/
│   ├── __init__.py
│   ├── base.py              # Protocols + BaseExtractor abc
│   ├── facts.py             # FactExtractor — text → (subj, pred, obj, ts)
│   ├── entities.py          # EntityResolver — surface forms → canonical IDs
│   ├── temporal.py          # TemporalParser — natural-language dates → ISO
│   └── ontology.py          # OntologyLinker — fact pairs → typed edges
├── consolidation/
│   ├── __init__.py
│   ├── contradictions.py    # ContradictionResolver
│   └── merge.py             # MergeJob — dedupe, promote, prune
├── reason/
│   ├── __init__.py
│   ├── planner.py           # QueryPlanner — query → search plan
│   ├── executor.py          # PlanExecutor — runs vector+bm25+graph
│   └── synthesizer.py       # Synthesizer — collate + rank + provenance
└── pipelines/
    ├── __init__.py
    └── ingest.py            # IngestPipeline — wires 1→2→3→4 end-to-end
```

### 3.2 Core APIs (Python)

```python
from feather_db import DB
from feather_db.extractors import FactExtractor, EntityResolver, OntologyLinker
from feather_db.consolidation import ContradictionResolver
from feather_db.reason import QueryPlanner, PlanExecutor, Synthesizer
from feather_db.pipelines import IngestPipeline
from feather_db.providers import ClaudeProvider

db = DB.open("memory.feather", dim=1536)
llm = ClaudeProvider(model="claude-haiku-4-5")

# ───────── Ingest pipeline (write side) ─────────
pipeline = IngestPipeline(
    db=db,
    embedder=OpenAIEmbedder(),
    extractors=[
        FactExtractor(provider=llm),
        EntityResolver(provider=llm),
        TemporalParser(),                              # rule-based, no LLM
    ],
    linkers=[
        OntologyLinker(provider=llm,
                       allowed_relations=["caused_by", "supersedes",
                                          "refers_to", "contradicts"]),
    ],
    consolidator=ContradictionResolver(provider=llm,
                                       on_conflict="supersede"),
)

pipeline.ingest(
    source="slack_export.zip",
    namespace="acme_corp",
    metadata={"channel": "#product"},
)

# ───────── Reasoner (read side) ─────────
planner   = QueryPlanner(provider=llm, db=db)
executor  = PlanExecutor(db=db)
synth     = Synthesizer(provider=llm)

result = synth.answer(
    question="Who pushed the auth-rewrite RFC and what did they
              decide about session storage?",
    plan=planner.plan_for(question, namespace="acme_corp"),
    executor=executor,
)
# result.answer        — synthesized string
# result.plan          — the steps Feather chose
# result.evidence      — list of (memory_id, score, why_chosen)
# result.contradictions — anything the synthesizer flagged
```

### 3.3 Critical design constraints

**Pluggable, not prescriptive.** Every step accepts an `LLMProvider`. Users on Claude/OpenAI/Gemini/Ollama all work without code change. No vendor lock-in.

**Read path stays sub-millisecond.** All LLM calls are write-path or query-time *planning*, never inside `db.search()`. The existing `hybrid_search` stays unchanged. The `Reasoner` is *above* it.

**Idempotent, resumable.** Every extraction writes a content-addressed log (`extraction_id` = sha256(source + provider + version)). Re-running on the same source is a no-op. Cheap in tokens, safe in failures.

**Provenance is non-negotiable.** Every extracted fact carries an edge `extracted_from: <source_memory_id>`. Every contradiction carries `supersedes: <old_id>` with a `_resolved_at` timestamp. Audit trails are first-class — same philosophy as `bench/results/`.

**Confidence scoring.** Each extracted triple carries a `confidence ∈ [0,1]`. Low-confidence triples are stored but down-weighted in retrieval ranking. No hallucinated "fact" gets equal status to a verbatim user statement.

**Cost-bounded.** `IngestPipeline` accepts a `max_tokens_per_source` budget. If a Slack export would burn $50 in extraction tokens, it stops, surfaces a quote, and asks for confirmation.

### 3.4 Failure modes we must handle

- **LLM hallucinated a fact.** Mitigation: confidence scoring + cross-validation pass on top-N facts before persisting. If the source text doesn't substantively support the triple, drop it.
- **Contradiction resolver itself contradicts.** Mitigation: human-review queue for high-stakes contradictions (e.g., changes to known canonical entities). Cloud has a UI; OSS exposes a hook.
- **Token budget exhausted mid-source.** Mitigation: checkpoint every N triples; resume on next run. Already shipped pattern (WAL).
- **Drift from the LLM provider.** Mitigation: prompts are versioned in code; every extraction logs `prompt_version`. When provider behavior drifts, we re-extract that version's outputs.

---

## 4. Cloud — vertical agents + brand context

This is what we sell. The cut is: *the things you can't replicate from OSS in a weekend.*

### 4.1 What a vertical agent is

A **vertical agent** is a triple of (curated ontology, brand-context bootstrap, tuned reasoning workflows) packaged for a specific business domain.

```
Marketing Agent
├── Ontology
│   - Entities: Brand, Campaign, Channel, Segment, Creative, Metric
│   - Relations: targets, runs_on, performs, supersedes_creative
│   - Time-aware: campaign_active_at, metric_observed_at
├── Brand-Context Bootstrap
│   - Connectors: GA4 / Meta Ads / Linear / Notion / Google Drive
│   - First-pass extraction: brand voice, audience, performance baselines
│   - Output: pre-warmed .feather file specific to this brand
└── Reasoning Workflows
    - "Why is FD CTR dropping?" → 5-step plan: trend → recent_changes →
      attribute → audit_creatives → recommend
    - Pre-tuned decay, k, hop_depth per workflow
```

Same pattern for: **DevTools, Customer Support, Finance, HR, Sales/CRM**. Each is a pluggable module on the cloud control plane.

### 4.2 Why vertical agents are valuable to customers

Today, every team building "AI memory" reinvents:
1. The connectors (Slack, GA4, etc.) — at least 2 weeks per source.
2. The extraction prompts — multiple iterations to stop hallucinations.
3. The ontology — bespoke per business, but ~70% of marketing ontologies are the same.
4. The reasoning workflows — "what's our top-performing campaign" gets reasked the same five ways across every customer.

Vertical agents collapse that work to a connector wizard + a one-day "load my data" run. Customer ships in a week instead of 6 months.

This is a real moat: ontology design is *boring craft work* nobody wants to redo, and it compounds with usage telemetry.

### 4.3 Bootstrap flow (cloud, customer-facing)

```
1. Sign up, pick vertical (Marketing | DevTools | Support | ...)
2. Connect data sources (OAuth: GA4, Slack, Notion, Drive, ...)
3. Cloud runs the OSS IngestPipeline against your data, biased by the
   vertical's ontology
4. ~30 min later, your tenant has a populated Feather DB with:
   - Atomic facts extracted
   - Entities canonicalized to your business namespace
   - Typed edges built per the vertical's ontology
5. UI shows: "We extracted 12,481 facts across 47 entities. Top
   contradictions detected: 3. Ready to query."
6. Customer queries via API or built-in chat UI
7. Cloud uses the OSS QueryPlanner + vertical's tuned workflows
```

### 4.4 Multi-tenant + ops layer

Standard SaaS but on top of Feather:
- Per-tenant `.feather` files in S3, warmed to local NVMe on access
- Stripe metered billing (per-token-extracted, per-query)
- API keys with scoped roles (read-only / read-write / admin)
- Audit log of every extraction + query
- SOC2 controls (Q4 2026)
- VPC deployment option for regulated tenants

---

## 5. The "Feather guides the LLM" pattern (Step 5 detailed)

This is the architectural inversion the user explicitly called out. Worth a section of its own.

### 5.1 Today's pattern (passive)

```
LLM → "search my memory for X" → db.hybrid_search(X, k=10) → 10 snippets
                                                              │
                                                              ▼
                                  LLM synthesizes from 10 snippets
```

The LLM does the search reasoning. Feather just returns rows.

### 5.2 Phase 9 pattern (active)

```
                LLM asks question Q
                         │
                         ▼
   ┌──────────────────────────────────────────────┐
   │           QueryPlanner (Feather)             │
   │   "Q is a temporal-knowledge-update query.   │
   │    Plan:                                     │
   │      step 1: bm25 on entity name (precision) │
   │      step 2: graph hop=2 from result         │
   │      step 3: vector(k=20, decay=0.6)         │
   │      step 4: filter (timestamp >= 2026-Q1)   │
   │      step 5: rank by recency-weighted score" │
   └──────────────────────────────────────────────┘
                         │
                         ▼
   ┌──────────────────────────────────────────────┐
   │     PlanExecutor — runs steps 1..5           │
   │     against the Feather core                 │
   └──────────────────────────────────────────────┘
                         │
                         ▼
   ┌──────────────────────────────────────────────┐
   │     Synthesizer (LLM)                        │
   │   - Receives plan + ranked evidence          │
   │   - Detects contradictions in the evidence   │
   │   - Drafts answer with inline citations      │
   │   - Returns: answer + plan + evidence        │
   └──────────────────────────────────────────────┘
                         │
                         ▼
                  Final response to user
```

**What "guide the LLM" means concretely:** Feather's QueryPlanner inspects the question (using a small, fast LLM) to decide *how* to search — vector vs BM25 vs graph hops vs decay-weighted vs filtered — *before* any retrieval happens. The LLM that *answers* the user gets the curated bundle, not the raw vector hits.

This is what specifically attacks our weakest LongMemEval axes:

- **Knowledge-update (0.714 today)** — planner detects "this is a current-state question" → uses decay-weighted retrieval, surfaces only the most recent + supersession-resolved facts.
- **Multi-session (0.606 today)** — planner detects "this requires synthesis" → graph BFS expansion before final ranking.
- **Temporal-reasoning (0.477 today)** — planner detects "anchored to specific time" → filtered retrieval by timestamp window first, then vector.

Expected lift on LongMemEval_S (best estimate, will validate):

| Axis | Today (GPT-4o) | After Phase 9 |
|---|---|---|
| knowledge-update | 0.714 | **0.85+** (closing Supermemory's gap) |
| multi-session | 0.606 | **0.78+** |
| temporal-reasoning | 0.477 | **0.70+** |
| **overall** | **0.693** | **0.80+** |

**Important caveat:** these projections are gut estimates. The whole point of Phase 9 is we'll *measure* against the existing harness on every step.

### 5.3 What this is NOT

- **Not** a chain-of-thought wrapper around an LLM. The planner is structured — it emits a typed plan, not free-form reasoning.
- **Not** a generic agent framework. We're solving "memory query," not "general task execution."
- **Not** a replacement for the answerer LLM. Synthesis still requires a strong reader. We just hand it dramatically better context.

---

## 6. Phased rollout (10–14 weeks)

Each phase has a measurable exit criterion against our existing benchmark harness. No phase ships without a number.

### Phase 9.1 — `feather_db.extractors` (4 weeks, OSS)

| Week | Deliverable | Exit criterion |
|---|---|---|
| 1 | `FactExtractor` + `EntityResolver` + `TemporalParser` | LongMemEval_S overall ≥ 0.72 (baseline 0.693) |
| 2 | `OntologyLinker` + `ContradictionResolver` | knowledge-update ≥ 0.80 (baseline 0.714) |
| 3 | `IngestPipeline` end-to-end + provenance edges | 100% facts have `extracted_from` edges |
| 4 | Cost guards + idempotency + tests + docs | Re-run on same source < 2% token cost |

**Ship as v0.9.0.** Headline: "Feather DB v0.9.0 — agentic ingestion. LongMemEval_S = X.XXX."

### Phase 9.2 — `feather_db.reason` (4 weeks, OSS)

| Week | Deliverable | Exit criterion |
|---|---|---|
| 5 | `QueryPlanner` (LLM-backed) | LongMemEval_S overall ≥ 0.78 |
| 6 | `PlanExecutor` (against existing search primitives) | per-axis: temporal ≥ 0.65, multi-session ≥ 0.75 |
| 7 | `Synthesizer` (with provenance + contradictions) | All answers cite evidence by ID |
| 8 | Reason CLI + REST endpoint + benchmarks | LongMemEval_S overall ≥ 0.80 |

**Ship as v0.9.1.** Headline: "Feather DB v0.9.1 — guided memory. LongMemEval_S = X.XXX, beats Supermemory."

### Phase 9.3 — Cloud verticals (parallel, separate private repo, 6 weeks)

| Week | Deliverable | Exit criterion |
|---|---|---|
| 5–6 | Cloud control plane MVP (multi-tenant, auth, billing) | First-tenant onboarding works end-to-end |
| 7–8 | First vertical: **Marketing Agent** (the one we already understand) | One pilot customer ingests + queries |
| 9–10 | Second vertical: **DevTools Agent** | Two pilots; reasoning workflows tuned |
| 11–12 | Self-serve signup + dashboard + observability | Public beta launch |

**Cloud beta in Q3 2026** as the original positioning doc said. Marketing + DevTools the two launch verticals; Support / Finance / HR follow.

### Phase 9.4 — Hardening (continuous)

- Auto-tuning per tenant (decay, k, hop_depth, planner temperature)
- VPC / on-prem deployment
- SOC2 compliance
- Customer dashboards / observability

---

## 7. Risks + open questions

### Risks

1. **LLM extractors hallucinate facts that look real.** Mitigation: confidence scoring + adversarial validation pass + provenance edges. Worst case: facts have low confidence, they're stored but down-weighted in retrieval. *Test plan:* re-run a known-clean dataset (LongMemEval gold answers) through the extractor and verify recovery rate.
2. **Token cost on real customer corpora is unpredictable.** Mitigation: cost guards (`max_tokens_per_source`), batch extraction, the cheaper-tier extractor LLM (Haiku, Flash) by default.
3. **Vertical agent ontologies become unmaintainable** as we add verticals. Mitigation: shared ontology kernel + per-vertical extensions, not standalone trees per vertical.
4. **Cloud customers ask "but I want this open"** — we lose the deal. Mitigation: clear messaging that the *engine* (extractors + reasoner) is open; only the verticals + ops are closed. Most customers will accept this; a few won't, and they'll self-host on OSS — that's fine.
5. **Cannibalization** — OSS extractors are good enough that nobody buys Cloud. Mitigation: Cloud's value isn't the extractors, it's the **time-to-first-query** (connectors + bootstrap + tuning), which OSS users still have to build themselves.

### Open questions

- **Should the QueryPlanner LLM be the same as the answerer?** Likely no — planner needs speed (Haiku/Flash); answerer can be slower (Sonnet/GPT-4o). But this doubles per-query LLM cost. Profile in Phase 9.2.
- **Do we expose the plan to end users, or hide it?** Default: expose (transparency, debuggability). Cloud may toggle hidden in production.
- **How do we handle multi-namespace queries?** Today each question gets a namespace; cross-namespace search is unimplemented. Likely needs explicit federation API.
- **Versioning extracted facts.** When prompts change, do we re-extract retroactively or only on new ingest? Probably new only, with migration tools.

---

## 8. Success criteria — what "Phase 9 worked" looks like

**Engineering:**
- LongMemEval_S overall ≥ **0.80** with the OSS-only stack (extractors + reasoner) on a flash-tier answerer.
- LongMemEval_S overall ≥ **0.85** with the OSS stack + GPT-4o answerer.
- Per-axis: knowledge-update ≥ 0.85, temporal-reasoning ≥ 0.70.
- Re-running ingestion on the same corpus costs < 2% of first run.

**Adoption:**
- v0.9.0 release → **+1,000 GitHub stars** within 30 days (vs baseline).
- v0.9.0 PyPI weekly downloads → **3× v0.8.0 baseline**.
- 100+ developers in the Discord / Github discussions actively using extractors.

**Cloud:**
- 5 paying pilot customers by end of Q3 2026.
- Marketing vertical is the lighthouse: ≥ 1 named customer with public quote.

**Narrative:**
- One published external benchmark (HN front page or AI newsletter feature) confirming our 0.80+ number.
- Cited by at least one academic paper as a memory layer.

---

## 9. What we ship in the launch

When v0.9.0 lands:

| Asset | Where |
|---|---|
| `feather_db.extractors` module | OSS (PyPI, Crates.io for Rust binding) |
| Reproduction harness extension (`bench/scenarios/longmemeval_phase9.py`) | OSS |
| Benchmark report update (per-axis lifts, before/after) | `docs/benchmarks/longmemeval-phase9.md` |
| arXiv paper §4.8 (Phase 9 results) | repo + arXiv |
| Blog post: "Feather DB v0.9.0 — agentic memory" | `docs/blog/v0.9.0.md` |
| Marketing pack (twitter, HN, Reddit) | `docs/marketing/v0.9.0/` |
| Cloud beta waitlist email blast | Cloud team |
| Two demo videos (extractors + reasoner) | YouTube |

---

## 10. Decision needed from you (founder)

To proceed, I need three calls from you:

1. **Confirm the OSS / Cloud cut.** Do extractors + reasoner ship OSS as proposed, or do you want some/all of them gated to Cloud?
2. **Pick the launch verticals.** I propose Marketing + DevTools (we have Marketing already in our codebase via `feather_db.domain_profiles.MarketingProfile`; DevTools is what most of your audience builds). Other candidates: Customer Support, Finance, HR.
3. **Approve the 14-week timeline.** Aggressive but doable. If we slip, the most likely slip is Phase 9.3 (Cloud), which is OK because OSS launches earlier and Cloud follows at its own pace.

When you approve all three, I'll:
- Open the `feather-cloud` private repo (separate from this OSS repo)
- Scaffold `feather_db/extractors/` here in the OSS
- Add Phase 9.1 sub-tasks to the task list
- Start Week 1 work: `FactExtractor` + `EntityResolver` + `TemporalParser`

---

*Last updated: 2026-04-27. Author: Hawky.ai engineering. Status: draft pending founder approval.*
