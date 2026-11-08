# Feather DB

**Agent memory in a single file**

*Part of [Hawky.ai](https://hawky.ai) — AI-Native Development Tools*

[![Tests](https://github.com/feather-store/feather/actions/workflows/test.yml/badge.svg)](https://github.com/feather-store/feather/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/feather-db)](https://pypi.org/project/feather-db/)
[![Crates.io](https://img.shields.io/crates/v/feather-db-cli)](https://crates.io/crates/feather-db-cli)
[![Python](https://img.shields.io/pypi/pyversions/feather-db)](https://pypi.org/project/feather-db/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Website](https://img.shields.io/badge/website-getfeather.store-blue)](https://www.getfeather.store/)
[![HuggingFace Space](https://img.shields.io/badge/demo-HuggingFace-yellow)](https://huggingface.co/spaces/Hawky-ai/feather-db)
[![Benchmark Results](https://img.shields.io/badge/dataset-benchmarks-orange)](https://huggingface.co/datasets/Hawky-ai/feather-db-benchmarks)

[![LongMemEval_S](https://img.shields.io/badge/LongMemEval__S-0.911%20%7C%20gemini--2.5--flash-success)](#memory-benchmark--longmemeval-xu-et-al-2024)
[![ANN p50](https://img.shields.io/badge/SIFT1M%20p50-0.19ms-blue)](docs/benchmarks/longmemeval.md)
[![ANN Recall](https://img.shields.io/badge/recall%4010-0.972-blue)](docs/benchmarks/longmemeval.md)

Feather is the memory layer that runs **inside** your agent — not a database you
deploy next to it. One `.feather` file on disk, no server, no daemon, no network
hop. It is an embedded vector store with the things a memory actually needs:
adaptive decay, trust levels, confidence, scoped working sets, and context
packets that refuse to drop a required constraint.

```python
import feather_db, numpy as np
db = feather_db.DB.open("memory.feather", dim=768)

meta = feather_db.Metadata()
meta.content    = "user prefers written async updates"
meta.confidence = 0.9                       # how likely it is TRUE
meta.importance = 1.0                       # how much it MATTERS - a separate axis
db.add(id=1, vec=embedding, meta=meta)

db.keyword_search("async updates", k=5)     # BM25, Porter-stemmed
db.hybrid_search(query_vec, "async", k=5)   # reciprocal rank fusion
db.close()
```

**What it is not.** Feather has no transactions, no query planner, and one
writer per file (readers are unlimited and lock-free). Prefiltered search is
metadata equality, not graph traversal. If you need a general-purpose graph or
OLTP database, use one — Feather is built for the narrower job of being what an
agent remembers into, and the limits are the price of being small enough to run
in-process.

**Memory semantics, concretely:**

| | |
|---|---|
| Adaptive decay | `stickiness = 1 + ln(1 + recall_count)` — recalled memories age slower |
| Trust and confidence | Separate from importance, and both filterable |
| Scoped pockets | Budget-fitted hot set over dotted, inheriting scopes |
| Context packets | Required constraints resolved first; **fails closed** if one is missing |
| Expiry | Per-record `ttl`, so a short-lived agent run scope cleans itself up |

Integrations that already exist: an **MCP server** for Claude Desktop and Claude
Code, and a **LangGraph `BaseStore`** — cross-thread agent memory in one line,
`builder.compile(store=FeatherStore(...))`.

---

## What's New in v0.22 — Valid Time & Structured Agent Memory

| Capability | Notes |
|---|---|
| **Valid time** | `Metadata.valid_at` records *when something happened*, beside `timestamp` (when it was recorded). Indexed per namespace (`db.newest(ns, n, by_valid=True)`), filterable (`FilterBuilder().valid_between(after, before)`), and exposed over the Cloud API and `RemoteDB`. Stored as a reserved attribute, so **no file-format or WAL change**. |
| **Fact routing into `AgentMemory`** | `mem.ingest_facts(facts)` sends each extracted fact to the kind whose lifetime fits: events → episodes (deduplicated), preferences → preferences, the rest → facts with supersession. Every write carries the fact's valid time, so sessions can be ingested in any order. |
| **Dated writes** | `mem.facts.state(..., at=ts)` and `mem.preferences.set(..., at=ts)`; an older `at` never overwrites a newer value. `episodes.recent()` / `all()` walk the valid-time index instead of sorting the namespace. |
| **`SessionFactExtractor`** | Dated facts from a whole session, each with `kind`, canonical `entity`, `amount` and own-vs-mentioned; `ledger()` and `profile()` render them for a prompt. |
| **File ingest (Cloud API)** | `POST /v1/{ns}/ingest_file` stores an image, video, audio, PDF or text file by what a multimodal model sees in it; there is an upload modal in the admin SPA. Hybrid search embeds the query server-side when no vector is given. |
| **Time travel** | `DB.open(path, as_of=t)` opens the memory as it was at a past moment: by time, by tag (`as_of="run-42"`) or by commit (`generation=n`). The handle is read-only and scores with its clock pinned to that moment, so a replay retrieves what the agent retrieved then. `db.history()`, `db.tag()`, `feather_db.diff(a, b)`. Costs 51 bytes a commit; opening the past costs what a read-only open costs. See [Time Travel](#time-travel). |
| **History retention** | `db.set_history_retention(seconds)` keeps the file a rewrite (vacuum, compact, purge) replaces, so `as_of` reaches back that far. Off by default; tags keep their commit regardless. `erase()` deletes all history. |
| **LongMemEval_S: 0.911** | Mean of three reportable runs on all 500 questions, official GPT-4o judge (0.916, 0.904, 0.914). See [Benchmarks](#memory-benchmark--longmemeval-xu-et-al-2024). |
| **Measured end to end** | Latency p50/p99 for every layer, from a live agent turn down to one search, plus LongMemEval, LoCoMo, contradiction and poisoning scores: [Performance](#performance) and [Benchmarks](#benchmarks). |

```python
from feather_db.integrations import AgentMemory
from feather_db.extractors import SessionFactExtractor

facts = SessionFactExtractor(provider).extract_session(turns, session_date)
mem.ingest_facts(facts)            # {'episodes': 3, 'preferences': 1, 'facts': 4, 'repeats': 0}

db.newest("user_842", 10, by_valid=True)                     # by when it happened
f = FilterBuilder().namespace("user_842").valid_between(after=t0).build()

db.tag("run-42")                                             # name this moment
past = feather_db.DB.open("memory.feather", as_of="run-42")  # memory as it was then
feather_db.diff(past, db)                                    # added / removed / changed
```

Nothing was removed or renamed. On `PUT`, an omitted `valid_at` keeps the stored
value, so clients that predate the field can't erase it. Builds older than 0.22
see it as an extra attribute named `"\0valid_at"` and write it back intact.

---

<details>
<summary><b>Release history — v0.10 through v0.16</b> (click to expand)</summary>

## What's New in v0.13–v0.16 — Ingestion, Memory, Claude & Fast Load (Phase 8)

| Capability | Version | Notes |
|---|---|---|
| **Persisted HNSW graph** | v0.16.0 | `save()` embeds the prebuilt graph (**file format v9**) so `load()` restores it instead of rebuilding — **5–25× faster cold load** (48 ms vs 2.7s/13.4s on 40k×128 clustered) and deterministic serial-build recall (0.988). Falls back to rebuild for dirty DBs / old files; ~25% larger files. |
| **Adaptive index capacity** | v0.15.3 | HNSW indices start at 4096 elements and grow via `resizeIndex()` on demand instead of preallocating 1M — **~7.7× less RAM** for many-namespace deployments (709→92 MB across 19 namespaces), no hard cap. |
| **Parallel HNSW load** | v0.13.0 | Graph rebuilt across a thread pool on open — **~4.7× faster load** (7.6s→1.7s for 40k×128), identical recall. `FEATHER_LOAD_THREADS` to cap. Still used for old files / DBs with pending deletions. |
| **Parallel batch ingest** | v0.13.0 | `DB.add_batch(ids, vecs, metas=None)` builds the graph in parallel with the GIL released — **~3.4× faster** bulk insert. |
| **SIMD on x86** | v0.13.0 | SSE/AVX L2 kernels (runtime-dispatched) compiled on x86_64; arm64 uses `-O3` NEON. `FEATHER_SIMD=none\|sse\|avx\|avx512`. |
| **In-RAM int8 quantization** | v0.15.0 | `set_int8_ram(modality)` stores vectors as int8 **in memory** — **~1.7× less RAM** (227→129 MB at 60k×768). Since 0.22 the scale is trained from the data, and search re-ranks by float originals read from the file (`refine=2`), so recall matches float. |
| **MCP connector for Claude** | v0.14.0 | `feather-serve` exposes Feather as a persona context engine to **Claude Desktop / Code** — local `.feather` *or* remote Cloud API (`--api-url`). |
| **Real embedders** | v0.15.1 | `feather-serve --embed-provider gemini\|openai\|voyage\|cohere\|ollama` — semantic recall over a hosted instance (Gemini `text-embedding-004` = native 768). |

```python
# Bulk-ingest a persona's history fast (parallel HNSW build)
db.add_batch(ids, vecs, metas)

# Store vectors as int8 in RAM — ~1.7x less memory. The scale is trained from
# the data; search re-ranks by float originals kept in the file (refine=2).
db.set_int8_ram("text")
```

```bash
# Claude Code → hosted Feather as a persona context engine (real embeddings)
GOOGLE_API_KEY=… claude mcp add feather -- feather-serve \
  --api-url http://HOST:8000 --namespace persona --dim 768 --embed-provider gemini
```

---

## What's New in v0.11–v0.12 — Query Performance & Compression (Phase 7)

| Capability | Version | Notes |
|---|---|---|
| **Secondary metadata indexes** | v0.11.0 | Inverted indexes on `namespace_id` / `entity_id` / `attributes` — namespace & attribute lookups go from O(n) scans to **O(matches)**. New `DB` methods: `ids_in_namespace`, `ids_for_entity`, `ids_with_attribute`, `namespace_size`, `list_namespaces`. |
| **Pre-filtered ANN search** | v0.11.0 | `search(filter=…)` with a `namespace`/`entity`/`attribute` constraint now ranks **exactly** over the indexed candidate set, returning a **complete top-k** instead of HNSW's `ef`-bounded under-return — and is O(matches), so selective filters are *faster*. |
| **Incremental auto-compaction** | v0.11.0 | `set_auto_compact(ratio)` rebuilds a modality index once its deleted/total ratio crosses a threshold, reclaiming `forget()`/`purge()`'d vectors automatically. `compact()` also fixed to reclaim forgotten records and never resurrect purged ones. |
| **On-disk int8 quantization** | v0.12.0 | `set_quantized(modality)` persists vectors as int8 + per-vector scale (**file format v7**) — **~2.5–4× smaller `.feather` files**, dequantized to float32 on load (search unchanged). |

```python
# Pre-filtered exact search — reliably returns a full k even under a selective filter
f = FilterBuilder().namespace("acme").attribute("channel", "instagram").build()
results = db.search(query_vec, k=10, filter=f)        # complete top-10, exact ranking

# Auto-compaction — reclaim deleted vectors past 20% dead
db.set_auto_compact(0.2)

# int8 on-disk compression — ~3x smaller files, opt-in per modality
db.set_quantized("text", True)
db.save()                                             # persisted as int8 (format v7)
```

> **Scope note:** int8 quantization reduces **disk footprint and load I/O**; the
> in-memory HNSW index remains float32. In-RAM int8 indexing is a future step.

---

## What's New in v0.10 — Feather DB Cloud Edition

The `feather-api/` package now ships a **production-ready admin SPA** + a
**pluggable embedding service**, so you can run Feather as a managed context
engine for downstream consumers (e.g. brand teams, agents, internal tools).

| Capability | Where | Notes |
|---|---|---|
| **Atlas-style admin SPA** | `/admin/` route on the FastAPI server | Custom HTML + Tailwind + Alpine.js, brand-aligned, zero build step |
| **Pluggable embeddings** | Settings → Embedding service | OpenAI · Azure OpenAI · Gemini · Voyage · Cohere · Ollama with curated model dropdowns |
| **Ingest text** | `POST /v1/{ns}/ingest_text` | Server embeds via the configured provider, then stores — single call |
| **Bulk import** | `POST /v1/{ns}/import` | JSON array of `{id, vector?, metadata}`; WAL-backed throttled saves for fast large loads, `flush` to compact — see [bulk ops](docs/bulk-operations.md) |
| **Bulk delete** | `POST /v1/{ns}/records/batch_delete` | Delete many ids and/or a whole `entity_id` in one call (one save) — use instead of looping single DELETE |
| **Upload `.feather`** | `POST /v1/admin/upload` · Namespaces → Import | Adopt a locally-built DB as a cloud namespace (graph + index intact) |
| **Any-dimension namespaces** | first vector or `dim` at create | Each namespace adopts its own dim (384/768/1024/1536…), no silent padding — see [dimensions](docs/dimensions.md) |
| **Filtered search** | `POST /v1/{ns}/search` · Search tab | Vector / keyword / hybrid, filtered by `namespace_id`, `entity_id`, and attributes |
| **Hierarchy navigator** | Namespace detail → Hierarchy tab | Brand → Channel → Campaign → AdSet → Ad → Creative tree from `metadata.attributes` |
| **Marketing profile card** | Record drawer | Auto-renders KPIs (CTR, ROAS, channel) when present |
| **Cmd-K palette** | Press ⌘K / Ctrl+K | Fuzzy search namespaces · `id:123` to open a record · `/seed`, `/import` actions |
| **Live observability** | Overview screen | p50 / p95 / p99 latency, ops-per-minute sparkline, recent activity feed |
| **Delete + purge + compact** | Namespace header buttons | Bulk delete by ids/entity, PURGE by `namespace_id`, COMPACT to reclaim |
| **Schema discovery** | Namespace detail → Schema tab | Distinct attribute keys + type inference + sample values |
| **Connection panel** | Settings → Connection | Copy-paste cURL / Python / JS snippets pre-filled with your URL |

See [docs/quickstart.md](docs/quickstart.md) for a self-hosted setup walkthrough.

**Guides:** [Vector dimensions](docs/dimensions.md) · [Bulk operations](docs/bulk-operations.md) (import / delete / upload) · [Integrations](docs/integrations.md)

> **Deployment note**: `feather-api/` runs single-tenant with one shared
> `FEATHER_API_KEY`, and **refuses to start without it** — an unset key would
> leave every namespace readable and deletable by anyone who can reach the
> port. For a local no-auth run, set `FEATHER_DEV_MODE=1` explicitly.
> Multi-tenant key isolation + HTTPS are on the roadmap.

---

</details>

## What's Inside

| Capability | Description |
|------------|-------------|
| **ANN Search** | Sub-millisecond approximate nearest-neighbor search via HNSW |
| **Multimodal Pockets** | Text, image, audio vectors per entity under a single ID |
| **Context Graph** | Typed + weighted edges, reverse index, auto-link by similarity |
| **Context Chain** | One call: vector search + n-hop BFS graph expansion |
| **Living Context** | Recall-count stickiness — frequently accessed items resist temporal decay |
| **Namespace / Entity / Attributes** | Generic partition + subject + KV metadata for any domain |
| **Graph Visualizer** | Self-contained D3 force-graph HTML — fully offline, no CDN |
| **LLM Agent Connectors** | Claude, OpenAI, Gemini tool-use/function-calling with 14 Feather tools |
| **MCP Server** | `feather-serve` — connects Feather to Claude Desktop, Cursor, and any MCP client |
| **LangChain / LlamaIndex** | Drop-in `FeatherVectorStore`, `FeatherMemory`, `FeatherRetriever` adapters |
| **Self-Aligned Context Engine** | LLM-powered ingestion: auto-classifies, scores, links, and namespaces every record |
| **Valid time** | `valid_at` per record — indexed, filterable, ordered by when it happened rather than when it was written |
| **Agent memory** | `Pocket` working sets, `AgentMemory`'s five kinds with enforced lifetimes, `ingest_facts()` routing |
| **Single-file persistence** | `.feather` binary format **v10**: append-only commits behind two checksummed headers, so a save writes only what changed; persisted HNSW graph, optional int8 on-disk and in-RAM; v3–v9 files load transparently |

---

## Installation

```bash
pip install feather-db            # core
pip install "feather-db[all]"     # + langchain, llamaindex, mcp extras
```

CLI (Rust):

```bash
cargo install feather-db-cli
```

Docker (official multi-arch image — amd64 + arm64):

```bash
docker run -d --name feather \
  -p 8000:8000 \
  -e FEATHER_API_KEY="feather-$(openssl rand -hex 16)" \
  ghcr.io/feather-store/feather-api:latest

curl http://localhost:8000/health   # → {"status":"ok"}
# Admin SPA: http://localhost:8000/admin/ · Swagger: http://localhost:8000/docs
```

> Images are published to GHCR on every version tag by
> `.github/workflows/docker.yml`. Pin a release in production
> (`ghcr.io/feather-store/feather-api:<version>`).

Build from source:

```bash
git clone https://github.com/feather-store/feather
cd feather
python setup.py build_ext --inplace
```

---

## Quick Start

```python
import feather_db
import numpy as np

# Open or create a database
db = feather_db.DB.open("context.feather", dim=768)

# Add a vector with metadata
meta = feather_db.Metadata()
meta.content = "User prefers dark mode"
meta.importance = 0.9
db.add(id=1, vec=np.random.rand(768).astype(np.float32), meta=meta)

# Semantic search
results = db.search(np.random.rand(768).astype(np.float32), k=5)
for r in results:
    print(r.id, r.score, r.metadata.content)

db.save()
```

---

## Self-Aligned Context Engine (v0.7.0)

The `ContextEngine` wraps `DB` with an LLM-powered ingestion pipeline. Drop in any text — the engine classifies it, scores it, links it to related records, and stores it in the right namespace. No schema to define upfront.

```python
from feather_db import ContextEngine, ClaudeProvider
import numpy as np, hashlib

def embed(text: str) -> np.ndarray:
    # replace with your real embedder
    vec = np.zeros(768, dtype=np.float32)
    for i, tok in enumerate(text.split()[:768]):
        vec[i % 768] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

engine = ContextEngine(
    db_path  = "knowledge.feather",
    dim      = 768,
    provider = ClaudeProvider(),   # or OpenAIProvider, GeminiProvider, OllamaProvider, None
    embedder = embed,
    namespace = "myapp",
)

nid = engine.ingest(
    "Competitor X launched a developer SDK with MIT license — 10k GitHub stars in 24 hours."
)
```

**The engine automatically:**
- Classifies entity type (`competitor_intel`, `user_feedback`, `strategy_brief`, …)
- Scores importance (0–1) and confidence (0–1)
- Assigns TTL and namespace
- Suggests and creates graph edges to related records

**Works offline too** — pass `provider=None` for a built-in heuristic classifier (no API key needed):

```python
engine = ContextEngine(db_path="k.feather", dim=768, provider=None, embedder=embed)
```

### Supported Providers

```python
from feather_db import ClaudeProvider, OpenAIProvider, OllamaProvider, GeminiProvider

ClaudeProvider(model="claude-haiku-4-5-20251001")         # Anthropic
OpenAIProvider(model="gpt-4o-mini")                        # OpenAI
OpenAIProvider(model="llama-3.3-70b-versatile",            # Groq
               base_url="https://api.groq.com/openai/v1",
               api_key=GROQ_KEY)
OllamaProvider(model="llama3.1:8b")                        # Ollama (local, no key)
GeminiProvider(model="gemini-2.0-flash")                   # Google Gemini
```

All providers share the same `LLMProvider` interface — swap at any time without changing the rest of your code.

---

## LLM Agent Connectors (v0.6.0)

Give any LLM agent native access to Feather DB via **14 built-in tools**.

### Claude (tool_use)

```python
import anthropic
from feather_db import ClaudeConnector

connector = ClaudeConnector(db=db, embedder=embed)
client    = anthropic.Anthropic()

messages = [{"role": "user", "content": "What competitor moves should I watch?"}]
reply    = connector.run_loop(client, messages, model="claude-opus-4-6")
print(reply)
```

### OpenAI / Groq / vLLM (function_calling)

```python
from openai import OpenAI
from feather_db import OpenAIConnector

connector = OpenAIConnector(db=db, embedder=embed)
client    = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Find records about onboarding friction"}],
    tools=connector.tools(),
)
result = connector.handle(resp.choices[0].message.tool_calls[0].function.name,
                          resp.choices[0].message.tool_calls[0].function.arguments)
```

### Available Tools (14)

| Tool | Description |
|------|-------------|
| `feather_search` | Semantic vector search |
| `feather_context_chain` | Vector search + graph BFS expansion |
| `feather_get_node` | Retrieve a single record by ID |
| `feather_get_related` | Get all graph-linked records |
| `feather_add_intel` | Store a new record with metadata |
| `feather_link_nodes` | Create a typed weighted edge |
| `feather_timeline` | Time-ordered records in a range |
| `feather_forget` | Drop a record by ID |
| `feather_health` | Database health report |
| `feather_why` | Explain why a record was retrieved |
| `feather_mmr_search` | Maximal marginal relevance search |
| `feather_consolidate` | Merge near-duplicate records |
| `feather_episode_get` | Retrieve an episode by ID |
| `feather_expire` | Purge records past their TTL |

---

## MCP Server (v0.6.0)

Connect Feather DB to **Claude Desktop**, **Cursor**, or any MCP-compatible client:

```bash
pip install "feather-db[mcp]"
feather-serve --db knowledge.feather --dim 768
```

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "feather": {
      "command": "feather-serve",
      "args": ["--db", "/path/to/knowledge.feather", "--dim", "768"]
    }
  }
}
```

Claude gets agent memory, not database verbs: the tools `remember`, `recall`,
`context`, `forget` and `memory_stats`, the prompts `/remember`,
`/what-do-you-know` and `/catch-up`, and the resource `feather://context` (the
current working set). `--scope org.brand.agent` separates agents sharing one
file. To serve a Cloud API namespace instead of a local file, use
`--api-url http://HOST:8000 --namespace NAME` (it is created if absent).
`feather-serve` and `feather-agent` are the same server.

---

## LangChain Integration (v0.6.0)

```python
from feather_db.integrations import FeatherVectorStore, FeatherMemory, FeatherRetriever

# Drop-in VectorStore
store     = FeatherVectorStore(db=db, embedder=embed)
retriever = store.as_retriever(search_kwargs={"k": 5})

# Semantic conversation memory with adaptive decay
memory = FeatherMemory(db=db, embedder=embed, k=5)

# context_chain retriever
retriever = FeatherRetriever(db=db, embedder=embed, k=5, hops=2)
```

---

## LlamaIndex Integration (v0.6.0)

```python
from feather_db.integrations import FeatherVectorStoreIndex, FeatherReader

# Index documents
index = FeatherVectorStoreIndex.from_documents(documents, db=db, embed_model=embed_model)
query_engine = index.as_query_engine()
response = query_engine.query("What is our retention strategy?")

# Load existing Feather DB as LlamaIndex Documents
reader = FeatherReader(db=db)
docs   = reader.load_data()
```

---

## Core Features

### Multimodal Pockets

Each named modality gets its own independent HNSW index and dimensionality. A single entity ID can hold text, visual, and audio vectors simultaneously.

```python
db.add(id=42, vec=text_vec,   modality="text")    # 768-dim
db.add(id=42, vec=image_vec,  modality="visual")  # 512-dim
db.add(id=42, vec=audio_vec,  modality="audio")   # 256-dim

results = db.search(query_vec, k=10, modality="visual")
```

### Context Graph

Typed, weighted edges between records. Nine built-in relationship types plus free-form strings.

```python
from feather_db import RelType

db.link(from_id=1, to_id=2, rel_type=RelType.CAUSED_BY, weight=0.9)
db.link(from_id=1, to_id=3, rel_type=RelType.SUPPORTS,  weight=0.7)

edges    = db.get_edges(1)      # outgoing edges
incoming = db.get_incoming(2)   # reverse index

db.auto_link(modality="text", threshold=0.85, rel_type=RelType.RELATED_TO)
```

Built-in types: `related_to`, `derived_from`, `caused_by`, `contradicts`, `supports`, `precedes`, `part_of`, `references`, `multimodal_of`.

### Context Chain

One call that combines semantic search with n-hop BFS graph traversal:

```python
result = db.context_chain(query=query_vec, k=5, hops=2, modality="text")

for node in result.nodes:
    print(node.id, node.score, node.hop_distance)
for edge in result.edges:
    print(edge.source_id, "->", edge.target_id, edge.rel_type)
```

### Filtered Search

```python
from feather_db import FilterBuilder

results = db.search(
    query_vec, k=10,
    filter=FilterBuilder()
        .namespace("acme")
        .entity("user_123")
        .attribute("channel", "instagram")
        .importance_gte(0.5)
        .build()
)
```

When the filter constrains a `namespace`, `entity`, or `attribute`, Feather
resolves the candidate set from its secondary indexes and ranks **exactly** over
just those vectors — so a selective filter returns a **complete top-k** (no
`ef`-bounded under-return) and runs in O(matches).

### Living Context / Adaptive Decay

```python
from feather_db import ScoringConfig

cfg     = ScoringConfig(half_life=30.0, weight=0.3, min=0.0)
results = db.search(query_vec, k=10, scoring=cfg)
```

Formula:
```
relevance     = (cosine - floor) / (1 - floor)   # floor: the model's background cosine
recalls       = recall_count * 0.5 ^ (days since last recall / 30)   # one H for all
stickiness    = 1 + log(1 + recalls)            # grows with use, fades without it
effective_age = age_in_days / stickiness        # age from valid_time()
recency       = 0.5 ^ (effective_age / half_life_days)
freshness     = (1 - time_weight) + time_weight * recency
final_score   = relevance * freshness * importance   # relevance <= 0 returned as is
```

Recency scales relevance rather than adding to it, so at `time_weight=0.3` a
record can lose at most 30% of its relevance to age, and vector length does
not change the ranking. Relevance is measured from the embedding model's
background cosine (`db.background_cosine()`, estimated from your vectors; an
unrelated pair of gemini-embedding-001 texts is at ~0.51, not 0), so a fresh
but unrelated record cannot overtake an old relevant one.
`ScoringConfig(..., relevance_floor=0.0)` scores by plain cosine. Recalls fade with time since the last one, so a record
heavily used a year ago is not hot now.

`touch()` is called automatically on every search hit.

### Time Travel

Every commit (`save()`) ends with a small marker, so any earlier commit the
file still holds can be opened as it was:

```python
import feather_db, time

db = feather_db.DB.open("memory.feather", dim=768)
db.set_history_retention(7 * 86400)        # keep a week of history across rewrites
db.tag("before-run-42")                    # commit now and name the commit

past = feather_db.DB.open("memory.feather", as_of="before-run-42")
past = feather_db.DB.open("memory.feather", as_of=time.time() - 3600)   # an hour ago
past = feather_db.DB.open("memory.feather", generation=17)              # one commit

past.search(query_vec, k=10, scoring=cfg)  # ranked with the clock of that moment
past.as_of()                               # {'generation', 'commit_time', 'clock', 'file'}
db.history()                               # every commit as_of can reach
feather_db.diff(past, db)                  # MemoryDiff(added=[...], removed=[...], changed=[...])
```

- **A replay sees what the agent saw.** A past handle is read-only, ignores
  the WAL, records no recalls, and pins the clock that decay scoring,
  `Pocket.hot()`, `PacketBuilder` and TTL checks read (`past.now()`).
- **How far back.** Back to the latest full rewrite of the file, or further
  while `set_history_retention()` or a tag keeps the replaced files (hard
  links under `<path>.history/`). A moment out of reach is a `ValueError`
  that says where the history begins, never the nearest state instead.
- **`erase()` deletes all history**, so erased data cannot come back through
  the past.
- **Over HTTP:** `RemoteDB(url, ns, as_of=...)` sends `X-Feather-As-Of`, and
  the Cloud API serves reads from that commit.

### Memory Layer (v0.6.0)

```python
from feather_db import MemoryManager

mm = MemoryManager(db)
print(mm.health_report())          # cluster stats + stale records
diverse = mm.search_mmr(vec, k=10) # maximal marginal relevance
mm.consolidate(threshold=0.95)     # merge near-duplicate records
mm.assign_tiers()                  # hot / warm / cold tiering by recall_count
```

### Episodes (v0.6.0)

Group related records into named episodes:

```python
from feather_db import EpisodeManager

em  = EpisodeManager(db)
eid = em.begin_episode("onboarding_analysis")
em.add_to_episode(eid, node_id)
ep  = em.get_episode(eid)
em.close_episode(eid)
```

### Triggers & Contradiction Detection (v0.6.0)

```python
from feather_db import WatchManager, ContradictionDetector

wm = WatchManager(db)
wm.watch(namespace="acme", callback=lambda record: print("New:", record.content))

cd = ContradictionDetector(db)
conflicts = cd.check(new_meta)     # returns list of conflicting record IDs
```

### Namespace / Entity / Attributes

```python
meta = feather_db.Metadata()
meta.namespace_id = "acme"
meta.entity_id    = "user_123"
meta.set_attribute("channel", "instagram")   # use this, NOT meta.attributes['k'] = v
val = meta.get_attribute("channel")
```

Domain profiles for typed helpers:

```python
from feather_db import MarketingProfile

p = MarketingProfile()
p.set_brand("nike")
p.set_user("user_8821")
p.set_channel("instagram")
p.set_ctr(0.045)
meta = p.to_metadata()
```

### Graph Visualization

```python
from feather_db.graph import visualize, export_graph

visualize(db, output_path="/tmp/graph.html")          # self-contained D3 HTML
data = export_graph(db, namespace_filter="nike")      # Python dict for D3/Cytoscape
```

---

## Rust CLI

```bash
feather add    --db my.feather --id 1 --vec "0.1,0.2,0.3" --modality text
feather search --db my.feather --vec "0.1,0.2,0.3" --k 5
feather link   --db my.feather --from 1 --to 2
feather save   --db my.feather
```

---

## Performance

Measured 2026-11-08 on an Apple M1 (8 cores, 16 GB), default settings (fsync
on, `ef=50`), one caller at a time unless noted; milliseconds. Full report:
[`bench/reports/e2e_latency_20261108.md`](./bench/reports/e2e_latency_20261108.md).

**An agent turn, end to end** (120 LoCoMo questions with live models:
gemini-embedding-001 for the question, gpt-4o-mini via OpenRouter for the answer):

| stage | p50 | p99 |
|---|---:|---:|
| embed the question (Gemini API) | 504 | 838 |
| **Feather `hybrid_search`, k=30** | **1.3** | **11.0** |
| the same search through the Cloud API (HTTP) | 5.8 | 19.3 |
| render the context | 0.2 | 1.1 |
| LLM answer | 846 | 1,674 |
| **total** | **1,352** | **2,350** |

Feather is 0.15% of the turn; the model calls are the rest.

**Engine** (100k × 768):

| operation | p50 | p99 |
|---|---:|---:|
| vector search, top-10 | 0.24 | 0.98 |
| search with a namespace filter (1% of records) | 0.18 | 0.32 |
| search with decay scoring | 0.23 | 0.43 |
| keyword search (BM25), warm, 100k real conversation turns | 1.85 | 3.61 |
| hybrid search, warm, same corpus | 2.33 | 3.82 |
| `context_chain` k=5, 2 hops | 0.24 | 0.42 |
| `get_metadata` | 0.001 | 0.016 |
| `update_metadata` | 0.16 | 0.26 |
| save, a 15-write commit | 0.52 | 1.23 |
| read-only open of the 100k file | 214 | 273 |

Search throughput: 4.4k queries/s on one thread, 16.1k on four. Ingest:
`add_batch` 4.1k vectors/s, `add` 1.3k/s (each one WAL-logged).

**Recall@10 on real embeddings** (33,849 gemini-embedding-001 vectors, 3072-d,
1,000 held-out queries against exact search):

| ef | recall@10 | p50 | p99 |
|---:|---:|---:|---:|
| 10 | 0.937 | 0.23 | 0.53 |
| **50 (default)** | **0.995** | 0.60 | 0.89 |
| 100 | 0.998 | 1.06 | 1.65 |

**Agent layer** (8,021 memories over three scope levels):

| operation | p50 | p99 |
|---|---:|---:|
| `Pocket.remember` | 1.00 | 2.35 |
| `Pocket.hot()` after a write / memoised | 1.25 / 0.0005 | 1.42 / 0.001 |
| `Pocket.recall` | 1.99 | 2.51 |
| `PacketBuilder.build(required, query)` | 2.90 | 8.34 |
| `FeatherStore.put` (a commit per write) | 0.70 | 3.11 |
| `FeatherStore.search(query)` | 0.63 | 0.98 |
| `AgentMemory.facts.state` | 3.80 | 7.66 |
| `AgentMemory.recall` | 0.36 | 0.45 |

**Cloud API over HTTP** (uvicorn, 20k × 768): `/search` 1.32 / 1.93,
`/hybrid_search` 1.43 / 2.19, `GET /records/{id}` 0.62 / 0.90 (p50 / p99).
`/search` throughput is ~1,560 requests/s from 4 to 16 clients.

**Time travel** (100k records, 200 commits): opening a past commit takes 201 ms
p50 (222 p99) against 214 ms for a plain read-only open; in a kept file, 204 ms
(711 p99, cold page cache). `history()` 2.1 ms, `diff()` of two 100k states
45 ms, 51 bytes a commit. A kept file is a whole copy on disk (416 MB here).

| Setting | Value |
|---|---|
| HNSW | M=16, ef_construction=200, ef=50 |
| Capacity per modality | unbounded (starts at 4096, grows on demand) |
| SIMD | picked at runtime: AVX-512 / AVX2 / SSE2 on x86-64, NEON on arm64 |
| File format | `.feather` v10, append-only: a save writes only what changed |

Reproduce: `benchmarks/e2e_latency.py`, `benchmarks/full_suite.py`,
`benchmarks/api_bench.py`.

---

## Benchmarks

### Memory benchmark — LongMemEval (Xu et al., 2024)

500-question end-to-end memory QA benchmark, the standard for long-term memory in chat assistants. Full report: [`docs/benchmarks/longmemeval.md`](./docs/benchmarks/longmemeval.md).

**0.911 on LongMemEval_S**: the mean of three runs on all 500 questions (0.916, 0.904, 0.914), scored by the paper's official judge (`gpt-4o-2024-08-06`), with gemini-2.5-flash as the reader. Retrieval used Gemini embeddings and the top 20 sessions, packed up to 160k characters. 0 failed questions; the evidence was fully in context for 99.4% of them. On the 400 held-out questions the mean is 0.908. Rerun noise, now measured: a standard deviation of 0.64 points between runs, and two runs disagree on 3.6% of questions; question sampling alone puts a 95% interval of about ±2.5 points on any one run. All three runs were made by `scripts/lme_final.sh` on committed code (`8160eca`), so the harness marks each one *reportable* (bench/README.md, "Evaluation protocol").

| Feather configuration (S, all 500, official judge) | Overall | |
|---|---|---|
| **Single reader + evidence packing** (v0.22 default) | **0.911** | mean of 3 runs; +0.7 vs baseline [−1.7, +3.2], 25 fixed, 23 broken (McNemar p = 0.89, within noise) |
| Single reader, baseline | 0.904 | |
| Routed reader + count map-reduce | 0.794 | a regression: **not shipped**, kept only to reproduce it |

By question type (mean of the three runs; baseline in brackets): knowledge-update 0.957 (0.910), temporal-reasoning 0.930 (0.910), single-session-user 0.976 (0.971), single-session-assistant 0.970 (0.982), multi-session 0.837 (0.827), single-session-preference 0.778 (0.900). Task-averaged 0.908, abstention 0.878. No per-type difference from the baseline is distinguishable from noise; preference (30 questions) moved most.

| System | Variant | Answerer | Overall |
|---|---|---|---|
| **Feather DB v0.22** | **S** | **gemini-2.5-flash** | **0.911** (mean of 3) |
| Supermemory | S | gpt-4o | 0.816 |
| Mem0 (prior algo) | S | gpt-4o-mini | 0.678 |
| Full-context GPT-4o (paper "ceiling") | S | gpt-4o + CoN | 0.640 |
| Zep (graphiti) | S | gpt-4o-mini | 0.638 |
| Full-context GPT-4o-mini | S | gpt-4o-mini | 0.554 |
| Feather DB v0.8.0 + decay (earlier) | S | gemini-2.5-flash | 0.657 |

Other systems' numbers are as they report them, mostly from one run. An earlier single run of this configuration scored 0.912 on an uncommitted tree; the three reportable runs above replace it. `scripts/lme_final.sh` refuses an uncommitted tree or a short or partly failed run, and compares the result with the 0.904 baseline on all 500 and on the held-out 400. Per-question records are in [`bench/results/`](./bench/results/) (`longmemeval__s__20261108_230418__questions.jsonl`, `..._232726__...`, `..._235119__...`).

Reproduce (needs `OPENROUTER_API_KEY` for the official judge and `GOOGLE_API_KEY` or `GEMINI_API_KEY`):
```bash
scripts/lme_final.sh               # all 500, official judge, checked for completeness
scripts/lme_dev.sh                 # the 100-question dev split, for trying a change first
```

### End-to-end quality, this build (dev splits, 2026-11-08)

A check of the current code on the frozen dev splits, one run each. Under the
harness's protocol these are not reportable headline numbers (dev split,
uncommitted tree); the 0.911 above is.

| Benchmark | Configuration | n | Score |
|---|---|---:|---:|
| LongMemEval_S dev | as for the headline runs, official GPT-4o judge | 100 | **0.930** (the earlier 0.912 run also scored 0.930 on these 100, the 0.904 baseline 0.920: the dev questions are easier than the full set) |
| LoCoMo dev | turns + extracted facts, gpt-4o-mini reader, Mem0's judge | 459 | J **81.5** |
| LoCoMo dev, sound key | the same, without the 99 known-wrong answers | 427 | J **83.8** |

LongMemEval dev by type (official judge): temporal 0.962, knowledge-update
0.938, multi-session 0.889, single-session-user 1.000, -assistant 1.000,
-preference 0.667; abstention 0.833. LoCoMo dev by category: single-hop 89.1,
multi-hop 78.5, temporal 74.4, open-domain 42.9. LoCoMo's dev split is 3 of the
10 conversations, so it is not comparable with numbers published on all 10.

**Contradiction**: when a fact about someone changed, does the reader
(gpt-4o-mini) answer with the current value? 20 trials × 8 people × 5
attributes, dense retrieval.

| How the memory was written | Correct |
|---|---:|
| notes + keys | **100%** |
| keyed facts (`AgentMemory.facts`) | 98% |
| free-text notes | 85% |
| free-text notes, written out of order, no valid time | 25% |

**Poisoning**: planted "SYSTEM OVERRIDE" memories from untrusted origins.
With the default trust policy none reaches a prompt as an instruction or
takes a binding slot; it is quoted as data. Whether gpt-4o-mini still obeys it:

| Surface | No defence | Default trust policy |
|---|---:|---:|
| `PacketBuilder` with a query | 25/25 | **0/25** |
| `PacketBuilder`, hot set | 19/20 | 5/20 |
| `Pocket.hot_text()` | 8/20 | **15/20** |

Quoting works in query packets and does not in `hot_text()`. Until that is
fixed, assemble prompts that may hold untrusted memory with `PacketBuilder`
and a query.

Reproduce: `python -m bench run longmemeval ... --split dev` (see
`scripts/lme_dev.sh`), `python -m bench.locomo run --memory index`,
`python -m bench.contradiction --embed gemini --model openrouter:openai/gpt-4o-mini`,
`python -m bench.poisoning --model openrouter:openai/gpt-4o-mini`.

### ANN benchmark — SIFT1M

Standard ANN benchmark. Full sweep results in [`bench/results/`](./bench/results/).

| Config | p50 | p99 | Recall@10 |
|---|---|---|---|
| 500K × 128, ef=10 | 0.07 ms | 0.13 ms | 0.774 |
| 500K × 128, ef=50 (default) | 0.19 ms | 0.23 ms | **0.972** |
| 500K × 128, ef=200 | 0.56 ms | 0.69 ms | 0.998 |

---

## Cloud Deployment (Azure / Docker)

Feather DB ships with a production-ready FastAPI wrapper and the **Atlas-style
admin SPA** (custom HTML + Tailwind + Alpine.js — no build step) you can deploy
on any Linux VM. No clone needed — pull the official image:

```bash
curl -fsSLO https://raw.githubusercontent.com/feather-store/feather/master/feather-api/docker-compose.yml
FEATHER_API_KEY="feather-$(openssl rand -hex 16)" docker compose up -d
```

Or plain `docker run`:

```bash
docker run -d --name feather \
  -p 8000:8000 \
  -v feather-data:/data \
  -e FEATHER_API_KEY="feather-$(openssl rand -hex 16)" \
  ghcr.io/feather-store/feather-api:latest
```

(Contributors building from source: `docker compose -f feather-api/docker-compose.yml up -d --build`.)

| URL | Description |
|-----|-------------|
| `http://<VM_IP>:8000/health` | Health check |
| `http://<VM_IP>:8000/docs` | Swagger / OpenAPI |
| `http://<VM_IP>:8000/admin/` | Admin SPA — namespaces, search, graph, embedding settings |

**Self-hosted walkthrough → [`docs/quickstart.md`](docs/quickstart.md)** ·
**Azure guide → [`docs/deploy-azure.md`](docs/deploy-azure.md)**

> Data is stored in a Docker named volume (`feather-data → /data`) and persists
> across restarts and rebuilds.

---

## Examples

| File | Description |
|------|-------------|
| `examples/context_engine_demo.py` | Self-Aligned Context Engine — all four providers + heuristic fallback |
| `examples/context_graph_demo.py` | Context graph — auto-link, context_chain, D3 HTML export |
| `examples/marketing_living_context.py` | Multi-brand namespace/entity/attribute filtering |
| `examples/feather_inspector.py` | Local HTTP inspector — force graph, PCA scatter, edit, delete |

Run:

```bash
python setup.py build_ext --inplace
python3 examples/context_engine_demo.py

# With a provider:
ANTHROPIC_API_KEY=sk-ant-... python3 examples/context_engine_demo.py
OLLAMA_MODEL=mistral:7b       python3 examples/context_engine_demo.py
```

---

## Architecture

```
[Generic Core — C++17]
feather::DB
  ├── modality_indices_  (unordered_map<string, ModalityIndex>)  — one HNSW per modality
  ├── metadata_store_    (unordered_map<uint64_t, Metadata>)     — shared metadata by ID
  └── Methods: add, search, link, context_chain, auto_link, export_graph_json, …

[Python Layer — feather_db]
  ├── DB, Metadata, ContextType, ScoringConfig
  ├── Edge, IncomingEdge, ContextNode, ContextEdge, ContextChainResult
  ├── FilterBuilder         — fluent search filter helper
  ├── DomainProfile         — generic namespace/entity/attributes base class
  ├── MarketingProfile      — digital marketing typed adapter
  ├── RelType               — standard relationship type constants
  ├── graph.visualize()     — D3 force-graph HTML exporter
  ├── MemoryManager         — health reports, MMR, consolidate, tiering
  ├── WatchManager          — namespace/entity watch callbacks
  ├── ContradictionDetector — conflict detection on ingest
  ├── EpisodeManager        — grouped episode records
  ├── merge()               — merge two .feather files
  ├── LLMProvider / ClaudeProvider / OpenAIProvider / OllamaProvider / GeminiProvider
  ├── ContextEngine         — self-aligned LLM-powered ingestion pipeline
  └── integrations/
      ├── ClaudeConnector   — Claude tool_use with 14 Feather tools
      ├── OpenAIConnector   — OpenAI/Groq/Mistral function_calling
      ├── GeminiConnector   — Gemini function_calling + GeminiEmbedder
      ├── FeatherVectorStore / FeatherMemory / FeatherRetriever  (LangChain)
      ├── FeatherVectorStoreIndex / FeatherReader                 (LlamaIndex)
      └── mcp_server        — feather-serve MCP endpoint

[Rust CLI]
feather-db-cli (FFI via extern "C" from src/feather_core.cpp)
```

---

## File Format

`.feather` **v10** is append-only. Two checksummed 64-byte headers sit at the
front, and each commit appends typed, length-prefixed sections, then switches
to the other header. A save writes only what changed: dirty graph slots,
metadata records and salience updates. A crash mid-commit falls back to the
previous header, and the WAL redoes the commit.

```
[0,    4096)  header slot 0      magic "FEAT" | version 10 | generation | committed end | xxh64
[4096, 8192)  header slot 1
[8192, end)   sections: [type u32][flags u32][payload_len u64][payload][xxh64]
              MODALITY · GRAPH · META · SALIENCE · VECTORS · VERSIONS · IDEMPOTENCY · FULLVEC · COMMIT
```

Each commit ends with a `COMMIT` marker (generation, time, tags), so the bytes
up to any marker are that commit: `DB.open(..., as_of=)` loads them. Builds
without the marker skip it.

Superseded bytes stay until a vacuum: `db.vacuum()`, or automatically once the
file passes `FEATHER_VACUUM_RATIO` × the live size. Read-only handles map the
vector image, so many reader processes share one copy through the page cache.
v3–v9 files load transparently, and the first `save()` rewrites them as v10.
`FEATHER_FORMAT=9` keeps writing v9. The full layout is in
[`include/format_v10.h`](./include/format_v10.h).

---

## Known Limitations

| Issue | Detail |
|-------|--------|
| One writer per file | Writes are serialised within a process; a second writer *process* is refused (readers are unlimited) |
| Soft deletes reclaimed on compaction | `forget()`/`purge()` mark vectors deleted; space is reclaimed by `compact()` or `set_auto_compact(ratio)` |
| int8 quantization (two kinds) | `set_quantized()` = smaller files; `set_int8_ram()` = ~1.7× less RAM (opt-in, lossy) |
| Index capacity | Adaptive (v0.15.3): starts at 4096, grows via `resizeIndex` on demand — no hard cap, RAM tracks the working set |
| `meta.attributes['k'] = v` silent no-op | pybind11 map copy; use `meta.set_attribute(k, v)` |
| Rust CLI missing v0.6.0+ features | namespace/entity/context_chain/integrations/time travel are Python-only |
| History ends at an erase | `erase()` deletes all history; `as_of` sees commits, not writes still only in the WAL |
| Kept history costs a full copy | each file kept by `set_history_retention()` is the whole DB as it was (off by default) |
| BM25 builds on first use | the first keyword or hybrid search after an open builds the index: 1.95 s and +82 MB per process at 100k conversation turns |
| Forgotten records slow search | 10% forgotten records take search p99 from 0.4 ms to 13.9 ms until `compact()` or `set_auto_compact()` |
| Quoted memory in `hot_text()` | a model still obeyed 15 of 20 planted instructions quoted there; query packets held at 0 of 25 |

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

---

## License

MIT — see [LICENSE](LICENSE)

---

## Acknowledgments

- HNSW algorithm: [hnswlib](https://github.com/nmslib/hnswlib)
- Python bindings: [pybind11](https://github.com/pybind/pybind11)
- Rust CLI: [clap](https://github.com/clap-rs/clap)
- Graph visualization: [D3.js](https://d3js.org/)
