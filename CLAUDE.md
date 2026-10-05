# Feather DB — Technical Reference for AI Assistants

> This document is the single authoritative guide for understanding, modifying, and extending Feather DB. Read it fully before making any changes.

---

## 1. Project Overview

**Feather DB** is a lightweight, embedded vector database and living context engine written in C++17 with Python and Rust bindings — zero-server, file-based, designed for AI context and embeddings.

**Core Value Proposition:**
- Sub-millisecond ANN (Approximate Nearest Neighbour) search via HNSW
- Native multimodal support (text, image, audio vectors per entity)
- Embedded Contextual Graph — typed+weighted edges, reverse index, auto-link by similarity
- Adaptive Decay / Living Context — frequently accessed items resist temporal decay
- Namespace + Entity + Attributes — generic partition + subject + KV metadata for any domain
- Graph visualizer — self-contained D3 force-graph HTML, fully offline
- Single file persistence (`.feather` binary format, v9 with persisted HNSW graph for fast cold load, optional int8 on-disk and in-RAM; v3–v8 files load transparently)

**Version:** `0.20.0` (Phase 9 — agent memory: pocket tiering, typed agent store, MCP over the real protocol, inter-process locking)

> **Phase 9 (v0.19–v0.20) additions** — the agent-memory layer. Four modules,
> all Python, sitting on the unchanged engine:
> - **`feather_db/pocket.py`** — per-agent working memory with HOT / WARM /
>   CACHE tiers. `hot()` returns a budget-fitted working set ranked by *heat*
>   (`recency × use_weight × inherit_weight × importance`, pinned ⇒ 1.0);
>   `recall()` reaches warm storage the budget could not carry. Scopes are
>   dotted and **inherit**: `org.brand.creative` reads its own scope and every
>   ancestor, decayed by `INHERIT_DECAY` per hop. `hot()` is memoised on
>   `(budget, write-generation, second)` — it is O(namespace) cold (171 ms at
>   10k) and 0.001 ms warm. `recall()` **invalidates** it, because `search()`
>   increments `recall_count`: a read that is really a write.
> - **`integrations/langgraph_store.py`** — `FeatherStore`, a LangGraph
>   `BaseStore` (long-term, cross-thread memory, *not* `BaseCheckpointSaver`).
>   Implements the two abstract methods, `batch`/`abatch`. Beyond the interface
>   it adds `scan()` / `count()` / `newest()`, because `search()` has no
>   ordering and no cheap count.
> - **`integrations/agent_memory.py`** — `AgentMemory`, the five kinds an agent
>   stores, with the lifetime of each enforced. See §5.5.
> - **`integrations/mcp_agent.py`** — `feather-agent`, the MCP server shaped
>   like agent memory (5 tools, 3 prompts, 2 resources). Replaces
>   `mcp_server.py`, which is **broken against SDK 2.0** (`Server.list_tools`
>   was removed, so `create_server()` raises on import).
> - **Inter-process locking** — see §9 "File locking". `db.close()` is now
>   required to release a file.

> **Phase 8 (v0.13–v0.15) additions** — see `include/feather.h`, `feather_db/integrations/`:
> - **Parallel HNSW load** (`parallel_add`): graph rebuilt across a thread pool on
>   open — ~4.7× faster. `FEATHER_LOAD_THREADS` caps it. Lazy `open()` (the modality
>   index is created on first `add()`, not eagerly — `dim()` falls back to `default_dim_`).
> - **Parallel batch ingest**: `add_batch(ids, vecs, metas)` builds the graph in
>   parallel with the GIL released (~3.4× faster bulk insert).
> - **SIMD on x86**: SSE/AVX L2 kernels compiled on x86_64 (runtime-dispatched in
>   `space_l2.h`); arm64 unchanged. `setup.py` is platform-conditional; `FEATHER_SIMD`.
> - **In-RAM int8** (`Int8L2Space`): `set_int8_ram(modality, max_abs)` stores int8 in
>   memory (global scale, integer-L2) → ~1.7× less RAM, **file format v8** (int8-ram
>   flag + scale persisted). `is_int8_ram()`. Lossy; best for embeddings.
> - **MCP connector for Claude** (`feather_db.integrations.mcp_server` / `mcp_remote`):
>   `feather-serve` exposes Feather to Claude Desktop/Code — local `--db` OR remote
>   `--api-url` (Cloud API). `--embed-provider gemini|openai|…` for real client-side
>   embeddings (`integrations/embedders.py`). Cloud API gained `/admin/index_stats`,
>   `/admin/auto_compact`, `/admin/quantize`; `/import` uses `add_batch`.
>
> **Phase 7 (v0.11–v0.12) additions** — see `include/feather.h`:
> - **Secondary metadata indexes** (`ns_index_`/`entity_index_`/`attr_index_`):
>   O(matches) lookups via `ids_in_namespace`/`ids_for_entity`/`ids_with_attribute`/
>   `namespace_size`/`list_namespaces`. Rebuilt on load, maintained on mutation.
> - **Pre-filtered ANN search**: `search()` ranks exactly over the indexed
>   candidate set for namespace/entity/attribute filters → complete top-k.
> - **Incremental auto-compaction**: `set_auto_compact(ratio)`; `compact()` now
>   reclaims forgotten records and is orphan-safe (won't resurrect purged ids).
> - **On-disk int8 quantization**: `set_quantized(modality)` → file format v7,
>   ~3× smaller `.feather`, dequantized to float32 on load (RAM index unchanged).

> **What changed since this doc was first written (v0.5.0 era):** the
> `feather-api/` package was rewritten in v0.10. The Gradio dashboard is gone;
> a custom HTML/Tailwind/Alpine.js SPA at `/admin/` replaces it. A pluggable
> embedding service supports OpenAI / Azure OpenAI / Gemini / Voyage / Cohere
> / Ollama. New modules: `app/embedding.py`, `app/metrics.py`. New endpoints:
> create/delete namespace, schema discovery, hierarchy, top-recalled, graph
> export, context_chain, purge, compact, ingest_text, import, ops_timeseries,
> connection_info, embedding_config. The C++ `save_vectors()` was fixed so
> `forget()`/`purge()` survive save+reload. Phase 9.1 Agentic Context Engine
> (FactExtractor, EntityResolver, OntologyLinker, ContradictionResolver,
> FeedbackLog, IngestPipeline, hierarchy module) shipped in v0.9.0 — see
> `feather_db.extractors`, `.feedback`, `.hierarchy`, `.pipelines`, `.reason`.

---

## 2. Repository Structure

```
feather/
├── include/                 # C++ headers (core logic lives here)
│   ├── feather.h            # ← MAIN DB CLASS (read this first)
│   ├── metadata.h           # Metadata struct + ContextType enum + Edge struct
│   ├── scoring.h            # Scorer + ScoringConfig (decay logic)
│   ├── filter.h             # SearchFilter struct
│   ├── hnswalg.h            # HNSW index algorithm (hnswlib fork)
│   ├── hnswlib.h            # hnswlib base interfaces
│   ├── space_l2.h           # L2/Euclidean distance (SIMD optimized)
│   ├── space_ip.h           # Inner Product distance space
│   ├── bruteforce.h         # Brute force fallback index
│   ├── visited_list_pool.h  # Visited node pool for HNSW search
│   └── stop_condition.h     # Search stop condition interface
├── src/
│   ├── feather_core.cpp     # C-compatible extern "C" wrappers (for Rust CLI)
│   ├── metadata.cpp         # Metadata serialize/deserialize
│   ├── filter.cpp           # Filter logic
│   └── scoring.cpp          # Scorer (thin wrapper; logic in .h)
├── bindings/
│   └── feather.cpp          # pybind11 Python bindings (the Python API bridge)
├── feather_db/
│   ├── __init__.py          # Python package exports (DB, Metadata, RelType, etc.)
│   ├── filter.py            # Python FilterBuilder helper class
│   ├── pocket.py            # ← Pocket: per-agent hot/warm/cache memory, scoped
│   ├── domain_profiles.py   # DomainProfile base + MarketingProfile adapter
│   ├── graph.py             # export_graph(), visualize(), RelType constants
│   ├── d3.min.js            # D3.js v7.9.0 inlined for offline visualization
│   └── integrations/
│       ├── langgraph_store.py  # ← FeatherStore: LangGraph BaseStore
│       ├── agent_memory.py     # ← AgentMemory: the five kinds (§5.5)
│       ├── mcp_agent.py        # ← feather-agent: MCP server for SDK 2.0
│       ├── mcp_server.py       #   BROKEN on SDK 2.0 — superseded, do not use
│       ├── mcp_remote.py       #   MCP against the Cloud API instead of a file
│       └── embedders.py        #   make_embedder() + default_embedder() (env)
├── feather-cli/             # Rust CLI crate (feather-db-cli)
│   ├── src/main.rs          # CLI entry point
│   ├── src/lib.rs           # CLI command implementations
│   ├── build.rs             # Rust build script (links C++ core)
│   └── Cargo.toml           # Rust package manifest (v0.20.0)
├── feather-api/             # FastAPI Cloud wrapper (v0.10 rewrite)
│   ├── app/main.py          # FastAPI app + all /v1/* routes
│   ├── app/db_manager.py    # DB lifecycle management + delete()
│   ├── app/models.py        # Pydantic request/response models
│   ├── app/embedding.py     # Pluggable embeddings (OpenAI/Azure/Gemini/Voyage/Cohere/Ollama)
│   ├── app/metrics.py       # In-memory ring buffer for ops + latency
│   ├── static/admin/        # Atlas-style admin SPA (mounted at /admin/)
│   │   └── index.html       # Single-page app (Tailwind + Alpine.js + D3, no build step)
│   ├── Dockerfile           # Multi-stage Docker build
│   └── docker-compose.yml   # Local compose stack
├── examples/                # Working Python usage examples
│   ├── context_graph_demo.py        # Full context graph demo
│   ├── marketing_living_context.py  # Namespace/entity/attribute filtering
│   └── feather_inspector.py         # Local HTTP inspector app
├── benchmarks/              # Performance benchmarks
│   └── stress_test.py       # 100k vector stress test
├── real_data/               # Real dataset files (not committed)
├── p-test/                  # Rust CLI integration tests
├── setup.py                 # Python build (compiles C++ extension)
├── pyproject.toml           # PEP 517 metadata (version: 0.20.0)
├── MANIFEST.in              # Source distribution manifest
└── CHANGELOG.md             # Version history
```

---

## 3. Core Architecture

### 3.1 The `feather::DB` Class (`include/feather.h`)

This is the single most important file. **All logic flows through here.**

```cpp
namespace feather {
class DB {
private:
    // Each named modality (e.g., "text", "visual", "audio") has its own HNSW index.
    // This is the "Multimodal Pocket" design.
    std::unordered_map<std::string, ModalityIndex> modality_indices_;
    std::string path_;
    // Centralized metadata store keyed by entity ID (uint64_t).
    // A record can span multiple modality indices but shares ONE Metadata entry.
    std::unordered_map<uint64_t, Metadata> metadata_store_;
    // Reverse edge index (built in-memory on load)
    std::unordered_map<uint64_t, std::vector<IncomingEdge>> incoming_index_;
};
}
```

**Key Design Decisions:**
- **Multimodal via multiple HNSW indices**: each call to `add(id, vec, meta, modality)` routes to the correct named `ModalityIndex`. New modalities are created on-demand.
- **Shared metadata by ID**: a single `Metadata` object tracks all cross-modal data (edges, recall_count, importance, timestamps) for a given entity ID.
- **HNSW params**: `M=16`, `ef_construction=200`. `max_elements` is **adaptive** (v0.15.3): each modality index starts at `INITIAL_MAX_ELEMENTS = 4096` and doubles via `resizeIndex()` on demand (`reserve()` is called before every insert), so RAM tracks the working set instead of preallocating 1M elements per index.
- **`ef` (search beam width)** defaults to `10`. Higher = more accurate but slower.
- **Reverse edge index**: rebuilt from `metadata_store_` edges on every `load()`. Not persisted separately.

### 3.2 File Format (`.feather` binary v9)

```
[magic: 4B = 0x46454154 "FEAT"] [version: 4B = 8]
--- Metadata Section ---
[meta_count: 4B]
  for each record:
    [id: 8B]
    [timestamp: 8B] [importance: 4B] [type: 4B]
    [source_len: 2B][source: N]
    [content_len: 4B][content: N]
    [tags_len: 2B][tags_json: N]
    [edge_count: 2B]
      for each edge: [target_id: 8B][rel_len: 1B][rel_type: N][weight: 4B]
    [recall_count: 4B] [last_recalled_at: 8B]
    [ns_len: 2B][namespace_id: N]
    [eid_len: 2B][entity_id: N]
    [attr_count: 2B]
      for each attr: [key_len: 2B][key: N][val_len: 4B][val: N]
--- Modality Indices Section ---
[modal_count: 4B]
  for each modality:
    [name_len: 2B] [name: N bytes]
    [dim: 4B] [quantized: 1B]                          # on-disk int8 flag (v7+)
    [int8_ram: 1B] [scale: 4B if int8_ram]             # in-RAM int8 flag + scale (v8+)
    [persist_graph: 1B]                                # v9: 1 → graph blob follows
    if persist_graph == 1:                             # fast path (clean, non-on-disk-quant)
      [HNSW graph blob via saveIndexStream]            # header + base layer (vectors) + link lists
    else:                                              # rebuild path (dirty DB / on-disk quant)
      [element_count: 4B]
      for each element:
        [id: 8B] then, per `quantized`:
          0 → [float32 vector: dim * 4 bytes]
          1 → [scale: 4B float] [int8 vector: dim bytes]  # set_quantized() — ~3x smaller
```

**Backward compatibility**: v3–v8 files load transparently — the `quantized` flag is read for v7+, the `int8_ram` flag + scale for v8+, the `persist_graph` flag for v9+ (`if (version >= 9)`); missing metadata fields default to empty via `if (is.read(...))` guards in `metadata.cpp`. When `persist_graph` is set, `load()` restores the graph via `loadIndexStream` (no rebuild) and calls `setEf(DEFAULT_EF)`; otherwise it reads vectors and rebuilds the HNSW graph (parallel). On-disk int8 vectors are dequantized to float32 on load; in-RAM int8 modalities persist/restore their int8 base layer directly (the graph blob is storage-agnostic — reconstructed against the matching `Int8L2Space`).

**When is the graph persisted?** Only when the index holds exactly the live set (`live_count == total`, i.e. no `forget()`/`purge()` nodes pending) **and** the modality isn't on-disk-quantized. A DB with pending deletions falls back to the rebuild path; `compact()` clears the dead nodes and re-enables fast load. The trade-off is ~25% larger files (the link lists) for a 5–25× faster cold load.

---

## 4. Data Structures

### 4.1 `Metadata` (`include/metadata.h`)

```cpp
struct Edge {
    uint64_t    target_id;
    std::string rel_type;  // e.g. "caused_by", "supports", free-form strings ok
    float       weight;    // [0.0-1.0]
};

struct IncomingEdge {
    uint64_t    source_id;
    std::string rel_type;
    float       weight;
};

struct Metadata {
    int64_t  timestamp;       // Unix timestamp of creation
    float    importance;      // Relevance weight [0.0–1.0], default 1.0
    ContextType type;         // FACT | PREFERENCE | EVENT | CONVERSATION
    std::string source;       // Origin identifier (e.g., "gpt-4o", "user")
    std::string content;      // Human-readable text content
    std::string tags_json;    // JSON array string: '["tag1","tag2"]'

    // Context Graph (v0.5.0)
    std::vector<Edge> edges;  // Typed weighted outgoing edges (replaces flat links)

    // Living Context
    uint32_t recall_count;        // Incremented on every search hit (via touch())
    uint64_t last_recalled_at;    // Unix timestamp of last retrieval

    // Namespace / Entity / Attributes (v0.4.0)
    std::string namespace_id;                            // partition key
    std::string entity_id;                               // subject key
    std::unordered_map<std::string,std::string> attributes; // domain KV pairs
};
```

**CRITICAL GOTCHA**: `meta.attributes['k'] = v` silently does nothing in Python (pybind11 returns a copy of the map). Always use `meta.set_attribute(key, value)` and `meta.get_attribute(key)`.

**Backward-compat**: Python `meta.links` property still works (returns list of target IDs from edges).

### 4.2 `ContextType` Enum

| Value          | int | Meaning                       |
|----------------|-----|-------------------------------|
| `FACT`         | 0   | Static knowledge               |
| `PREFERENCE`   | 1   | User preference or setting     |
| `EVENT`        | 2   | Time-bound occurrence          |
| `CONVERSATION` | 3   | Dialog turn or message         |

### 4.3 `SearchFilter` (`include/filter.h`)

All fields are `std::optional` — only set fields are evaluated.

```cpp
struct SearchFilter {
    optional<vector<ContextType>> types;
    optional<string>              source;
    optional<string>              source_prefix;
    optional<int64_t>             timestamp_after;
    optional<int64_t>             timestamp_before;
    optional<float>               importance_gte;
    optional<vector<string>>      tags_contains;
    // v0.4.0 additions:
    optional<string>              namespace_id;       // exact namespace match
    optional<string>              entity_id;          // exact entity match
    optional<unordered_map<string,string>> attributes_match; // all KV must match
};
```

### 4.4 `ScoringConfig` + `Scorer` (`include/scoring.h`)

The **Adaptive Decay** formula:

```
stickiness      = 1 + log(1 + recall_count)    # grows with access frequency
effective_age   = age_in_days / stickiness      # sticky items age slower
recency         = 0.5 ^ (effective_age / half_life_days)
final_score     = ((1 - time_weight) * similarity + time_weight * recency) * importance
```

**Default config**: `half_life=30 days`, `time_weight=0.3`, `min_weight=0.0`

---

## 5. Python API (`feather_db`)

### Install
```bash
pip install feather-db  # v0.20.0 on PyPI (binary wheels, cp39-cp314)
# or from source:
python setup.py build_ext --inplace
```

### Python Package Exports

```python
from feather_db import (
    DB, ContextType, Metadata, ScoringConfig,
    Edge, IncomingEdge,
    ContextNode, ContextEdge, ContextChainResult,
    FilterBuilder,
    DomainProfile, MarketingProfile,
    visualize, export_graph, RelType,
)
```

### Quick Reference

```python
import feather_db, numpy as np, time

db = feather_db.DB.open("my_context.feather", dim=768)

# --- Metadata ---
meta = feather_db.Metadata()
meta.timestamp = int(time.time())
meta.importance = 0.9
meta.type = feather_db.ContextType.FACT
meta.source = "pipeline-v1"
meta.content = "User prefers dark mode"
meta.namespace_id = "acme"
meta.entity_id = "user_123"
meta.set_attribute("channel", "instagram")  # ← use this, NOT meta.attributes['k'] = v

db.add(id=42, vec=np.random.rand(768).astype(np.float32), meta=meta)

# --- Multimodal ---
db.add(id=42, vec=np.random.rand(512).astype(np.float32), modality="visual")

# --- Search ---
results = db.search(query_vec, k=10)
results = db.search(query_vec, k=5, modality="visual")

# --- Filtered search ---
from feather_db import FilterBuilder
f = FilterBuilder().namespace("acme").entity("user_123").attribute("channel", "instagram").build()
results = db.search(query_vec, k=10, filter=f)

# --- Scored search (adaptive decay) ---
cfg = feather_db.ScoringConfig(half_life=30.0, weight=0.3, min=0.0)
results = db.search(query_vec, k=10, scoring=cfg)

# SearchResult fields: r.id, r.score, r.metadata
for r in results:
    print(r.id, r.score, r.metadata.content)

# --- Typed graph edges ---
db.link(from_id=1, to_id=2, rel_type="caused_by", weight=0.9)
edges    = db.get_edges(1)      # list[Edge]
incoming = db.get_incoming(2)   # list[IncomingEdge]

# --- Auto-link by similarity ---
db.auto_link(modality="text", threshold=0.85, rel_type="related_to")

# --- Context chain (vector search + BFS graph expansion) ---
result = db.context_chain(query=query_vec, k=5, hops=2, modality="text")
for node in result.nodes:
    print(node.id, node.score, node.hop_distance)

# --- Export / import ---
json_str = db.export_graph_json(namespace_filter="acme", entity_filter="")
vec      = db.get_vector(id=42, modality="text")   # returns np.ndarray
ids      = db.get_all_ids(modality="visual")        # returns list[int]

# --- Metadata-only updates (no HNSW touch) ---
db.update_metadata(id=42, meta=new_meta)
db.update_importance(id=42, importance=0.95)

# --- Salience ---
db.touch(id=42)           # manual boost; called automatically on search hits
meta = db.get_metadata(42)

db.save()
```

### 5.5 Agent memory — `Pocket`, `FeatherStore`, `AgentMemory`

Three layers, each usable alone:

```python
# ── Pocket: one agent's working set, budget-fitted ───────────────────────
from feather_db.pocket import pocket
pkt = pocket(db, ("org", "brand", "creative"), budget_tokens=4000)
pkt.remember("tone", "captions stay lowercase", pinned=True)
pkt.hot()            # what to carry now, ranked by heat, trimmed to budget
pkt.hot_text()       # the same, as a prompt-ready string
pkt.recall("tone")   # reaches WARM storage the budget could not carry
pkt.stats()          # hot/warm/cache counts, token usage, inherited scopes

# ── FeatherStore: LangGraph long-term memory ─────────────────────────────
from feather_db.integrations.langgraph_store import FeatherStore
store = FeatherStore("agent.feather", dim=768)       # embed= defaults to env
graph = builder.compile(store=store)                 # ← the whole integration

# ── AgentMemory: the five kinds, lifetimes enforced ─────────────────────
from feather_db.integrations import AgentMemory
mem = AgentMemory(store, org="hawky", user="user_842", agent="creative", brand="nike")
mem.preferences.set("comms", "prefers written async")   # revised in place
mem.episodes.record("report shipped late", outcome="miss")  # append-only
mem.facts.state("positioning", "premium")               # corrected, keeps history
mem.procedures.define("weekly", ["pull", "send"])       # versioned as name@N
mem.entities.enrich("nike", {"industry": "apparel"})    # merged, org-wide
mem.observations.note("style", "decisive", evidence=["comms"])
mem.about()                                             # all kinds, one prefix read
mem.recall("how to contact", kinds=["preferences"])
```

```python
# ── PacketBuilder: required context guaranteed, the rest budgeted ───────
from feather_db import PacketBuilder, RequiredRule
builder = PacketBuilder(pkt, budget_tokens=4000,
                        resolve_required=mongo_rule_lookup)   # optional
packet = builder.build(required=["compliance"], query="which hook")
if not packet.may_mutate:
    raise PolicyBlocked(packet.why_blocked())     # fail closed
prompt = packet.text
packet.manifest()    # refs + versions + omissions, for replay
```

**`hot()` is not a policy path.** It fits a budget by heat, so a binding rule
can be crowded out by ordinary notes — silently. Anything that gates a mutation
must go through `PacketBuilder`, which puts required refs first, judges the
required set **whole**, and raises `RequiredContextUnavailable` rather than
returning a partial constraint set. `allow_degraded=True` is for read-only help
only.

**`resolve_required=` makes an external store authoritative, with no pocket
fallback.** That is deliberate: Feather is derived retrieval and indexing is
asynchronous, so a pocket fallback would (a) turn index lag into "rule missing"
and block every mutation, and (b) let a stale Feather copy of a just-changed
rule be enforced as binding. A key the resolver does not return is missing.

| kind | write semantics | get it wrong and… |
|---|---|---|
| `preferences` | upsert by key | six contradictory answers, none current |
| `episodes` | append-only, key embeds the event time | the history that made it an episode is gone |
| `facts` | upsert, previous kept in `supersedes` | cannot answer "since when" |
| `procedures` | new version each write, `name@N` retained | a regression cannot be diffed against what worked |
| `entities` | merge into existing attributes | additions overwrite instead of accumulating |
| `observations` | cites `evidence` keys | a drawn conclusion cannot be traced or retracted |

**Namespace ordering is load-bearing.** Search takes a *prefix*, so
`(org, subject, kind)` makes "everything about this user" a prefix read while
`(org, kind, subject)` makes it a filter over every user in the org. Same data;
the second does not scale. `AgentMemory` owns the ordering so a caller cannot
invert it.

**Never use `store.search(ns, query=None, limit=n)` to mean "the newest n".**
`search` has no ordering, so it returns an arbitrary `n`; sorting those is
sorting the wrong set. At 300 episodes it returned 299, 298, 296, 295, 293 —
silently skipping two. Use `store.newest(ns, n, key=…)`, and `store.count(ns)`
rather than `len(search(...))`. `count` cannot use the engine's
`namespace_size()`, which counts tombstones until compaction.

### Domain Profiles

```python
from feather_db import MarketingProfile

p = MarketingProfile()
p.set_brand("nike")
p.set_user("user_8821")
p.set_channel("instagram")
p.set_ctr(0.045)
p.set_roas(3.2)
meta = p.to_metadata()
```

### Graph Visualization

```python
from feather_db.graph import visualize, export_graph

visualize(db, output_path="/tmp/graph.html")  # self-contained D3 HTML
data = export_graph(db, namespace_filter="nike")  # Python dict
```

---

## 6. Rust CLI (`feather-db-cli`)

**Crate on Crates.io:** `feather-db-cli` v0.20.0

```bash
feather add    --db my.feather --id 1 --vec "0.1,0.2,0.3" --modality text
feather search --db my.feather --vec "0.1,0.2,0.3" --k 5
feather link   --db my.feather --from 1 --to 2
feather save   --db my.feather
```

The Rust CLI calls the `extern "C"` C ABI from `src/feather_core.cpp` via the FFI bridge in `build.rs`. The CLI does **not** yet expose v0.4.0/v0.5.0 graph features (namespace, attributes, context_chain) — those are Python-only for now.

---

## 7. pybind11 Bindings (`bindings/feather.cpp`)

Key points:
- `DB` is bound with `py::nodelete` to prevent Python from double-deleting (destructor calls `save()`).
- `add()` accepts `py::array_t<float>` (NumPy arrays) and copies data into `std::vector<float>`.
- `search()` accepts optional raw pointers to `SearchFilter` and `ScoringConfig`.
- `get_vector()` returns `py::array_t<float>`.
- `meta.attributes` map mutation via `meta.attributes['k'] = v` silently does nothing — pybind11 returns a copy. Use `meta.set_attribute(k, v)`.

---

## 8. Build System

### Python Extension
```bash
# Development build (inplace, fastest)
python setup.py build_ext --inplace

# Production wheel
python setup.py sdist bdist_wheel
```

`setup.py` compiles:
- `bindings/feather.cpp` (pybind11 entry point)
- `src/filter.cpp`, `src/metadata.cpp`, `src/scoring.cpp`
- Flags: `-O3 -std=c++17`

### SIMD (runtime-dispatched since 0.19)
Distance kernels live in `include/feather_simd.h` and are picked at RUNTIME from
CPUID: AVX-512F / AVX2+FMA / SSE2 / scalar on x86-64, NEON on arm64. Wide
kernels use per-function `FEATHER_TARGET("avx2,fma")` attributes — never add a
global `-mavx`/`-march` to the default build (it makes wheels crash on older
CPUs). Build modes: `FEATHER_SIMD=auto` (default) | `native` | `none`;
`FEATHER_SIMD_RUNTIME=scalar|sse|avx2|avx512` caps the level for A/B tests;
`feather_db.core.simd_info()` reports it. A new kernel needs a numpy
equivalence case in `tests/test_simd.py` (it runs every level the CPU has).

### Rust CLI
```bash
cd feather-cli
cargo build --release
cargo publish  # requires cargo login
```

---

## 9. Common Patterns & Gotchas

### Multimodal pockets have independent dims
Each modality gets its own HNSW index. You cannot search text vectors against the visual index.

```python
db.add(id=1, vec=np.rand(768), modality="text")    # 768-dim
db.add(id=1, vec=np.rand(512), modality="visual")  # 512-dim, independent
```

### `meta.attributes` pybind11 copy gotcha
```python
# WRONG — silently does nothing
meta.attributes["channel"] = "instagram"

# CORRECT
meta.set_attribute("channel", "instagram")
value = meta.get_attribute("channel", default="")
```

### `touch()` is called automatically on search
Every `search()` increments `recall_count` for all returned records. Call `touch()` manually only to boost salience outside of search.

### Import cache when recompiling C++ `.so`
`importlib.reload()` does NOT reload compiled `.so` files. Start a fresh Python process to pick up recompiled bindings.

### `load_complete_` — why the destructor is guarded
`~DB()` only checkpoints when `load_complete_` is true, and `save_vectors()`
throws `std::logic_error` otherwise. `DB::open()` holds a `unique_ptr`, so a
throw inside `load_vectors()` unwinds directly into the destructor — without the
flag, a file that failed to parse got a half-parsed fragment written back over
it and its WAL cleared. If you add an early return or a new throw site in the
load path, `load_complete_ = true` must stay the **last** statement of
`load_vectors()`; anything after it is code that ran on a DB already declared
loaded.

Corollary for new format-reading code: every count read off disk is a loop trip
count. Bound it against the bytes actually remaining before looping (see
`meta_count` at `feather.h:969` and the `dim`/`element_count` guards below it),
or a corrupt file becomes a hang rather than an error.

### WAL durability — what the guarantee actually is
The WAL is format **v2**: an 8-byte header (`FWAL` + version) then records of
`[op:1][id:8][plen:4][payload][crc32:4]`. Two rules if you touch it:

- **Verify before applying.** `replay_wal()` checks each record's CRC and stops
  at the first mismatch. A length check alone only catches a short tail; it
  cannot see a record that is the right size with the wrong bytes, and replaying
  one writes garbage into the next checkpoint.
- **Append inside the lock, wait for durability outside it (group commit, 0.19).**
  `wal_append()` writes + flushes and returns the record's LSN; call it while
  holding `mutex_` exclusively so WAL order == apply order. After releasing
  `mutex_`, call `wal_wait_durable(lsn)` — a leader fsyncs everything appended
  so far and releases every writer it covers. A new mutation follows this shape:

  ```cpp
  const std::string payload = encode_...(...);          // no lock
  uint64_t lsn = 0;
  {
      std::unique_lock<RWMutex> lock(mutex_);
      ensure_open();
      /* validate BEFORE logging */  lsn = wal_append(WalOp::X, id, payload);
      /* apply in memory */
      if (wal_strict()) wal_wait_durable(lsn);           // FEATHER_WAL_STRICT=1
  }
  wal_wait_durable(lsn);                                // never inside mutex_ otherwise
  ```

  Bulk mutations append all records and wait once on the last LSN. Never fsync
  (or call `wal_wait_durable`) inside the exclusive lock outside strict mode:
  it blocks every reader for the disk flush (measured −68% reads, 1 writer).

v1 WALs (≤0.17.0, no header, no CRC) still replay — the header sniff is exact
because `'F'` cannot be a v1 opcode. Don't remove that path while any deployed
build can still leave a v1 WAL behind. `FEATHER_WAL_SYNC=0` disables fsync.

**Checkpoints (0.19).** `save()` writes the snapshot under the SHARED lock, then
`wal_rotate()` renames `<wal>` → `<wal>.old`, and the fsync + atomic replace of
the base file happen after the lock is released; `<wal>.old` is deleted last.
`load_vectors()` replays `<wal>.old` then `<wal>`. Replay must stay idempotent
(a crash after the rename replays `<wal>.old` over a base that already has it).
Every new WalOp must be replayable twice without changing the result.

### Locking model — which lock does your new method need?
`mutex_` is a `FairSharedMutex` (phase-fair, atomic reader fast path; alias
`RWMutex`; any change to it must pass `tests/cpp/lock_stress.cpp`, including
under `-fsanitize=thread`) — glibc's
`std::shared_mutex` prefers readers and starved writers under read load. It is
**not recursive**: never take it shared while already holding it shared (a
waiting writer would deadlock you). Lock order is `save_mutex_` → `mutex_` →
`wal_mutex_`; never join the compactor thread while holding `mutex_`.
Take `std::shared_lock<RWMutex>` if the method only
reads (all const accessors, `search`, `keyword_search`, `hybrid_search`,
`context_chain`, `save`'s snapshot); take `std::unique_lock<RWMutex>` if it mutates
`metadata_store_`, any derived index, or the HNSW graph — including anything
that can trigger `resizeIndex`, which is not thread-safe. The one exception is
salience: `Metadata::recall_count` / `last_recalled_at` are `mutable
std::atomic`, so `touch_nolock()` is `const` and legal under a shared lock.
Don't mutate anything else from a shared section.

### Editing headers? The build may not notice (fixed, but know why)
Nearly the whole engine is in `include/*.h`. setuptools only stat-checks the
listed `.cpp` sources, so before `depends=glob("include/*.h")` was added to
`setup.py`, an incremental `build_ext` after a header edit reused stale objects
and produced a build that looked fresh but contained none of the changes. If you
ever see a change "not take effect", `rm -rf build` and rebuild.

### Max elements per index (adaptive since v0.15.3)
Each modality index starts at `INITIAL_MAX_ELEMENTS = 4096` and grows on demand: `reserve()` calls `resizeIndex()` (doubling) before any insert exceeds capacity. There is no hard ceiling — indices grow to fit whatever you store. `resizeIndex` is **not thread-safe**, so `reserve()` runs before `parallel_add` for the full batch size, never from inside it.

### File locking — `close()` is now required (v0.20.0)
`DB.open()` takes an **exclusive** advisory `flock` on `<path>.lock` for the
handle's lifetime. A second *process* is refused, naming the holder's pid.
`DB.open(..., read_only=True)` takes **no lock at all** and is never blocked,
even while a writer holds the file. Every mutation on that handle raises at the
call site.

That is safe, not sloppy: `save_vectors()` writes `<path>.tmp` and `rename()`s
it over the original, so `open()` returns either the complete old inode or the
complete new one, and a reader keeps reading its own consistent snapshot until
it reopens. WAL records carry a CRC32 and replay stops at the first mismatch, so
replaying a log the writer is appending to yields a clean prefix; a read-only
handle never calls `save_vectors()`, so it never clears the writer's WAL. What a
reader gives up is **freshness, not integrity** — reopen to advance.

A shared lock was the first implementation and was wrong: `LOCK_SH` conflicts
with the writer's `LOCK_EX`, so every reader was refused while the writer merely
held the handle open. That blocks the one topology this is for — a single writer
service plus many agent processes reading the same file.

**What this does NOT do:** let two processes write one file. It cannot. Each DB
holds the whole dataset in RAM and `save_vectors()` rewrites the file from that
view, so serialising the saves would only decide *whose* records vanish
(measured before the lock: 20 of 20 lost, both processes exiting 0). Concurrent
writing needs merge-on-save or a paged store — a format change.

Three consequences you will hit:

1. **`db.close()` is required, not decorative.** The Python binding holds `DB`
   with `py::nodelete`, so `~DB()` never runs from Python. Without `close()` the
   lock lives until process exit and the file cannot be handed to a subprocess.
   `del db` does **not** release it. `with DB.open(...) as db:` works.
2. **The lock is reentrant within one process.** A strict lock would be a
   breaking change for exactly the reason above — it refused programs their own
   file, and broke 26 tests + 34 errors. Same-process double-open therefore
   remains the documented footgun: use **one handle** and separate agents by
   namespace/scope, or `mcp_agent.build(db=...)`.
3. **`FEATHER_LOCK=0`** disables it (flock is unreliable on NFS and some overlay
   mounts). The caller then owns the consequence.

Keep the registry key in `lock_key()` on `realpath(dirname) + basename`, never
on the file: `realpath` only resolves an existing file, and on macOS rewrites
`/var/…` → `/private/var/…`, so keying on the file gave a different key before
and after the first save — reentrancy missed and the handle reported *itself* as
"another process (pid \<ourselves\>)".

`close(save=False)` releases the handle WITHOUT checkpointing, so the WAL is
kept. That is how a test simulates a crash. `close()` also stops the background
compactor first, so call it before handing a file to another process.

### File saved on close
`feather::DB::~DB()` calls `save()`. Call `db.save()` explicitly in long-running
processes, and `db.close()` when done with the file.

### Dangling edges in `export_graph_json`
If a record exists in the edge list but not in the metadata store (e.g., added without metadata), `export_graph_json` filters those dangling edges automatically via the `exported_ids` set.

---

## 10. Adding New Features — Checklist

When adding a new feature to Feather DB, touch these files **in order**:

1. **`include/metadata.h`** — Add new field to `Metadata` struct
2. **`src/metadata.cpp`** — Update `serialize()` and `deserialize()`
3. **`include/feather.h`** — Add new method to `DB` class
4. **`src/feather_core.cpp`** — Add `extern "C"` wrapper for Rust/FFI
5. **`bindings/feather.cpp`** — Expose to Python via pybind11
6. **`feather_db/__init__.py`** — Export from Python package
7. **`feather-cli/src/lib.rs`** — Add CLI command in Rust
8. **`examples/`** — Add a usage example
9. **`CHANGELOG.md`** — Document the change
10. **`scripts/sync-cpp.sh`** — run it if you touched `include/` or `src/`; CI
    fails on drift between `feather-cli/cpp/` and the originals
11. **`./scripts/verify.sh`** — the gate. Nothing ships red.

If the feature is agent-facing, it probably belongs in `feather_db/pocket.py`,
`integrations/agent_memory.py` or `integrations/mcp_agent.py` rather than on
`DB` — the engine stays generic and the memory model lives above it.

---

## 11. Known Issues & Limitations

| Issue | Details |
|-------|---------|
| ~~Two processes destroy each other's writes~~ | **Fixed in v0.20.0**: enforced single-writer / many-reader via `flock`. Still NOT concurrent multi-process *writing* — the second writer is refused, not queued. `db.close()` required to release. |
| Writes are serialized (reads are not) | `mutex_` is a `std::shared_mutex`: retrieval + const accessors take it **shared** (concurrent), mutations take it **exclusively** (one writer at a time, and no reader runs alongside). Salience (`recall_count`/`last_recalled_at`) is `mutable std::atomic` so a query can record its hit under the shared lock. |
| Soft deletes reclaimed on compaction | `forget()`/`purge()` mark vectors deleted; space is reclaimed by `compact()` or `set_auto_compact(ratio)` (Phase 7) |
| int8 quantization (two modes) | `set_quantized()` shrinks the file; `set_int8_ram()` shrinks RAM (~1.7×, opt-in, lossy) via `Int8L2Space` |
| `tags_json` is a raw string | Tag filtering uses substring search, not JSON parsing |
| ~~Max 1M vectors per modality~~ | **Fixed in v0.15.3**: index capacity is adaptive (starts 4096, grows via `resizeIndex`); no hard cap |
| `meta.attributes['k'] = v` no-op | pybind11 map copy; use `set_attribute()` |
| Load time for large attribute DBs | v4/v5 attribute map deserialization is O(n * attrs); namespace/attribute *lookups* are O(matches) via secondary indexes (Phase 7) |
| Rust CLI missing v0.5.0 features | namespace/entity/context_chain are Python-only for now |
| Rust CLI now participates in locking | if a long-running Python process holds a DB, `feather add --db that.feather` is refused rather than silently discarding one side |
| `mcp_server.py` is broken on SDK 2.0 | `Server.list_tools` was removed; `create_server()` raises on import. Use `mcp_agent.py` (`feather-agent`) |
| BM25 fallback does no stemming | without an embedder, `allergic` matches but `allergy` does not, and the key is not indexed. A paraphrased query returns "nothing found", indistinguishable from "never stored" — set `FEATHER_EMBED_PROVIDER` for anything relying on paraphrase |
| `all()` ties within one second | the engine timestamp is whole seconds, so items written in the same second order arbitrarily. `_Episodes` overrides this using the event time in the key; sub-second ordering needs format v10's second timestamp |

---

## 12. Testing

```bash
# One command, the pre-release gate — build, suite, format scripts, the four
# data-loss regressions, MCP over the protocol, vendored C++ sync, version
# consistency, release prereqs, live API contract. Exits non-zero on anything.
./scripts/verify.sh            # or `./scripts/verify.sh quick`

pytest tests -q --deselect tests/test_engine.py::TestProviderInterface::test_provider_str
# 488 passed as of v0.20.0. The deselected test needs `openai` installed.

# The agent-memory suites specifically:
pytest tests/test_pocket.py tests/test_agent_memory.py \
       tests/test_langgraph_store.py tests/test_file_locking.py -q
# MCP over a real ClientSession (not direct calls) — must RUN, not skip:
pytest tests/test_mcp_protocol.py tests/test_mcp_personas.py -q

source repro_venv/bin/activate

python3 examples/langgraph_agent_memory.py   # memory added to an agent in 1 line
python3 examples/context_graph_demo.py
python3 examples/marketing_living_context.py
python3 examples/feather_inspector.py   # local inspector at http://localhost:7777

python3 benchmarks/stress_test.py

cd p-test && ./run_tests.sh   # Rust CLI tests
```

---

## 13. Phase Roadmap

| Phase | Status | Highlights |
|-------|--------|-----------|
| Phase 1 | Done | Basic HNSW, L2 search, `.feather` file format |
| Phase 2 | Done | Context Engine: metadata, types, time-decay scoring, filtered search |
| Phase 3 | Done | Multimodal pockets, graph links, adaptive decay |
| Phase 4a | Done | Generic namespace/entity/attributes (v0.4.0) |
| Phase 4b | Done | FastAPI + Docker cloud wrapper in `feather-api/` |
| Phase 5 | Done | Typed edges, reverse index, auto_link, context_chain, D3 visualizer (v0.5.0) |
| Phase 6 | Done | LLM connectors, MCP, LangChain/LlamaIndex, ContextEngine (v0.6–v0.9) |
| Cloud | Done | FastAPI admin SPA + pluggable embeddings (v0.10 Cloud Edition) |
| Phase 7 | Done | Secondary metadata indexes, pre-filtered ANN, auto-compaction (v0.11.0), on-disk int8 quantization / format v7 (v0.12.0) |
| Phase 8 | Done | Parallel load + `add_batch`, SIMD-on-x86 (v0.13.0), in-RAM int8 / format v8 (v0.15.0), Claude MCP connector + real embedders (v0.14–v0.15), adaptive index capacity (v0.15.3), persisted HNSW graph / format v9 (v0.16.0) |
| Phase 9 | Done | Agent memory: `Pocket` hot/warm/cache tiering, `FeatherStore` (LangGraph `BaseStore`), `AgentMemory`'s five kinds, `feather-agent` MCP server for SDK 2.0 tested over the real protocol, inter-process file locking (v0.19–v0.20) |
| Phase 10 | Planned | Format v10 (length-prefixed records, deletion flags, metric tag, typed attributes, bitemporal, footer) — which also turns three `FeatherStore` Python workarounds into engine features: namespace-prefix indexing, typed attributes for real comparisons, a second timestamp. Plus multi-tenant auth, in-RAM int8 SIMD distance |
