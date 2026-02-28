# Feather DB — Technical Reference for AI Assistants

> This document is the single authoritative guide for understanding, modifying, and extending Feather DB. Read it fully before making any changes.

---

## 1. Project Overview

**Feather DB** is a lightweight, embedded vector database written in C++17 with Python and Rust bindings. It is designed as "SQLite for Vectors" — a fast, zero-server, file-based store for AI context and embeddings.

**Core Value Proposition:**
- Sub-millisecond ANN (Approximate Nearest Neighbour) search via HNSW
- Native multimodal support (text, image, audio vectors per entity)
- Embedded Contextual Graph — typed+weighted edges, reverse index, auto-link by similarity
- Adaptive Decay / Living Context — frequently accessed items resist temporal decay
- Namespace + Entity + Attributes — generic partition + subject + KV metadata for any domain
- Graph visualizer — self-contained D3 force-graph HTML, fully offline
- Single file persistence (`.feather` binary format, v5; v3/v4 files load transparently)

**Version:** `0.5.0` (Context Graph + Living Context Engine)

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
│   ├── domain_profiles.py   # DomainProfile base + MarketingProfile adapter
│   ├── graph.py             # export_graph(), visualize(), RelType constants
│   └── d3.min.js            # D3.js v7.9.0 inlined for offline visualization
├── feather-cli/             # Rust CLI crate (feather-db-cli)
│   ├── src/main.rs          # CLI entry point
│   ├── src/lib.rs           # CLI command implementations
│   ├── build.rs             # Rust build script (links C++ core)
│   └── Cargo.toml           # Rust package manifest (v0.5.0)
├── feather-api/             # FastAPI Cloud wrapper
│   ├── app/main.py          # FastAPI app
│   ├── app/db_manager.py    # DB lifecycle management
│   ├── app/models.py        # Pydantic models
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
├── pyproject.toml           # PEP 517 metadata (version: 0.5.0)
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
- **HNSW params**: `M=16`, `ef_construction=200`, `max_elements=1,000,000` per modality index (hardcoded in `get_or_create_index`).
- **`ef` (search beam width)** defaults to `10`. Higher = more accurate but slower.
- **Reverse edge index**: rebuilt from `metadata_store_` edges on every `load()`. Not persisted separately.

### 3.2 File Format (`.feather` binary v5)

```
[magic: 4B = 0x46454154 "FEAT"] [version: 4B = 5]
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
    [dim: 4B] [element_count: 4B]
    for each element:
      [id: 8B] [float32 vector: dim * 4 bytes]
```

**Backward compatibility**: v3 and v4 files load transparently — missing fields default to empty via `if (is.read(...))` guards in `metadata.cpp`.

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
pip install feather-db  # v0.5.0 on PyPI
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

**Crate on Crates.io:** `feather-db-cli` v0.5.0

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

### Enabling SIMD (Performance Optimization)
```python
# In setup.py extra_compile_args:
"-DUSE_AVX", "-march=native", "-ffast-math"
```

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

### Max elements per index
Each modality index is initialized with `max_elements=1,000,000`. Inserting beyond this crashes. Modify `get_or_create_index()` in `include/feather.h` to change this limit.

### File saved on close
`feather::DB::~DB()` calls `save()`. Call `db.save()` explicitly in long-running processes.

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

---

## 11. Known Issues & Limitations

| Issue | Details |
|-------|---------|
| No concurrent writes | HNSW is not thread-safe for simultaneous `addPoint` calls |
| No vector deletion | HNSW marks deletions but data stays |
| `tags_json` is a raw string | Tag filtering uses substring search, not JSON parsing |
| Max 1M vectors per modality | Hardcoded in `get_or_create_index` |
| `meta.attributes['k'] = v` no-op | pybind11 map copy; use `set_attribute()` |
| Load time for large attribute DBs | v4/v5 attribute map deserialization is O(n * attrs) |
| Rust CLI missing v0.5.0 features | namespace/entity/context_chain are Python-only for now |

---

## 12. Testing

```bash
source repro_venv/bin/activate

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
| Phase 6 | Planned | SIMD tuning, parallel ingestion, load-time optimization |
