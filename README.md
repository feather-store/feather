# Feather DB

**Embedded vector database + living context engine**

*Part of [Hawky.ai](https://hawky.ai) — AI-Native Digital Marketing OS*

[![PyPI](https://img.shields.io/pypi/v/feather-db)](https://pypi.org/project/feather-db/)
[![Crates.io](https://img.shields.io/crates/v/feather-db-cli)](https://crates.io/crates/feather-db-cli)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Website](https://img.shields.io/badge/website-getfeather.store-blue)](https://www.getfeather.store/)

Feather DB is an embedded vector database and living context engine — zero-server, file-based, with a built-in knowledge graph and adaptive memory decay. No separate database server required.

---

## What's Inside (v0.5.0)

| Capability | Description |
|------------|-------------|
| **ANN Search** | Sub-millisecond approximate nearest-neighbor search via HNSW |
| **Multimodal Pockets** | Text, image, audio vectors stored per entity under a single ID |
| **Context Graph** | Typed + weighted edges, reverse index, auto-link by similarity |
| **Living Context** | Recall-count-based sticky memory — frequently accessed items resist decay |
| **Namespace / Entity / Attributes** | Generic partition + subject + KV metadata for any domain |
| **Graph Visualizer** | Self-contained D3 force-graph HTML — fully offline, no CDN |
| **Single-file persistence** | `.feather` binary format (v5); v3/v4 files load transparently |

---

## Installation

```bash
pip install feather-db
```

CLI (Rust):

```bash
cargo install feather-db-cli
```

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

## Core Features

### Multimodal Pockets

Each named modality gets its own independent HNSW index with its own dimensionality. A single entity ID can hold text, visual, and audio vectors simultaneously.

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

# Link records with typed relationships
db.link(from_id=1, to_id=2, rel_type=RelType.CAUSED_BY, weight=0.9)
db.link(from_id=1, to_id=3, rel_type=RelType.SUPPORTS,  weight=0.7)

# Query graph structure
edges    = db.get_edges(1)          # outgoing edges
incoming = db.get_incoming(2)       # reverse index

# Auto-create edges by vector similarity
db.auto_link(modality="text", threshold=0.85, rel_type=RelType.RELATED_TO)
```

Built-in relationship types: `related_to`, `derived_from`, `caused_by`, `contradicts`, `supports`, `precedes`, `part_of`, `references`, `multimodal_of`.

### Context Chain (Vector Search + Graph Expansion)

One call that combines semantic vector search with n-hop BFS graph traversal:

```python
result = db.context_chain(
    query=query_vec,
    k=5,           # seed nodes from vector search
    hops=2,        # BFS graph expansion depth
    modality="text"
)

for node in result.nodes:
    print(node.id, node.score, node.hop_distance)

for edge in result.edges:
    print(edge.source_id, "->", edge.target_id, edge.rel_type)
```

Score = `similarity × hop_decay × importance × stickiness`

### Namespace / Entity / Attributes

Generic partitioning for multi-tenant, multi-domain use:

```python
from feather_db import FilterBuilder, MarketingProfile

# Build metadata with domain profile
profile = feather_db.MarketingProfile()
profile.set_brand("nike")
profile.set_user("user_8821")
profile.set_channel("instagram")
profile.set_ctr(0.045)
meta = profile.to_metadata()

db.add(id=100, vec=vec, meta=meta)

# Filter by namespace + entity + attribute
f = FilterBuilder().namespace("nike").entity("user_8821").attribute("channel", "instagram").build()
results = db.search(query_vec, k=10, filter=f)
```

Works for any domain — healthcare, e-commerce, finance — by subclassing `DomainProfile`.

### Living Context / Adaptive Decay

Records accessed more frequently resist temporal decay:

```python
from feather_db import ScoringConfig

cfg = ScoringConfig(half_life=30.0, weight=0.3, min=0.0)
results = db.search(query_vec, k=10, scoring=cfg)
```

Formula:
```
stickiness    = 1 + log(1 + recall_count)
effective_age = age_in_days / stickiness
recency       = 0.5 ^ (effective_age / half_life_days)
final_score   = ((1 - time_weight) * similarity + time_weight * recency) * importance
```

`touch()` is called automatically on every search hit. Call `db.touch(id)` manually to boost salience.

### Graph Visualization

Exports a self-contained, offline D3 force-graph HTML — no CDN, no server:

```python
from feather_db.graph import visualize, export_graph

# Interactive HTML force graph
visualize(db, output_path="/tmp/graph.html")

# JSON for D3 / Cytoscape (namespace-filtered)
data = export_graph(db, namespace_filter="nike")
```

### Import / Export

```python
# D3 / Cytoscape-compatible JSON
json_str = db.export_graph_json(namespace_filter="nike", entity_filter="user_8821")

# Raw vector retrieval
vec   = db.get_vector(id=42, modality="text")
ids   = db.get_all_ids(modality="visual")

# Metadata update without touching HNSW index
db.update_metadata(id=42, meta=new_meta)
db.update_importance(id=42, importance=0.95)
```

---

## Filtered Search

```python
from feather_db import FilterBuilder

results = db.search(
    query_vec, k=10,
    filter=FilterBuilder()
        .namespace("nike")
        .entity("user_8821")
        .attribute("channel", "instagram")
        .source("pipeline-v1")
        .importance_gte(0.5)
        .build()
)
```

---

## Metadata Fields

```python
meta = feather_db.Metadata()
meta.timestamp      = int(time.time())    # Unix timestamp
meta.importance     = 0.9                 # [0.0–1.0]
meta.type           = feather_db.ContextType.FACT  # FACT | PREFERENCE | EVENT | CONVERSATION
meta.source         = "pipeline-v1"
meta.content        = "Human-readable content"
meta.tags_json      = '["tag1","tag2"]'
meta.namespace_id   = "nike"             # partition key
meta.entity_id      = "user_8821"        # subject key
meta.set_attribute("channel", "instagram")   # safe KV setter (use this, not meta.attributes['k']=v)
val = meta.get_attribute("channel")
```

---

## Rust CLI

```bash
# Add a record
feather add --db my.feather --id 1 --vec "0.1,0.2,0.3" --modality text

# Search
feather search --db my.feather --vec "0.1,0.2,0.3" --k 5

# Link two records
feather link --db my.feather --from 1 --to 2
```

---

## Performance

| Metric | Value |
|--------|-------|
| Add rate | 2,000–5,000 vectors/sec |
| Search latency (k=10) | 0.5–1.5 ms |
| Max vectors per modality | 1,000,000 (configurable) |
| HNSW params | M=16, ef_construction=200 |
| File format | Binary `.feather` v5 |

SIMD (AVX2/AVX512) optimizations are available in `space_l2.h`. Enable with `-DUSE_AVX -march=native` in `setup.py`.

---

## File Format

```
[magic: 4B = "FEAT"] [version: 4B = 5]
--- Metadata Section ---
[meta_count: 4B]
  for each record:
    [id: 8B] [serialized Metadata including namespace/entity/attributes/edges]
--- Modality Indices Section ---
[modal_count: 4B]
  for each modality:
    [name_len: 2B] [name: N bytes]
    [dim: 4B] [element_count: 4B]
    for each element:
      [id: 8B] [float32 vector: dim * 4 bytes]
```

v3 and v4 files load transparently — missing fields default to empty.

---

## Examples

| File | Description |
|------|-------------|
| `examples/context_graph_demo.py` | Full context graph demo — auto-link, context_chain, D3 HTML export |
| `examples/marketing_living_context.py` | Multi-brand namespace/entity/attribute filtering + importance feedback |
| `examples/feather_inspector.py` | Local HTTP inspector — force graph, PCA scatter, edit, delete |

Run any example:

```bash
python setup.py build_ext --inplace
python3 examples/context_graph_demo.py
```

---

## Architecture

```
[Generic Core — C++17]
feather::DB
  ├── modality_indices_  (unordered_map<string, ModalityIndex>)  — one HNSW per modality
  ├── metadata_store_    (unordered_map<uint64_t, Metadata>)     — shared metadata by ID
  └── Methods: add, search, link, context_chain, auto_link, export_graph_json ...

[Python Layer]
feather_db (pybind11)
  ├── DB, Metadata, ContextType, ScoringConfig
  ├── Edge, IncomingEdge, ContextNode, ContextEdge, ContextChainResult
  ├── FilterBuilder       — fluent search filter helper
  ├── DomainProfile       — generic namespace/entity/attributes base class
  ├── MarketingProfile    — digital marketing typed adapter
  ├── RelType             — standard relationship type constants
  └── graph.visualize()   — D3 force-graph HTML exporter

[Rust CLI]
feather-db-cli (FFI via extern "C" from src/feather_core.cpp)
```

---

## Known Limitations

| Issue | Detail |
|-------|--------|
| No concurrent writes | HNSW is not thread-safe for simultaneous adds |
| No vector deletion | HNSW marks deletions; data stays until compaction |
| Max 1M vectors/modality | Hardcoded in `get_or_create_index`; increase `max_elements` to raise |
| `meta.attributes['k'] = v` silent no-op | pybind11 map copy; use `meta.set_attribute(k, v)` |
| tags_json is raw string | Tag filtering uses substring search, not proper JSON parsing |

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

MIT — see [LICENSE](LICENSE)

---

## Acknowledgments

- HNSW algorithm: [hnswlib](https://github.com/nmslib/hnswlib)
- Python bindings: [pybind11](https://github.com/pybind/pybind11)
- Rust CLI: [clap](https://github.com/clap-rs/clap)
- Graph visualization: [D3.js](https://d3js.org/)
