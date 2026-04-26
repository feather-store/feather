# Feather DB

**Embedded vector database + self-aligned context engine**

*Part of [Hawky.ai](https://hawky.ai) — AI-Native Development Tools*

[![PyPI](https://img.shields.io/pypi/v/feather-db)](https://pypi.org/project/feather-db/)
[![Crates.io](https://img.shields.io/crates/v/feather-db-cli)](https://crates.io/crates/feather-db-cli)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Website](https://img.shields.io/badge/website-getfeather.store-blue)](https://www.getfeather.store/)
[![HuggingFace Space](https://img.shields.io/badge/demo-HuggingFace-yellow)](https://huggingface.co/spaces/Sri-Vigneshwar-DJ/feather-db)

Feather DB is an embedded vector database and living context engine — zero-server, file-based, with a built-in knowledge graph, adaptive memory decay, LLM agent connectors, and a self-aligned ingestion engine that organises data automatically.

---

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
| **Single-file persistence** | `.feather` binary format (v5); v3/v4 files load transparently |

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

All 14 tools become available to Claude Desktop immediately — no code required.

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

### Living Context / Adaptive Decay

```python
from feather_db import ScoringConfig

cfg     = ScoringConfig(half_life=30.0, weight=0.3, min=0.0)
results = db.search(query_vec, k=10, scoring=cfg)
```

Formula:
```
stickiness    = 1 + log(1 + recall_count)
effective_age = age_in_days / stickiness
recency       = 0.5 ^ (effective_age / half_life_days)
final_score   = ((1 - time_weight) * similarity + time_weight * recency) * importance
```

`touch()` is called automatically on every search hit.

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

| Metric | Value |
|--------|-------|
| Add rate | 2,000–5,000 vectors/sec |
| Search latency p50 (k=10, 500K × 128-dim, real SIFT data) | **0.19 ms** |
| Search latency p99 (k=10, 500K × 128-dim, real SIFT data) | **0.13 ms @ ef=10**, **1.03 ms @ ef=200** |
| Recall@10 (500K × 128-dim, ef=50, real SIFT) | **0.972** |
| Max vectors per modality | 1,000,000 (configurable) |
| HNSW params | M=16, ef_construction=200, ef=50 (default in v0.8.0) |
| File format | Binary `.feather` v6 |

SIMD (AVX2/AVX512) optimizations are available in `space_l2.h`. Enable with `-DUSE_AVX -march=native` in `setup.py`.

Reproducible benchmark harness lives in [`bench/`](./bench/). Run any benchmark with `python -m bench run <scenario>`.

---

## Benchmarks

### Memory benchmark — LongMemEval (Xu et al., 2024)

500-question end-to-end memory QA benchmark, the standard for long-term memory in chat assistants. Full report: [`docs/benchmarks/longmemeval.md`](./docs/benchmarks/longmemeval.md).

| Run | Variant | Answerer | Overall | Notes |
|---|---|---|---|---|
| Feather DB v0.8.0 + decay | **S** | gpt-4o | **0.693** | best run; same model as Supermemory |
| Feather DB v0.8.0 + decay | **S** | gemini-2.5-flash | **0.657** | cheap-tier; ~$2.40 per full run |
| Feather DB v0.8.0 + decay | oracle | gemini-2.5-flash | 0.670 | retrieval-easy ceiling |

| System | Variant | Answerer | Overall |
|---|---|---|---|
| **Feather DB v0.8.0 + decay** | **S** | **gemini-2.5-flash** | **0.657** |
| Zep (graphiti) | S | gpt-4o-mini | 0.638 |
| Full-context GPT-4o (paper "ceiling") | S | gpt-4o + CoN | 0.640 |
| Full-context GPT-4o-mini | S | gpt-4o-mini | 0.554 |
| Mem0 (prior algo) | S | gpt-4o-mini | 0.678 |
| Supermemory | S | gpt-4o | 0.816 |

Cost for the full Feather S run: **~$2.40** (Azure embeddings + Gemini answer + judge). Wall time 4.5 hours. 5 failures / 500 questions.

Reproduce:
```bash
python -m bench run longmemeval --dataset s --limit 0 \
  --embedder openai --judge llm \
  --judge-provider gemini --judge-model gemini-2.0-flash \
  --answerer-provider gemini --answerer-model gemini-2.5-flash \
  --decay-half-life 14 --decay-time-weight 0.4 --k 10
```

### ANN benchmark — SIFT1M

Standard ANN benchmark. Full sweep results in [`bench/results/`](./bench/results/).

| Config | p50 | p99 | Recall@10 |
|---|---|---|---|
| 500K × 128, ef=10 | 0.07 ms | 0.13 ms | 0.774 |
| 500K × 128, ef=50 (default) | 0.19 ms | 0.23 ms | **0.972** |
| 500K × 128, ef=200 | 0.56 ms | 0.69 ms | 0.998 |

---

## Cloud Deployment (Azure / Docker)

Feather DB ships with a production-ready FastAPI wrapper and Gradio management dashboard you can deploy on any Linux VM.

```bash
git clone https://github.com/feather-store/feather.git
cd feather
export FEATHER_API_KEY="your-secret-key"
docker compose -f feather-api/docker-compose.yml up -d --build
```

| URL | Description |
|-----|-------------|
| `http://<VM_IP>:8000/health` | Health check |
| `http://<VM_IP>:8000/docs` | Swagger / OpenAPI |
| `http://<VM_IP>:8000/dashboard` | Management Dashboard UI |

**Full Azure deployment guide → [`docs/deploy-azure.md`](docs/deploy-azure.md)**

> Data is stored in a Docker named volume (`feather-data → /data`) and persists across restarts and rebuilds.

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

## Known Limitations

| Issue | Detail |
|-------|--------|
| No concurrent writes | HNSW is not thread-safe for simultaneous adds |
| No vector deletion | HNSW marks deletions; data stays until compaction |
| Max 1M vectors/modality | Hardcoded in `get_or_create_index`; increase `max_elements` to raise |
| `meta.attributes['k'] = v` silent no-op | pybind11 map copy; use `meta.set_attribute(k, v)` |
| Rust CLI missing v0.6.0+ features | namespace/entity/context_chain/integrations are Python-only |

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
