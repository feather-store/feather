---
title: Feather DB — Living Context Engine
emoji: 🪶
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: true
license: mit
tags:
  - vector-database
  - embeddings
  - knowledge-graph
  - hnsw
  - context-graph
  - llm
  - rag
  - agents
  - mcp
---

# 🪶 Feather DB — Living Context Engine

Embedded vector database with **sub-millisecond HNSW search**, typed context graph, adaptive decay, and a **brand-aligned admin UI** for running it as a managed Cloud service.

> **Now in v0.10 — Cloud Edition**: a custom admin SPA + pluggable embeddings
> (OpenAI · Azure OpenAI · Gemini · Voyage · Cohere · Ollama). Stand up the
> full stack with one `docker compose up`. See the
> [quickstart](https://github.com/feather-store/feather/blob/main/docs/quickstart.md).

## What this demo shows

| Tab | What it does |
|-----|-------------|
| 🔍 Semantic Search | Find nodes by meaning — with namespace/entity/product filters |
| 🕸️ Context Chain | Vector search + BFS graph expansion — traces root causes across edges |
| 🔬 Why Retrieved? | Score breakdown: similarity × stickiness × recency × importance |
| 🩺 Graph Health | Tier distribution, orphan nodes, recall histogram |
| ➕ Add Intel | Ingest a new node — immediately searchable |

## Three ways to use Feather

### Embedded (Python)

```bash
pip install feather-db
```

```python
import feather_db, numpy as np
db = feather_db.DB.open("my.feather", dim=768)
db.add(id=1, vec=np.random.rand(768).astype(np.float32))
results = db.search(np.random.rand(768).astype(np.float32), k=5)
```

### Self-hosted Cloud (Docker)

```bash
git clone https://github.com/feather-store/feather
cd feather
FEATHER_API_KEY="feather-$(openssl rand -hex 16)" \
  docker compose -f feather-api/docker-compose.yml up -d
open http://localhost:8000/admin/      # admin SPA
```

Then configure an embedding provider in Settings → Embedding service and
hit `POST /v1/{ns}/ingest_text` from anywhere.

### As an LLM tool surface (5 lines)

```python
from feather_db.integrations import ClaudeConnector
conn = ClaudeConnector(db_path="my.feather", dim=3072, embedder=embed_fn)
result = conn.run_loop(client,
    messages=[{"role": "user", "content": "Why is our CTR dropping?"}],
    model="claude-opus-4-6")
```

Works with **Claude, OpenAI, Gemini, Groq, Mistral, Ollama** and any MCP-compatible agent (Claude Desktop, Cursor).

## Links

- [PyPI](https://pypi.org/project/feather-db/) · [Crates.io](https://crates.io/crates/feather-db-cli)
- [GitHub](https://github.com/feather-store/feather)
- [Quickstart](https://github.com/feather-store/feather/blob/main/docs/quickstart.md) · [Integrations Guide](https://github.com/feather-store/feather/blob/main/docs/integrations.md) · [Changelog](https://github.com/feather-store/feather/blob/main/CHANGELOG.md)
- [Hawky.ai](https://hawky.ai)
