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

Embedded vector database with **sub-millisecond HNSW search**, typed context graph, and adaptive decay.

## What this demo shows

| Tab | What it does |
|-----|-------------|
| 🔍 Semantic Search | Find nodes by meaning — with namespace/entity/product filters |
| 🕸️ Context Chain | Vector search + BFS graph expansion — traces root causes across edges |
| 🔬 Why Retrieved? | Score breakdown: similarity × stickiness × recency × importance |
| 🩺 Graph Health | Tier distribution, orphan nodes, recall histogram |
| ➕ Add Intel | Ingest a new node — immediately searchable |

## Install

```bash
pip install feather-db
```

## Connect to any LLM in 5 lines

```python
from feather_db.integrations import ClaudeConnector
conn = ClaudeConnector(db_path="my.feather", dim=3072, embedder=embed_fn)
result = conn.run_loop(client,
    messages=[{"role": "user", "content": "Why is our FD CTR dropping?"}],
    model="claude-opus-4-6")
```

Works with **Claude, OpenAI, Gemini, Groq, Mistral, Ollama** and any MCP-compatible agent (Claude Desktop, Cursor).

## Links

- [PyPI](https://pypi.org/project/feather-db/)
- [GitHub](https://github.com/feather-store/feather)
- [Integrations Guide](https://github.com/feather-store/feather/blob/main/docs/integrations.md)
- [Hawky.ai](https://hawky.ai)
