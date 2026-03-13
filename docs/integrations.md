# Feather DB — LLM Integrations Guide

Connect Feather DB's living context engine to any LLM — Claude, Gemini, OpenAI, Groq, Mistral, or any MCP-compatible agent — in under 10 lines of code.

**Version:** v0.6.0 · [GitHub](https://github.com/feather-store/feather) · [PyPI](https://pypi.org/project/feather-db/)

---

## Table of Contents

1. [How it works](#how-it-works)
2. [Installation](#installation)
3. [Claude (Anthropic)](#claude-anthropic)
4. [OpenAI and compatible APIs](#openai-and-compatible-apis)
5. [Gemini (Google)](#gemini-google)
6. [MCP Server — Claude Desktop & Cursor](#mcp-server)
7. [LangChain](#langchain)
8. [LlamaIndex](#llamaindex)
9. [Tool Reference — all 14 tools](#tool-reference)
10. [Choosing an embedder](#choosing-an-embedder)
11. [Building a knowledge graph to query](#building-a-knowledge-graph)
12. [Production patterns](#production-patterns)

---

## How it works

Every connector follows the same pattern:

```
Your question
     │
     ▼
LLM  (Claude / Gemini / GPT-4o / Llama …)
     │  decides it needs context → calls a Feather tool
     ▼
Feather DB Connector
     │  translates the tool call → db.search() / context_chain() / …
     │  returns structured JSON back to the LLM
     ▼
Feather DB  (your .feather file on disk)
     │  HNSW vector search + typed graph + adaptive decay
     ▼
LLM writes its final answer, grounded in retrieved context
```

The LLM decides **which tools to call and in what order**. The connector handles the translation and all multi-turn rounds. You call one method — `run_loop()` — and get back the final answer.

### The tool loop in detail

```
Round 1  →  LLM calls feather_search("FD CTR drop")
         ←  Feather returns top-4 nodes as JSON
Round 2  →  LLM calls feather_context_chain("budget day competitor", hops=2)
         ←  Feather returns 8 nodes + 7 causal edges
Round 3  →  LLM writes final answer  (no more tool calls)
         ←  run_loop() returns the text
```

Each round is handled automatically. `run_loop()` exits when the LLM stops calling tools.

---

## Installation

```bash
# Core (always needed)
pip install feather-db numpy

# Per provider — install only what you use
pip install anthropic                  # Claude
pip install openai                     # OpenAI / Azure / Groq / Mistral / Together
pip install google-genai               # Gemini
pip install mcp                        # MCP server (Claude Desktop / Cursor)
pip install langchain langchain-core   # LangChain
pip install llama-index llama-index-core  # LlamaIndex

# Or install all optional deps at once
pip install "feather-db[all]"
```

---

## Claude (Anthropic)

### Minimal example

```python
import anthropic
from feather_db.integrations import ClaudeConnector

conn = ClaudeConnector(
    db_path  = "my.feather",
    dim      = 3072,
    embedder = my_embed_fn,      # text → np.ndarray
)

client = anthropic.Anthropic(api_key="sk-ant-...")

result = conn.run_loop(
    client,
    messages = [{"role": "user", "content": "Why is our FD CTR dropping?"}],
    model    = "claude-opus-4-6",
)
print(result)
```

### With system prompt and options

```python
result = conn.run_loop(
    client,
    messages   = [{"role": "user", "content": "Why is our FD CTR dropping?"}],
    model      = "claude-opus-4-6",
    max_tokens = 4096,
    max_rounds = 8,           # max tool-call rounds before giving up
    system     = (
        "You are a performance marketing analyst with access to a Feather DB "
        "knowledge graph containing campaign intel, competitor signals, and "
        "social trends. Always retrieve context before answering. Cite node IDs."
    ),
    verbose    = True,        # prints tool calls and results as they happen
)
```

### Single API call (no loop)

If you want to manage the loop yourself:

```python
import anthropic

response = client.messages.create(
    model      = "claude-opus-4-6",
    max_tokens = 4096,
    tools      = conn.tools(),    # list of tool definitions
    messages   = [{"role": "user", "content": "Why is our FD CTR dropping?"}],
)

done, tool_messages = conn.process_response(response)
# done=True  → response.content has the final text
# done=False → append tool_messages to messages and call again
```

### How tools are formatted for Claude

Feather tools use Anthropic's `input_schema` format:

```python
[
  {
    "name": "feather_search",
    "description": "Semantic vector search over the Feather DB knowledge graph...",
    "input_schema": {
      "type": "object",
      "properties": {
        "query": {"type": "string",  "description": "Natural language search query"},
        "k":     {"type": "integer", "description": "Number of results (default 5)"},
        ...
      },
      "required": ["query"]
    }
  },
  ...  # 14 tools total
]
```

---

## OpenAI and compatible APIs

`OpenAIConnector` works with any OpenAI-spec API: OpenAI, Azure OpenAI, Groq, Mistral, Together AI, Ollama, and more. Change only the `client` — the connector is identical.

### OpenAI

```python
from openai import OpenAI
from feather_db.integrations import OpenAIConnector

conn   = OpenAIConnector(db_path="my.feather", dim=3072, embedder=my_embed_fn)
client = OpenAI(api_key="sk-...")

result = conn.run_loop(
    client,
    messages = [{"role": "user", "content": "Why is our FD CTR dropping?"}],
    model    = "gpt-4o",
)
```

### Groq (Llama 3.3 70B)

```python
from openai import OpenAI

client = OpenAI(
    api_key  = "gsk_...",
    base_url = "https://api.groq.com/openai/v1",
)

result = conn.run_loop(
    client,
    messages = [{"role": "user", "content": "Why is our FD CTR dropping?"}],
    model    = "llama-3.3-70b-versatile",
)
```

### Mistral

```python
client = OpenAI(
    api_key  = "...",
    base_url = "https://api.mistral.ai/v1",
)
result = conn.run_loop(client, messages, model="mistral-large-latest")
```

### Ollama (local)

```python
client = OpenAI(
    api_key  = "ollama",       # any string
    base_url = "http://localhost:11434/v1",
)
result = conn.run_loop(client, messages, model="llama3.2")
```

### With system prompt

```python
result = conn.run_loop(
    client,
    messages = [{"role": "user", "content": "Analyse FD campaign performance"}],
    model    = "gpt-4o",
    system   = "You are a marketing analyst. Use Feather tools before answering.",
    max_rounds  = 8,
    max_tokens  = 4096,
    verbose     = True,
)
```

### How tools are formatted for OpenAI

```python
[
  {
    "type": "function",
    "function": {
      "name": "feather_search",
      "description": "Semantic vector search over the Feather DB knowledge graph...",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string",  "description": "Natural language search query"},
          "k":     {"type": "integer", "description": "Number of results (default 5)"},
          ...
        },
        "required": ["query"]
      }
    }
  },
  ...  # 14 tools total
]
```

---

## Gemini (Google)

### Option A — with Gemini Embedding 2 (recommended)

Use Gemini's own 3072-dim multimodal embedder for both ingestion and queries. Text, image descriptions, and video transcripts all land in the same vector space.

```python
from google import genai
from feather_db.integrations import GeminiConnector, GeminiEmbedder

# 1. Create the embedder
emb = GeminiEmbedder(api_key="AIza...")

# 2. Create the connector — use the same embedder for queries
conn = GeminiConnector(
    db_path  = "my.feather",
    dim      = emb.dim,          # 3072 for Gemini Embedding 2
    embedder = emb.embed_text,
)

# 3. Create a Gemini client and chat session
client = genai.Client(api_key="AIza...")
chat   = client.chats.create(
    model  = "gemini-2.0-flash",
    config = conn.chat_config(),  # attaches all 14 tools
)

# 4. Run
result = conn.run_loop(chat, "Which competitor moves should I worry about?")
print(result)
```

### Option B — with any embedder

```python
from feather_db.integrations import GeminiConnector

conn = GeminiConnector(
    db_path  = "my.feather",
    dim      = 768,               # match your embedder's output dimension
    embedder = my_embed_fn,
)
```

### With system prompt

```python
chat = client.chats.create(
    model  = "gemini-2.0-flash",
    config = conn.chat_config(
        system = (
            "You are a performance marketing analyst. "
            "Use Feather DB tools to retrieve context before answering."
        )
    ),
)
result = conn.run_loop(chat, "Analyse FD campaign performance", max_rounds=8)
```

### Multimodal ingestion with GeminiEmbedder

```python
emb = GeminiEmbedder(api_key="AIza...")

# Embed different modalities — all land in the same 3072-dim space
text_vec  = emb.embed_text("FD 8.5% interest rate campaign")
image_vec = emb.embed_image(image_path="ad_creative.jpg")
video_vec = emb.embed_video_transcript("0:00 Are your savings safe? 0:12 Guaranteed 8.5%")

# Cross-modal: search text, find image creatives
results = db.search(text_vec, k=5)   # returns text AND image nodes

# Offline / mock mode (no API key needed for testing)
emb_mock = GeminiEmbedder(mock=True)
```

### How tools are formatted for Gemini

```python
[
  types.Tool(function_declarations=[
    types.FunctionDeclaration(
      name        = "feather_search",
      description = "Semantic vector search over the Feather DB knowledge graph...",
      parameters  = types.Schema(
        type       = "OBJECT",
        properties = {
          "query": types.Schema(type="STRING", description="Natural language search query"),
          "k":     types.Schema(type="INTEGER", description="Number of results"),
          ...
        },
        required = ["query"]
      )
    ),
    ...  # 14 tools total
  ])
]
```

---

## MCP Server

The MCP server exposes all 14 Feather tools as first-class MCP tool definitions. Any MCP-compatible agent — Claude Desktop, Cursor, or a custom agent — can connect without writing any Python code.

### Install

```bash
pip install mcp feather-db
```

### Run

```bash
feather-serve --db my.feather --dim 3072
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | required | Path to `.feather` file (created if absent) |
| `--dim` | `3072` | Vector dimension |
| `--name` | `feather-db` | Server name shown in agent UIs |
| `--embedder` | built-in mock | Path to a Python file exporting `embed(text) -> list[float]` |

### Connect Claude Desktop

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "feather": {
      "command": "feather-serve",
      "args": ["--db", "/path/to/my.feather", "--dim", "3072"]
    }
  }
}
```

Restart Claude Desktop. All 14 Feather tools appear in the tool picker immediately.

### Connect Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "feather": {
      "command": "feather-serve",
      "args": ["--db", "/path/to/my.feather", "--dim", "3072"]
    }
  }
}
```

### Use a custom embedder with MCP

Create `my_embedder.py`:

```python
# Must export a function named `embed`
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str) -> list[float]:
    return _model.encode(text).tolist()
```

Run with:

```bash
feather-serve --db my.feather --dim 384 --embedder my_embedder.py
```

### Programmatic MCP server

```python
from feather_db.integrations.mcp_server import create_server
import asyncio
from mcp.server.stdio import stdio_server

server = create_server(db_path="my.feather", dim=3072)
asyncio.run(stdio_server(server))
```

---

## LangChain

### FeatherVectorStore

Drop-in `VectorStore` for use anywhere LangChain expects a vector store.

```python
from feather_db.integrations import FeatherVectorStore

store = FeatherVectorStore(
    db_path   = "my.feather",
    dim       = 3072,
    embed_fn  = my_embed_fn,
    namespace = "langchain",
)

# Add documents
store.add_texts(
    texts     = ["FD rate 8.5%", "Competitor launched 8.75%"],
    metadatas = [{"product": "FD"}, {"source": "competitor"}],
)

# Search
docs = store.similarity_search("FD interest rate", k=5)
for doc in docs:
    print(doc.page_content, doc.metadata["score"])

# Search with scores
results = store.similarity_search_with_score("FD interest rate", k=5)

# Create from existing texts
store = FeatherVectorStore.from_texts(
    texts     = ["FD rate 8.5%", "Competitor launched 8.75%"],
    embedding = my_embedding_model,
    db_path   = "my.feather",
)

# Create from LangChain Documents
store = FeatherVectorStore.from_documents(documents, embedding=model)
```

### FeatherMemory

Conversation memory that retrieves semantically relevant history — not just last-N turns — with adaptive decay so recent and frequently accessed turns surface first.

```python
from feather_db.integrations import FeatherMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

memory = FeatherMemory(
    db_path    = "my.feather",
    dim        = 3072,
    embed_fn   = my_embed_fn,
    k          = 5,             # retrieve top-5 relevant past turns
    namespace  = "memory",
    memory_key = "history",
)

chain = ConversationChain(
    llm    = ChatOpenAI(model="gpt-4o"),
    memory = memory,
)

response = chain.predict(input="Why is FD CTR dropping?")
```

### FeatherRetriever

Wraps `context_chain()` as a LangChain `BaseRetriever`. Returns graph-expanded nodes, not just top-k similarity — traces causal connections automatically.

```python
from feather_db.integrations import FeatherRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

retriever = FeatherRetriever(
    db_path  = "my.feather",
    dim      = 3072,
    embed_fn = my_embed_fn,
    k        = 5,
    hops     = 2,               # expand 2 graph hops from seed nodes
    modality = "text",
)

qa_chain = RetrievalQA.from_chain_type(
    llm       = ChatOpenAI(model="gpt-4o"),
    retriever = retriever,
)

answer = qa_chain.run("What caused the FD CTR spike on Budget Day?")
```

---

## LlamaIndex

### FeatherVectorStoreIndex

Drop-in `VectorStore` for LlamaIndex.

```python
from feather_db.integrations import FeatherVectorStoreIndex
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.vector_stores.types import VectorStoreQuery

store = FeatherVectorStoreIndex(
    db_path   = "my.feather",
    dim       = 3072,
    embed_fn  = my_embed_fn,
    namespace = "llamaindex",
)

# Add LlamaIndex nodes
store.add(nodes)

# Query directly
query  = VectorStoreQuery(query_embedding=vec, similarity_top_k=5)
result = store.query(query)
for node_with_score in result.nodes:
    print(node_with_score.node.text, node_with_score.score)

# Use as a LlamaIndex storage context
storage_context = StorageContext.from_defaults(vector_store=store)
index = VectorStoreIndex(nodes, storage_context=storage_context)
query_engine = index.as_query_engine()
response = query_engine.query("Why did FD CTR spike?")
```

### FeatherReader

Load an existing `.feather` file as LlamaIndex Documents.

```python
from feather_db.integrations import FeatherReader
from llama_index.core import VectorStoreIndex

reader = FeatherReader()
docs   = reader.load_data(
    db_path          = "my.feather",
    dim              = 3072,
    namespace_filter = "marketing",   # optional
    min_importance   = 0.5,           # optional
)

# Build a LlamaIndex index from existing Feather data
index        = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
response     = query_engine.query("What competitor threats exist for FD?")
```

---

## Tool Reference

All 14 tools are available across every connector — Claude, OpenAI, Gemini, and MCP.

### Search tools

#### `feather_search`
Semantic vector search. Converts the query to an embedding and returns the k most similar nodes.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | yes | Natural language search query |
| `k` | integer | no | Results to return (default 5) |
| `namespace` | string | no | Filter by namespace |
| `entity` | string | no | Filter by entity type (e.g. `competitor_intel`) |
| `product` | string | no | Filter by product attribute (e.g. `FD`, `CC`) |

#### `feather_mmr_search`
Same as `feather_search` but post-processes results with Maximal Marginal Relevance to balance relevance and diversity. Use when you need a broad view rather than the single most relevant cluster.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | yes | Natural language search query |
| `k` | integer | no | Results (default 5) |
| `diversity` | number | no | 0.0 = pure similarity, 1.0 = max diversity (default 0.5) |

#### `feather_context_chain`
Two-phase retrieval: vector search for k seeds (hop=0) then BFS graph expansion for `hops` levels. Returns all reached nodes with hop distance. Use to trace root causes.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | yes | Seed query |
| `k` | integer | no | Vector-search seeds (default 3) |
| `hops` | integer | no | Graph expansion hops (default 2) |

### Node tools

#### `feather_get_node`
Fetch full metadata for a node by ID — content, attributes, edges in and out.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | integer | yes | Node ID |

#### `feather_get_related`
Get direct graph neighbours — nodes connected by typed edges.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | integer | yes | Source node ID |
| `rel_type` | string | no | Filter by edge type (e.g. `caused_by`) |
| `direction` | string | no | `outgoing` (default), `incoming`, or `both` |

#### `feather_why`
Score breakdown explaining why a node would be retrieved for a query. Returns similarity, stickiness (recall bonus), recency, importance, confidence, and the final formula.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | integer | yes | Node ID to explain |
| `query` | string | yes | Query text to score against |

### Write tools

#### `feather_add_intel`
Ingest a new intelligence node. Use when the agent discovers information worth persisting.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | yes | Full text of the intelligence note |
| `entity_type` | string | yes | Type label (e.g. `competitor_intel`, `strategy_intel`) |
| `product` | string | no | Product this intel relates to |
| `importance` | number | no | Relevance weight 0–1 (default 0.8) |

#### `feather_link_nodes`
Create a typed weighted edge between two existing nodes.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `from_id` | integer | yes | Source node ID |
| `to_id` | integer | yes | Target node ID |
| `rel_type` | string | yes | `caused_by`, `supports`, `contradicts`, `derived_from`, `references`, `same_ad`, `related_to` |
| `weight` | number | no | Edge weight 0–1 (default 0.8) |

#### `feather_forget`
Soft-delete a node — removes it from search results but preserves graph edges.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | integer | yes | Node ID to forget |

### Memory management tools

#### `feather_timeline`
Chronological node list (newest first) for a product or entity type.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `product` | string | no | Filter by product |
| `entity_type` | string | no | Filter by entity type |
| `limit` | integer | no | Max nodes (default 10) |

#### `feather_health`
Knowledge graph health report — tier distribution, orphan nodes, recall histogram, avg importance/confidence.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `modality` | string | no | Modality to inspect (default `text`) |

#### `feather_consolidate`
Cluster similar nodes in a namespace and merge each cluster into a summary node. Reduces noise, improves retrieval quality over time.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `namespace` | string | yes | Namespace to consolidate |
| `since_hours` | number | no | Only consolidate recent nodes (default 24) |
| `threshold` | number | no | Cosine similarity to cluster (default 0.85) |

#### `feather_episode_get`
Retrieve all nodes in a named episode, ordered by timestamp.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `episode_id` | string | yes | Episode identifier |

#### `feather_expire`
Scan and soft-delete all nodes that have exceeded their TTL (time-to-live). Call periodically to keep the graph clean.

No parameters required.

---

## Choosing an embedder

The embedder converts text queries (and ingested content) into vectors. Use the same embedder for both ingestion and querying.

### Gemini Embedding 2 (recommended for Gemini projects)

```python
from feather_db.integrations import GeminiEmbedder

emb = GeminiEmbedder(api_key="AIza...")
# dim=3072, multimodal (text + image + video in same space)
embed_fn = emb.embed_text
```

### OpenAI Embeddings

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed_fn(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-large", input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)
# dim=3072 for text-embedding-3-large
```

### Sentence Transformers (local, no API cost)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # dim=384

def embed_fn(text: str) -> np.ndarray:
    return model.encode(text, normalize_embeddings=True).astype(np.float32)
```

### Mock embedder (offline testing, CI)

All connectors include a built-in deterministic mock embedder — no API key or model required. Leave `embedder=None` to use it:

```python
conn = ClaudeConnector(db_path="my.feather", dim=3072)  # uses mock embedder
```

---

## Building a knowledge graph

Before querying, you need nodes in the DB. Here is the typical ingestion pattern:

```python
import feather_db, numpy as np, time

db = feather_db.DB.open("my.feather", dim=3072)

def add_node(id, content, entity_type, product=None, importance=0.9):
    vec  = embed_fn(content)
    meta = feather_db.Metadata()
    meta.timestamp    = int(time.time())
    meta.importance   = importance
    meta.confidence   = 0.9
    meta.type         = feather_db.ContextType.FACT
    meta.source       = "pipeline-v1"
    meta.content      = content
    meta.namespace_id = "my_brand"
    meta.set_attribute("entity_type", entity_type)
    if product:
        meta.set_attribute("product", product)
    db.add(id=id, vec=vec, meta=meta)

# Add intelligence nodes
add_node(1001, "FD video ad — CTR 3.2%, ROAS 4.1. Senior couple hook.",      "ad_creative",    "FD")
add_node(9001, "Competitor launched 8.75% APY FD on Budget Day.",             "competitor_intel","FD")
add_node(9101, "Budget Day: RBI held repo rate. FD appeal increased.",         "market_signal",  "FD")

# Add causal edges
db.link(9001, 1001, "contradicts", 0.9)   # competitor undercuts our rate
db.link(9101, 1001, "supports",    0.85)  # budget day supports FD demand

# Working memory with TTL (auto-expires in 1 hour)
meta_temp = feather_db.Metadata()
meta_temp.content      = "Live social trend: #BudgetDay2026 trending"
meta_temp.ttl          = 3600   # expires in 1 hour
meta_temp.namespace_id = "live_feed"
db.add(id=8001, vec=embed_fn(meta_temp.content), meta=meta_temp)

db.save()
```

---

## Production patterns

### Pattern 1 — Shared connector, multiple agents

```python
# Create once at startup, reuse across requests
conn = ClaudeConnector(db_path="production.feather", dim=3072, embedder=embed_fn)

# Each request gets a fresh message list but the same connector
def handle_request(user_question: str) -> str:
    return conn.run_loop(
        client,
        messages = [{"role": "user", "content": user_question}],
        model    = "claude-opus-4-6",
    )
```

### Pattern 2 — Namespace isolation per tenant

```python
# Each brand / customer gets their own namespace
def build_connector(brand_id: str) -> ClaudeConnector:
    return ClaudeConnector(
        db_path  = f"{brand_id}.feather",
        dim      = 3072,
        embedder = embed_fn,
    )
```

### Pattern 3 — Keep DB fresh with triggers

```python
from feather_db import WatchManager, ContradictionDetector

wm = WatchManager()
cd = ContradictionDetector()

# Alert when a new node matches a critical pattern
wm.watch(
    db,
    query_text = "competitor launched new FD product",
    threshold  = 0.75,
    callback   = lambda nid, sim: alert_team(f"⚡ New competitor intel: node {nid}"),
    embed_fn   = embed_fn,
)

def ingest(content: str, entity_type: str):
    # Add to DB
    nid = next_id()
    db.add(nid, embed_fn(content), make_meta(content, entity_type))
    # Check triggers
    wm.check_triggers(db, nid, embed_fn=embed_fn)
    # Check for contradictions
    cd.check(db, nid, auto_link=True)
```

### Pattern 4 — Periodic memory maintenance

```python
import schedule, time
from feather_db import MemoryManager

def maintain():
    db.forget_expired()                                    # remove TTL nodes
    MemoryManager.consolidate(db, namespace="live_feed")   # merge similar nodes
    MemoryManager.assign_tiers(db)                         # classify hot/warm/cold

schedule.every(6).hours.do(maintain)
while True:
    schedule.run_pending()
    time.sleep(60)
```

### Pattern 5 — Explain retrieval decisions

```python
from feather_db import MemoryManager

# After an agent response, explain why node 1001 was retrieved
breakdown = MemoryManager.why_retrieved(db, node_id=1001, query_vec=query_vec)
print(f"Retrieved because: {breakdown['formula']}")
# → Retrieved because: ((1-0.3) × 0.81 + 0.3 × 0.94) × 0.95 = 0.84
```

---

## Quick reference

```python
from feather_db.integrations import (
    ClaudeConnector,      # Anthropic tool_use format
    OpenAIConnector,      # OpenAI function-calling format (+ Groq, Mistral, etc.)
    GeminiConnector,      # Google FunctionDeclaration format
    GeminiEmbedder,       # Gemini Embedding 2 (3072-dim, multimodal)
    FeatherVectorStore,   # LangChain VectorStore
    FeatherMemory,        # LangChain BaseMemory
    FeatherRetriever,     # LangChain BaseRetriever
    FeatherVectorStoreIndex,  # LlamaIndex VectorStore
    FeatherReader,            # LlamaIndex BaseReader
)

# MCP server
# feather-serve --db my.feather --dim 3072
```

---

*Feather DB is built by [Hawky.ai](https://hawky.ai) · [GitHub](https://github.com/feather-store/feather) · [PyPI](https://pypi.org/project/feather-db/)*
