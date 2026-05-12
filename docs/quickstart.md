# Feather DB Cloud — Self-hosted Quickstart

Deploy the `feather-api` server with the brand-aligned admin SPA on your own
host (local Docker, Azure VM, AWS EC2, etc.). End-to-end: under 5 minutes.

---

## 1 · Prerequisites

- Docker 20+ with `docker compose`
- A workstation that can reach your host on port 8000 (or any port you map)
- Optional: an embedding-provider API key (OpenAI / Azure OpenAI / Gemini /
  Voyage / Cohere — pick one) for real semantic search. You can skip this
  step initially and the server will produce random vectors.

---

## 2 · Run the server

From the repo root:

```bash
# build the image (uses the multi-stage Dockerfile that compiles the C++
# wheel inside the container)
docker compose -f feather-api/docker-compose.yml build

# start it
FEATHER_API_KEY="feather-$(openssl rand -hex 16)" \
  docker compose -f feather-api/docker-compose.yml up -d
```

Save the generated `FEATHER_API_KEY` somewhere — you'll need it to call any
`/v1/*` endpoint and to log into the admin SPA.

Health check:

```bash
curl -sf http://localhost:8000/health
# → {"status":"ok","version":"0.10.10","namespaces_loaded":0}
```

The admin SPA is at `http://localhost:8000/admin/`. Log in with the key.

---

## 3 · Configure an embedding provider (optional but recommended)

Open the admin SPA → **Settings** → **Embedding service**:

- **OpenAI** — provider `OpenAI`, model `text-embedding-3-small` (1536 d) or
  `text-embedding-3-large` (3072 d), paste API key, dim must match your
  namespace (defaults to 768 — pad/truncate is automatic).
- **Azure OpenAI** — provider `Azure OpenAI`, set Endpoint URL
  (`https://<resource>.openai.azure.com`), Deployment name, API version,
  Azure API key.
- **Gemini** — provider `Google AI · Gemini`, model `gemini-embedding-001`
  (current — `text-embedding-004` is deprecated), Google AI API key.
- **Voyage / Cohere / Ollama** — similar shape; Ollama runs locally so no
  key needed, just a Base URL like `http://localhost:11434`.

Click **Test** to verify credentials end-to-end. After it shows `OK`, click
**Save**.

---

## 4 · Create a namespace and ingest

### From the admin SPA

1. Sidebar → **Namespaces** → **+ New namespace** → name `acme`
2. Header → **Ingest text** → paste a real document → **Embed + store**

### From a script

```python
import requests

BASE = "http://localhost:8000"
KEY  = "feather-..."  # the value of FEATHER_API_KEY

# create namespace
requests.post(
    f"{BASE}/v1/namespaces",
    headers={"X-API-Key": KEY},
    json={"name": "acme"},
).raise_for_status()

# ingest a single text (requires embedding provider configured)
r = requests.post(
    f"{BASE}/v1/acme/ingest_text",
    headers={"X-API-Key": KEY},
    json={
        "text": "Summer 2026 Instagram ad — sandals collection, 30% off, women 25–40.",
        "metadata": {
            "namespace_id": "acme",
            "entity_id":    "creative_001",
            "attributes":   {"brand": "acme", "channel": "instagram", "campaign": "summer-2026"},
        },
    },
)
print(r.json())   # → {"id": ..., "namespace": "acme", "embedded": true, "dim": 768}
```

### Bulk import

If you already have vectors:

```python
requests.post(
    f"{BASE}/v1/acme/import",
    headers={"X-API-Key": KEY},
    json={
        "items": [
            {"id": 1, "vector": [0.1, 0.2, ...], "metadata": {"content": "first",  "namespace_id": "acme"}},
            {"id": 2, "vector": [0.3, 0.4, ...], "metadata": {"content": "second", "namespace_id": "acme"}},
        ]
    },
)
```

Vector dim must match the namespace's dim (768 by default; set with
`FEATHER_DB_DIM`).

---

## 5 · Search

```python
# vector search (random vector here for illustration)
r = requests.post(
    f"{BASE}/v1/acme/search",
    headers={"X-API-Key": KEY},
    json={"vector": [0.1] * 768, "k": 5},
)
for hit in r.json()["results"]:
    print(hit["id"], hit["score"], hit["metadata"]["content"])
```

Filter by attribute:

```python
{
    "vector": [...],
    "k": 5,
    "namespace_id": "acme",
    "attributes_match": {"channel": "instagram"}
}
```

Hybrid (vector + BM25 keyword):

```python
# POST /v1/acme/hybrid_search   body: { "vector": [...], "query": "summer sandals", "k": 5 }
```

---

## 6 · Inspect via the admin SPA

- **Schema tab** — discover what attribute keys exist in a namespace.
- **Hierarchy tab** — see the Brand → Channel → Campaign → … tree if you've
  set those attributes on records.
- **Graph tab** — D3 force-directed view of typed edges; drag nodes, click
  to drill in.
- **Context tab** — `context_chain` runner: vector search for top-k, then
  BFS-expand through graph edges for n hops.
- **Console tab** — raw HTTP tester (handy when integrating).

---

## 7 · Delete and clean up

```bash
# per-record
curl -X DELETE -H "X-API-Key: $KEY" $BASE/v1/acme/records/1

# bulk by namespace_id
curl -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  $BASE/v1/acme/purge -d '{"namespace_id":"acme"}'

# physically drop the namespace + .feather file
curl -X DELETE -H "X-API-Key: $KEY" $BASE/v1/namespaces/acme

# rebuild HNSW dropping soft-deleted residuals (run after lots of deletes)
curl -X POST -H "X-API-Key: $KEY" $BASE/v1/acme/compact
```

---

## 8 · Production checklist

The defaults are fine for testing on `localhost`. For real production:

| Item | Why | How |
|---|---|---|
| **HTTPS / TLS** | API keys travel as headers — never plain HTTP for production | Front the container with Caddy or nginx + Let's Encrypt |
| **Custom domain** | Friendly URL like `feather.yourcompany.cloud` | DNS A record → reverse proxy |
| **Rotate `FEATHER_API_KEY`** | The default in `docker-compose.yml` is a placeholder | Generate one and pass via env var |
| **Persistent volume** | Records live in `.feather` files; container restart wipes ephemeral storage | Mount `/data` to a persistent volume |
| **Backups** | `.feather` files are atomic; periodic copy is enough | `rsync` or cloud snapshots |
| **Resource sizing** | HNSW + Gradio + Python = ~3 GB RAM per million 768-d vectors | 8 GB RAM minimum for production |
| **Multi-tenant keys** | One shared key gives access to all namespaces | Roadmap for v0.11 — for now, run one container per tenant or proxy with per-key authz |

---

## 9 · Endpoint reference

The full OpenAPI is at `http://localhost:8000/docs`. Highlights:

```
POST   /v1/namespaces                         create namespace
DELETE /v1/namespaces/{ns}                    hard-delete namespace
GET    /v1/namespaces                         list
GET    /v1/namespaces/{ns}/stats              dim + record count

POST   /v1/{ns}/vectors                       add a record with vector
POST   /v1/{ns}/ingest_text                   embed text via configured provider + store
POST   /v1/{ns}/import                        bulk import [{id, vector, metadata}, ...]
POST   /v1/{ns}/seed                          generate N random records (testing)

POST   /v1/{ns}/search                        vector search
POST   /v1/{ns}/hybrid_search                 vector + BM25
POST   /v1/{ns}/keyword_search                BM25 only
POST   /v1/{ns}/context_chain                 vector search + BFS expansion

GET    /v1/{ns}/records?limit=&after=         paginate
GET    /v1/{ns}/records/{id}                  fetch one
PUT    /v1/{ns}/records/{id}                  update metadata
DELETE /v1/{ns}/records/{id}                  soft-delete (HNSW markDelete)
POST   /v1/{ns}/records/{id}/link             add typed edge

GET    /v1/{ns}/records/{id}/edges            outgoing + incoming edges
GET    /v1/{ns}/schema                        attribute discovery
GET    /v1/{ns}/hierarchy                     Brand → Channel → … tree
GET    /v1/{ns}/top_recalled                  recall_count desc
GET    /v1/{ns}/graph                         D3-shaped nodes + edges
POST   /v1/{ns}/purge                         bulk delete by namespace_id
POST   /v1/{ns}/compact                       rebuild HNSW
POST   /v1/{ns}/save                          flush WAL to disk

GET    /v1/admin/overview                     cluster stats
GET    /v1/admin/metrics                      latency + ops counts
GET    /v1/admin/activity                     recent ops feed
GET    /v1/admin/ops_timeseries               buckets for sparkline
GET    /v1/admin/connection_info              copy-paste code samples
GET    /v1/admin/embedding_models             per-provider model catalog
GET    /v1/admin/embedding_config             current config (key never echoed)
PUT    /v1/admin/embedding_config             configure provider + key
```

---

## 10 · Where to go next

- **Embeddings:** if you skipped step 3, configure a provider; everything that
  uses `ingest_text` depends on it.
- **Memory / decay:** see [README → Living Context](../README.md#living-context--adaptive-decay).
- **Self-aligned ingestion:** see [PHASE2_GUIDE.md](../PHASE2_GUIDE.md) for the
  LLM-powered extraction pipeline.
- **MCP server:** see [README → MCP Server](../README.md#mcp-server-v060) to
  expose Feather as a tool surface for Claude Desktop, Cursor, or any MCP
  client.
