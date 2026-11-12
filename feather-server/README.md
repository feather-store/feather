# feather-server

The Cloud API's hot routes in Rust. It reads and writes the same `.feather`
files as `feather-api`, sends the same JSON, uses the same status codes and
error bodies, and links the engine directly through the C ABI
(`src/feather_core.cpp`), with no Python in the request path.

**Status: experimental.** `feather-api` (Python) is still the reference
server and the only one with the full API. This server covers the search
routes and **every route an agent session sends** (AgentMemory, Pocket,
PacketBuilder and FeatherStore over `RemoteDB`), so a namespace that agents
use can move here. It exists for the namespaces Python cannot carry: one
namespace on feather-api tops out near 50 agent turns/s (the GIL), against
~350 here. See "Measured" below.

Nothing in feather-api or the SDK changes: a client keeps its URL and code,
and a namespace is served by whichever server opens its file.

## Run

```bash
cd feather-server
cargo build --release
FEATHER_DATA_DIR=/data FEATHER_API_KEY=... ./target/release/feather-server --port 8000
```

| variable | default | |
|---|---|---|
| `FEATHER_DATA_DIR` | `/data` | one `<namespace>.feather` per namespace, as feather-api |
| `FEATHER_API_KEY` | (none) | required unless `FEATHER_DEV_MODE=1`; sent as `X-API-Key` |
| `FEATHER_DB_DIM` | 768 | reported dim of a new, empty namespace |
| `FEATHER_SERVER_THREADS` | cores | async worker threads |
| `FEATHER_WRITE_THREADS` | 8 | writes in flight at once |
| `FEATHER_CHECKPOINT_WAL_MB` / `_MAX_AGE_S` / `_POLL_S` | 64 / 300 / 5 | background checkpoints, as feather-api |

**Never run it and feather-api on the same data directory at once.** One
process holds a namespace's write lock, and the second is refused (503
`namespace_held`). This is the engine's single-writer rule, not a choice
made here.

## Routes

| route | |
|---|---|
| `GET /health` | |
| `GET /v1/namespaces` | |
| `POST /v1/{ns}/search` | filters, scoring, `raw_score`, `track`, `vector` or `vector_b64` |
| `POST /v1/{ns}/keyword_search` | |
| `POST /v1/{ns}/hybrid_search` | with `vector` / `vector_b64` only (see below) |
| `GET /v1/{ns}/records/{id}` | `include_vector`, `include_deleted`, `modality`; `ETag` |
| `POST /v1/{ns}/records/get_many` | |
| `POST /v1/{ns}/vectors` | atomic and versioned; `Idempotency-Key` honoured, `ETag` |
| `POST /v1/namespaces` | create, with an optional `dim` |
| `POST /v1/{ns}/batch` | put / upsert / update / link / forget, by id or by key; atomic, compare-and-set, `Idempotency-Key` |
| `GET /v1/{ns}/by_key`, `POST /v1/{ns}/by_keys` | records and ids by native key |
| `GET /v1/{ns}/newest`, `GET /v1/{ns}/count_live`, `GET /v1/{ns}/ids` | ordered, counted and listed reads |
| `GET /v1/{ns}/metadata_namespaces` | |
| `POST /v1/{ns}/heat_select` | Pocket.hot() in the engine |
| `POST /v1/{ns}/records/touch_many`, `POST /v1/{ns}/records/{id}/touch` | salience |
| `POST /v1/{ns}/records/{id}/link` | one edge; `Idempotency-Key`, `ETag` |
| `POST /v1/{ns}/normalize_vectors`, `POST /v1/{ns}/save` | setup and close |
| `GET /v1/{ns}/admin/engine` | what `RemoteDB.dim()`, `storage_stats()` etc. read |
| `PUT /v1/{ns}/records/{id}` | replace metadata, keeping omitted edges / key / valid_at; `If-Match` (412), `Idempotency-Key`, `ETag` |
| `DELETE /v1/{ns}/records/{id}` | forget; `cascade` (default true) prunes edges into it atomically; `Idempotency-Key` |
| `POST /v1/{ns}/erase` | erase the selection and what derives from or cites it; the receipt |

From the batch route down, these are what an agent session sends besides
search and get_many: recorded from AgentMemory, Pocket, PacketBuilder and
FeatherStore over `RemoteDB` (setup, every turn, close), and outside a turn
the corrections and erasure: `Pocket.pin` / `keys.write` (PUT),
`Pocket.forget`, `PocketSession.close()`, `maintain()` (DELETE), and
`Pocket.erase`, `AgentMemory.erase` / `erase_subject`, `FeatherStore.erase`
and the MCP `forget` tools (erase).
`tests/agent_session.py` runs such a session against both servers over HTTP
and fails if any request it sends is not served, or if what the agent reads
back differs. **A namespace is served by one process (its write lock), so a
namespace moved here must answer every route its clients send: a 501 is an
agent that crashes.** Run it after changing the agent layer.

Any other `/v1` route answers **501 `not_implemented`** and names the
Python server. It never answers 404, so a client pointed at the wrong
server can tell what went wrong.

## What differs from feather-api

Checked by `tests/parity.py`, which sends 193 requests to both servers
(reads, filters, scoring, errors, writes, idempotent replays, corrections,
erasure) and compares
status, error type, `ETag` / `Idempotent-Replayed` / `Retry-After`, the JSON
bodies, and the files both servers leave after shutdown:

```bash
.venv/bin/python feather-server/tests/parity.py          # 193 requests, route by route
.venv/bin/python feather-server/tests/agent_session.py   # one agent session, end to end
```

Deliberate differences:

- **`hybrid_search` without a vector** is 501. This server has no embedding
  providers. feather-api embeds `query` itself.
- **`X-Feather-As-Of`** (time travel) is 501 on reads and 400 `read_only`
  on writes, the same as feather-api for writes. Serving a historical read
  from the live namespace would be a wrong answer with a 200.
- **A 422's `detail`** carries serde's wording, not pydantic's. The status,
  `error` and `retryable` fields match.
- **Lax coercion is not accepted**: `"5"` for 5, `"true"` for true,
  `["0.1"]` for a float, or `NaN` literals. feather-api accepts all of these
  for old clients. `feather_db.remote` sends none of them.
- **A `POST /vectors` with invalid metadata** is refused before the
  namespace is created. feather-api creates the file first.
- **Not implemented**: admission control (429s, per-tenant fairness), idle
  close (`FEATHER_IDLE_CLOSE_S`, `FEATHER_MAX_OPEN`), sharding (421),
  uploads, the admin SPA, metrics, and every route not listed above. Of the
  routes `RemoteDB` can call, the ones left are maintenance and history:
  importance, edges, purge, compact, vacuum, forget_expired, auto_link,
  context_chain, graph, history, as_of, record_versions, tags,
  admin/settings. An agent session sends none of them.
- **`POST /erase`** does what feather-api's does to the file, the WAL and
  the graph. feather-api also closes its `as_of` handles and empties its
  embedding cache; this server holds neither.

## Load test

```bash
.venv/bin/python benchmarks/http_server_load.py --rounds 3
.venv/bin/python benchmarks/http_server_load.py --route agent_turn --dim 768   # a recorded agent turn
```

`feather-loadgen` (in this crate) is an open-loop generator. A dedicated
thread schedules requests at a fixed rate, and latency is measured from
the *scheduled* time, so a server that falls behind is charged for the
queue it builds. A closed-loop client would just send less. The script
samples the server process's CPU time around each run, which gives CPU
milliseconds per request.

## Measured

### Agent turns (`bench/reports/http_agent_turn_20261110.md`)

One agent turn (AgentMemory.recall, Pocket.hot, Pocket.recall,
PacketBuilder.build, one episode written) is **18–20 requests over 9
routes**. The 18-request turn replayed at a fixed rate on one namespace
(2,000 episodes, 768-d):

| | feather-api | feather-server |
|---|---:|---:|
| throughput once overloaded | 850–920 req/s = **47–51 turns/s** | ~6,300 req/s = **~350 turns/s** |
| highest offered rate fully served (≥98%) | 500 req/s | 4,000 req/s |
| CPU per request | 1.2–1.5 ms | 0.22–0.30 ms |
| cores busy at the limit | 1.07 (GIL-bound) | 1.55 |

Python's limit is the same at 32-d (44–48 turns/s): it is framework work
under the GIL, not the engine. Run it with
`benchmarks/http_server_load.py --route agent_turn`.

### Search alone

`bench/reports/http_server_20261110.md`: search, 20k x 768, k=10, three
alternating rounds on a loaded M1 laptop (absolute numbers are slow there;
the ratios held in every round).

| | feather-api | feather-server |
|---|---:|---:|
| CPU per search at 1,500/s | 0.89 ms | 0.59 ms |
| of which HTTP layer (engine 0.48 ms) | ~0.41 ms | ~0.11 ms |
| highest rate kept up, per namespace | ~1,500–2,000/s (GIL-bound, 2.2 cores) | 4,000/s (core-bound, 3.5 cores) |
| p50 at 1,500/s | 1.10 ms | 0.68 ms |

The Python server's ceiling comes from its per-request framework work,
which holds the GIL. More uvicorn workers cannot serve one namespace,
because one process holds its write lock. Use this server for a namespace
that needs more than that; below it, the gain is about a third of the CPU
and is not felt as latency.
