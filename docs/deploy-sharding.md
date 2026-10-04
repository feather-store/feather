# Scaling the Cloud API out: namespace sharding

One `feather-api` process runs Python, so the work around each request (parsing, validation, JSON) is bound to a single core by the GIL. The engine calls themselves release the GIL. In 0.18 one process served about 490 searches/s at 128-d, whatever the client concurrency. To use more cores or more machines, run several API processes, give each one a subset of the namespaces, and put `feather-gateway` in front.

```
                 ┌──────────────┐
  clients ──────►│   gateway    │  owner(ns) = crc32(ns) % N
                 └──┬───┬───┬───┘  fan-out: /v1/namespaces, /v1/admin/overview
                    │   │   │
              ┌─────▼┐ ┌▼────┐ ┌▼─────┐
              │ api 0│ │api 1│ │api N-1│   FEATHER_SHARD_COUNT=N, FEATHER_SHARD_INDEX=i
              └──┬───┘ └──┬──┘ └──┬────┘
                 └────────┴───────┘
                  shared /data volume (or one volume per shard)
```

## Why this is safe

- **Each namespace has exactly one owner.** A shard only opens namespaces where `crc32(name) % N == its index`, and answers **HTTP 421** (with `owner_shard`) for any other namespace.
- **The engine enforces the same rule on disk.** Opening a `.feather` file takes an exclusive OS lock on `<ns>.feather.lock`. If a routing mistake does happen, a second process gets an error instead of corrupting the file. The OS releases the lock when a process dies, so a crashed shard never leaves a stale lock behind.

## Running it

With Docker Compose (4 shards):

```bash
cd feather-api
FEATHER_API_KEY=... docker compose -f docker-compose.sharded.yml up
```

By hand:

```bash
for i in 0 1 2 3; do
  FEATHER_SHARD_COUNT=4 FEATHER_SHARD_INDEX=$i FEATHER_API_KEY=... FEATHER_DATA_DIR=/data \
    uvicorn app.main:app --port $((8000+i)) --workers 1 &
done
FEATHER_SHARDS=http://127.0.0.1:8000,http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003 \
  uvicorn gateway:app --port 8080          # from feather-gateway/
```

`FEATHER_SHARDS` must list the shard URLs **in shard-index order**.

## What the gateway does

| Request | Routed to |
|---|---|
| `/v1/{ns}/…`, `/v1/namespaces/{ns}[/…]` | the owning shard (the response carries an `X-Feather-Shard` header) |
| `POST /v1/namespaces` (`{"name": ns}`) | the owning shard |
| `GET /v1/namespaces` | every shard, with the lists merged |
| `GET /v1/admin/overview` | every shard: totals summed, top namespaces merged |
| `PUT /v1/admin/embedding_config` | broadcast to every shard |
| `POST /v1/admin/upload` | the owning shard. **Add `?namespace=<ns>` or an `X-Feather-Namespace` header**, because the gateway does not parse multipart bodies. |
| other `/v1/admin/*` (metrics, activity, …) | the shard given by `?shard=i`, default 0. These are per-process numbers. |
| `/health` | the gateway plus every shard. Returns 503 if any shard is down. |
| `/admin` SPA, docs | shard 0 |

## Changing the shard count

The ownership rule depends on N, so changing N moves namespaces between shards:

1. Stop sending traffic to the gateway.
2. Stop every shard. A clean shutdown checkpoints and closes each namespace, which releases the file locks.
3. Set the new `FEATHER_SHARD_COUNT` on every shard, and update `FEATHER_SHARDS` on the gateway.
4. Start the shards, then the gateway.

With a shared data volume no files need to move. With one volume per shard, move each `<ns>.feather` file, together with any `.wal` / `.wal.old` files, to the volume of its new owner before starting. Online resharding is not supported.

## Limits

- **API keys.** There is still one API key for the whole deployment, now checked by every shard. The gateway forwards the header unchanged.
- **Per-shard admin views.** Metrics and the activity feed are per shard (`?shard=i`).
- **Cross-namespace operations** such as merging namespaces are not coordinated across shards.
- **Each shard is still one process with one GIL.** Sharding multiplies throughput by the number of shards. Within one shard, 0.19 makes the request path cheaper (binary `vector_b64` input, `include_metadata=false`, orjson responses, a separate write pool, background checkpoints).
