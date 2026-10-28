# Scaling the Cloud API out: namespace sharding

One `feather-api` process runs Python, so the work around each request (parsing, validation, JSON) is bound to a single core by the GIL. The engine calls themselves release the GIL. In 0.18 one process served about 490 searches/s at 128-d, whatever the client concurrency. To use more cores or more machines, run several API processes, give each one a subset of the namespaces, and put `feather-gateway` in front.

```
                 ┌──────────────┐
  clients ──────►│   gateway    │  owner(ns) = placement override, else rendezvous hash
                 └──┬───┬───┬───┘  fan-out: /v1/namespaces, /v1/admin/overview
                    │   │   │
              ┌─────▼┐ ┌▼────┐ ┌▼─────┐
              │ api a│ │api b│ │api c │   FEATHER_SHARD_IDS=a,b,c, FEATHER_SHARD_ID=a
              └──┬───┘ └──┬──┘ └──┬────┘
                 └────────┴───────┘
                  shared /data volume (or one volume per shard)
```

## Why this is safe

- **Each namespace has exactly one owner.** A shard only opens namespaces it owns, and answers **HTTP 421** (with `owner_shard`, a shard id) for any other namespace.
  - The owner is the namespace's entry in the placement table, if it has one. Otherwise it is the shard with the highest `blake2b(shard_id, namespace)`: **rendezvous hashing**.
  - A 421 is refused before any work, so the gateway resends it once to the shard it names. Requests keep landing while the gateway's view lags the shards' (mid-move, mid-reshard).
  - Clients that talk to a shard directly get the 421 and should go to `owner_shard` (or the gateway), not retry.
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

`FEATHER_SHARDS` lists the shards as `id=url` (`a=http://api0:8000,b=...`), or as bare URLs, which take ids `0`, `1`, … in order. The old `FEATHER_SHARD_COUNT=N` / `FEATHER_SHARD_INDEX=i` mean ids `"0".."N-1"`.

## Namespaces open on use and close when idle

A shard opens a namespace on its first request, not at startup. It closes one
that has been idle for `FEATHER_IDLE_CLOSE_S` (default 900; 0 = never),
checkpointing it and releasing its file lock. Past `FEATHER_MAX_OPEN` open
namespaces (default 0 = no cap), the least recently used are closed first.

- **Why.** Memory, not CPU, limits a shard, and only open namespaces hold it. 40
  tenants of 5k × 256 (`benchmarks/idle_namespaces.py`): 564 MB with all of
  them open, which is what startup used to do; 77 MB once idle; 116 MB with 4
  hot. That is 73 → 354 tenants per GB, at ~27 ms to reopen one.
- **Closed namespaces stay visible.** They are still listed. `/v1/admin/overview`
  reports a closed one from the stats it had when it closed, so it does not
  open them to count.
- **Racing a close.** A request that took a namespace's handle just before it
  closed is run again on a fresh one: the engine raises `ClosedError` before
  doing anything, and the closer holds the namespace's write lock, so no write
  is half done.

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

## Adding or removing a shard

Rendezvous hashing moves only the namespaces the change has to move: adding a
fifth shard to four moves about 20% of them, all to the new shard. The
`crc32 % N` rule before 0.22 moved about 80%.

1. Stop sending traffic, stop every shard (a clean shutdown checkpoints and
   releases the file locks).
2. Set the new `FEATHER_SHARD_IDS` on every shard and `FEATHER_SHARDS` on the
   gateway.
3. Start the shards, then the gateway.

**Upgrading from `crc32 % N`.** The first start with rendezvous hashing moves
about N/(N+1) of the namespaces once. With a shared volume no files need to
move. Otherwise set `FEATHER_SHARD_HASH=modulo` on the shards and the gateway
to keep the old placement until you can reshuffle.

With one volume per shard, move each `<ns>.feather` (with any `.wal` /
`.wal.old`) to the volume of its new owner before starting.

## Moving one namespace, with no restart

```bash
curl -X POST http://gateway:8080/v1/admin/placement/move \
     -H 'Content-Type: application/json' -d '{"namespace": "nike", "shard": "c"}'
```

For a hot tenant that needs a shard of its own. The gateway:

1. **Gives the target the new placement table.** Its epoch is one higher, and
   from here on the target owns the namespace.
2. **Gives the source the table.** The source lets in-flight writes finish,
   checkpoints and closes the namespace, releasing its file lock, then answers
   421.
3. **Adopts the table itself, then gives it to the other shards.**

**During the move**, requests are either served by the source or redirected to
the target, which opens the file on its first request. Writes during a move:
none lost and none applied twice (`tests/test_gateway.py`).

**Moving back.** Moving a namespace to its hash owner removes its override.
`GET /v1/admin/placement` shows the table.

**Needs a volume both shards see.** A target that cannot see the namespace's
file is refused with 409.

## Limits

- **API keys.** There is still one API key for the whole deployment, now checked by every shard. The gateway forwards the header unchanged.
- **Per-shard admin views.** Metrics and the activity feed are per shard (`?shard=<id>`).
- **Placement is copied to every shard, not consensus.** The gateway pushes the table to every shard; with a shared volume they share one file. A shard that missed an update still gets corrected by 421s, and the gateway reloads the newest table before each move.
- **Cross-namespace operations** such as merging namespaces are not coordinated across shards.
- **Each shard is still one process with one GIL.** Sharding multiplies throughput by the number of shards. Within one shard, 0.19 makes the request path cheaper (binary `vector_b64` input, `include_metadata=false`, orjson responses, a separate write pool, background checkpoints).
