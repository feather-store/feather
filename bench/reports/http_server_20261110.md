# HTTP server: Python (feather-api) vs Rust (feather-server), fixed arrival rate

2026-11-10. Question: is CPU per request what limits the Cloud API, and does
a Rust server change it? Raw rows: `bench/results/http_load__search__20261110.json`.
Harness: `benchmarks/http_server_load.py` (open loop, `feather-loadgen`;
latency from the scheduled send time).

## Setup

- `POST /v1/{ns}/search`, k=10, `vector_b64` (what `feather_db.remote` sends),
  2,000 distinct queries, salience tracking on (the default).
- One namespace, 20,000 x 768 unit vectors with ~300-char agent text, a
  namespace, an entity and two attributes per record.
- feather-api: uvicorn, 1 worker (the Dockerfile's CMD), uvloop + httptools.
  feather-server: tokio, engine calls on the blocking pool. Same engine build
  flags (-O3, NDEBUG), same file.
- 64 keep-alive connections, 15 s per rate, 5 s warm-up per server, 3 rounds
  with the order alternated.
- **Machine: Apple M1 (4P + 4E), on battery (30%), load average 15 from other
  work (a VM, browsers), loadgen on the same machine (2 threads).** Absolute
  numbers are slow, and the tails come from the machine, not the servers: the
  slow requests arrive in bursts that hit every request in flight at once,
  at 8 connections as at 64. Rerun on a quiet Linux host before quoting
  absolutes. The ratios held across all three rounds.

## Results

Median of the rounds at each rate. "kept up": rounds whose achieved rate was
at least 98% of the offered rate (a round stops one rate past the first miss,
so later rates have fewer runs).

| server | offered/s | kept up | p50 ms | p99 ms, median (range) | CPU ms / request | cores |
|---|---:|---:|---:|---:|---:|---:|
| Python | 500 | 2/2 | 1.54 | 171 (96–247) | 1.14 | 0.56 |
| Python | 1,000 | 2/2 | 19.9 | 992 (139–1,845) | 1.10 (0.89–1.32) | 1.09 |
| Python | 1,500 | 1/1 | 1.10 | 63 | 0.89 | 1.32 |
| Python | 2,000 | 1/1 | **43.3** | 617 | 1.10 | 2.17 |
| Python | 3,000 | 0/1 | 3,889 | 8,181 | 1.25 | 2.40 |
| Rust | 500 | 3/3 | 1.20 | 12 (7–135) | 0.83 | 0.41 |
| Rust | 1,000 | 3/3 | 0.81 | 157 (4–947) | 0.68 (0.61–0.76) | 0.67 |
| Rust | 1,500 | 3/3 | 0.68 | 29 (25–105) | 0.59 (0.58–0.66) | 0.88 |
| Rust | 2,000 | 3/3 | 0.65 | 13 (9–687) | 0.60 | 1.19 |
| Rust | 3,000 | 3/3 | 0.76 | 215 (142–532) | 0.74 | 2.18 |
| Rust | 4,000 | 3/3 | 2.30 | 546 (43–605) | 0.88 | 3.47 |
| Rust | 6,000 | 0/3 | 2,049 | 3,632 | 1.07 | 5.15 |

Python fell behind early in two of its three rounds: at 250/s in round 1
(216/s achieved, a multi-second stall; cause not found) and at 1,000/s in
round 3. In the round where it held, it held 1,500/s. Its CPU per request
rises once it queues (1.32 ms in round 3 at 1,000/s, against 0.89 in round 2).

The engine alone, in-process on the same file and queries: **0.48 ms CPU
per search** (0.45–0.55 over three runs of 5,000; wall p50 0.43–0.50 ms).

## What the numbers say

1. **Per-request CPU at 1,500/s, with both servers keeping up:**

   | | total | engine | HTTP layer |
   |---|---:|---:|---:|
   | Python | 0.89 ms | 0.48 | ~0.41 |
   | Rust | 0.59 ms | 0.48 | ~0.11 |

   Rust cuts the HTTP layer about 4x, but total CPU per search only 1.5x
   (−34%), because the engine is now most of the cost.

2. **What limits Python is the serialized part of that CPU, not the total.**
   It saturated at ~2,000/s using 2.2 cores. The engine call releases the
   GIL, so it runs in parallel, but the ~0.4 ms of framework work per request
   does not. A namespace cannot be spread over uvicorn workers either,
   because one process holds its write lock. So **~1,500–2,000 searches/s is
   a per-namespace ceiling for feather-api.** Rust kept up at 4,000/s in
   every round and saturated at ~4,800/s on 5.2 cores. That limit is the
   cores this machine had free, not the server.

3. **Latency is not the reason.** p50 at 1,500/s: 1.10 → 0.68 ms. The 0.4 ms
   saved is 0.05% of an agent turn (Feather's share of a turn is 0.15%,
   bench/reports/e2e_latency_20261108.md). On this machine the p99s are
   machine noise for both servers.

4. Running reads inline on the async workers instead of the blocking pool
   made no difference (0.57 vs 0.59 ms CPU per search at 1,500/s, saturation
   within noise), so feather-server keeps only the pool.

## Verdict

CPU per request limits feather-api only for a single namespace that needs
more than ~1,500 searches/s. Below that, the Rust server saves about a third
of the CPU per search and about 0.4 ms at p50, which matters for the bill at
very large scale and not for anyone waiting on a reply. feather-server
covers the hot routes with the same responses (`feather-server/tests/parity.py`,
65/65 identical). Deploy it for read-heavy namespaces that hit the ceiling.
feather-api stays the reference server for everything else.

## Cost

Local only, no API calls. About 25 minutes of machine time.
