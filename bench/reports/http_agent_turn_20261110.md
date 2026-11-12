# Agent turns per namespace: Python (feather-api) vs Rust (feather-server)

2026-11-10. Question: the search-only test (`http_server_20261110.md`) found
a per-namespace ceiling for feather-api near 1,500–2,000 searches/s. What
does that mean for agents, which send more than searches, and does the Rust
server move it? Raw rows: `bench/results/http_load__agent_turn__20261110_135847.json`
(768-d), `..._135450.json` (32-d, 2 rounds), `..._rust_32d_ceiling.json`.
Harness: `benchmarks/http_server_load.py --route agent_turn`.

## What an agent turn sends

Recorded from the SDK as agents use it (`RemoteDB` under FeatherStore,
AgentMemory, Pocket and PacketBuilder):

| step | requests |
|---|---|
| `AgentMemory.recall(query, kinds=["preferences"])` | search, then by_keys + get_many for its hits |
| `Pocket.hot()` | heat_select, get_many |
| `Pocket.recall(query, k=5)` | 3 × search (own scope and 2 ancestors), touch_many |
| `PacketBuilder.build(query=...)` | 2 × ids, 3 × search, touch_many |
| `AgentMemory.episodes.record(...)` | by_key, records/{id}, by_keys, get_many, batch |

**18–20 requests over 9 routes per turn** (20 when recall finds something).
The replayed turn is the 18-request one: its recall had no hits, so the
turns/s below are, if anything, slightly high for both servers. Setup adds admin/engine, by_key,
metadata_namespaces, normalize_vectors and POST /v1/namespaces; close adds
save; `observations.note` adds records/{id}/link. All of them are now
served by feather-server (`feather-server/tests/agent_session.py` runs a
whole session against both servers: 294 requests over 17 routes, every one
served, the agent's reads identical).

## Setup

- One namespace: 2,000 episodes over 10 users, 500 pocket notes,
  preferences and facts, written through the SDK on the Python server;
  the file copied to each server. 768-d unit vectors (deterministic per
  text), plus a 32-d run.
- The recorded turn replayed by `feather-loadgen --requests` at a fixed
  rate, each turn writing a new episode (its key numbered). 404 is a served
  answer: the turn looks up the episode before writing it.
- 64 keep-alive connections, 8 s per rate, 5 s warm-up per server.
  feather-api: uvicorn, 1 worker. feather-server: defaults.
- **Machine: Apple M1 (4P + 4E), on battery, load average 14–16 from other
  work, loadgen on the same machine.** Absolute numbers are slow and tails
  are noisy. The Python figures agree across all three runs; rerun on a
  quiet Linux host before quoting absolutes.

## Results, 768-d (one round)

| server | offered req/s | achieved req/s | turns/s | p50 ms | p99 ms | CPU ms/req | cores |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python | 250 | 247 | 14 | 1.35 | 28 | 1.54 | 0.36 |
| Python | 500 | 494 | 27 | 3.39 | 120 | 1.37 | 0.64 |
| Python | 1,000 | **919** | **51** | 207 | 650 | 1.21 | 1.06 |
| Python | 1,500 | 853 | 47 | 3,648 | 5,959 | 1.29 | 1.07 |
| Rust | 1,000 | 991 | 55 | 0.31 | 19 | 0.30 | 0.28 |
| Rust | 2,000 | 1,973 | 110 | 0.14 | 140 | 0.25 | 0.46 |
| Rust | 4,000 | 3,950 | 219 | 1.30 | 152 | 0.26 | 0.96 |
| Rust | 6,000 | 5,858 | 325 | 1.26 | 141 | 0.23 | 1.25 |
| Rust | 8,000 | **6,279** | **349** | 1,044 | 3,085 | 0.26 | 1.55 |

32-d, two rounds alternated: Python fell behind at 1,000/s in both (789 and
871 achieved: 44 and 48 turns/s, 0.96–0.99 cores). Rust held 6,000/s in one
round (333 turns/s, 0.56 cores) and missed at 2,000/s in the other with a
2-second stall while its p50 was 0.16 ms (the machine). A Rust-only run then
held 16,000/s (888 turns/s, 1.63 cores) without saturating.

## What the numbers say

1. **The ceiling that fails first at scale is one namespace's agent
   traffic on feather-api: about 50 turns/s.** It is the same at 32-d and
   768-d, and the server sits at one core when it hits it: framework work
   per request under the GIL (1.2–1.5 ms of CPU per request on these routes,
   more than search's 0.89 ms because the bodies are larger: get_many,
   batch, heat_select). An org's agents share one namespace, and a
   namespace cannot be spread over workers (one process holds its file), so
   neither more workers nor more cores move it.
2. **feather-server carries ~7× that on one namespace** (~350 turns/s at
   768-d) at a fifth of the CPU per request. It saturated with 1.55 cores
   busy, so its limit on this run was not cores; the cause was not
   investigated (a loaded machine; one turn in 18 is a write, which takes
   the namespace's write lock and the WAL fsync).
3. **Latency is not the reason**, as before: an agent turn spends ~1 s in
   the LLM and ~0.5 s embedding; the 18 requests are a few ms either way.

## Cost

Local only, no API calls. About 12 minutes of machine time.
