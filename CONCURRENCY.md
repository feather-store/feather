# Feather DB Under Concurrent Load: Problems, Locations and Fixes

> **Scope.** This document covers where Feather DB stops scaling when many requests arrive at once. For each problem it gives the exact place in the code, explains why it hurts, and proposes a fix, ordered from cheap to structural.
>
> **Status: fixed.** Every problem below has been fixed on branch `perf/concurrency`, and the measured before/after numbers are in [CONCURRENCY_RESULTS.md](CONCURRENCY_RESULTS.md). This document describes the v0.18.2 code, and its line numbers refer to that version.

---

## Summary

| # | Problem | Where | Severity | Fix effort |
|---|---|---|---|---|
| 1 | Every write fsyncs the WAL while holding the exclusive lock, blocking all readers | `include/feather.h` `add`, `add_batch`, `update_metadata`, `forget`, … | **High** | Medium |
| 2 | Compaction rebuilds every index under the exclusive lock | `include/feather.h` `compact`, `compact_nolock`, `maybe_auto_compact_nolock` | **High** (with auto-compact on) | Low → Medium |
| 3 | `save()` rewrites the whole file while writers wait | `include/feather.h` `save`, `save_vectors`; `feather-api/app/main.py` `_throttled_save` | Medium | Low → High |
| 4 | `update_metadata` scans the entire reverse index under the exclusive lock | `include/feather.h` `update_metadata` | Medium | Low |
| 5 | The API is one process, with no protection against a second process opening the same file | `feather-api/Dockerfile`, `feather-api/app/db_manager.py` | **High** (safety + scale) | Low → Medium |
| 6 | Python per-request overhead dominates the ~0.2 ms engine search | `feather-api/app/main.py` search routes, `bindings/feather.cpp` | Medium | Low → High |
| 7 | Blocking embedding HTTP calls can starve the shared thread pool | `feather-api/app/main.py` `ingest_text`, `feather-api/app/embedding.py` | Medium | Low |
| – | *(Durability, found along the way)* `save()` renames the file and deletes the WAL without fsyncing the new file | `include/feather.h` `save_vectors` | **High** | Low |
| 8 | **Measured, new:** the reader-preferring `std::shared_mutex` starves writers under read load (768-d: 191 → 0.6 writes/s) | `include/feather.h:69` | **High** | Medium (needs Phase 3 first) |
| 9 | **Measured, new:** `add_batch` builds the HNSW graph inside the exclusive lock (reads −99% while batches run) | `include/feather.h:1160-1209` | **High** | Medium |

> Measured numbers for every row are in [CONCURRENCY_BASELINE.md](CONCURRENCY_BASELINE.md). Rows 8–9 were found by that benchmark; fixes for both are in its §5.

---

## Background: the locking model

Each `feather::DB` has **one** readers-writer lock:

```cpp
// include/feather.h:69
mutable std::shared_mutex mutex_;
// include/feather.h:72
mutable std::mutex save_mutex_;   // serialises save() vs save(); taken BEFORE mutex_
```

- **Shared mode:** `search`, `keyword_search`, `hybrid_search`, `context_chain`, the const accessors and `save`. Any number of these run at the same time. Salience updates are atomic, so they are safe under the shared lock.
- **Exclusive mode:** every mutation. While a writer holds the exclusive lock, **no reader runs**.

Reads scale well because of this design. The problems below all come from how long the exclusive lock is held.

In the Cloud API every namespace is a separate `DB`, so each namespace has its own lock. Load on one namespace does not block another.

---

## 1. Writes hold the exclusive lock through an fsync

### Where

`add()`, in [include/feather.h:1110-1151](include/feather.h#L1110-L1151):

```cpp
std::unique_lock<std::shared_mutex> lock(mutex_);   // 1113  exclusive — all readers now wait
{
    ...                                             // serialize payload (CPU work, under lock)
    wal_append(WalOp::ADD, id, ws.str());           // 1125  fwrite + fflush
    wal_sync();                                     // 1126  fsync / _commit  ← disk latency, under lock
}
... HNSW insert, metadata, indexes ...
```

Other places with the same pattern:

| Method | Lock | fsync |
|---|---|---|
| `add_batch` | [feather.h:1160](include/feather.h#L1160) | [feather.h:1207](include/feather.h#L1207) (once per batch) |
| `update_metadata` | [feather.h:1500](include/feather.h#L1500) | [feather.h:1506](include/feather.h#L1506) |
| `link` | [feather.h:1226](include/feather.h#L1226) | WAL block after line 1233 |
| `forget` | [feather.h:1941](include/feather.h#L1941) | [feather.h:1943](include/feather.h#L1943) |
| `forget_expired` | [feather.h:2016](include/feather.h#L2016) | [feather.h:2040](include/feather.h#L2040) |

`wal_sync()` is defined at [feather.h:661-669](include/feather.h#L661-L669).

### Why it hurts

- An fsync costs about 50 µs to 1 ms on local NVMe, and **1 to 10+ ms** on cloud block storage (EBS, Azure Disk, Fly volumes).
- For that whole time the exclusive lock is held, so **every search on that DB stalls**.
- With `W` writes per second at `t` ms each, readers are blocked for `W × t` ms per second. At 200 writes/s and 3 ms each, readers are locked out about 60% of the time.
- Writes are also fully serialized. Two concurrent `add()` calls pay for two fsyncs one after the other, even though a single fsync could make both durable.

### Fix A — Group commit, with the fsync outside the data lock (recommended)

1. Build the WAL payload **before** taking the lock. It only depends on the arguments.
2. Under the exclusive lock, append the record (`fwrite` + `fflush`, which are cheap), apply the change in memory, and take a **log sequence number** (LSN).
3. Release the lock. Readers resume.
4. Wait until the WAL is durable up to your LSN. One thread performs the fsync, and every writer whose record was appended before it started is covered by that same fsync.

```cpp
// new members
mutable std::mutex              wal_mutex_;       // guards wal_file_, LSNs, syncing flag
mutable std::condition_variable wal_cv_;
uint64_t wal_written_lsn_ = 0;                    // appended + fflush'd
uint64_t wal_synced_lsn_  = 0;                    // on stable storage
bool     wal_syncing_     = false;

// wal_append now returns the record's LSN (takes wal_mutex_ internally)
uint64_t wal_append(WalOp op, uint64_t id, const std::string& payload);

void wal_wait_durable(uint64_t lsn) {
    if (!wal_sync_enabled()) return;
    std::unique_lock<std::mutex> lk(wal_mutex_);
    while (wal_synced_lsn_ < lsn) {
        if (wal_syncing_) { wal_cv_.wait(lk); continue; }   // someone else is syncing
        wal_syncing_ = true;
        uint64_t target = wal_written_lsn_;                  // covers every writer so far
        int fd = fileno(wal_file_);
        lk.unlock();
        platform_fsync(fd);                                  // ONE fsync, many writers
        lk.lock();
        wal_syncing_    = false;
        wal_synced_lsn_ = std::max(wal_synced_lsn_, target);
        wal_cv_.notify_all();
    }
}

void add(uint64_t id, const std::vector<float>& vec, const Metadata& meta, const std::string& modality) {
    std::string payload = build_add_payload(modality, vec, meta);   // no lock held
    uint64_t lsn;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        lsn = wal_append(WalOp::ADD, id, payload);                  // cheap
        /* ... existing in-memory apply, unchanged ... */
    }                                                               // readers resume here
    wal_wait_durable(lsn);                                          // durable before returning
}
```

**Things to handle:**
- **Lock order.** Always take `mutex_` before `wal_mutex_`. The fsync path holds only `wal_mutex_`, and releases even that during the fsync itself, so the two cannot deadlock.
- **`wal_clear()` and `wal_close()`.** They are called from `save()` and the destructor. They must take `wal_mutex_` and wait for `!wal_syncing_` before calling `fclose`. Otherwise an in-flight fsync could run on a closed descriptor. After a checkpoint, set `wal_synced_lsn_ = wal_written_lsn_`, because the base file now holds those records.
- **Semantic change: a write becomes visible before it is durable.** A reader can see a record whose `add()` has not returned yet. If the machine crashes during that window, the record is lost even though someone already read it. The writer still gets its guarantee, since `add()` only returns once the record is durable. Most databases behave this way; Postgres calls it `synchronous_commit` with group commit. If the stricter guarantee is required, keep the fsync inside the lock and use only the group-commit part. That still batches concurrent writers.
- Apply the same pattern to `add_batch`, `update_metadata`, `update_importance`, `link`, `forget` and `forget_expired`.

### Fix B — Cheap mitigations available today

- `FEATHER_WAL_SYNC=0` ([feather.h:610-616](include/feather.h#L610-L616)) turns off the fsync entirely. Writes are then flush-only: they survive a process crash but not a machine crash.
- Use `add_batch` instead of repeated `add` calls. It pays for one fsync per batch.

### Fix C — Finer-grained write locking (later, needs research)

hnswlib has its own per-node locks, and upstream allows `addPoint` to run concurrently with `searchKnn`. **Verify that this holds in our fork before relying on it.** Even if it does, `metadata_store_` and the derived indexes are plain `std::unordered_map`s. They would need concurrent maps, or sharded locks keyed by id, before writers could run alongside readers. This is a large change, so do it only if Fix A turns out not to be enough.

---

## 2. Compaction freezes the DB

### Where

- `compact()` at [include/feather.h:2048-2051](include/feather.h#L2048-L2051) takes the **exclusive** lock and calls `compact_nolock()`.
- `compact_nolock()` at [include/feather.h:381-425](include/feather.h#L381-L425) does all of the following under that lock:
  1. copies every surviving vector out of every modality (`read_vector_internal`)
  2. builds a brand-new HNSW index for each modality, **one point at a time and single-threaded** (`add_point` loop, lines 416-417)
  3. rebuilds the reverse index, the secondary indexes **and the entire BM25 index** (lines 421-423)
- `maybe_auto_compact_nolock()` at [include/feather.h:428-437](include/feather.h#L428-L437) runs **inline**, at the end of:
  - `forget`, [feather.h:1958](include/feather.h#L1958)
  - `purge`, [feather.h:2009](include/feather.h#L2009)
  - `forget_expired`, [feather.h:2041](include/feather.h#L2041)

### Why it hurts

- A rebuild is O(N·log N) HNSW inserts, plus a re-tokenization of every document for BM25. On hundreds of thousands of records that means **seconds with no reads at all**.
- With auto-compaction enabled (`set_auto_compact`), the rebuild fires in the middle of whichever caller's `forget()` crossed the threshold. That request then takes seconds, and so does every request queued behind it.

### Fix A — Stop rebuilding indexes that are already correct (trivial)

`forget`, `purge`, `forget_expired` and `update_metadata` already remove dead records from the secondary indexes and BM25 at the moment they happen:
- `deindex_meta` + `remove_from_bm25_index` at [feather.h:1949-1952](include/feather.h#L1949-L1952), [1979-1982](include/feather.h#L1979-L1982) and [2031-2032](include/feather.h#L2031-L2032)
- the dead-state handling at [feather.h:1517](include/feather.h#L1517) and [1527-1528](include/feather.h#L1527-L1528)

So in `compact_nolock()`:
- Drop `build_secondary_indexes()` and `rebuild_bm25_index()` (lines 422-423).
- Replace `build_reverse_index()` (line 421) with an incremental cleanup: erase `reverse_index_[id]` for each dead id, and remove any entries whose source is a dead id.
- Use `parallel_add` instead of the serial `add_point` loop at lines 416-417, the same way the load path does ([feather.h:1083](include/feather.h#L1083)).

Keep a debug assertion, or a test, that the derived indexes match a full rebuild after compaction.

### Fix B — Reuse deleted slots so compaction is rarely needed (low effort)

Our hnswlib fork already supports this ([include/hnswalg.h:68](include/hnswalg.h#L68), [1074](include/hnswalg.h#L1074)):

```cpp
// constructor flag
HierarchicalNSW(space, max_elements, M, ef_construction, /*random_seed*/100,
                /*allow_replace_deleted=*/true);
// insert
index->addPoint(vec, id, /*replace_deleted=*/true);
```

A new insert then takes over the slot of a vector that was marked deleted, so dead space stops piling up. The places to change are every index constructor (`get_or_create_index` [feather.h:148-164](include/feather.h#L148-L164), `set_int8_ram` [feather.h:2098](include/feather.h#L2098), `compact_nolock` [feather.h:409](include/feather.h#L409)) and `add_point` ([feather.h:180-192](include/feather.h#L180-L192)).

Caveats:
- hnswlib notes that replacing a slot assumes "no concurrent operations on deleted element". Our writers hold the exclusive lock, but `parallel_add` inserts from many threads at once. **Verify** that concurrent replacement is safe in the fork, or turn replacement off inside `parallel_add`.
- Graph quality slowly degrades with heavy replacement, so an occasional full compaction is still worth running.

### Fix C — Background compaction with a short swap (structural)

Move the expensive work out of the lock:

1. **Shared lock:** snapshot the live `(id, vector)` pairs for each modality and start a *delta log*. From then on, every writer records the ids it adds or forgets in that log, under its own exclusive lock.
2. **No lock:** build the new indexes with `parallel_add`.
3. **Exclusive lock, short:** replay the delta onto the new indexes (adds → `add_point`, forgets → `markDelete`), swap the index pointers, erase dead metadata, and clean up the reverse index incrementally.

In addition, change `maybe_auto_compact_nolock()` so it only **signals** a background compaction thread (a flag plus a condition variable) instead of compacting inline. The `forget()` that crosses the threshold then returns immediately.

---

## 3. `save()` rewrites the entire file while writers wait

### Where

- `save()` at [include/feather.h:2112-2121](include/feather.h#L2112-L2121) holds `save_mutex_` and `mutex_` in **shared** mode while `save_vectors()` runs.
- `save_vectors()` at [include/feather.h:836-937](include/feather.h#L836-L937) writes **all** metadata and **all** vectors or graphs to `.tmp`, then renames it. The cost is O(DB size), paid on every save.
- The API calls it:
  - after every single-record mutation: for example the delete route (`db.save()` near [feather-api/app/main.py:819](feather-api/app/main.py#L819)), plus unlink, purge and compact
  - at most every `FEATHER_IMPORT_SAVE_INTERVAL_S` (30 s) during bulk writes, via `_throttled_save` at [feather-api/app/main.py:343-349](feather-api/app/main.py#L343-L349), which runs **inside** `manager.lock(namespace)` (for example [main.py:1449](feather-api/app/main.py#L1449))

### Why it hurts

- Readers keep going during a save, because it only takes the shared lock. **Writers are blocked** for the whole file write, which can take seconds for a large namespace.
- A busy namespace under import therefore has its writes stalled every 30 s.
- Saving on every delete makes one delete cost O(DB size).

### Fixes, from cheapest

1. **Save less often and let the WAL provide durability.** The WAL already makes each mutation crash-safe, so a full save is a *checkpoint*, not a durability step. In the API:
   - Remove the per-mutation `db.save()` calls, since the WAL covers them.
   - Checkpoint based on WAL size (for example, once the WAL passes 64 MB) or on a timer from a background task that does not hold `manager.lock`.
2. **Make `save()` non-blocking for writers.** Under the lock, take a cheap snapshot (copy the metadata map and the vector buffers), release the lock, then serialize outside it. This roughly doubles peak memory for the duration of the save. The WAL must only be truncated up to the LSN the snapshot captured, not cleared completely, because new writes have landed since. This depends on the LSN from §1.
3. **Segmented storage (structural).** Write immutable segment files plus the WAL, and have checkpoints write only new segments, merging them in the background (LSM-style). This is the long-term answer if namespaces grow large. It is also a file-format change, so it would be format v10.

### Durability bug found while reviewing `save_vectors`

[feather.h:931-936](include/feather.h#L931-L936):

```cpp
f.close();                                   // data may still be in the OS page cache
std::rename(tmp_path.c_str(), path_.c_str());
wal_clear();                                 // WAL deleted
```

The `.tmp` file is **never fsynced** before the rename, and neither is the directory after it. If the machine loses power soon after a save, the renamed file can come back empty or partially written. The WAL is already gone, so data acknowledged earlier is lost.

**Fix:**
1. Write the file with `FILE*` / `fd` instead of `std::ofstream`, so we have a descriptor to sync.
2. `fsync` the `.tmp` file.
3. `rename` it.
4. On POSIX, `fsync` the parent directory. On Windows, use `MoveFileEx(..., MOVEFILE_WRITE_THROUGH)`.
5. Only then call `wal_clear()`.

This is a small change with a large benefit, and it is **not** a concurrency issue. Fix it regardless of the other work in this document.

---

## 4. `update_metadata` scans the whole reverse index

### Where

[include/feather.h:1518-1523](include/feather.h#L1518-L1523), inside the exclusive lock:

```cpp
for (auto& [target, incoming_list] : reverse_index_) {       // every target in the DB
    incoming_list.erase(std::remove_if(... ie.source_id == id ...));
}
```

### Why it hurts

This costs O(total edges) on **every** metadata update, and readers wait the whole time. The API calls it on a record `PUT`, and `MemoryManager.assign_tiers()` calls it once per node.

### Fix

The old record's own `edges` list already says exactly which targets to touch:

```cpp
if (old != metadata_store_.end())
    for (const auto& e : old->second.edges) {
        auto rit = reverse_index_.find(e.target_id);
        if (rit == reverse_index_.end()) continue;
        auto& v = rit->second;
        v.erase(std::remove_if(v.begin(), v.end(),
                [id](const IncomingEdge& ie) { return ie.source_id == id; }), v.end());
    }
```

That makes the update O(edges of this record). Add a test that edits a record's edges and then checks `get_incoming`.

---

## 5. The API is a single process, with no guard against a second one

### Where

- [feather-api/Dockerfile:62](feather-api/Dockerfile#L62) runs `uvicorn ... --workers 1`.
- [feather-api/deploy/railway.json:9](feather-api/deploy/railway.json#L9) also uses `--workers 1`.
- In [feather-api/app/db_manager.py:65-67](feather-api/app/db_manager.py#L65-L67), open handles and locks live in plain in-process dicts and `threading.Lock`s.
- [db_manager.py:106-116](feather-api/app/db_manager.py#L106-L116): `get()` opens the file and caches the handle in this process only.
- **There is no file lock anywhere**, in either the API or the C++ `DB::open` ([feather.h:1093-1105](include/feather.h#L1093-L1105)).

### Why it hurts

- **Scale.** One process means one GIL. Everything except the C++ search call (see §6) runs on one core, and you can only scale by moving to a bigger machine.
- **Safety.** If anyone raises `--workers`, or points the MCP server (`feather-serve --db`) or a script at a file the API already has open, two processes will:
  - each replay the same WAL
  - each append to it
  - each overwrite the `.feather` file on `save()`

  The result is silent data loss or corruption.

### Fix A — An exclusive file lock in `DB::open` (do this first)

Put the lock in C++, so it protects every consumer: the API, MCP, the Rust CLI and scripts.

```cpp
// on open: create/open  path + ".lock"  and take an exclusive, non-blocking lock
//   POSIX:   flock(fd, LOCK_EX | LOCK_NB)
//   Windows: LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, ...)
// if it fails → throw std::runtime_error("database is open in another process: " + path)
// release in ~DB()  (note: py::nodelete means Python never runs ~DB — also release
//                    in an explicit close() method exposed to Python)
```

The OS releases the lock automatically when the process dies, so a crash cannot leave a stale lock behind.

The same `py::nodelete` detail matters in the API too: `DBManager.adopt()` and `delete()` drop handles without destroying them ([db_manager.py:142-145](feather-api/app/db_manager.py#L142-L145)). They will need an explicit `db.close()` that releases the lock, or re-adopting a namespace will fail.

### Fix B — Shard namespaces across processes (scale out)

Namespaces are already independent files, so this fits naturally:

- Run **N** API processes. Each owns the namespaces where `hash(ns) % N == i`.
- Put a router in front (nginx, Envoy or a small gateway) that picks the backend from the `/v1/{namespace}/…` path segment using consistent hashing.
- Cross-namespace routes such as `GET /v1/namespaces` and `GET /v1/admin/overview` need to fan out to every process and merge the results.
- The file lock from Fix A makes a routing mistake fail loudly instead of corrupting data.

This gives multi-core, and multi-machine with shared or attached disks, without changing the engine.

---

## 6. Python overhead around a 0.2 ms search

### Where

Search route, [feather-api/app/main.py:677-715](feather-api/app/main.py#L677-L715):

| Step | Location | GIL held |
|---|---|---|
| JSON body → Pydantic `SearchRequest`, including a `List[float]` of 768–3072 floats | `models.py`, FastAPI | yes |
| `_build_filter`, `_build_scoring`, `_check_query_dim` | [main.py:683-685](feather-api/app/main.py#L683-L685) | yes |
| Python list → `std::vector<float>` | [bindings/feather.cpp:192-194](bindings/feather.cpp#L192-L194) | yes |
| **C++ search** | [bindings/feather.cpp:198-199](bindings/feather.cpp#L198-L199) | **no** |
| C++ `SearchResult` → Python objects (each holds a full `Metadata` copy) | pybind return | yes |
| `raw_score`: `get_vector()` plus a numpy cosine **per hit** | [main.py:692-706](feather-api/app/main.py#L692-L706) | yes |
| `SearchResultItem` + `_meta_to_model()`, which copies the attributes dict and links | [main.py:708-712](feather-api/app/main.py#L708-L712), [main.py:214-228](feather-api/app/main.py#L214-L228) | yes |
| `response_model` validation + JSON encoding | FastAPI | yes |
| Metrics middleware (`@app.middleware("http")`, which uses Starlette's `BaseHTTPMiddleware`) | [main.py:96-111](feather-api/app/main.py#L96-L111) | yes |

`keyword_search` ([main.py:1063](feather-api/app/main.py#L1063)) and `hybrid_search` ([main.py:1080](feather-api/app/main.py#L1080)) follow the same shape.

### Why it hurts

Only one row runs in parallel. Everything else serializes on the GIL. Parsing and validating a 1536-float JSON array alone likely costs more than the search. The process therefore hits 100% of **one** core while the C++ engine sits mostly idle.

### Fixes, from cheapest

1. **Binary vector input.** Accept an optional `vector_b64` field containing little-endian float32 bytes, decoded with `np.frombuffer(base64.b64decode(...), dtype=np.float32)`. Keep `vector` for compatibility. This skips parsing and validating thousands of floats.
2. **Skip response validation.** Return an `ORJSONResponse` built from plain dicts instead of going through `response_model`, or use `Model.model_construct(...)`. `orjson` is several times faster than the standard encoder.
3. **Compute `raw_score` in C++.** Add a flag to `DB::search` that returns the true cosine alongside the score, using the vector the engine already has. This removes one `get_vector` copy and one numpy call per hit.
4. **Let callers skip metadata.** An `include_metadata=false` option, or a binding that returns only numpy arrays of `ids` and `scores`, avoids copying `Metadata` into Python when the caller only needs ids.
5. **Replace the metrics middleware** with a pure ASGI middleware. `BaseHTTPMiddleware` adds measurable per-request overhead.
6. **Scale out** with §5B, which gives multiple processes and so multiple GILs.
7. **Structural option: a native server.** Write an async Rust server (axum/tokio) or a Go server that calls the existing C ABI in [src/feather_core.cpp](src/feather_core.cpp). That ABI must first be extended with the hybrid, keyword, filter, namespace and graph functions it currently lacks. This removes the GIL completely.

   **Do not rewrite the engine itself in Rust for this.** The engine is already native C++, and none of the problems in this document are caused by the language.

---

## 7. Blocking embedding calls can starve the thread pool

### Where

- `ingest_text` at [feather-api/app/main.py:1422-1425](feather-api/app/main.py#L1422-L1425) calls `EMBEDDING.embed(req.text)` synchronously.
- `embed()` at [feather-api/app/embedding.py:117](feather-api/app/embedding.py#L117) uses `urllib.request.urlopen(..., timeout=...)` at [embedding.py:162](feather-api/app/embedding.py#L162). That means **a new connection and TLS handshake on every call**.
- Every route is a sync `def`, so FastAPI runs them all in **one shared AnyIO thread pool**, which defaults to 40 threads.

### Why it hurts

Embedding providers take 50–500 ms per call. Forty concurrent `ingest_text` requests occupy all 40 threads, and **searches start queueing behind them** even though a search needs about 1 ms. A slow provider therefore degrades every endpoint.

### Fixes

1. **Make `ingest_text` async.** Declare it `async def` and call the provider with a shared `httpx.AsyncClient`. The client's connection pool keeps connections alive, which removes the per-call TLS handshake. Then hand only the engine write to a thread, with `await run_in_threadpool(db.add, ...)`.
2. **Micro-batch.** Collect concurrent `ingest_text` texts for about 5–10 ms and send them as one batched provider request. OpenAI, Voyage and Cohere all accept a list of inputs. This cuts both latency and cost.
3. **Isolate the capacity.** If the route stays synchronous, put embedding calls behind their own bounded executor or semaphore, as `/import` already does ([main.py:307-321](feather-api/app/main.py#L307-L321)), so they can never occupy the whole shared pool.
4. **Cache.** Keep a small LRU keyed by `(provider, model, sha256(text))`. Agents often re-ingest identical text.

---

## 8. Recommended order of work

| Step | Change | Section | Why this order |
|---|---|---|---|
| 1 | fsync before rename in `save_vectors` | §3 | Fixes a data-loss bug, and it's tiny |
| 2 | File lock in `DB::open` + `close()` binding | §5A | Prevents corruption before anyone tries scaling out |
| 3 | O(edges) reverse-index cleanup in `update_metadata` | §4 | Tiny, and removes an O(N) exclusive section |
| 4 | Drop redundant index rebuilds, and use `parallel_add` in `compact_nolock` | §2A | Tiny; shortens the compaction stall a lot |
| 5 | Remove per-mutation `save()` from the API; checkpoint by WAL size | §3 | Removes O(DB) work from single-record requests |
| 6 | Group commit with fsync outside `mutex_` | §1A | The biggest read-latency win under write load |
| 7 | Async `ingest_text` + httpx pool + micro-batching | §7 | Stops embedding latency from spilling into search |
| 8 | Binary vectors in, orjson out, cosine in C++ | §6 | Raises API throughput per core |
| 9 | Background compaction / slot reuse | §2B–C | Removes the remaining multi-second stalls |
| 10 | Namespace sharding across processes | §5B | Multi-core and multi-machine scale |
| 11 | *(Only if still needed)* Native Rust/Go server over the C ABI | §6.7 | Largest effort; do it once 1–10 are measured |

---

## 9. Measure first

Run this before changing anything, then again after each step. It measures how much searches stall while writes are running. **The script has not been run yet.**

```python
# bench_concurrency.py — run from feather/ after `python setup.py build_ext --inplace`
import os, time, threading, statistics, tempfile, numpy as np, feather_db as fdb

DIM, N, READERS, SECONDS = 768, 50_000, 8, 10
path = tempfile.mktemp(suffix=".feather")
db = fdb.DB.open(path, dim=DIM)
rng = np.random.default_rng(0)
db.add_batch(np.arange(N, dtype=np.uint64), rng.random((N, DIM), dtype=np.float32))

def run(writers: int):
    lat, stop, nid = [], threading.Event(), [N]
    def reader():
        q = rng.random(DIM, dtype=np.float32)
        while not stop.is_set():
            t = time.perf_counter(); db.search(q, k=10, record_salience=False)
            lat.append((time.perf_counter() - t) * 1000)
    def writer():
        while not stop.is_set():
            nid[0] += 1; db.add(id=nid[0], vec=rng.random(DIM, dtype=np.float32))
    ts = [threading.Thread(target=reader) for _ in range(READERS)] + \
         [threading.Thread(target=writer) for _ in range(writers)]
    [t.start() for t in ts]; time.sleep(SECONDS); stop.set(); [t.join() for t in ts]
    lat.sort()
    print(f"writers={writers}  searches={len(lat)}  "
          f"p50={lat[len(lat)//2]:.2f}ms  p99={lat[int(len(lat)*.99)]:.2f}ms  "
          f"max={lat[-1]:.1f}ms  WAL_SYNC={os.getenv('FEATHER_WAL_SYNC','1')}")

run(0); run(1); run(4)
t = time.perf_counter()
for i in range(0, N, 3): db.forget(i)
print("forget 1/3 done"); t = time.perf_counter(); db.compact()
print(f"compact() held the exclusive lock for {(time.perf_counter()-t):.2f}s")
```

To compare durability settings, run it twice: once as is, and once with `FEATHER_WAL_SYNC=0`. Run it on the same kind of disk production uses, because fsync cost varies greatly between laptops and cloud volumes.

For the API, drive `POST /v1/{ns}/search` with a load generator such as `oha` or `locust`. Run one test with only searches, and another where `ingest_text` traffic runs alongside them. Compare p99 latency and CPU usage per core.
