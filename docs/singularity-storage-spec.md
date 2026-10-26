# Feather as Singularity's memory substrate

Feasibility, storage spec, and an answer on deployment count. Written 5 Oct 2026
against Feather at `master` (0.20.0 + 7 unreleased commits), in reply to the six
proposed decisions from the memory study.

Everything numeric below was measured on this machine today, not estimated.
Where a figure is extrapolated it says so.

---

## 0. Measurements the rest of this document rests on

Apple arm64, Python 3.11, `dim=768`, float32, persisted HNSW graph (format v9),
`add_batch` with the GIL released.

| records | file size | build | save | **reopen** | search k=10 |
|--------:|----------:|------:|-----:|-----------:|------------:|
| 10,000  | 34.1 MB   | 7.5 s | 0.06 s | **0.05 s** | 1.74 ms |
| 100,000 | 341.3 MB  | 130.4 s | 0.64 s | **0.62 s** | 2.27 ms |
| 1,000,000 | ~3.4 GB *(extrapolated)* | ~22 min | ~6 s | **~6 s** | ~3 ms |

Namespace prefix resolution (`list_namespaces()` + prefix match, which is how a
scoped read is resolved today):

| namespaces in file | prefix scan |
|-------------------:|------------:|
| 5     | 0.003 ms |
| 105   | 0.013 ms |
| 1,005 | 0.132 ms |
| 5,005 | 0.500 ms |

Three facts drive most of what follows:

1. **Size is the vector.** 3.4 KB/record at dim 768 is 3,072 bytes of float32
   plus graph links and metadata. Record count barely matters; dimensionality
   and quantization do.
2. **`save()` rewrites the whole file.** 0.64 s at 100k, ~6 s at 1M. There is no
   partial write. This is the single most important constraint on the outbox
   design.
3. **Reopen is cheap until it isn't.** 50 ms at 10k is nothing; ~6 s at 1M makes
   a read-your-writes reopen unusable. This bounds brand-file size more tightly
   than disk does.

---

## (a) Feasibility, item by item

### 1. Scopes inside one brand file — **works today, 0.5 days**

`Metadata.namespace_id` is a free-form string with a secondary index, so
`brand.{id}.core`, `.graph`, `.shared.findings`, `.agent.{node}`,
`.agent.{node}.run.{run}`, `.thread.{thread}`, `.user.{user}` are all just
namespace values. `Pocket` already treats dotted scopes as a hierarchy with
inheritance, and `scoped_id(namespace, key)` gives deterministic ids per scope.
A read-only `global.feather` also works: since the unreleased lock change,
`read_only=True` takes no lock, so any number of processes can open it
concurrently.

**One objection, and it is smaller than I expected.** The engine's namespace
index is exact-match, so a *prefix* read (`brand.b1.agent.n3.*`) is resolved by
listing all namespaces and filtering — O(namespaces in file). I expected this to
be the headline problem with run-scoped namespaces, which are unbounded: one new
namespace per agent per run, forever. Measured, it is 0.5 ms at 5,000
namespaces. Linear, so ~5 ms at 50,000 and ~50 ms at 500,000.

That is tolerable for a long time, and the fix is cheap when it stops being:
cache the namespace list and invalidate it on the write generation, exactly as
`Pocket.hot()` already does (171 ms → 0.001 ms by that same trick). **No format
change needed.** Half a day, and I would not do it until a brand actually
crosses ~20k live scopes.

What *does* bite is item 6's GC: without expiry, run scopes accumulate
permanently and every scoped read pays for every run ever executed. The TTL
machinery below handles it, so items 1 and 6 should ship together.

### 2. Per-brand writer service fed by an outbox — **right call, and it fits. 2 days**

**I withdraw my earlier recommendation of the brand VM owning the file.** The
writer service is better, for a reason I had not weighed: a VM owns a *machine*,
while a `.feather` needs a single *process*. A VM running several uvicorn workers
breaks single-writer at startup — the second worker is refused, which looks like
a broken deploy rather than a design violation. A named writer service makes the
one-process constraint explicit and schedulable.

Three things to get right, one of which is a real constraint:

- **Do not checkpoint per commit.** `save()` rewrites the entire file: 0.64 s and
  341 MB of I/O at 100k records. An outbox committing at any rate will spend all
  its time rewriting. Feather already has the answer: the **WAL is durable per
  mutation** (write + fsync, CRC32 per record, v2 format), so a commit is
  recoverable without a checkpoint. Batch `save()` on an interval or a size
  threshold and let the WAL carry durability in between. `add_batch()` already
  does one fsync per batch rather than per record.
- **`feather.committed{seq}` should be emitted by the writer, not read from the
  file.** A reader's view is a snapshot as of its open, so it cannot learn the
  current `seq` from the file without reopening — which is the very thing `seq`
  is supposed to tell it whether to do. Publish the commit sequence out of band
  (the writer already knows it); keep a `seq` record in the file only as a
  recovery/audit anchor.
- **`min_seq` + reopen is viable to ~100k records per brand, not to 1M.** Reopen
  is 50 ms at 10k and 620 ms at 100k — fine on a read-your-writes path. At 1M it
  is ~6 s, which no request can absorb. See the sizing section for how to stay
  under that.

### 3. Per-record `scope`, `trust`, `source_ref`, `version`, `as_of`, `seq` — **storable today, not queryable. 1 day now, the rest needs v10**

`Metadata.attributes` is `string → string`, so all six fields can be written and
read immediately, and `scope` is already `namespace_id`. Storage is not the
problem; **filtering is**. `SearchFilter.attributes_match` is exact string
equality only. The engine's sole numeric comparisons are `timestamp_after` /
`timestamp_before` (int64) and `importance_gte` (float).

So, concretely:

| field | store today | filter today |
|---|---|---|
| `scope` | yes — `namespace_id` | yes, exact + prefix |
| `trust` | yes — attribute | yes, exact match (it is an enum, so exact is enough) |
| `source_ref` | yes — attribute | exact only, which is all it needs |
| `version` | yes — attribute | **no** — `version >= n` is not expressible |
| `as_of` | yes — attribute | **no** — a time window is not expressible |
| `seq` | yes — attribute | **no** — `seq > min_seq` is not expressible |

The three "no"s all mean the same thing: post-filter in Python over the
candidate set, which is correct but O(candidates) and defeats pre-filtered ANN.
They are exactly the fields format v10 was scoped to add as typed and indexed.

**One cheap win available now.** `Metadata.confidence` (float) already exists and
is persisted, but is absent from `SearchFilter`. Adding `confidence_gte`
mirrors `importance_gte` almost line for line — a few hours, no format change —
and gives findings a real ranking filter.

**On trust and taint, a boundary worth stating plainly:** "untrusted content
never satisfies a required rule" is not a storage guarantee and should not be
implemented as one. It is already enforced in the packet layer, which is the
right place — `PacketBuilder`'s `resolve_required` takes constraints from the
authoritative store with **no fallback**, so a `third_party_untrusted` record
cannot become a required rule regardless of what it claims about itself. Taint
*propagation* is likewise application logic: the store can carry a `trust`
attribute faithfully, but it cannot know that a summary derived from an
untrusted page inherits its taint. Put propagation in the extraction path and
treat the attribute as a label the store preserves, never as something the store
enforces.

### 4. `handoff/v1` — **mostly built. 2 days**

`PacketBuilder` already does the hard parts: required refs resolved from their
authority, failing closed with the arithmetic when they are missing or will not
fit; `omitted` recording everything dropped with a reason; `manifest()` storing
`source/key@version` references rather than content, so a packet is revalidated
at action time and cannot carry revoked authority.

Missing, all additive Python: `from`/`to` agent and run, `goal ref@version`,
findings as structured `{claim, evidence_refs, as_of, confidence, trust}`,
`artifacts` as `adam://…@rev`, `entities`, `open_questions`, `allowed_writes`,
`min_seq`, and immutable storage (a packet written to
`brand.{id}.handoff` with `ttl=0` and never updated — the key must include the
run so an upsert cannot silently rewrite history).

One note: `allowed_writes` is a capability grant. Feather cannot enforce it — any
holder of a writable handle can write anything. It must be checked by the writer
service, which is the only process that can.

### 5. Bi-temporal entity graph — **the one real blocker. v10, or 3 days of a workaround**

`Edge` is `{target_id, rel_type, weight}`. There is no room for `valid_from`,
`valid_to`, `recorded_at` or `retracted_at`, and edges are serialised inline in
the owning record, so widening the struct **is** a format change.

Three options, in the order I would consider them:

1. **Wait for v10** and add typed temporal fields to `Edge`. Correct, and it is
   already on the v10 list.
2. **Edges as records.** Represent each edge as its own record in a
   `brand.{id}.graph` scope with `entity_id = "{src}→{dst}"` and the four
   timestamps as attributes. Works today, keeps bi-temporality honest, and loses
   the native reverse index and `context_chain()` traversal — those only
   understand `Edge`. ~3 days including a traversal helper.
3. **Encode times in `rel_type`.** Do not. It makes the string the schema and
   every query a prefix match on formatted dates.

I would do (2) only if the graph is needed before v10, and I would write it so
the record layout maps onto v10's `Edge` later without a data migration.

**Structural-vs-proposal is easy and should be enforced:** `trust=authoritative`
for Mongo-derived structural edges, `trust=third_party_untrusted` (or a new
`llm_proposed`) for extracted ones, and the packet layer already refuses to let
the latter satisfy a constraint.

### 6. Run-scope lifecycle — **best fit of the six. 1 day**

This is nearly free. `Metadata.ttl` (seconds from `timestamp`, `0` = never) is
already a persisted field, and `forget_expired()` already scans and soft-deletes
everything past its TTL. So:

- **TTL**: set `ttl` when writing into `brand.{id}.agent.{node}.run.{run}`.
- **Promotion**: copy the record into `brand.{id}.shared.findings` with `ttl=0`
  and a `promoted_from` attribute. A copy, not a move — the run scope stays an
  audit record until it expires.
- **GC**: `forget_expired()` then `compact()`. Note that `forget_expired()` soft
  deletes; space returns only on `compact()`, and `set_auto_compact(ratio)`
  automates it.

Two traps worth writing down:

- **Soft-deleted records stay visible in the namespace index until compaction.**
  A GC'd run scope still appears in `list_namespaces()`, so scope enumeration
  must filter on `source != "_forgotten"`. This has already bitten me once:
  a retracted rule satisfied its own requirement because the tombstone resolved.
- **Pending deletions disable the fast load path.** The persisted HNSW graph is
  only written when the index holds exactly the live set, so a file with
  un-compacted tombstones falls back to rebuilding the graph on open — turning a
  620 ms reopen at 100k into something much worse. **Compact before the window
  where `min_seq` reopens matter.**

### Summary

| # | item | verdict | effort |
|---|---|---|---|
| 1 | scopes in one file | works today | 0.5 d |
| 2 | writer service + outbox | right call, fits | 2 d |
| 3 | per-record fields | store yes, filter no | 1 d + v10 |
| 4 | `handoff/v1` | mostly built | 2 d |
| 5 | bi-temporal edges | **needs v10** or 3 d workaround | 3 d / v10 |
| 6 | run lifecycle | near-free | 1 d |

Items 1, 2, 3 (storage half) and 6 are ~4.5 days and unblock the architecture.
Item 5 is the one I would not start before v10.

---

## (b) Storage spec

### File layout on disk

```
/var/lib/feather/
  global.feather                 read-only, opened read_only=True by everyone
  brands/
    b_<brand_id>/
      memory.feather             the brand's single writable file
      memory.feather.wal         durable per-mutation log (v2, CRC32/record)
      memory.feather.lock        advisory lock; holder pid written inside
      memory.feather.bak         one rolling backup, written on import/replace
  snapshots/
    b_<brand_id>/
      <utc-iso>.feather          checkpoint copies, pruned by retention
```

One file per brand, one writer process per file. `global.feather` is written by a
separate offline job and published read-only; readers never lock it.

### Index types per scope

One HNSW index per *modality*, not per scope — scopes are metadata partitions
inside the shared index, resolved through the namespace secondary index.

| scope | lookup pattern | index used |
|---|---|---|
| `.core` | semantic + scope filter | HNSW (`text`) + `ns_index_` |
| `.graph` | id and edge traversal | `metadata_store_` + `incoming_index_` |
| `.shared.findings` | semantic, `confidence_gte` | HNSW + post-filter |
| `.agent.{node}` | scope-exact | `ns_index_` (O(matches)) |
| `.agent.{node}.run.{run}` | scope-exact, TTL'd | `ns_index_` + `ttl` sweep |
| `.thread.{thread}` | scope-exact, recency | `ns_index_`, sort by `timestamp` |
| `.user.{user}` | scope-exact | `ns_index_` |
| `global.feather` | semantic only | its own HNSW |

Do not create one modality per scope: each HNSW index carries its own base layer
and link lists, so per-scope indices multiply RAM for no retrieval benefit.

### Record schema

Engine fields carry what the engine can filter; attributes carry the rest.

| Singularity field | where it lives | filterable |
|---|---|---|
| `scope` | `namespace_id` | exact + prefix |
| id | record id = `scoped_id(scope, key)` | direct |
| `key` | `entity_id` | exact |
| text | `content` | BM25 + vector |
| `as_of` | `timestamp` (int64) *and* `_as_of` attribute | **window: yes, via `timestamp`** |
| `trust` | `_trust` attribute | exact |
| `source_ref` | `_source_ref` attribute | exact |
| `version` | `_version` attribute | no |
| `seq` | `_seq` attribute | no |
| confidence | `confidence` (float) | once `confidence_gte` lands |
| TTL | `ttl` (int64) | swept by `forget_expired()` |
| taint | `_taint` attribute (csv of upstream refs) | exact |

Reserve the `_` prefix for Singularity-owned attribute keys so they cannot
collide with domain attributes. **Put `as_of` on `timestamp`**, not only in an
attribute — it is the one temporal field the engine can range-query, and
spending it on `as_of` (when the fact was true) rather than on write time is the
right trade, because write time is recoverable from `seq`.

### Sizes per brand

Measured at dim 768 float32; 1M extrapolated.

| chunks | float32 | int8 on disk (`set_quantized`) | int8 in RAM (`set_int8_ram`) |
|------:|--------:|-------------------------------:|----------------------------:|
| 10k   | 34 MB   | ~12 MB  | RAM ~1.7× lower |
| 100k  | 341 MB  | ~115 MB | ~200 MB resident |
| 1M    | ~3.4 GB | ~1.1 GB | ~2 GB resident |

**Recommendation: cap a brand file at ~250k chunks.** Not for disk — for the
reopen cost that `min_seq` depends on. At 250k reopen is ~1.5 s, which a
read-your-writes path can just about absorb; at 1M it is ~6 s, which it cannot.
Past the cap, shard by scope family (a separate `history.feather` for cold
threads and expired runs) rather than growing one file. Also note `set_quantized`
is lossy and **disables the persisted graph**, so the file reopens by rebuilding
— do not use it on the hot brand file.

### Compaction and snapshots

- **Checkpoint** (`save()`) on an interval or write-count threshold, never per
  commit. Full rewrite: 0.64 s / 341 MB at 100k.
- **Compact** (`compact()`) after GC, before the next read-heavy window. Required
  to reclaim tombstones *and* to re-enable the fast load path.
- `set_auto_compact(ratio)` automates it at a dead-record ratio.
- **Snapshot** = checkpoint, then copy the file. Because `save()` is
  tmp-write-plus-`rename`, a copy taken at any moment is a complete file; it may
  be one checkpoint stale, which is what the WAL is for.
- Snapshot *with* its `.wal` to capture commits since the last checkpoint.

### Backup to S3

```
s3://<bucket>/feather/<brand_id>/<utc-iso>/memory.feather{,.wal}
```

- After each checkpoint, upload if the content hash changed.
- Keep the WAL alongside: file + WAL is the only pair that reproduces the exact
  committed state.
- `.lock` is never backed up — it is process state, and restoring one would make
  a fresh file look locked by a dead pid.
- Restore = download both, open once (WAL replays automatically), `save()`,
  verify `size()` against the recorded count.
- Verify restores on a schedule. A `.feather` whose header parses can still be
  truncated; `DB::open()` guards against that and leaves the original intact
  (`load_complete_`), but an unverified backup is not a backup.

---

## (d) Three Feather deployments — my view

**Two, not three, and for a reason about write patterns rather than tidiness.**

| deployment | shape | verdict |
|---|---|---|
| brand-kit vectors (`ck_{brand_id}`) | per-brand, HTTP, read-heavy | **merge into the brand memory file** |
| memory | per-brand, one writer, read-heavy | the same thing |
| MCP activity ledger | cross-brand, append-only, write-heavy | **keep separate** |

Brand-kit vectors and brand memory have the same tenancy boundary (one brand),
the same access pattern (read-heavy, semantic) and the same writer topology (one
service per brand). They are the same deployment with different scopes —
`brand.{id}.assets` alongside `brand.{id}.core`. Merging them removes a whole
service, and makes "everything we know about this brand" one prefix read instead
of a join across two systems.

The ledger is genuinely different and should not be forced in. It is
append-only, cross-brand, and write-heavy — the exact pattern that conflicts with
a read-heavy single-writer file, because every append either pays a full-file
checkpoint or defers it and grows the WAL. It is also already designed to fall
back to Mongo `mcp_usage_events` when unset, which is the correct home for an
append-only event log.

**Caveat on merging, which is not free.** The brand-kit client is live and
depends on an HTTP contract I have verified exists: `POST /v1/{ns}/vectors`,
`/search`, `/records/{id}`, `/records/{id}/importance`, `/records/{id}/link`,
`DELETE /records/{id}`, `/save`, `/namespaces/{ns}/stats`, plus `x-api-key`
auth (the `Authorization: Bearer` header the client also sends is ignored).
Merging means those routes keep working against a file that now holds memory
too. Do it by adding scopes to the existing namespace, not by migrating the
namespace — `ck_{brand_id}` can stay the namespace name forever.

---

## What (c) needs, and what is not started

Items 1–3 against test data is roughly 3.5 days of the 4.5 above, and I have not
started it. It is also sequenced behind two things that are not mine to decide:

1. **0.21.0 is unreleased**, and it carries the fix for a bug that is live on
   PyPI (`recent()` and `len()` returning wrong answers past a handful of
   records). Building new surface on top of an unreleased stack means Singularity
   would integrate against something no one can `pip install`.
2. **Item 5 should wait for v10**, and v10 also removes the three "not
   filterable" rows in item 3. Doing item 3's post-filter workaround first and
   v10 second means writing the same code twice.

So the order I would propose: release 0.21.0 → items 1, 2, 6 and the
`confidence_gte` addition (~3.5 days, no format change) → format v10 → items 3
(filtering half), 4 and 5 on top of it.
