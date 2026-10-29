# Feather's contracts

What Feather promises, the limits of each promise, and the tests that hold it
to the promise. A change that breaks one of these fails a named test. Changing
a contract means changing this file, its tests and the CHANGELOG together.

## 1. Exact results by default

Every query answers exactly unless the caller opted into approximation:

- filtered vector search (namespace, namespace prefix, entity, attribute, type,
  tags, time, importance, confidence): the true nearest k among the records
  that match, and never fewer than k when k match;
- keyword search, hybrid search's keyword half, `newest()`, key lookups, id
  listings, versions.

**Approximate by design:** unfiltered vector search, which walks an HNSW graph
and finds the true nearest neighbours most of the time, not always (recall@10
≈ 0.99 at the default `ef`; raise it with `set_ef`).

**Opt-ins that trade exactness for speed:**

- `FEATHER_NS_ANN_MIN` / `set_namespace_ann_min(n)`: per-namespace graphs.
- `FEATHER_PREFILTER_MODE=auto`: filtered walks of the global graph.
- `set_int8_ram`: in-RAM int8 vectors. `refine` re-ranks by the float
  originals.

Each one is off until asked for. `FEATHER_PREFILTER_MODE=exact` forces exact
filtered search everywhere.

Tests:

- `test_contracts.py`: `test_filtered_search_is_exact`,
  `test_keyword_search_and_newest_are_exact`, `test_namespace_graphs_are_opt_in`;
- `test_filtered_search_fast.py::test_default_route_is_exact`.

## 2. One writer, many readers

- **One writer.** One process at a time holds a file for writing: an exclusive
  `flock` on `<path>.lock`. A second writer is refused with the holder's pid.
  Writes in that process are serialised; reads run concurrently with each
  other.
- **Readers.** Any number of processes may open the file `read_only=True`.
  They take no lock, are never blocked by the writer, and every mutation on
  their handle raises.

**Limits:**

- Not two writers on one file: the second is refused, not queued.
- The lock is reentrant within one process. Use one handle per file there.
- `FEATHER_LOCK=0` turns locking off (NFS, some overlay mounts). The caller
  then owns the consequence.

Tests: `test_contracts.py::test_one_writer_many_readers`, `test_file_locking.py`.

## 3. Read-only means a consistent snapshot

A read-only open sees exactly the state of one commit, plus a prefix of the
writes logged after it. Atomic batches are whole or absent. It never sees a
commit's base together with later writes but without the writes in between.
The snapshot is the moment of the open: reopen to see newer writes.

**How:** every WAL file starts with a `WALGEN` record naming the commit
generation its writes follow. A reader checks that base → `<wal>.old` →
`<wal>` is unbroken, and starts over when a commit finished between its reads.
A chain that is broken the same way twice is the files as they are (a crash,
a WAL removed by hand), not a race, and is loaded as a writer would load it.

**Before this, 187 of 7,395 read-only opens under a saving writer saw a torn
view.**

**Limits:**

- Files written with `FEATHER_FORMAT=9` (the rollback format) carry no
  generations, so they get no check.
- A WAL written before 0.22 has no `WALGEN` record and is not judged.

Test: `test_contracts.py::test_a_read_only_open_is_a_consistent_snapshot`. The
writer commits and vacuums while readers open. A gap, a partial batch or a
wrong record fails it.

## 4. One feature set in every mode

Embedded Feather (`feather_db.DB`) and the Cloud API (`feather_db.RemoteDB`, or
plain HTTP) offer the same operations with the same results:

- **Every operation is mapped.** Every public `DB` method has a route.
  `feather_db.remote.PARITY` maps each one, or states why the handle itself
  has no HTTP form: `open`, `close`, `write_error`, `idempotent_result`, which
  is the `Idempotency-Key` header.
- **Same calls.** `RemoteDB` takes the same arguments, returns objects with the
  same attributes, and raises the same exceptions: `VersionConflictError`,
  `DuplicateKeyError`, `IdempotencyKeyReusedError`, `WriteFailedError`,
  `ValueError`.
- **One defaulting rule.** A field a request omits takes the API's default; a
  value it gives (`0` and `""` included) is stored as given.

**Rules both modes share:**

- **A forgotten record is gone.** `get_metadata` returns `None`, the same before
  and after a reload. `include_dead=True` shows the tombstone until the next
  save.
- **`link(a, b)`** needs a live `a` and returns whether it linked; `b` may be a
  forward reference.
- **A `WriteBatch` link** needs both ends live unless `require_target=False`.
- **What a crash recovers** is what the process had, record versions included.

**Not covered:**

- The Rust CLI is a deliberately small subset (add, search, link, save).
- The MCP servers expose agent-shaped tools, not the `DB` API.

Tests:

- `test_parity.py`: a public `DB` method without a route, a mapped route that
  does not exist, or a `RemoteDB` method with different arguments.
- `test_contracts.py`: every test taking `db` runs against both modes.
