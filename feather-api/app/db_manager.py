"""
DBManager — Multi-tenant DB router.

Each namespace maps to its own .feather file under FEATHER_DATA_DIR.
This provides full storage-level isolation between tenants.

  FEATHER_DATA_DIR/
    nike.feather
    adidas.feather
    hospital_a.feather
    ...

A namespace name IS its file name: validate_namespace() refuses anything else
(400), and _claim_file() refuses a name the filesystem folds onto another open
namespace's file (409). Never map two names to one file — each would hold its
own handle, and each checkpoint would overwrite the other's acknowledged writes.

Thread safety: reads are safe; writes use a per-namespace lock.

Single owner: the engine takes an exclusive OS lock on <ns>.feather.lock when a
namespace is opened (0.19), so two processes can never both serve — and
corrupt — the same file. Handles must therefore be closed (not just dropped)
before a file is replaced or deleted.

Sharding (scale-out): run N API processes behind feather-gateway, each
serving only the namespaces it owns and answering 421 for the rest. Owner =
rendezvous hash over the shard ids, overridden by the placement table
(app/placement.py) — adding a shard moves ~1/(N+1) of the namespaces, and one
namespace can be moved alone, with no restart.

Namespaces open on first use and close when idle (FEATHER_IDLE_CLOSE_S,
default 900) or when more than FEATHER_MAX_OPEN are open (LRU). A shard is
limited by memory, which only open namespaces hold, so idle ones cost a
file on disk and a line in the stats sidecar.
"""

import json
import os
import sys
import struct
import threading
import time
import zlib
from typing import Dict, Optional
import feather_db
from feather_db import DB

from .placement import Placement, owner as placement_owner, shard_config


DATA_DIR = os.getenv("FEATHER_DATA_DIR", "/data")
DEFAULT_DIM = int(os.getenv("FEATHER_DB_DIM", "768"))
SHARD_COUNT = max(1, int(os.getenv("FEATHER_SHARD_COUNT", "1")))
SHARD_INDEX = int(os.getenv("FEATHER_SHARD_INDEX", "0"))
SHARD_HASH = os.getenv("FEATHER_SHARD_HASH", "rendezvous").strip().lower()

# Open on use, close when idle. 0 turns either off.
IDLE_CLOSE_S = float(os.getenv("FEATHER_IDLE_CLOSE_S", "900"))
MAX_OPEN = int(os.getenv("FEATHER_MAX_OPEN", "0"))
STATS_FILE = "_namespace_stats.json"

# .feather binary format: [magic 4B = "FEAT"] [version 4B]. We accept any
# on-disk format this build can load (v3–v10; load() is backward-compatible).
FEATHER_MAGIC = 0x46454154   # "FEAT"
MAX_FORMAT_VERSION = 10

# After a WAL write or fsync fails, the engine refuses every write on that
# handle until the file is reopened. get() reopens it; a reopen that fails
# (the disk is still broken) is retried at most this often, not per request.
RECOVER_RETRY_S = float(os.getenv("FEATHER_RECOVER_RETRY_S", "5"))

# Every file that belongs to a namespace besides <ns>.feather itself.
_SIDECARS = (".wal", ".wal.old", ".tmp", ".lock")


def shard_owner(namespace: str, shard_count: int = SHARD_COUNT) -> int:
    """The owner's index among shards "0".."N-1", by the configured hash and
    no overrides — the rule for a deployment that names shards by index."""
    ids = [str(k) for k in range(max(1, shard_count))]
    return int(placement_owner(namespace, ids, {}, SHARD_HASH))


class NotOwnedError(Exception):
    def __init__(self, namespace: str, owner: str, this_shard: str):
        super().__init__(namespace)
        self.namespace, self.owner, self.this_shard = namespace, owner, this_shard


class NamespaceLoadingError(Exception):
    """The namespace is being loaded (an upload is being adopted). HTTP 503
    with Retry-After: the request is fine, the namespace is not ready yet."""

    def __init__(self, namespace: str):
        super().__init__(f"namespace '{namespace}' is loading; retry shortly")
        self.namespace = namespace


class NamespaceHeldError(Exception):
    """Another process holds the namespace's write lock: during a placement
    move, the source shard until its checkpoint finishes. HTTP 503 with
    Retry-After. If it persists, something else has the file open."""

    def __init__(self, namespace: str, why: str):
        super().__init__(f"namespace '{namespace}' is held by another process; "
                         f"retry shortly ({why.splitlines()[0]})")
        self.namespace = namespace


class InvalidNamespaceError(ValueError):
    """A namespace name that is not exactly its own file name. HTTP 400."""


class NamespaceAliasError(InvalidNamespaceError):
    """A valid name that this filesystem folds onto another open namespace's
    file (case, Unicode form). HTTP 409."""


# The longest name the engine derives from <ns>: <ns>.feather.wal.old.
_LONGEST_SUFFIX = ".feather" + max(_SIDECARS, key=len)
_NAME_MAX = 255                          # bytes per path component (ext4, APFS, NTFS)


def validate_namespace(namespace: str) -> str:
    """Return `namespace` unchanged, or raise InvalidNamespaceError.

    Allowed: letters, digits (Unicode, as str.isalnum() decides), '-' and '_'
    — exactly what the old sanitiser kept, so every name it left alone, and
    every file it ever created, is still valid.

    What changed is the rest. They used to be stripped silently, so `ni.ke`,
    `n.i.k.e` and `nike` were three namespaces — three handles, one per name —
    on ONE file, each saving its own view over the others': a write
    acknowledged on one was gone after the other's next checkpoint, and
    deleting `ni.ke` deleted `nike`'s data. A name is never rewritten now."""
    if not isinstance(namespace, str) or not namespace:
        raise InvalidNamespaceError("namespace must be a non-empty string")
    bad = "".join(sorted({c for c in namespace if not (c.isalnum() or c in "-_")}))
    if bad:
        raise InvalidNamespaceError(
            f"Invalid namespace {namespace!r}: only letters, digits, '-' and '_' "
            f"are allowed (found {bad!r}). Names are not rewritten: an older "
            f"server stored such a name under the name with those characters "
            f"removed, and its data is there.")
    if len((namespace + _LONGEST_SUFFIX).encode("utf-8")) > _NAME_MAX:
        raise InvalidNamespaceError(
            f"Invalid namespace: {len(namespace.encode('utf-8'))} bytes; at most "
            f"{_NAME_MAX - len(_LONGEST_SUFFIX)} fit in a file name with its WAL.")
    return namespace


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _close_quietly(db: DB, save: bool) -> None:
    try:
        db.close(save=save)
    except Exception as e:  # noqa: BLE001
        print(f"[db_manager] close failed: {e}", file=sys.stderr)


def write_error(db: DB) -> Optional[str]:
    """Why `db` refuses writes, or None. An engine older than the
    write-failure handling has no write_error() and never refuses: None."""
    fn = getattr(db, "write_error", None)
    return fn() if fn is not None else None


def _validate_feather_header(path: str) -> int:
    """Read & validate the .feather magic + version. Returns the format version.
    Raises ValueError if the file isn't a loadable .feather."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
    except OSError as e:
        raise ValueError(f"could not read uploaded file: {e}")
    if len(head) < 8:
        raise ValueError("not a .feather file (too small / truncated)")
    magic, version = struct.unpack("<II", head)
    if magic != FEATHER_MAGIC:
        raise ValueError("bad magic bytes — this is not a .feather file")
    if version < 1 or version > MAX_FORMAT_VERSION:
        raise ValueError(
            f"unsupported .feather format v{version} "
            f"(this server loads up to v{MAX_FORMAT_VERSION})"
        )
    return version


class DBManager:
    def __init__(self, data_dir: str = DATA_DIR, default_dim: int = DEFAULT_DIM,
                 idle_close_s: float = IDLE_CLOSE_S, max_open: int = MAX_OPEN):
        self._data_dir = data_dir
        self._default_dim = default_dim
        os.makedirs(data_dir, exist_ok=True)
        me, ids, mode = shard_config()
        self.placement = Placement(data_dir, me, ids, mode)
        self._idle_close_s = idle_close_s
        self._max_open = max_open
        self._dbs: Dict[str, DB] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._recover_after: Dict[str, float] = {}   # ns -> next reopen attempt
        self._file_ids: Dict[str, tuple] = {}         # ns -> (st_dev, st_ino) it opened
        # Namespaces whose file is being loaded OUTSIDE _global_lock (adopt):
        # reserved, so nothing else opens, replaces or deletes them meanwhile.
        self._loading: set = set()
        # Opening or closing on demand, outside _global_lock: a getter waits
        # for the event rather than open a second handle on the same file
        # (the flock is reentrant within a process, so nothing else stops it).
        self._transit: Dict[str, threading.Event] = {}
        self._known: set = set()                      # owned namespaces with a file
        self._last_used: Dict[str, float] = {}
        self._stats: Dict[str, dict] = {}             # as of each namespace's last close
        self._stats_path = os.path.join(data_dir, STATS_FILE)
        self._stop = threading.Event()
        self._sweeper: Optional[threading.Thread] = None
        self._load_stats()
        self._scan_existing()
        if self._idle_close_s > 0 or self._max_open > 0:
            self._sweeper = threading.Thread(target=self._sweep_loop, daemon=True,
                                             name="feather-ns-sweeper")
            self._sweeper.start()

    # ── sharding ──────────────────────────────────────────────────────────
    def owns(self, namespace: str) -> bool:
        return self.placement.owns(namespace)

    def _check_owner(self, namespace: str) -> None:
        owner = self.placement.owner(namespace)
        if owner != self.placement.me:
            raise NotOwnedError(namespace, owner, self.placement.me)

    def shard_info(self) -> dict:
        st = self.placement.state()
        return {"shard_id": st["shard"], "shard_ids": st["shards"], "hash": st["hash"],
                "placement_epoch": st["epoch"]}

    def adopt_placement(self, epoch: int, overrides: Dict[str, str]) -> dict:
        """Take a newer placement table, then hand over every open namespace
        this shard no longer owns: drained (its write lock taken, so no write
        is half done), checkpointed and closed, which releases the file for
        the new owner. Requests for it are answered 421 from here on."""
        adopted = self.placement.adopt(epoch, overrides)
        released = []
        for ns in list(self._dbs):
            if not self.owns(ns) and self.close_namespace(ns, blocking=True):
                released.append(ns)
        for ns in list(self._known):
            if not self.owns(ns):
                self._known.discard(ns)
        if adopted:
            self._scan_existing()          # namespaces it gained
        return {**self.placement.state(), "adopted": adopted, "released": released}

    # ── lifecycle ─────────────────────────────────────────────────────────
    def _scan_existing(self):
        """Note every owned, validly named .feather file in data_dir. Nothing
        is opened: a namespace opens on its first request (it used to open
        all of them at startup, so a shard's memory was the sum of every
        tenant it had ever held)."""
        for fname in sorted(os.listdir(self._data_dir)):
            if not fname.endswith(".feather"):
                continue
            ns = fname[:-len(".feather")]
            try:
                validate_namespace(ns)
            except InvalidNamespaceError:
                print(f"[db_manager] skipping namespace '{ns}': not a valid namespace name",
                      file=sys.stderr)
                continue
            if self.owns(ns):
                self._known.add(ns)

    # ── stats of closed namespaces ────────────────────────────────────────
    def _load_stats(self):
        try:
            with open(self._stats_path) as f:
                self._stats = {str(k): v for k, v in json.load(f).items()}
        except (OSError, ValueError):
            self._stats = {}

    def _save_stats(self):
        tmp = f"{self._stats_path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self._stats, f)
            os.replace(tmp, self._stats_path)
        except OSError:
            pass

    def stats_of(self, namespace: str) -> Optional[dict]:
        """{records, dim, modalities, closed_at} as of the last close."""
        return self._stats.get(namespace)

    def data_dir(self) -> str:
        return self._data_dir

    def _namespace_path(self, namespace: str) -> str:
        return os.path.join(self._data_dir, f"{validate_namespace(namespace)}.feather")

    def _claim_file(self, namespace: str, path: str):
        """Refuse a name that reaches a file another open namespace serves.

        Validation makes the name the file name, but the filesystem can still
        fold two names onto one entry: case on macOS/Windows (`Nike`, `nike`),
        Unicode normalisation on APFS. Two handles on one file each save their
        own view, so one side's confirmed writes vanish. The names are compared
        by what they open — the inode of `<path>.lock`, which the engine creates
        at open anyway — so every kind of aliasing is caught, and a filesystem
        that keeps the names apart (Linux) keeps both namespaces working."""
        marker = path + ".lock"
        try:
            os.close(os.open(marker, os.O_RDWR | os.O_CREAT, 0o644))
            st = os.stat(marker)
        except OSError:
            return None                  # unwritable dir: DB.open reports it
        ident = (st.st_dev, st.st_ino)
        for other, other_ident in self._file_ids.items():
            if other_ident == ident and other != namespace:
                raise NamespaceAliasError(
                    f"Namespace {namespace!r} is the same file as namespace "
                    f"{other!r} on this filesystem (names that differ only in "
                    f"case or Unicode form share one file here). Use {other!r}.")
        return ident

    def _open_namespace(self, namespace: str, dim: Optional[int] = None) -> DB:
        path = self._namespace_path(namespace)
        ident = self._claim_file(namespace, path)
        # `dim` only sets the reported default for a brand-new empty namespace;
        # an existing file keeps its own dim, and the first inserted vector is
        # what truly fixes it. So passing dim here never overrides real data.
        db = DB.open(path, dim=dim or self._default_dim)
        self._dbs[namespace] = db
        if ident is not None:
            self._file_ids[namespace] = ident
        # Kept across a recovery reopen: a writer still holding the old lock
        # must exclude one that takes it after the swap.
        self._locks.setdefault(namespace, threading.Lock())
        return db

    def get(self, namespace: str, create: bool = True,
            dim: Optional[int] = None) -> DB:
        """Return the DB for this namespace, opening it on first use and
        creating it if `create`. A handle that stopped accepting writes is
        replaced by a fresh one first."""
        self._last_used[namespace] = time.monotonic()   # before the read: see close_namespace
        db = self._dbs.get(namespace)
        if db is not None and write_error(db) is None:
            return db
        validate_namespace(namespace)    # every open name was validated; check the rest
        self._check_owner(namespace)
        while True:
            with self._global_lock:
                if namespace in self._loading:
                    raise NamespaceLoadingError(namespace)
                busy = self._transit.get(namespace)
                if busy is None:
                    db = self._dbs.get(namespace)
                    if db is not None:
                        return self._recover_locked(namespace, db)
                    path = self._namespace_path(namespace)
                    if not create and not os.path.exists(path):
                        raise KeyError(f"Namespace '{namespace}' not found")
                    # Reserve the name and the file, then load OUTSIDE the
                    # global lock: a large file takes seconds, and every other
                    # namespace's open, close and delete waits on that lock.
                    ident = self._claim_file(namespace, path)
                    if ident is not None:
                        self._file_ids[namespace] = ident
                    done = self._transit[namespace] = threading.Event()
                    break
            busy.wait()                   # another thread is opening or closing it
        try:
            db = DB.open(path, dim=dim or self._default_dim)
        except Exception as e:
            with self._global_lock:
                self._file_ids.pop(namespace, None)
                self._transit.pop(namespace, None)
            done.set()
            if "holds the write lock" in str(e):
                raise NamespaceHeldError(namespace, str(e)) from e
            raise
        with self._global_lock:
            self._dbs[namespace] = db
            self._locks.setdefault(namespace, threading.Lock())
            self._known.add(namespace)
            self._last_used[namespace] = time.monotonic()
            self._transit.pop(namespace, None)
        done.set()
        return db

    # ── closing idle namespaces ───────────────────────────────────────────
    def close_namespace(self, namespace: str, *, blocking: bool = False,
                        idle_s: Optional[float] = None) -> bool:
        """Checkpoint and close an open namespace; it reopens on its next use.
        With `idle_s`, only if unused for that long. Not `blocking`: skip it
        if a write holds its lock. Returns whether it was closed.

        A request that took the handle just before the close sees ClosedError
        on its next call — raised before the call does anything — and the API
        runs it again on a fresh handle (main._retry_closed)."""
        lock = self._locks.get(namespace)
        if lock is None or not lock.acquire(blocking=blocking):
            return False
        try:
            with self._global_lock:
                db = self._dbs.get(namespace)
                if db is None or namespace in self._transit or namespace in self._loading:
                    return False
                # Re-checked under the lock, after get() stamped it: a request
                # that arrived meanwhile keeps the namespace open.
                if idle_s is not None and time.monotonic() - self._last_used.get(namespace, 0) < idle_s:
                    return False
                del self._dbs[namespace]
                self._file_ids.pop(namespace, None)
                done = self._transit[namespace] = threading.Event()
            try:
                try:
                    self._stats[namespace] = {
                        "records": len([1 for i in db.all_ids() if db.get_metadata(i) is not None]),
                        "dim": db.dim(), "modalities": db.modality_names(),
                        "closed_at": int(time.time())}
                except Exception:  # noqa: BLE001 — stats are a courtesy
                    pass
                _close_quietly(db, save=True)
                self._save_stats()
            finally:
                with self._global_lock:
                    self._transit.pop(namespace, None)
                done.set()
            return True
        finally:
            lock.release()

    def sweep(self) -> list:
        """Close namespaces idle past FEATHER_IDLE_CLOSE_S, then the least
        recently used while more than FEATHER_MAX_OPEN are open."""
        closed = []
        if self._idle_close_s > 0:
            for ns in list(self._dbs):
                if self.close_namespace(ns, idle_s=self._idle_close_s):
                    closed.append(ns)
        if self._max_open > 0 and len(self._dbs) > self._max_open:
            for ns in sorted(self._dbs, key=lambda n: self._last_used.get(n, 0)):
                if len(self._dbs) <= self._max_open:
                    break
                if self.close_namespace(ns, idle_s=1.0):
                    closed.append(ns)
        return closed

    def _sweep_loop(self):
        period = max(1.0, min(30.0, (self._idle_close_s or 60.0) / 4))
        while not self._stop.wait(period):
            try:
                self.sweep()
            except Exception as e:  # noqa: BLE001
                print(f"[db_manager] idle sweep failed: {e}", file=sys.stderr)

    def recover(self, namespace: str) -> Optional[DB]:
        """Reopen `namespace` if its handle stopped accepting writes. Returns
        the handle now serving it (None if it is not open)."""
        with self._global_lock:
            db = self._dbs.get(namespace)
            return None if db is None else self._recover_locked(namespace, db)

    def _recover_locked(self, namespace: str, db: DB) -> DB:
        """Caller holds _global_lock. The new handle is opened BEFORE the old
        one is retired (the file lock is reentrant within this process, and the
        failed handle never writes again), so if the reopen fails the old one
        stays: reads keep being served and writes keep returning its error.

        The old handle is not close()d: requests already running hold it, and
        a search on a closed handle would fail. Dropping our reference frees
        it (no checkpoint, it has none it may make) when the last one ends."""
        why = write_error(db)
        if why is None or time.monotonic() < self._recover_after.get(namespace, 0.0):
            return db
        print(f"[db_manager] namespace '{namespace}' stopped accepting writes "
              f"({why}); reopening to replay its WAL", file=sys.stderr)
        try:
            fresh = self._open_namespace(namespace)
        except Exception as e:  # noqa: BLE001 — keep serving reads from the old handle
            self._recover_after[namespace] = time.monotonic() + RECOVER_RETRY_S
            print(f"[db_manager] reopening '{namespace}' failed: {e}", file=sys.stderr)
            return db
        self._recover_after.pop(namespace, None)
        return fresh                     # _open_namespace replaced self._dbs[namespace]

    def peek(self, namespace: str) -> Optional[DB]:
        """The open handle, or None — never opens or creates."""
        return self._dbs.get(namespace)

    def adopt(self, namespace: str, staged_path: str, overwrite: bool = False) -> DB:
        """Adopt an uploaded .feather file as `namespace`.

        `staged_path` is a fully-written temp file (ideally already inside
        data_dir so the final move is an atomic rename). On success the temp
        file is moved into place and the namespace is opened/served. The temp
        file is always cleaned up on validation failure.

        Raises ValueError (bad file or name) or FileExistsError (exists, no
        overwrite).
        """
        try:
            validate_namespace(namespace)
        except InvalidNamespaceError:
            _safe_remove(staged_path)
            raise
        self._check_owner(namespace)
        # Cheap structural check (magic + version) before we touch anything live.
        try:
            _validate_feather_header(staged_path)
        except ValueError:
            _safe_remove(staged_path)
            raise

        with self._global_lock:
            if namespace in self._loading or namespace in self._transit:
                _safe_remove(staged_path)
                raise NamespaceLoadingError(namespace)
            dest = self._namespace_path(namespace)
            already = namespace in self._dbs or os.path.exists(dest)
            if already and not overwrite:
                _safe_remove(staged_path)
                raise FileExistsError(namespace)

            # Close any live handle first, WITHOUT a checkpoint: the uploaded
            # file replaces its state, and saving would write the old state back
            # over it. Closing also releases the single-owner file lock so the
            # adopted file can be opened.
            old = self._dbs.pop(namespace, None)
            self._file_ids.pop(namespace, None)
            self._locks.pop(namespace, None)
            self._recover_after.pop(namespace, None)
            if old is not None:
                _close_quietly(old, save=False)

            # Back up the existing file so a bad upload (or a regretted overwrite)
            # is recoverable. One rolling backup per namespace.
            backup = None
            if os.path.exists(dest):
                backup = dest + ".bak"
                _safe_remove(backup)
                try:
                    os.replace(dest, backup)
                except OSError:
                    backup = None

            # The uploaded file is authoritative; a stale WAL would replay old
            # ops on top of it, so it goes — with the backup, not away. The old
            # handle was closed without a checkpoint, so writes since its last
            # save exist only in that WAL, and a rollback (or a hand-restored
            # .bak) without it silently lost them.
            for suffix in (".wal", ".wal.old"):
                if backup is not None:
                    _safe_remove(backup + suffix)
                    if os.path.exists(dest + suffix):
                        os.replace(dest + suffix, backup + suffix)
                else:
                    _safe_remove(dest + suffix)
            _safe_remove(dest + ".tmp")

            os.replace(staged_path, dest)   # atomic within the same filesystem
            # Reserve the name AND the file: an alias of it (another case on
            # macOS) opened while this one loads is refused, as it would be
            # once it is open.
            ident = self._claim_file(namespace, dest)
            if ident is not None:
                self._file_ids[namespace] = ident
            self._loading.add(namespace)

        # Load OUTSIDE the global lock. A large upload takes seconds to load,
        # and every other namespace's open, recovery and delete waits on that
        # lock; requests for this namespace get NamespaceLoadingError (503).
        # The C++ loader's guards reject corrupt/oversized headers here.
        try:
            db = DB.open(dest, dim=self._default_dim)
        except Exception as e:  # noqa: BLE001
            # Roll back to the backup so the namespace is never left broken.
            with self._global_lock:
                self._file_ids.pop(namespace, None)
                self._loading.discard(namespace)
                _safe_remove(dest)
                _safe_remove(dest + ".wal")
                if backup and os.path.exists(backup):
                    try:
                        for suffix in (".wal", ".wal.old"):
                            if os.path.exists(backup + suffix):
                                os.replace(backup + suffix, dest + suffix)
                        os.replace(backup, dest)
                        self._open_namespace(namespace)
                    except Exception:  # noqa: BLE001
                        self._dbs.pop(namespace, None)
                        self._locks.pop(namespace, None)
                        self._file_ids.pop(namespace, None)
            raise ValueError(f"could not load uploaded .feather: {e}")
        with self._global_lock:
            self._dbs[namespace] = db
            self._locks.setdefault(namespace, threading.Lock())
            self._loading.discard(namespace)
            self._known.add(namespace)
            self._last_used[namespace] = time.monotonic()
        return db

    def lock(self, namespace: str) -> threading.Lock:
        """Return the write lock for this namespace."""
        self.get(namespace)   # ensure it exists
        return self._locks[namespace]

    def list_namespaces(self):
        """Every namespace this shard serves, open or not."""
        return sorted(self._known | set(self._dbs))

    def open_namespaces(self):
        return list(self._dbs.keys())

    def save_all(self) -> Dict[str, str]:
        """Checkpoint every namespace. One that fails is logged and skipped,
        not allowed to stop the rest; returns {namespace: error}."""
        failed: Dict[str, str] = {}
        for ns, db in list(self._dbs.items()):
            try:
                db.save()
            except Exception as e:  # noqa: BLE001
                failed[ns] = str(e)
                print(f"[db_manager] checkpoint of '{ns}' failed: {e}", file=sys.stderr)
        return failed

    def close_all(self):
        """Checkpoint and close every namespace (shutdown). Releases the file
        locks so a restarted or replacement process can take over at once."""
        self._stop.set()
        if self._sweeper is not None:
            self._sweeper.join(timeout=60)
        with self._global_lock:
            for ns, db in list(self._dbs.items()):
                _close_quietly(db, save=True)
            self._dbs.clear()
            self._file_ids.clear()
            self._locks.clear()

    def save(self, namespace: str):
        if namespace in self._dbs:
            self._dbs[namespace].save()

    def delete(self, namespace: str) -> bool:
        """Hard-delete a namespace: close it + remove .feather, WAL and sidecars.
        Returns True if anything was removed.
        """
        with self._global_lock:
            if namespace in self._loading:
                raise NamespaceLoadingError(namespace)
            if namespace in self._transit:
                raise NamespaceLoadingError(namespace)
            removed = False
            db = self._dbs.pop(namespace, None)
            self._file_ids.pop(namespace, None)
            self._locks.pop(namespace, None)
            self._recover_after.pop(namespace, None)
            self._known.discard(namespace)
            self._last_used.pop(namespace, None)
            if self._stats.pop(namespace, None) is not None:
                self._save_stats()
            if db is not None:
                _close_quietly(db, save=False)   # its files are deleted next
                removed = True
            path = self._namespace_path(namespace)
            for p in (path,) + tuple(path + s for s in _SIDECARS):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        removed = True
                    except Exception:
                        pass
            return removed
