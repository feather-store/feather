"""
DBManager — Multi-tenant DB router.

Each namespace maps to its own .feather file under FEATHER_DATA_DIR.
This provides full storage-level isolation between tenants.

  FEATHER_DATA_DIR/
    nike.feather
    adidas.feather
    hospital_a.feather
    ...

Open on first use. Startup only lists the files on disk; a namespace is opened
the first time a request needs it. Before this, startup opened every file
serially before the API answered anything, so time-to-ready grew with the total
data size (one 3.4 GB namespace held the whole server down while it loaded).

One lock per namespace. Opening, evicting, adopting and deleting a namespace
take only that namespace's lock, never a global one — a slow open of one
database blocks requests to that database and nothing else.

Memory follows use. With FEATHER_MEMORY_LIMIT_MB set, once the open namespaces'
estimated footprint passes the limit, the least-recently-used ones idle for at
least FEATHER_EVICT_IDLE_S are closed (checkpointed first if their WAL is
non-empty); the next request reopens them. The footprint is the on-disk size
(.feather + WAL), which tracks the in-RAM index closely. RSS does not: the
allocator keeps freed pages, so it would never drop back under the limit.

Single owner: the engine takes an exclusive OS lock on <ns>.feather.lock when a
namespace is opened (0.19), so two processes can never both serve — and
corrupt — the same file. Handles must therefore be closed (not just dropped)
before a file is replaced or deleted.

Sharding (scale-out): run N API processes with FEATHER_SHARD_COUNT=N and
FEATHER_SHARD_INDEX=0..N-1 behind feather-gateway. Each process serves only the
namespaces it owns, owner(ns) = crc32(ns) % N, and answers 421 for the rest.
"""

import os
import sys
import struct
import threading
import time
import zlib
from typing import Dict, List, Optional, Tuple
from feather_db import DB


DATA_DIR = os.getenv("FEATHER_DATA_DIR", "/data")
DEFAULT_DIM = int(os.getenv("FEATHER_DB_DIM", "768"))
SHARD_COUNT = max(1, int(os.getenv("FEATHER_SHARD_COUNT", "1")))
SHARD_INDEX = int(os.getenv("FEATHER_SHARD_INDEX", "0"))
# 0 = no limit: a namespace stays open once opened.
MEMORY_LIMIT_BYTES = int(float(os.getenv("FEATHER_MEMORY_LIMIT_MB", "0")) * 1024 * 1024)
# A namespace used more recently than this is never evicted — stops a namespace
# that is merely between requests from thrashing (close, reopen, close). Safety
# for a request still running is pin()/unpin(), not this.
EVICT_IDLE_S = float(os.getenv("FEATHER_EVICT_IDLE_S", "60"))

# .feather binary format: [magic 4B = "FEAT"] [version 4B]. We accept any
# on-disk format this build can load (v3–v9; load() is backward-compatible).
FEATHER_MAGIC = 0x46454154   # "FEAT"
MAX_FORMAT_VERSION = 9

# Every file that belongs to a namespace besides <ns>.feather itself.
_SIDECARS = (".wal", ".wal.old", ".tmp", ".lock")

_LOCK_HELD = ("holds the write lock", "holds it exclusively", "holds the lock")


def shard_owner(namespace: str, shard_count: int = SHARD_COUNT) -> int:
    """Stable namespace -> shard mapping. crc32, not hash(): Python's str hash
    is randomised per process, so two processes would disagree."""
    return zlib.crc32(namespace.encode("utf-8")) % max(1, shard_count)


class NotOwnedError(Exception):
    def __init__(self, namespace: str, owner: int, this_shard: int):
        super().__init__(namespace)
        self.namespace, self.owner, self.this_shard = namespace, owner, this_shard


class NamespaceUnavailable(Exception):
    """The namespace is on disk but cannot be opened (corrupt, or held by
    another process). Startup used to skip such files and log; with lazy opening
    the failure surfaces on the first request that needs it."""
    def __init__(self, namespace: str, reason: str):
        super().__init__(f"namespace '{namespace}' is unavailable: {reason}")
        self.namespace, self.reason = namespace, reason


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _close_quietly(db: DB, save: bool) -> None:
    try:
        db.close(save=save)
    except Exception as e:  # noqa: BLE001
        print(f"[db_manager] close failed: {e}", file=sys.stderr)


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
                 shard_count: int = SHARD_COUNT, shard_index: int = SHARD_INDEX,
                 memory_limit_bytes: int = MEMORY_LIMIT_BYTES,
                 evict_idle_s: float = EVICT_IDLE_S):
        self._data_dir = data_dir
        self._default_dim = default_dim
        self._shard_count = max(1, shard_count)
        self._shard_index = shard_index
        if not 0 <= shard_index < self._shard_count:
            raise ValueError(f"FEATHER_SHARD_INDEX={shard_index} outside 0..{self._shard_count - 1}")
        self._memory_limit = max(0, memory_limit_bytes)
        self._evict_idle_s = evict_idle_s

        self._dbs: Dict[str, DB] = {}
        self._last_used: Dict[str, float] = {}
        # Per-namespace locks; dict.setdefault is atomic under the GIL, so
        # creating one never needs a global lock.
        #   _ns_locks:    open / evict / adopt / delete, and the get() lookup
        #                 itself, so eviction cannot race a handle being handed out
        #   _write_locks: what lock() gives routes; survives eviction + reopen
        self._ns_locks: Dict[str, threading.Lock] = {}
        self._write_locks: Dict[str, threading.Lock] = {}
        # Last open failure, keyed by the file's (mtime, size), so a corrupt
        # multi-GB file is not re-parsed on every request until it changes.
        self._failed: Dict[str, Tuple[Tuple[int, int], str]] = {}
        # Requests in flight per namespace (pin/unpin). Never evicted while > 0.
        self._inflight: Dict[str, int] = {}
        self._inflight_lock = threading.Lock()   # held for a counter bump only
        self._evicting = threading.Lock()
        self._evict_again = False
        self._opener = DB.open   # seam: tests make one namespace's open slow

        os.makedirs(data_dir, exist_ok=True)
        # Nothing is opened here — see the module docstring.

    # ── sharding ──────────────────────────────────────────────────────────
    def owns(self, namespace: str) -> bool:
        return shard_owner(namespace, self._shard_count) == self._shard_index

    def _check_owner(self, namespace: str) -> None:
        if not self.owns(namespace):
            raise NotOwnedError(namespace, shard_owner(namespace, self._shard_count),
                                self._shard_index)

    def shard_info(self) -> dict:
        return {"shard_index": self._shard_index, "shard_count": self._shard_count}

    # ── lifecycle ─────────────────────────────────────────────────────────
    def data_dir(self) -> str:
        return self._data_dir

    def _namespace_path(self, namespace: str) -> str:
        # Sanitize: only allow alphanumeric, dash, underscore
        safe = "".join(c for c in namespace if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError(f"Invalid namespace: {namespace!r}")
        return os.path.join(self._data_dir, f"{safe}.feather")

    def _ns_lock(self, namespace: str) -> threading.Lock:
        return self._ns_locks.setdefault(namespace, threading.Lock())

    def _on_disk(self) -> List[str]:
        try:
            names = os.listdir(self._data_dir)
        except OSError:
            return []
        return [f[:-len(".feather")] for f in names
                if f.endswith(".feather") and self.owns(f[:-len(".feather")])]

    def _open_namespace(self, namespace: str, dim: Optional[int] = None) -> DB:
        """Open and register. Caller holds this namespace's _ns_lock."""
        path = self._namespace_path(namespace)
        sig = None
        if os.path.exists(path):
            st = os.stat(path)
            sig = (st.st_mtime_ns, st.st_size)
            failed = self._failed.get(namespace)
            if failed is not None and failed[0] == sig:
                raise NamespaceUnavailable(namespace, failed[1])
        # `dim` only sets the reported default for a brand-new empty namespace;
        # an existing file keeps its own dim, and the first inserted vector is
        # what truly fixes it. So passing dim here never overrides real data.
        try:
            db = self._opener(path, dim=dim or self._default_dim)
        except Exception as e:  # noqa: BLE001
            if any(s in str(e) for s in _LOCK_HELD):
                # Transient — never cached, the next request tries again.
                raise NamespaceUnavailable(
                    namespace, "held by another process (is a second server, worker "
                               "or feather-serve using this data dir?)") from e
            reason = f"unloadable: {e}"
            if sig is not None:
                self._failed[namespace] = (sig, reason)
            print(f"[db_manager] namespace '{namespace}' {reason}", file=sys.stderr)
            raise NamespaceUnavailable(namespace, reason) from e
        self._failed.pop(namespace, None)
        self._dbs[namespace] = db
        self._last_used[namespace] = time.monotonic()
        self._write_locks.setdefault(namespace, threading.Lock())
        return db

    def get(self, namespace: str, create: bool = True,
            dim: Optional[int] = None) -> DB:
        """Return the DB for this namespace, opening it on first use and
        creating it if `create`. Raises KeyError if it doesn't exist and
        create=False, NamespaceUnavailable if its file can't be opened."""
        self._check_owner(namespace)
        opened = False
        with self._ns_lock(namespace):
            db = self._dbs.get(namespace)
            if db is None:
                if not create and not os.path.exists(self._namespace_path(namespace)):
                    raise KeyError(f"Namespace '{namespace}' not found")
                db = self._open_namespace(namespace, dim=dim)
                opened = True
            # Stamped under the namespace lock: eviction re-checks idleness
            # under the same lock, so it cannot close a handle being handed out.
            self._last_used[namespace] = time.monotonic()
        if opened and self._memory_limit:
            threading.Thread(target=self.evict, daemon=True,
                             name="feather-evict").start()
        return db

    def pin(self, namespace: str) -> None:
        """Mark a request in flight on `namespace`; eviction skips it until the
        matching unpin(). A request that pins before eviction takes the
        namespace lock is protected outright; one that pins after it simply
        waits in get() for the close and reopens."""
        with self._inflight_lock:
            self._inflight[namespace] = self._inflight.get(namespace, 0) + 1

    def unpin(self, namespace: str) -> None:
        with self._inflight_lock:
            n = self._inflight.get(namespace, 0) - 1
            if n > 0:
                self._inflight[namespace] = n
            else:
                self._inflight.pop(namespace, None)

    def peek(self, namespace: str) -> Optional[DB]:
        """The open handle, or None — never opens or creates."""
        return self._dbs.get(namespace)

    def adopt(self, namespace: str, staged_path: str, overwrite: bool = False) -> DB:
        """Adopt an uploaded .feather file as `namespace`.

        `staged_path` is a fully-written temp file (ideally already inside
        data_dir so the final move is an atomic rename). On success the temp
        file is moved into place and the namespace is opened/served. The temp
        file is always cleaned up on validation failure.

        Raises ValueError (bad file) or FileExistsError (exists, no overwrite).
        """
        self._check_owner(namespace)
        # Cheap structural check (magic + version) before we touch anything live.
        try:
            _validate_feather_header(staged_path)
        except ValueError:
            _safe_remove(staged_path)
            raise

        with self._ns_lock(namespace):
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
            if old is not None:
                _close_quietly(old, save=False)
            self._failed.pop(namespace, None)

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
            # ops on top of it, so remove it.
            for stale in (dest + ".wal", dest + ".wal.old", dest + ".tmp"):
                _safe_remove(stale)

            os.replace(staged_path, dest)   # atomic within the same filesystem

            # Open exactly once — the C++ loader's guards reject corrupt/oversized
            # headers here. On failure, roll back to the backup so the namespace
            # is never left broken.
            try:
                return self._open_namespace(namespace)
            except Exception as e:  # noqa: BLE001
                self._dbs.pop(namespace, None)
                self._failed.pop(namespace, None)
                _safe_remove(dest)
                _safe_remove(dest + ".wal")
                if backup and os.path.exists(backup):
                    try:
                        os.replace(backup, dest)
                        self._open_namespace(namespace)
                    except Exception:  # noqa: BLE001
                        pass
                cause = e.reason if isinstance(e, NamespaceUnavailable) else e
                raise ValueError(f"could not load uploaded .feather: {cause}")

    def lock(self, namespace: str) -> threading.Lock:
        """Return the write lock for this namespace."""
        self.get(namespace)   # ensure it exists
        return self._write_locks[namespace]

    def list_namespaces(self) -> List[str]:
        """Every namespace this process serves — on disk or open, loaded or not."""
        return sorted(set(self._on_disk()) | set(self._dbs))

    def loaded_namespaces(self) -> List[str]:
        """Only the namespaces currently open in memory."""
        return list(self._dbs)

    def footprint(self, namespace: str) -> int:
        """Estimated resident bytes of a namespace: its size on disk."""
        path = self._namespace_path(namespace)
        return _size(path) + _size(path + ".wal") + _size(path + ".wal.old")

    def memory_info(self) -> dict:
        return {"limit_bytes": self._memory_limit,
                "loaded_bytes": sum(self.footprint(ns) for ns in list(self._dbs)),
                "loaded": len(self._dbs)}

    def evict(self) -> List[str]:
        """Close least-recently-used idle namespaces until the loaded footprint
        is under the memory limit. Returns the namespaces closed. Never waits on
        a request: a namespace whose lock or write lock is busy is skipped."""
        if not self._memory_limit:
            return []
        # A pass already running is asked to go round once more instead of
        # this call being dropped: it may have sized the set before our open.
        # The re-check after release closes the gap between a pass's last look
        # at the flag and its release of the lock.
        self._evict_again = True
        evicted: List[str] = []
        while self._evict_again and self._evicting.acquire(blocking=False):
            try:
                while self._evict_again:
                    self._evict_again = False
                    sizes = {ns: self.footprint(ns) for ns in list(self._dbs)}
                    total = sum(sizes.values())
                    for ns in sorted(sizes, key=lambda n: self._last_used.get(n, 0.0)):
                        if total <= self._memory_limit:
                            break
                        if self._evict_one(ns):
                            evicted.append(ns)
                            total -= sizes[ns]
            finally:
                self._evicting.release()
        for ns in evicted:
            print(f"[db_manager] evicted idle namespace '{ns}'", file=sys.stderr)
        return evicted

    def _evict_one(self, namespace: str) -> bool:
        ns_lock = self._ns_lock(namespace)
        if not ns_lock.acquire(blocking=False):
            return False                      # being opened, adopted or handed out
        try:
            db = self._dbs.get(namespace)
            if db is None:
                return False
            if self._inflight.get(namespace):
                return False                  # a request is still using the handle
            if time.monotonic() - self._last_used.get(namespace, 0.0) < self._evict_idle_s:
                return False
            wlock = self._write_locks.get(namespace)
            if wlock is not None and not wlock.acquire(blocking=False):
                return False                  # a write is in progress
            try:
                del self._dbs[namespace]
                # Checkpoint only if there is something to checkpoint: save
                # rewrites the whole file, which for a multi-GB namespace is
                # exactly the cost an idle eviction should not pay.
                try:
                    dirty = db.wal_size() > 0
                except Exception:  # noqa: BLE001
                    dirty = True
                _close_quietly(db, save=dirty)
            finally:
                if wlock is not None:
                    wlock.release()
            return True
        finally:
            ns_lock.release()

    def save_all(self):
        for db in list(self._dbs.values()):
            db.save()

    def close_all(self):
        """Checkpoint and close every namespace (shutdown). Releases the file
        locks so a restarted or replacement process can take over at once."""
        for ns in list(self._dbs):
            with self._ns_lock(ns):
                db = self._dbs.pop(ns, None)
                if db is not None:
                    _close_quietly(db, save=True)

    def save(self, namespace: str):
        db = self._dbs.get(namespace)
        if db is not None:
            db.save()

    def delete(self, namespace: str) -> bool:
        """Hard-delete a namespace: close it + remove .feather, WAL and sidecars.
        Returns True if anything was removed.
        """
        with self._ns_lock(namespace):
            removed = False
            db = self._dbs.pop(namespace, None)
            self._last_used.pop(namespace, None)
            self._failed.pop(namespace, None)
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
