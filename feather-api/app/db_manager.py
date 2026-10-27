"""
DBManager — Multi-tenant DB router.

Each namespace maps to its own .feather file under FEATHER_DATA_DIR.
This provides full storage-level isolation between tenants.

  FEATHER_DATA_DIR/
    nike.feather
    adidas.feather
    hospital_a.feather
    ...

Thread safety: reads are safe; writes use a per-namespace lock.

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
import zlib
from typing import Dict, Optional
from feather_db import DB


DATA_DIR = os.getenv("FEATHER_DATA_DIR", "/data")
DEFAULT_DIM = int(os.getenv("FEATHER_DB_DIM", "768"))
SHARD_COUNT = max(1, int(os.getenv("FEATHER_SHARD_COUNT", "1")))
SHARD_INDEX = int(os.getenv("FEATHER_SHARD_INDEX", "0"))

# .feather binary format: [magic 4B = "FEAT"] [version 4B]. We accept any
# on-disk format this build can load (v3–v9; load() is backward-compatible).
FEATHER_MAGIC = 0x46454154   # "FEAT"
MAX_FORMAT_VERSION = 9

# Every file that belongs to a namespace besides <ns>.feather itself.
_SIDECARS = (".wal", ".wal.old", ".tmp", ".lock")


def shard_owner(namespace: str, shard_count: int = SHARD_COUNT) -> int:
    """Stable namespace -> shard mapping. crc32, not hash(): Python's str hash
    is randomised per process, so two processes would disagree."""
    return zlib.crc32(namespace.encode("utf-8")) % max(1, shard_count)


class NotOwnedError(Exception):
    def __init__(self, namespace: str, owner: int, this_shard: int):
        super().__init__(namespace)
        self.namespace, self.owner, self.this_shard = namespace, owner, this_shard


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
                 shard_count: int = SHARD_COUNT, shard_index: int = SHARD_INDEX):
        self._data_dir = data_dir
        self._default_dim = default_dim
        self._shard_count = max(1, shard_count)
        self._shard_index = shard_index
        if not 0 <= shard_index < self._shard_count:
            raise ValueError(f"FEATHER_SHARD_INDEX={shard_index} outside 0..{self._shard_count - 1}")
        self._dbs: Dict[str, DB] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

        os.makedirs(data_dir, exist_ok=True)
        self._load_existing()

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
    def _load_existing(self):
        """Load every owned .feather file in data_dir on startup. A single
        corrupt file must not take down the whole server — skip + log it."""
        for fname in sorted(os.listdir(self._data_dir)):
            if fname.endswith(".feather"):
                ns = fname[:-len(".feather")]
                if not self.owns(ns):
                    continue
                try:
                    self._open_namespace(ns)
                except Exception as e:  # noqa: BLE001 — never crash startup on one bad file
                    self._dbs.pop(ns, None)
                    self._locks.pop(ns, None)
                    why = ("held by another process (is a second server, worker or "
                           "feather-serve using this data dir?)"
                           if any(s in str(e) for s in ("holds the write lock",
                                                        "holds it exclusively",
                                                        "holds the lock"))
                           else f"unloadable: {e}")
                    print(f"[db_manager] skipping namespace '{ns}': {why}", file=sys.stderr)

    def data_dir(self) -> str:
        return self._data_dir

    def _namespace_path(self, namespace: str) -> str:
        # Sanitize: only allow alphanumeric, dash, underscore
        safe = "".join(c for c in namespace if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError(f"Invalid namespace: {namespace!r}")
        return os.path.join(self._data_dir, f"{safe}.feather")

    def _open_namespace(self, namespace: str, dim: Optional[int] = None) -> DB:
        path = self._namespace_path(namespace)
        # `dim` only sets the reported default for a brand-new empty namespace;
        # an existing file keeps its own dim, and the first inserted vector is
        # what truly fixes it. So passing dim here never overrides real data.
        db = DB.open(path, dim=dim or self._default_dim)
        self._dbs[namespace] = db
        self._locks[namespace] = threading.Lock()
        return db

    def get(self, namespace: str, create: bool = True,
            dim: Optional[int] = None) -> DB:
        """Return the DB for this namespace, creating it if needed."""
        db = self._dbs.get(namespace)
        if db is not None:
            return db
        self._check_owner(namespace)
        with self._global_lock:
            if namespace in self._dbs:
                return self._dbs[namespace]
            if not create:
                raise KeyError(f"Namespace '{namespace}' not found")
            return self._open_namespace(namespace, dim=dim)

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

        with self._global_lock:
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
            self._locks.pop(namespace, None)
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
                self._locks.pop(namespace, None)
                _safe_remove(dest)
                _safe_remove(dest + ".wal")
                if backup and os.path.exists(backup):
                    try:
                        os.replace(backup, dest)
                        self._open_namespace(namespace)
                    except Exception:  # noqa: BLE001
                        pass
                raise ValueError(f"could not load uploaded .feather: {e}")

    def lock(self, namespace: str) -> threading.Lock:
        """Return the write lock for this namespace."""
        self.get(namespace)   # ensure it exists
        return self._locks[namespace]

    def list_namespaces(self):
        return list(self._dbs.keys())

    def save_all(self):
        for db in list(self._dbs.values()):
            db.save()

    def close_all(self):
        """Checkpoint and close every namespace (shutdown). Releases the file
        locks so a restarted or replacement process can take over at once."""
        with self._global_lock:
            for ns, db in list(self._dbs.items()):
                _close_quietly(db, save=True)
            self._dbs.clear()
            self._locks.clear()

    def save(self, namespace: str):
        if namespace in self._dbs:
            self._dbs[namespace].save()

    def delete(self, namespace: str) -> bool:
        """Hard-delete a namespace: close it + remove .feather, WAL and sidecars.
        Returns True if anything was removed.
        """
        with self._global_lock:
            removed = False
            db = self._dbs.pop(namespace, None)
            self._locks.pop(namespace, None)
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
