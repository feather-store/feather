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
"""

import os
import sys
import struct
import threading
from typing import Dict, Optional
from feather_db import DB


DATA_DIR = os.getenv("FEATHER_DATA_DIR", "/data")
DEFAULT_DIM = int(os.getenv("FEATHER_DB_DIM", "768"))

# .feather binary format: [magic 4B = "FEAT"] [version 4B]. We accept any
# on-disk format this build can load (v3–v9; load() is backward-compatible).
FEATHER_MAGIC = 0x46454154   # "FEAT"
MAX_FORMAT_VERSION = 9


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


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
    def __init__(self, data_dir: str = DATA_DIR, default_dim: int = DEFAULT_DIM):
        self._data_dir = data_dir
        self._default_dim = default_dim
        self._dbs: Dict[str, DB] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

        os.makedirs(data_dir, exist_ok=True)
        self._load_existing()

    def _load_existing(self):
        """Load all .feather files found in data_dir on startup. A single
        corrupt file must not take down the whole server — skip + log it."""
        for fname in sorted(os.listdir(self._data_dir)):
            if fname.endswith(".feather"):
                ns = fname[:-len(".feather")]
                try:
                    self._open_namespace(ns)
                except Exception as e:  # noqa: BLE001 — never crash startup on one bad file
                    self._dbs.pop(ns, None)
                    self._locks.pop(ns, None)
                    print(f"[db_manager] skipping unloadable namespace '{ns}': {e}",
                          file=sys.stderr)

    def data_dir(self) -> str:
        return self._data_dir

    def _namespace_path(self, namespace: str) -> str:
        # Sanitize: only allow alphanumeric, dash, underscore
        safe = "".join(c for c in namespace if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError(f"Invalid namespace: {namespace!r}")
        return os.path.join(self._data_dir, f"{safe}.feather")

    def _open_namespace(self, namespace: str) -> DB:
        path = self._namespace_path(namespace)
        db = DB.open(path, dim=self._default_dim)
        self._dbs[namespace] = db
        self._locks[namespace] = threading.Lock()
        return db

    def get(self, namespace: str, create: bool = True) -> DB:
        """Return the DB for this namespace, creating it if needed."""
        if namespace in self._dbs:
            return self._dbs[namespace]
        with self._global_lock:
            if namespace in self._dbs:
                return self._dbs[namespace]
            if not create:
                raise KeyError(f"Namespace '{namespace}' not found")
            return self._open_namespace(namespace)

    def adopt(self, namespace: str, staged_path: str, overwrite: bool = False) -> DB:
        """Adopt an uploaded .feather file as `namespace`.

        `staged_path` is a fully-written temp file (ideally already inside
        data_dir so the final move is an atomic rename). On success the temp
        file is moved into place and the namespace is opened/served. The temp
        file is always cleaned up on validation failure.

        Raises ValueError (bad file) or FileExistsError (exists, no overwrite).
        """
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

            # Drop any live handle first. DB is bound py::nodelete, so dropping
            # the Python reference does NOT call ~DB()/save() — the in-memory
            # state can't clobber the file we're about to move into place.
            self._dbs.pop(namespace, None)
            self._locks.pop(namespace, None)

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
            for stale in (dest + ".wal", dest + ".tmp"):
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
        for db in self._dbs.values():
            db.save()

    def save(self, namespace: str):
        if namespace in self._dbs:
            self._dbs[namespace].save()

    def delete(self, namespace: str) -> bool:
        """Hard-delete a namespace: drop in-memory state + remove .feather and WAL.
        Returns True if anything was removed.
        """
        with self._global_lock:
            removed = False
            if namespace in self._dbs:
                # Trigger destructor flush, then drop the reference so the file
                # isn't reopened from a stale handle.
                try:
                    self._dbs[namespace].save()
                except Exception:
                    pass
                del self._dbs[namespace]
                self._locks.pop(namespace, None)
                removed = True
            path = self._namespace_path(namespace)
            for p in (path, path + ".wal", path + ".tmp"):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        removed = True
                    except Exception:
                        pass
            return removed
