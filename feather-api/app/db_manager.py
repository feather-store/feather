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
import threading
from typing import Dict, Optional
from feather_db import DB


DATA_DIR = os.getenv("FEATHER_DATA_DIR", "/data")
DEFAULT_DIM = int(os.getenv("FEATHER_DB_DIM", "768"))


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
        """Load all .feather files found in data_dir on startup."""
        for fname in os.listdir(self._data_dir):
            if fname.endswith(".feather"):
                ns = fname[:-len(".feather")]
                self._open_namespace(ns)

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
