"""Which shard owns a namespace: rendezvous hashing, plus a placement table.

**Rendezvous (highest-random-weight) hashing.** Each shard has a stable id.
A namespace belongs to the shard whose `hash(shard_id, namespace)` is
highest. Adding a shard moves only the namespaces the new shard now wins —
about 1/(N+1) of them, 20% when going from 4 shards to 5. `crc32(ns) % N`,
the rule before this, moves about N/(N+1) — 80% — because every namespace's
remainder changes.

**The placement table** overrides the hash for named namespaces: a hot one
can be moved onto a shard of its own, then back. It carries an epoch.
- A shard adopts a table only when its epoch is newer, so the order updates
  arrive in does not matter.
- A shard persists the table in its data dir. With a shared volume, the
  shards share one file.
- On adoption, a shard closes, cleanly, every open namespace it no longer
  owns. That releases the file lock, and the new owner opens the file on its
  next request. That handoff is what lets a namespace move with no restart:
  requests in the gap are answered 421 (go to the new owner), which the
  gateway follows.

`FEATHER_SHARD_HASH=modulo` keeps the old crc32 % N rule (shard ids in the
order given) for a deployment that cannot reshuffle yet.

The gateway carries an identical copy of `owner()` (feather-gateway/gateway.py);
tests/test_gateway.py checks the two agree.
"""
import hashlib
import json
import os
import threading
import zlib
from typing import Dict, List, Optional

PLACEMENT_FILE = "_placement.json"


def rendezvous_owner(namespace: str, shard_ids: List[str]) -> str:
    """The shard with the highest weight for `namespace`. blake2b, not hash():
    Python's str hash is randomised per process, so two processes would
    disagree."""
    def weight(sid: str) -> int:
        h = hashlib.blake2b(f"{sid}\x00{namespace}".encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big")
    return max(shard_ids, key=lambda sid: (weight(sid), sid))


def modulo_owner(namespace: str, shard_ids: List[str]) -> str:
    """The rule before 0.22: crc32(namespace) % N over the ids in order."""
    return shard_ids[zlib.crc32(namespace.encode("utf-8")) % len(shard_ids)]


def owner(namespace: str, shard_ids: List[str], overrides: Dict[str, str],
          mode: str = "rendezvous") -> str:
    o = overrides.get(namespace)
    if o is not None and o in shard_ids:
        return o
    return (modulo_owner if mode == "modulo" else rendezvous_owner)(namespace, shard_ids)


def shard_config(env=os.environ):
    """(this shard's id, every shard's id, hash mode) from the environment.

    FEATHER_SHARD_IDS=a,b,c with FEATHER_SHARD_ID=b names the shards. Without
    them, FEATHER_SHARD_COUNT=N / FEATHER_SHARD_INDEX=i (the old variables)
    mean ids "0".."N-1", this one str(i)."""
    ids = [s.strip() for s in env.get("FEATHER_SHARD_IDS", "").split(",") if s.strip()]
    if ids:
        me = env.get("FEATHER_SHARD_ID", "").strip()
        if me not in ids:
            raise ValueError(f"FEATHER_SHARD_ID={me!r} is not one of FEATHER_SHARD_IDS {ids}")
    else:
        n = max(1, int(env.get("FEATHER_SHARD_COUNT", "1")))
        i = int(env.get("FEATHER_SHARD_INDEX", "0"))
        if not 0 <= i < n:
            raise ValueError(f"FEATHER_SHARD_INDEX={i} outside 0..{n - 1}")
        ids, me = [str(k) for k in range(n)], str(i)
    if len(set(ids)) != len(ids):
        raise ValueError(f"shard ids repeat: {ids}")
    mode = env.get("FEATHER_SHARD_HASH", "rendezvous").strip().lower()
    if mode not in ("rendezvous", "modulo"):
        raise ValueError(f"FEATHER_SHARD_HASH={mode!r}: use rendezvous or modulo")
    return me, ids, mode


class Placement:
    """The shard ids, the hash mode, and the override table, persisted."""

    def __init__(self, data_dir: str, me: str, shard_ids: List[str], mode: str = "rendezvous"):
        self.me, self.shard_ids, self.mode = me, list(shard_ids), mode
        self._path = os.path.join(data_dir, PLACEMENT_FILE)
        self._lock = threading.Lock()
        self.epoch = 0
        self.overrides: Dict[str, str] = {}
        self._mtime = None
        self._reload()

    def _reload(self) -> None:
        """Pick up a table another shard wrote to a shared volume."""
        try:
            mtime = os.stat(self._path).st_mtime_ns
        except OSError:
            return
        if mtime == self._mtime:
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        self._mtime = mtime
        if int(data.get("epoch", 0)) > self.epoch:
            self.epoch = int(data["epoch"])
            self.overrides = {str(k): str(v) for k, v in data.get("overrides", {}).items()}

    def owner(self, namespace: str) -> str:
        self._reload()
        return owner(namespace, self.shard_ids, self.overrides, self.mode)

    def owns(self, namespace: str) -> bool:
        return self.owner(namespace) == self.me

    def adopt(self, epoch: int, overrides: Dict[str, str]) -> bool:
        """Take a newer table. False (and nothing changes) for an older or
        equal epoch, so updates may arrive in any order."""
        with self._lock:
            self._reload()
            if epoch <= self.epoch:
                return False
            unknown = sorted(set(overrides.values()) - set(self.shard_ids))
            if unknown:
                raise ValueError(f"placement names shards this deployment does not have: {unknown}")
            self.epoch, self.overrides = int(epoch), dict(overrides)
            tmp = f"{self._path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump({"epoch": self.epoch, "overrides": self.overrides}, f)
            os.replace(tmp, self._path)
            self._mtime = os.stat(self._path).st_mtime_ns
            return True

    def state(self) -> dict:
        self._reload()
        return {"shard": self.me, "shards": self.shard_ids, "hash": self.mode,
                "epoch": self.epoch, "overrides": dict(self.overrides)}
