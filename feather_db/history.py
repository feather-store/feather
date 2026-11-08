"""Compare two states of memory.

Every record carries a version that changes on every write and is never
reused (``Metadata.version``), and a handle opened with
``DB.open(path, as_of=...)`` holds the versions as they were. So the
difference between two moments is a comparison of two ``{id: version}`` maps:

    before = feather_db.DB.open(path, as_of="run-41")
    after  = feather_db.DB.open(path, as_of="run-42")
    d = feather_db.diff(before, after, namespace="org.user_842")
    d.added, d.removed, d.changed        # record ids

Works on any two handles with ``record_versions()``: embedded, read-only,
``as_of`` and ``RemoteDB`` alike. A recall (search hit, ``touch()``) is not a
change; versions move only when a record is written.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryDiff:
    added: list = field(default_factory=list)     # in `after`, not in `before`
    removed: list = field(default_factory=list)   # in `before`, not in `after`
    changed: list = field(default_factory=list)   # in both, written in between

    def __len__(self) -> int:
        return len(self.added) + len(self.removed) + len(self.changed)

    def __bool__(self) -> bool:
        return len(self) > 0

    def summary(self) -> str:
        return f"+{len(self.added)} -{len(self.removed)} ~{len(self.changed)}"


def diff(before, after, namespace: str = "") -> MemoryDiff:
    """What changed from `before` to `after` under a namespace prefix ("" = all)."""
    a = before.record_versions(namespace)
    b = after.record_versions(namespace)
    return MemoryDiff(
        added=sorted(i for i in b if i not in a),
        removed=sorted(i for i in a if i not in b),
        changed=sorted(i for i, v in b.items() if i in a and a[i] != v),
    )
