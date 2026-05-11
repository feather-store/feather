"""Lightweight in-memory metrics + activity feed for the admin dashboard.

Single-process only. Uses a deque ring buffer; no persistence (acceptable
for a dashboard that just needs "what happened in the last hour").
"""
from collections import deque
from threading import Lock
import time
from typing import Deque, Dict, List


class Metrics:
    def __init__(self, max_events: int = 2000):
        self._events: Deque[Dict] = deque(maxlen=max_events)
        self._counts: Dict[str, int] = {}
        self._lat_recent: Dict[str, Deque[float]] = {}
        self._lock = Lock()

    def record(self, op: str, latency_ms: float, namespace: str = "", status: int = 200):
        with self._lock:
            self._events.append({
                "ts": time.time(),
                "op": op,
                "namespace": namespace,
                "latency_ms": round(latency_ms, 2),
                "status": status,
            })
            self._counts[op] = self._counts.get(op, 0) + 1
            d = self._lat_recent.setdefault(op, deque(maxlen=500))
            d.append(latency_ms)

    def snapshot(self, since_seconds: float = 3600.0) -> Dict:
        cutoff = time.time() - since_seconds
        with self._lock:
            recent = [e for e in self._events if e["ts"] >= cutoff]
        ops_total = len(recent)
        errors = sum(1 for e in recent if e["status"] >= 400)
        if recent:
            lat_sorted = sorted(e["latency_ms"] for e in recent)
            p50 = lat_sorted[len(lat_sorted) // 2]
            p95 = lat_sorted[max(0, int(len(lat_sorted) * 0.95) - 1)]
            p99 = lat_sorted[max(0, int(len(lat_sorted) * 0.99) - 1)]
        else:
            p50 = p95 = p99 = 0.0
        by_op: Dict[str, int] = {}
        for e in recent:
            by_op[e["op"]] = by_op.get(e["op"], 0) + 1
        return {
            "windowSeconds": int(since_seconds),
            "opsTotal": ops_total,
            "errors": errors,
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "byOp": by_op,
        }

    def activity(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            tail = list(self._events)[-limit:]
        return list(reversed(tail))

    def bucketed(self, bucket_seconds: int = 60, since_seconds: float = 3600.0) -> List[Dict]:
        """Aggregate ops/errors into fixed-width time buckets for sparkline rendering."""
        if bucket_seconds <= 0:
            bucket_seconds = 60
        now = time.time()
        cutoff = now - since_seconds
        with self._lock:
            recent = [e for e in self._events if e["ts"] >= cutoff]
        n_buckets = max(1, int(since_seconds // bucket_seconds))
        buckets = [{"ts": int(now - (n_buckets - i) * bucket_seconds),
                    "ops": 0, "errors": 0} for i in range(n_buckets)]
        for e in recent:
            idx = int((e["ts"] - cutoff) // bucket_seconds)
            if 0 <= idx < n_buckets:
                buckets[idx]["ops"] += 1
                if e["status"] >= 400:
                    buckets[idx]["errors"] += 1
        return buckets


METRICS = Metrics()


def classify(method: str, path: str) -> str:
    """Map a request to a coarse op label for metrics."""
    p = path.lower()
    if "/search" in p:        return "search"
    if "/hybrid_search" in p: return "search"
    if "/keyword_search" in p:return "search"
    if "/vectors" in p:       return "add"
    if "/seed" in p:          return "seed"
    if "/purge" in p:         return "purge"
    if "/compact" in p:       return "compact"
    if "/link" in p:          return "link"
    if "/context_chain" in p: return "context_chain"
    if "/edges" in p:         return "edges"
    if "/records" in p and method.upper() == "DELETE": return "delete"
    if "/records" in p and method.upper() == "GET":    return "get"
    if "/records" in p and method.upper() == "PUT":    return "update"
    if "/records" in p:       return "records"
    if "/save" in p:          return "save"
    if "/admin" in p:         return "admin"
    if "/health" in p:        return "health"
    return "other"


def namespace_from_path(path: str) -> str:
    """Extract /v1/<namespace>/... → '<namespace>'."""
    parts = path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "v1" and parts[1] not in ("namespaces", "admin"):
        return parts[1]
    return ""
