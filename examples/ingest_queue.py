"""Write memory now, index it later: the ingest queue.

Each write returns at once; extraction (one model call per memory or
session) runs on worker threads, retries on failure, and survives a restart
because its jobs are records in the file. Runs offline with a stand-in
extractor; with a model, use SessionFactExtractor:

    from feather_db.extractors import SessionFactExtractor
    q = IngestQueue(pkt, SessionFactExtractor(provider))

    python3 examples/ingest_queue.py
"""
import os
import re
import tempfile
import time

import feather_db
from feather_db.extractors.base import Fact
from feather_db.ingest import IngestQueue


class StandIn:
    """One fact per "<name> moved to <city>": what SessionFactExtractor would
    return for it, without a model. Its first call fails, to show a retry."""
    name, version = "stand_in", "1"

    def __init__(self):
        self.calls = 0

    def extract(self, text, *, context=None):
        self.calls += 1
        if self.calls == 1:
            context["on_error"]("model timed out")   # retried, not "no facts"
            return []
        return [Fact(subject=m.group(1).lower(), predicate="home city", object=m.group(2),
                     metadata={"statement": f"{m.group(1)} lives in {m.group(2)}",
                               "replaces": True})
                for m in re.finditer(r"(\w+) moved to (\w+)", text)]


with tempfile.TemporaryDirectory() as d:
    db = feather_db.DB.open(os.path.join(d, "agent.feather"), dim=8)
    pkt = feather_db.Pocket(db, ("acme", "alice"))
    with IngestQueue(pkt, StandIn(), backoff=0.2) as q:
        t = time.time()
        # valid_at: when each became true. Without it, the order things are
        # current in is the order their indexes happened to be written.
        q.remember(pkt, "move-1", "Alice moved to Denver last spring",
                   valid_at=time.time() - 200 * 86400)
        q.remember(pkt, "move-2", "Alice moved to Austin in June",
                   valid_at=time.time() - 150 * 86400)
        print(f"written in {1000 * (time.time() - t):.1f} ms, searchable now:",
              [i.key for i in pkt.recall("Austin", k=2)])
        print("queue:", {k: v for k, v in q.stats().items() if k in ("pending", "running")})
        q.drain(timeout=30)
        print("after drain:", {k: v for k, v in q.stats().items()
                               if k in ("pending", "done", "retried", "failed")})
    print("keys on file:", pkt.known_keys())
    for item in pkt.recall("where does Alice live", k=2):
        print(" ", item.content, "| earlier:", [e.content for e in item.earlier])
    db.close()
