"""The BM25 index built in parallel at the first keyword search is the
index built record by record as writes arrive: same hits, same scores."""
import random

import numpy as np

import feather_db
from feather_db import DB, FilterBuilder

WORDS = ["allergy", "allergic", "running", "runs", "creative", "creatives", "policy",
         "policies", "denver", "austin", "dog", "dogs", "café", "naïve", "x1", "a",
         "the", "of", "yyyyyyyy", "data", "model", "models", "memory", "memories"]


def test_a_rebuilt_index_scores_as_the_incremental_one(tmp_path):
    rng = random.Random(4)
    path = str(tmp_path / "b.feather")
    live = DB.open(path, dim=4)
    live.keyword_search("warm", k=1)              # built now: every write below is incremental
    ids = []
    for i in range(3000):
        m = feather_db.Metadata()
        m.timestamp = 1_700_000_000 + i
        m.namespace_id = f"ns{i % 4}"
        m.key = f"k{i}" if rng.random() < 0.7 else ""
        m.content = " ".join(rng.choice(WORDS) for _ in range(rng.randint(0, 40)))
        ids.append(live.upsert(m, np.ones(4, dtype=np.float32))[0] if m.key
                   else (live.add(10_000_000 + i, np.ones(4, dtype=np.float32), m)
                         or 10_000_000 + i))
    for rid in rng.sample(ids, 400):              # rewrites and forgets, indexed as they come
        if rng.random() < 0.5:
            live.forget(rid)
        else:
            m = live.get_metadata(rid)
            if m is not None:
                m.content = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 30)))
                live.update_metadata(rid, m)
    live.save()
    fresh = DB.open(path, dim=4, read_only=True)  # builds the index from scratch, in parallel
    for q in ["allergy", "running dogs", "creative policy", "café naïve memory", "model",
              "denver austin x1", "yyyyyyyy", "the of a"] + [
              " ".join(rng.sample(WORDS, 3)) for _ in range(30)]:
        for f in (None, FilterBuilder().namespace("ns2").build()):
            a = [(r.id, r.score) for r in live.keyword_search(q, k=25, filter=f,
                                                              record_salience=False)]
            b = [(r.id, r.score) for r in fresh.keyword_search(q, k=25, filter=f,
                                                               record_salience=False)]
            assert a == b, q
    fresh.close()
    live.close()
