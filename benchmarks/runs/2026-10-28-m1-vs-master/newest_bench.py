"""FeatherStore.newest(ns, 10) on a namespace of N items: whatever build is importable."""
import json, os, sys, tempfile, time
import numpy as np
import feather_db
from feather_db.integrations.langgraph_store import FeatherStore
out = {}
for N in (1_000, 10_000, 100_000):
    d = tempfile.mkdtemp(); p = os.path.join(d, "s.feather")
    db = feather_db.DB.open(p, dim=8)
    now = int(time.time()); metas = []
    for i in range(N):
        m = feather_db.Metadata(); m.namespace_id = "org.u1.episodes"; m.timestamp = now - (N - i)
        m.source = "langgraph"; m.entity_id = f"k{i}"; m.set_attribute("_lg_value", json.dumps({"n": i}))
        metas.append(m)
    db.add_batch(list(range(N)), np.random.rand(N, 8).astype(np.float32), metas); db.close()
    st = FeatherStore(p, dim=8, embed=None, auto_save=False)
    ns = ("org", "u1", "episodes")
    got = [i.value["n"] for i in st.newest(ns, 10)]
    assert got == list(range(N - 1, N - 11, -1)), got
    ts = []
    for _ in range(5 if N >= 100_000 else 20):
        t = time.perf_counter(); st.newest(ns, 10); ts.append(time.perf_counter() - t)
    out[N] = round(float(np.median(ts)) * 1e3, 3)
    st.close()
print(json.dumps({"version": feather_db.__version__, "newest10_ms": out}))
