"""Targeted recheck of the rows that regressed on the branch. Same data and
calls as full_suite, smaller set; run master first to undo order bias."""
import json, os, sys, tempfile, time
import numpy as np
import feather_db as fc
N, D = 100_000, 128
rng = np.random.default_rng(0)
data = rng.standard_normal((N, D)).astype(np.float32)
qs = rng.standard_normal((300, D)).astype(np.float32)
metas = []
for i in range(N):
    m = fc.Metadata(); m.namespace_id = f"ns{i % 100}"; m.entity_id = f"e{i % 5000}"
    m.content = f"record {i} about topic {i % 37}"; m.set_attribute("color", ["red", "green", "blue", "black"][i % 4])
    metas.append(m)
p = os.path.join(tempfile.mkdtemp(), "r.feather")
db = fc.DB.open(p, dim=D); db.add_batch(list(range(N)), data, metas)
def pct(xs):
    a = np.array(xs); return {"p50": round(float(np.percentile(a, 50)), 4), "p99": round(float(np.percentile(a, 99)), 4), "mean": round(float(a.mean()), 4)}
def each(fn, items):
    out = []
    for it in items:
        s = time.perf_counter(); fn(it); out.append((time.perf_counter() - s) * 1e3)
    return pct(out)
R = {}
db.set_ef(50) if hasattr(db, "set_ef") else None
R["search_ef50"] = each(lambda q: db.search(q, k=10, record_salience=False), qs)
ids = rng.choice(N, 2000, replace=False).tolist()
R["get_vector"] = each(db.get_vector, ids)
R["update_importance"] = each(lambda i: db.update_importance(i, 0.5), ids)
R["ids_in_namespace"] = each(db.ids_in_namespace, [f"ns{i}" for i in range(100)] * 5)
t = time.perf_counter(); db.export_graph_json("ns7", ""); R["export_graph_json_s"] = round(time.perf_counter() - t, 4)
kill = rng.choice(N, N // 10, replace=False).tolist()
t = time.perf_counter()
for i in kill: db.forget(i)
R["forget_10k_s"] = round(time.perf_counter() - t, 3)
for r in range(3):
    R[f"search_tombstones_round{r}"] = each(lambda q: db.search(q, k=10, record_salience=False), qs)
    time.sleep(1)
db.close()
print(json.dumps(R))
