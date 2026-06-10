"""Test/benchmark Opt 2: parallel batch ingestion (add_batch)."""
import os, sys, glob, time, tempfile, importlib.util, numpy as np

_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
_cands = [p for p in glob.glob("feather_db/core*.so") if _tag in p]
_so = (_cands or sorted(glob.glob("feather_db/core*.so")))[0]
_spec = importlib.util.spec_from_file_location("core", _so)
fc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fc)
Metadata, ContextType = fc.Metadata, fc.ContextType

DIM, N = 128, 40000
rng = np.random.default_rng(9)
K = 400
centers = (rng.standard_normal((K, DIM)) * 10.0).astype(np.float32)
assign = rng.integers(0, K, size=N)
V = (centers[assign] + rng.standard_normal((N, DIM)).astype(np.float32) * 0.5).astype(np.float32)
IDS = list(range(N))
fails = []
def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}: {name} {extra}")
    if not cond: fails.append(name)
def cleanup(p):
    for x in (p, p + ".wal", p + ".tmp"):
        if os.path.exists(x): os.remove(x)

print(f"ingesting {N}x{DIM} two ways ...")
# serial add()
ps = tempfile.mktemp(suffix=".feather"); dbs = fc.DB.open(ps, dim=DIM)
t0 = time.perf_counter()
for i in range(N): dbs.add(id=i, vec=V[i])
t_serial = time.perf_counter() - t0
print(f"  serial add() loop: {t_serial*1000:8.1f} ms")

# parallel add_batch()
pb = tempfile.mktemp(suffix=".feather"); dbb = fc.DB.open(pb, dim=DIM)
t0 = time.perf_counter()
dbb.add_batch(IDS, V)
t_batch = time.perf_counter() - t0
print(f"  add_batch():       {t_batch*1000:8.1f} ms   ({t_serial/t_batch:.2f}x faster)")

print("correctness")
check("batch inserted all N", dbb.size() == N, f"got {dbb.size()}")
def recall(db, queries, k=10):
    hits = 0
    for q in queries:
        exact = set(np.argsort(np.sum((V - q) ** 2, axis=1))[:k].tolist())
        got = set(r.id for r in db.search(q.astype(np.float32), k=k))
        hits += len(exact & got)
    return hits / (k * len(queries))
qs = (centers[rng.integers(0, K, size=80)] + rng.standard_normal((80, DIM)).astype(np.float32)*0.5).astype(np.float32)
rb = recall(dbb, qs)
print(f"  batch recall@10 = {rb:.3f}")
check("batch recall@10 >= 0.88", rb >= 0.88, f"{rb:.3f}")
check("add_batch faster than serial add loop", t_batch < t_serial, f"{t_serial/t_batch:.2f}x")

print("metadata + secondary indexes via add_batch")
pm = tempfile.mktemp(suffix=".feather"); dbm = fc.DB.open(pm, dim=DIM)
metas = []
for i in range(1000):
    m = Metadata(); m.type = ContextType.FACT
    m.namespace_id = "acme" if i % 2 == 0 else "globex"
    m.set_attribute("tier", "vip" if i % 5 == 0 else "std")
    metas.append(m)
dbm.add_batch(list(range(1000)), V[:1000], metas)
check("ns index populated (acme=500)", dbm.namespace_size("acme") == 500, f"got {dbm.namespace_size('acme')}")
check("attr index populated (tier=vip=200)", len(dbm.ids_with_attribute("tier","vip")) == 200,
      f"got {len(dbm.ids_with_attribute('tier','vip'))}")
# pre-filtered search works over batch-ingested data
f = fc.SearchFilter(); f.namespace_id = "globex"
res = dbm.search(V[1], k=10, filter=f)
check("pre-filtered search over batch data returns 10", len(res) == 10, f"got {len(res)}")
check("all results in globex", all(r.id % 2 == 1 for r in res))

print("edge cases")
pe = tempfile.mktemp(suffix=".feather"); dbe = fc.DB.open(pe, dim=DIM)
dbe.add_batch([], np.zeros((0, DIM), dtype=np.float32))   # empty -> no-op
check("empty batch is no-op", dbe.size() == 0)
try:
    dbe.add_batch([1, 2], V[:3])     # id/vec mismatch
    check("size mismatch raises", False)
except Exception:
    check("size mismatch raises", True)

for p in (ps, pb, pm, pe): cleanup(p)
print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
exit(1 if fails else 0)
