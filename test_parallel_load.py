"""Test/benchmark Opt 1: parallel load (threaded HNSW graph rebuild).

Builds a DB, then reloads it serially vs in parallel, asserting identical
results and reporting the speedup. FEATHER_LOAD_THREADS controls the pool.
"""
import os, sys, glob, time, tempfile, importlib.util, numpy as np

_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
_cands = [p for p in glob.glob("feather_db/core*.so") if _tag in p]
_so = (_cands or sorted(glob.glob("feather_db/core*.so")))[0]
_spec = importlib.util.spec_from_file_location("core", _so)
fc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fc)

DIM, N = 128, 40000
rng = np.random.default_rng(5)
fails = []
def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}: {name} {extra}")
    if not cond: fails.append(name)

path = tempfile.mktemp(suffix=".feather")
print(f"building {N}x{DIM} DB ...")
db = fc.DB.open(path, dim=DIM)
# Clustered data (well-separated centers) so nearest neighbours are well-defined
# and ANN recall is high — random Gaussian in 128-d is a near-worst case for ANN.
K = 400
centers = (rng.standard_normal((K, DIM)) * 10.0).astype(np.float32)
assign = rng.integers(0, K, size=N)
V = (centers[assign] + rng.standard_normal((N, DIM)).astype(np.float32) * 0.5).astype(np.float32)
for i in range(N):
    db.add(id=i, vec=V[i])
db.save()
del db
sz = os.path.getsize(path)
print(f"  saved {sz/1e6:.1f} MB")

def timed_load(threads):
    os.environ["FEATHER_LOAD_THREADS"] = str(threads)
    t0 = time.perf_counter()
    d = fc.DB.open(path, dim=DIM)
    dt = time.perf_counter() - t0
    return d, dt

print("loading serial (1 thread) ...")
db1, t_serial = timed_load(1)
print(f"  serial load:   {t_serial*1000:8.1f} ms")

print("loading parallel (8 threads) ...")
db8, t_par = timed_load(8)
print(f"  parallel load: {t_par*1000:8.1f} ms   ({t_serial/t_par:.2f}x faster)")

print("correctness")
check("serial loaded all N", db1.size() == N, f"got {db1.size()}")
check("parallel loaded all N", db8.size() == N, f"got {db8.size()}")
# HNSW is approximate & order-dependent, so serial and parallel builds are two
# different *valid* graphs. Validate both against brute-force ground truth
# (averaged over several queries) rather than against each other.
def recall(db, queries, k=10):
    hits = tot = 0
    for q in queries:
        exact = set(np.argsort(np.sum((V - q) ** 2, axis=1))[:k].tolist())
        got = set(r.id for r in db.search(q.astype(np.float32), k=k))
        hits += len(exact & got); tot += k
    return hits / tot
# queries drawn from the data distribution (near clusters), else NN is ill-defined
qs = (centers[rng.integers(0, K, size=20)]
      + rng.standard_normal((20, DIM)).astype(np.float32) * 0.5).astype(np.float32)
r1, r8 = recall(db1, qs), recall(db8, qs)
print(f"  recall@10 — serial={r1:.3f}  parallel={r8:.3f}")
check("serial recall@10 >= 0.90", r1 >= 0.90, f"{r1:.3f}")
check("parallel recall@10 >= 0.90", r8 >= 0.90, f"{r8:.3f}")
check("parallel recall ~ serial (within 0.05)", abs(r1 - r8) < 0.05,
      f"|{r1:.3f}-{r8:.3f}|")
check("parallel load faster than serial", t_par < t_serial,
      f"{t_serial/t_par:.2f}x")

for x in (path, path + ".wal", path + ".tmp"):
    if os.path.exists(x): os.remove(x)
print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
exit(1 if fails else 0)
