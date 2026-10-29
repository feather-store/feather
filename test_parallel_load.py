"""Test: load paths — persisted graph (v10) + parallel rebuild fallback.

A DB persists its prebuilt HNSW graph (format v10; v9 when clean), so reload
restores it verbatim (no rebuild, FEATHER_LOAD_THREADS irrelevant) and keeps
the serial-build recall. The parallel *rebuild* still runs where no graph is
persisted — a v9 file whose modality is on-disk-quantized (a v9 file with
pending deletions used to be one too; it now keeps its graph). We write one
with FEATHER_FORMAT=9 + set_quantized and check the threaded rebuild is faster
than serial and still high-recall.
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
# Clustered data so nearest neighbours are well-defined and ANN recall is high.
K = 400
centers = (rng.standard_normal((K, DIM)) * 10.0).astype(np.float32)
assign = rng.integers(0, K, size=N)
V = (centers[assign] + rng.standard_normal((N, DIM)).astype(np.float32) * 0.5).astype(np.float32)
for i in range(N):
    db.add(id=i, vec=V[i])
db.save()
db.close(save=False)
sz = os.path.getsize(path)
print(f"  saved {sz/1e6:.1f} MB (graph persisted)")

def timed_load(threads):
    os.environ["FEATHER_LOAD_THREADS"] = str(threads)
    t0 = time.perf_counter()
    d = fc.DB.open(path, dim=DIM)
    dt = time.perf_counter() - t0
    return d, dt

def recall(db, queries, k=10):
    hits = tot = 0
    for q in queries:
        exact = set(np.argsort(np.sum((V - q) ** 2, axis=1))[:k].tolist())
        got = set(r.id for r in db.search(q.astype(np.float32), k=k))
        hits += len(exact & got); tot += k
    return hits / tot
qs = (centers[rng.integers(0, K, size=80)]
      + rng.standard_normal((80, DIM)).astype(np.float32) * 0.5).astype(np.float32)

# ── 1) persisted-graph fast path ────────────────────────────────────────
with open(path, "rb") as fh:
    fh.read(4); ver = int.from_bytes(fh.read(4), "little")
check("file format v10 (persisted graph)", ver == 10, f"got v{ver}")
print("persisted-graph reload ...")
dbp, t_persist = timed_load(1)
print(f"  persisted load: {t_persist*1000:8.1f} ms")
check("persisted load restored all N", dbp.size() == N, f"got {dbp.size()}")
rp = recall(dbp, qs)
print(f"  recall@10 (persisted, == serial-build quality) = {rp:.3f}")
check("persisted recall@10 >= 0.88", rp >= 0.88, f"{rp:.3f}")
dbp.close(save=False)

# ── 2) parallel rebuild fallback (v9 + on-disk int8 => graph not persisted) ──
print("forcing rebuild fallback (v9, quantized on disk) ...")
os.environ["FEATHER_FORMAT"] = "9"      # only a v9 file still takes the rebuild path
d = fc.DB.open(path, dim=DIM); d.set_quantized("text", True); d.save(); d.close(save=False)
del os.environ["FEATHER_FORMAT"]
with open(path, "rb") as fh:
    fh.read(4); ver2 = int.from_bytes(fh.read(4), "little")
print(f"  re-saved (v{ver2}, rebuild path: quantized on disk)")
db1, t_serial = timed_load(1)
r1 = recall(db1, qs)
db1.close(save=False)          # single owner: release before the second load
db8, t_par = timed_load(8)
print(f"  serial rebuild:   {t_serial*1000:8.1f} ms")
print(f"  parallel rebuild: {t_par*1000:8.1f} ms   ({t_serial/t_par:.2f}x faster)")
check("rebuild loaded all N", db8.size() == N, f"got {db8.size()}")
r8 = recall(db8, qs)
print(f"  recall@10 — serial={r1:.3f}  parallel={r8:.3f}")
# Ground truth is the float vectors; the rebuilt ones went through int8 on
# disk, which costs a few points. What matters is parallel == serial quality.
check("serial rebuild recall@10 >= 0.85 (int8 on disk)", r1 >= 0.85, f"{r1:.3f}")
check("parallel rebuild recall@10 >= 0.85 (int8 on disk)", r8 >= 0.85, f"{r8:.3f}")
check("parallel rebuild recall within 0.03 of serial", abs(r8 - r1) <= 0.03, f"{r8:.3f} vs {r1:.3f}")
check("parallel rebuild faster than serial", t_par < t_serial, f"{t_serial/t_par:.2f}x")

for x in (path, path + ".wal", path + ".tmp"):
    if os.path.exists(x): os.remove(x)
print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
exit(1 if fails else 0)
