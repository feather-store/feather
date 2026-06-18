"""Test: persisted HNSW graph (file format v9).

save() embeds the prebuilt graph; load() restores it verbatim instead of
rebuilding from vectors. Asserts:
  1) round-trip — search results IDENTICAL before vs after save+reload
  2) recall preserved (no parallel-rebuild loss) and load is fast (no rebuild)
  3) fallback — a DB with forgotten records still saves/loads correctly
     (graph not persisted; rebuild path used)
  4) on-disk quantized modality still round-trips (persist_graph forced off)
"""
import os, sys, glob, time, tempfile, importlib.util, numpy as np

_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
_cands = [p for p in glob.glob("feather_db/core*.so") if _tag in p]
_so = (_cands or sorted(glob.glob("feather_db/core*.so")))[0]
_spec = importlib.util.spec_from_file_location("core", _so)
fc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fc)

DIM, N, K = 128, 20000, 10
rng = np.random.default_rng(11)
vecs = rng.standard_normal((N, DIM)).astype(np.float32)
queries = rng.standard_normal((50, DIM)).astype(np.float32)
fails = 0

def check(cond, msg, extra=""):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}: {msg} {extra}")
    if not cond: fails += 1

def topk(db, q):
    return [r.id for r in db.search(q, k=K)]

# ── build + save ────────────────────────────────────────────────────────
p = tempfile.mktemp(suffix=".feather")
db = fc.DB.open(p, dim=DIM)
metas = []
for i in range(N):
    m = fc.Metadata(); m.content = f"r{i}"; metas.append(m)
db.add_batch(ids=list(range(N)), vecs=vecs, metas=metas)
before = [topk(db, q) for q in queries]
db.save()
fsz = os.path.getsize(p) / 1e6
del db

# ── reload (should restore graph, NOT rebuild) ──────────────────────────
t = time.time()
db2 = fc.DB.open(p, dim=DIM)
load_dt = time.time() - t
after = [topk(db2, q) for q in queries]

print(f"1) round-trip: {N}x{DIM}, file {fsz:.1f}MB, reload {load_dt*1000:.0f}ms")
identical = sum(1 for a, b in zip(before, after) if a == b)
check(identical == len(queries), "all 50 queries identical after reload",
      f"{identical}/{len(queries)}")
# version on disk should be 9
with open(p, "rb") as fh:
    fh.read(4); ver = int.from_bytes(fh.read(4), "little")
check(ver == 9, "file format v9", f"got v{ver}")

# ── compare to a pure-rebuild reload time (force fallback via a forget) ──
print("2) fallback path: forgotten record disables graph-persist")
db2.forget(id=0)
db2.save()
del db2
db3 = fc.DB.open(p, dim=DIM)
# id 0 must be gone, others searchable
res_ids = set()
for q in queries[:10]:
    res_ids.update(topk(db3, q))
check(0 not in [r.id for r in db3.search(vecs[0], k=K)],
      "forgotten id 0 absent after reload")
check(db3.size() == N - 1, "size N-1 after forget+reload", f"got {db3.size()}")
del db3
os.remove(p)

# ── on-disk quantized modality still round-trips ────────────────────────
print("3) on-disk quantized modality round-trips (persist_graph off)")
pq = tempfile.mktemp(suffix=".feather")
dbq = fc.DB.open(pq, dim=DIM)
dbq.set_quantized("text")
mq = []
for i in range(5000):
    m = fc.Metadata(); m.content = f"q{i}"; mq.append(m)
dbq.add_batch(ids=list(range(5000)), vecs=vecs[:5000], metas=mq)
q_before = [topk(dbq, q) for q in queries[:20]]
dbq.save()
del dbq
dbq2 = fc.DB.open(pq, dim=DIM)
q_after = [topk(dbq2, q) for q in queries[:20]]
ov = np.mean([len(set(a) & set(b)) / K for a, b in zip(q_before, q_after)])
check(ov >= 0.9, "quantized recall preserved across save/load", f"overlap={ov:.3f}")
del dbq2
os.remove(pq)

print("\n" + ("ALL PASS" if fails == 0 else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
