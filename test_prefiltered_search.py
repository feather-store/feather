"""Test feature A: pre-filtered ANN search.

Verifies that a selective filter returns up to k COMPLETE, EXACT results
(ranked over the full filtered set), rather than HNSW's ef-bounded under-return.
Loads the .so directly to skip the heavy feather_db package init.
"""
import os, sys, glob, tempfile, importlib.util, numpy as np

_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
_cands = [p for p in glob.glob("feather_db/core*.so") if _tag in p]
_so = (_cands or sorted(glob.glob("feather_db/core*.so")))[0]
_spec = importlib.util.spec_from_file_location("core", _so)
fc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fc)
Metadata, ContextType = fc.Metadata, fc.ContextType

DIM = 16
rng = np.random.default_rng(7)
fails = []
def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}: {name} {extra}")
    if not cond: fails.append(name)

def mk(ns, **attrs):
    m = Metadata(); m.type = ContextType.FACT; m.namespace_id = ns
    for k, v in attrs.items(): m.set_attribute(k, str(v))
    return m

path = tempfile.mktemp(suffix=".feather")
db = fc.DB.open(path, dim=DIM)

# 600 "big" records + 6 "rare" records, all in one index.
# Rare records are deliberately placed FAR from the query, so a plain
# ef-bounded HNSW filtered search would explore big-namespace neighbours
# first and return < k rare hits. Pre-filtered must still return all 6.
q = rng.random(DIM).astype(np.float32)
big_vecs, rare = {}, {}
for i in range(600):
    v = rng.random(DIM).astype(np.float32)
    big_vecs[i] = v
    db.add(id=i, vec=v, meta=mk("big", vip="no"))
for j in range(6):
    rid = 10000 + j
    v = (q + 5.0 + rng.random(DIM)).astype(np.float32)   # far from q
    rare[rid] = v
    db.add(id=rid, vec=v, meta=mk("rare", vip="yes"))

def brute_topk(vecs, k):
    d = sorted(((float(np.sum((q - v) ** 2)), i) for i, v in vecs.items()))
    return [i for _, i in d[:k]]

print("1) selective namespace filter returns COMPLETE k (the core bug)")
f = fc.SearchFilter(); f.namespace_id = "rare"
res = db.search(q, k=5, filter=f)
check("returns 5 (not under-returned)", len(res) == 5, f"got {len(res)}")
check("all results from 'rare'", all(r.id in rare for r in res))
check("exact match vs brute force", [r.id for r in res] == brute_topk(rare, 5),
      f"\n      got   {[r.id for r in res]}\n      brute {brute_topk(rare, 5)}")

print("2) k larger than match count returns all matches (no padding from 'big')")
res = db.search(q, k=20, filter=f)
check("returns all 6 rare", len(res) == 6, f"got {len(res)}")
check("none from 'big'", all(r.id in rare for r in res))

print("3) attribute filter path")
fa = fc.SearchFilter(); fa.attributes_match = {"vip": "yes"}
res = db.search(q, k=10, filter=fa)
check("vip=yes returns 6", len(res) == 6, f"got {len(res)}")
check("vip exact vs brute", [r.id for r in res] == brute_topk(rare, 6))

print("4) combined indexed + non-indexed predicate (ns=big AND importance high)")
# default importance is 1.0, so lower MOST big records, keep 3 high — then the
# importance_gte predicate (non-indexed) must prune within the ns candidate set.
hi = list(big_vecs.keys())[:3]
for i in big_vecs:
    db.update_importance(i, 0.99 if i in hi else 0.10)
fc2 = fc.SearchFilter(); fc2.namespace_id = "big"; fc2.importance_gte = 0.9
res = db.search(q, k=10, filter=fc2)
check("only high-importance big records", set(r.id for r in res) == set(hi),
      f"got {sorted(r.id for r in res)}")

print("5) empty namespace -> empty result")
fe = fc.SearchFilter(); fe.namespace_id = "ghost"
check("ghost ns empty", db.search(q, k=5, filter=fe) == [])

print("6) forgotten record drops out of filtered results")
db.forget(10000)
res = db.search(q, k=20, filter=f)
check("rare now 5 after forget", len(res) == 5, f"got {len(res)}")
check("forgotten id absent", 10000 not in [r.id for r in res])

print("7) unfiltered search still works (no regression)")
res = db.search(q, k=5)
check("unfiltered returns 5", len(res) == 5, f"got {len(res)}")
check("unfiltered nearest is a 'big' record", res[0].id in big_vecs)

for p in (path, path + ".wal", path + ".tmp"):
    if os.path.exists(p): os.remove(p)
print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
exit(1 if fails else 0)
