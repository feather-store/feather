"""Test feature D: incremental auto-compaction (+ broadened, orphan-safe compact)."""
import os, sys, glob, tempfile, importlib.util, numpy as np

_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
_cands = [p for p in glob.glob("feather_db/core*.so") if _tag in p]
_so = (_cands or sorted(glob.glob("feather_db/core*.so")))[0]
_spec = importlib.util.spec_from_file_location("core", _so)
fc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fc)
Metadata, ContextType = fc.Metadata, fc.ContextType

DIM = 8
rng = np.random.default_rng(3)
def vec(): return rng.random(DIM).astype(np.float32)
fails = []
def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}: {name} {extra}")
    if not cond: fails.append(name)

def fresh(**kw):
    p = tempfile.mktemp(suffix=".feather")
    return fc.DB.open(p, dim=DIM), p

def cleanup(p):
    for x in (p, p + ".wal", p + ".tmp"):
        if os.path.exists(x): os.remove(x)

print("0) default: auto-compaction disabled")
db, p = fresh()
check("get_auto_compact == 0", db.get_auto_compact() == 0.0)
for i in range(5): db.add(id=i, vec=vec(), meta=Metadata())
db.forget(0)
check("forgotten shell still present (no auto-compact)", db.get_metadata(0) is not None)
check("size unchanged at 5", db.size() == 5, f"got {db.size()}")
cleanup(p)

print("1) manual compact() reclaims forgotten records (broadened criterion)")
db, p = fresh()
for i in range(6): db.add(id=i, vec=vec(), meta=Metadata())
db.forget(1); db.forget(4)
removed = db.compact()
check("compact removed 2", removed == 2, f"got {removed}")
check("forgotten id 1 erased", db.get_metadata(1) is None)
check("forgotten id 4 erased", db.get_metadata(4) is None)
check("live id 2 survives", db.get_metadata(2) is not None)
check("size now 4", db.size() == 4, f"got {db.size()}")
check("survivors searchable", len(db.search(vec(), k=10)) == 4)
cleanup(p)

print("2) compact() is orphan-safe after purge (no resurrection)")
db, p = fresh()
for i in range(6):
    m = Metadata(); m.namespace_id = "x" if i < 3 else "y"
    db.add(id=i, vec=vec(), meta=m)
db.purge("x")                       # hard-deletes 0,1,2 (metadata erased, vectors marked)
db.compact()
ids = sorted(db.get_all_ids())
check("purged ids not resurrected", ids == [3, 4, 5], f"got {ids}")
check("size 3 after purge+compact", db.size() == 3, f"got {db.size()}")
cleanup(p)

print("3) auto-compaction triggers past threshold")
db, p = fresh()
db.set_auto_compact(0.25)
check("threshold stored", abs(db.get_auto_compact() - 0.25) < 1e-6)
for i in range(12): db.add(id=i, vec=vec(), meta=Metadata())
# forget 2/12 = 0.167 < 0.25 -> no compaction yet
db.forget(0); db.forget(1)
check("below threshold: shells remain", db.get_metadata(0) is not None and db.size() == 12,
      f"size {db.size()}")
# forget a 3rd -> 3/12 = 0.25 >= 0.25 -> auto-compact fires, dead erased
db.forget(2)
check("at threshold: auto-compacted (dead erased)",
      db.get_metadata(0) is None and db.get_metadata(2) is None)
check("size dropped to 9", db.size() == 9, f"got {db.size()}")
check("live records still searchable", len(db.search(vec(), k=20)) == 9)
cleanup(p)

print("4) persistence: compacted state round-trips")
db, p = fresh()
for i in range(5): db.add(id=i, vec=vec(), meta=Metadata())
db.forget(2); db.compact(); db.save()
del db
db2 = fc.DB.open(p, dim=DIM)
check("reload size 4", db2.size() == 4, f"got {db2.size()}")
check("reload forgotten gone", db2.get_metadata(2) is None)
cleanup(p)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
exit(1 if fails else 0)
