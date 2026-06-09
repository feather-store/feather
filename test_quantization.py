"""Test feature B: on-disk int8 quantization (file format v7)."""
import os, sys, glob, tempfile, importlib.util, numpy as np

_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
_cands = [p for p in glob.glob("feather_db/core*.so") if _tag in p]
_so = (_cands or sorted(glob.glob("feather_db/core*.so")))[0]
_spec = importlib.util.spec_from_file_location("core", _so)
fc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fc)
Metadata = fc.Metadata

DIM, N = 128, 500
rng = np.random.default_rng(11)
VECS = {i: rng.standard_normal(DIM).astype(np.float32) for i in range(N)}
Q = rng.standard_normal(DIM).astype(np.float32)
fails = []
def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}: {name} {extra}")
    if not cond: fails.append(name)

def build(path, quantize):
    db = fc.DB.open(path, dim=DIM)
    if quantize: db.set_quantized("text", True)
    for i, v in VECS.items(): db.add(id=i, vec=v, meta=Metadata())
    db.save()
    return db

def cleanup(p):
    for x in (p, p + ".wal", p + ".tmp"):
        if os.path.exists(x): os.remove(x)

print("0) API")
p0 = tempfile.mktemp(suffix=".feather")
db = fc.DB.open(p0, dim=DIM)
check("is_quantized default False", db.is_quantized("text") is False)
db.set_quantized("text", True)
check("is_quantized True after set", db.is_quantized("text") is True)
db.set_quantized("text", False)
check("is_quantized False after unset", db.is_quantized("text") is False)
cleanup(p0)

print("1) file-size reduction (quantized vs float32)")
pf = tempfile.mktemp(suffix=".feather"); pq = tempfile.mktemp(suffix=".feather")
build(pf, False); build(pq, True)
sz_f, sz_q = os.path.getsize(pf), os.path.getsize(pq)
ratio = sz_f / sz_q
check("quantized file >2.5x smaller", ratio > 2.5,
      f"float={sz_f}B quant={sz_q}B ratio={ratio:.2f}x")

print("2) quant flag persists across reload")
dbq = fc.DB.open(pq, dim=DIM)
check("reloaded is_quantized True", dbq.is_quantized("text") is True)
check("reloaded size N", dbq.size() == N, f"got {dbq.size()}")

print("3) reconstruction fidelity (dequantized vector ~ original)")
errs = []
for i in [0, 7, 123, 499]:
    rec = dbq.get_vector(i)
    orig = VECS[i]
    errs.append(float(np.max(np.abs(rec - orig)) / (np.max(np.abs(orig)) + 1e-9)))
check("max relative element error < 1%", max(errs) < 0.01, f"max_rel_err={max(errs):.4f}")

print("4) recall preserved vs float32 baseline")
dbf = fc.DB.open(pf, dim=DIM)
top_f = [r.id for r in dbf.search(Q, k=10)]
top_q = [r.id for r in dbq.search(Q, k=10)]
overlap = len(set(top_f) & set(top_q))
check("top-10 overlap >= 8", overlap >= 8, f"overlap={overlap}/10\n      float={top_f}\n      quant={top_q}")

print("5) quantized search returns full k")
check("returns 10 results", len(dbq.search(Q, k=10)) == 10)

cleanup(pf); cleanup(pq)
print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
exit(1 if fails else 0)
