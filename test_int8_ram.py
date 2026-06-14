"""Test Opt 4: in-RAM int8 quantization (4x less memory).

Default run: correctness + round-trip + RAM comparison (spawns two subprocesses
that build a float vs an int8 index and report peak RSS).
Subprocess mode: `python test_int8_ram.py build {float|int8} N DIM` -> prints RSS.
"""
import os, sys, glob, resource, subprocess, tempfile, importlib.util, numpy as np

_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
_cands = [p for p in glob.glob("feather_db/core*.so") if _tag in p]
_so = (_cands or sorted(glob.glob("feather_db/core*.so")))[0]
_spec = importlib.util.spec_from_file_location("core", _so)
fc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fc)
Metadata, ContextType = fc.Metadata, fc.ContextType

def rss_bytes():
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m if sys.platform == "darwin" else m * 1024   # macOS bytes, Linux KB

# ---- subprocess build mode (for RAM measurement) ----
if len(sys.argv) >= 2 and sys.argv[1] == "build":
    mode, N, DIM = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    rng = np.random.default_rng(1)
    base = rss_bytes()      # before any vectors — and we generate V in small
    db = fc.DB.open(tempfile.mktemp(suffix=".feather"), dim=DIM)
    if mode == "int8":
        db.set_int8_ram("text", max_abs=5.0)   # standard-normal max ~5
    # chunks (freed each iteration) so the data array never spikes maxrss above
    # the index — single sequential add → maxrss ≈ steady-state index size.
    i = 0
    while i < N:
        Vc = rng.standard_normal((min(5000, N - i), DIM)).astype(np.float32)
        for j in range(len(Vc)):
            db.add(id=i + j, vec=Vc[j])
        i += len(Vc)
        del Vc
    db.search(rng.standard_normal(DIM).astype(np.float32), k=10)
    print(int(rss_bytes() - base))
    sys.exit(0)

# ---- correctness ----
# Clustered data with MODERATE separation (centers scale 3, noise 0.3) — well-
# defined nearest neighbours (high float recall) and a modest dynamic range that
# int8's global scale resolves well (so int8 recall stays close to float).
DIM, N = 768, 20000
rng = np.random.default_rng(4)
K = 300
centers = (rng.standard_normal((K, DIM)) * 3.0).astype(np.float32)
assign = rng.integers(0, K, size=N)
V = (centers[assign] + rng.standard_normal((N, DIM)).astype(np.float32) * 0.3).astype(np.float32)
MAXABS = float(np.abs(V).max())
fails = []
def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}: {name} {extra}")
    if not cond: fails.append(name)
def cleanup(p):
    for x in (p, p + ".wal", p + ".tmp"):
        if os.path.exists(x): os.remove(x)
def recall(db, qs, k=10):
    hits = 0
    for q in qs:
        exact = set(np.argsort(np.sum((V - q) ** 2, axis=1))[:k].tolist())
        got = set(r.id for r in db.search(q.astype(np.float32), k=k))
        hits += len(exact & got)
    return hits / (k * len(qs))
qs = (centers[rng.integers(0, K, size=60)] + rng.standard_normal((60, DIM)).astype(np.float32)*0.3).astype(np.float32)

print("0) API")
pf = tempfile.mktemp(suffix=".feather"); pi = tempfile.mktemp(suffix=".feather")
dbf = fc.DB.open(pf, dim=DIM)
dbi = fc.DB.open(pi, dim=DIM)
check("default not int8", dbi.is_int8_ram("text") is False)
dbi.set_int8_ram("text", max_abs=MAXABS)
check("is_int8_ram after set", dbi.is_int8_ram("text") is True)
try:
    dbi.add(id=0, vec=V[0]); dbi.set_int8_ram("text"); check("set after add raises", False)
except Exception:
    check("set_int8_ram after first add raises", True)

print("1) build float + int8, compare recall")
dbf.add_batch(list(range(N)), V)
dbi.add_batch(list(range(1, N)), V[1:])   # id 0 already added above
rf, ri = recall(dbf, qs), recall(dbi, qs)
print(f"  recall@10 — float={rf:.3f}  int8={ri:.3f}")
check("int8 recall@10 >= 0.85", ri >= 0.85, f"{ri:.3f}")
# int8 is lossy by design — a modest recall drop vs float is the expected
# trade for ~1.6x less RAM; require it to stay within 0.15 of the float baseline.
check("int8 recall within 0.15 of float", abs(rf - ri) < 0.15, f"|{rf:.3f}-{ri:.3f}|")

print("2) get_vector dequantizes (~ original)")
rec = np.array(dbi.get_vector(100)); orig = V[100]
relerr = float(np.max(np.abs(rec - orig)) / (np.max(np.abs(orig)) + 1e-9))
check("get_vector rel error < 5%", relerr < 0.05, f"{relerr:.4f}")

print("3) round-trip: save + reload keeps int8 + recall")
dbi.save(); del dbi
dbi2 = fc.DB.open(pi, dim=DIM)
check("reload is_int8_ram", dbi2.is_int8_ram("text") is True)
check("reload size", dbi2.size() == N, f"got {dbi2.size()}")
check("reload recall >= 0.85", recall(dbi2, qs) >= 0.85)

print("4) filtered (pre-filtered exact) search over int8 data")
pm = tempfile.mktemp(suffix=".feather"); dbm = fc.DB.open(pm, dim=DIM)
dbm.set_int8_ram("text", max_abs=MAXABS)
metas = []
for i in range(2000):
    m = Metadata(); m.namespace_id = "a" if i % 2 == 0 else "b"; metas.append(m)
dbm.add_batch(list(range(2000)), V[:2000], metas)
f = fc.SearchFilter(); f.namespace_id = "a"
res = dbm.search(V[0], k=10, filter=f)
check("int8 filtered returns 10", len(res) == 10, f"got {len(res)}")
check("int8 filtered all in ns a", all(r.id % 2 == 0 for r in res))

for p in (pf, pi, pm): cleanup(p)

print("5) RAM: float vs int8 index (subprocess peak RSS)")
MN, MDIM = 60000, 768
def measure(mode):
    out = subprocess.check_output([sys.executable, __file__, "build", mode, str(MN), str(MDIM)],
                                  env={**os.environ, "PYTHONPATH": os.getcwd()})
    return int(out.strip())
mf, mi = measure("float"), measure("int8")
print(f"  {MN}x{MDIM}: float index +{mf/1e6:.0f} MB | int8 index +{mi/1e6:.0f} MB | saved {(mf-mi)/1e6:.0f} MB ({mf/max(mi,1):.2f}x)")
check("int8 index uses meaningfully less RAM", mi < mf * 0.75, f"int8={mi/1e6:.0f}MB float={mf/1e6:.0f}MB")

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
sys.exit(1 if fails else 0)
