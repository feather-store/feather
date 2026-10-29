import os, sys, tempfile, time, numpy as np
import feather_db as fc
N, D = 50000, 768
rng = np.random.default_rng(0)
B = rng.standard_normal((64, D)).astype(np.float32)
V = (rng.standard_normal((N, 64)).astype(np.float32) @ B + 0.3 * rng.standard_normal((N, D)).astype(np.float32)) / 8
Q = (rng.standard_normal((300, 64)).astype(np.float32) @ B + 0.3 * rng.standard_normal((300, D)).astype(np.float32)) / 8
gt = [set(np.argsort(((V - q) ** 2).sum(1))[:10].tolist()) for q in Q]
def run(tag, refine=None):
    p = os.path.join(tempfile.mkdtemp(), "i.feather")
    db = fc.DB.open(p, dim=D)
    if refine is not None:
        if refine == "master": db.set_int8_ram("text", float(np.abs(V).max()))
        else: db.set_int8_ram("text", float(np.abs(V).max()), refine=refine)
    db.add_batch(list(range(N)), V); db.save()
    db.set_ef(50)
    for q in Q[:20]: db.search(q, k=10, record_salience=False)
    ts, rc = [], 0
    for q, g in zip(Q, gt):
        s = time.perf_counter(); r = db.search(q, k=10, record_salience=False); ts.append((time.perf_counter() - s) * 1e3)
        rc += len(g & {x.id for x in r})
    print(f"{tag:22s} p50 {np.percentile(ts,50):.3f} ms  recall@10 {rc/3000:.3f}", flush=True)
    db.close()
run("float")
if len(sys.argv) > 1: run("int8 (master)", "master")
else:
    run("int8 refine0", 0); run("int8 refine2", 2)
