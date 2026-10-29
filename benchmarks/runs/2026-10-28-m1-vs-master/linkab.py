import os, sys, tempfile, time, threading, numpy as np
import feather_db as fc
N, D = 50000, 128
rng = np.random.default_rng(0)
p = os.path.join(tempfile.mkdtemp(), "l.feather")
db = fc.DB.open(p, dim=D)
metas = []
for i in range(N):
    m = fc.Metadata(); m.content = f"r{i}"; metas.append(m)
db.add_batch(list(range(1, N + 1)), rng.standard_normal((N, D)).astype(np.float32), metas)
Q = rng.standard_normal((500, D)).astype(np.float32)
def run(readers, writers, secs=3.0):
    stop = threading.Event(); cnt = {"r": 0, "w": 0}; lat = []
    def rd(w):
        i = w; n = 0
        while not stop.is_set():
            db.search(Q[i % 500], k=10, record_salience=False); i += 7; n += 1
        cnt["r"] += n
    def wr(w):
        pr = np.random.default_rng(w).integers(1, N + 1, size=(4096, 2)); n = 0; mine = []
        while not stop.is_set():
            a, b = pr[n % 4096]; s = time.perf_counter()
            db.link(int(a), int(b), "related_to", 0.5); mine.append(time.perf_counter() - s); n += 1
        cnt["w"] += n; lat.extend(mine)
    ts = [threading.Thread(target=rd, args=(i,)) for i in range(readers)] + [threading.Thread(target=wr, args=(i,)) for i in range(writers)]
    for t in ts: t.start()
    time.sleep(secs); stop.set()
    for t in ts: t.join()
    p99 = np.percentile(lat, 99) * 1e3 if lat else 0
    print(f"  readers={readers} writers={writers}: reads {cnt['r']/secs:8.0f}/s  links {cnt['w']/secs:8.0f}/s  link p99 {p99:.3f} ms", flush=True)
for r, w in ((0, 1), (0, 4), (8, 4), (8, 1)): run(r, w)
db.close()
