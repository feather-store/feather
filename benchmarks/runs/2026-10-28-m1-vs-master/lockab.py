import os, sys, tempfile, time, threading, numpy as np
import feather_db as fc
N, D = 50000, 128
rng = np.random.default_rng(0)
p = os.path.join(tempfile.mkdtemp(), "l.feather")
db = fc.DB.open(p, dim=D)
metas = []
words = "alpha beta gamma delta creative policy brand tone budget".split()
for i in range(N):
    m = fc.Metadata(); m.content = " ".join(rng.choice(words, 6)); metas.append(m)
db.add_batch(list(range(1, N + 1)), rng.standard_normal((N, D)).astype(np.float32), metas)
Q = rng.standard_normal((500, D)).astype(np.float32)
nid = [N + 10]; lk = threading.Lock()
def run(name, readers, writers, wkind, rkind="search", secs=4.0):
    stop = threading.Event(); cnt = {"r": 0, "w": 0}; rl = []
    def rd(w):
        i = w; n = 0; mine = []
        while not stop.is_set():
            s = time.perf_counter()
            if rkind == "search": db.search(Q[i % 500], k=10, record_salience=False)
            else: db.hybrid_search(Q[i % 500], "creative policy", k=10)
            mine.append(time.perf_counter() - s); i += 7; n += 1
        cnt["r"] += n; rl.extend(mine)
    def wr(w):
        r = np.random.default_rng(w); pr = r.integers(1, N + 1, size=(4096, 2)); v = r.standard_normal((256, D)).astype(np.float32); n = 0
        while not stop.is_set():
            if wkind == "link":
                a, b = pr[n % 4096]; db.link(int(a), int(b), "related_to", 0.5)
            else:
                with lk: nid[0] += 1; i = nid[0]
                m = fc.Metadata(); m.content = "new record creative"; db.add(i, v[n % 256], m)
            n += 1
        cnt["w"] += n
    ts = [threading.Thread(target=rd, args=(i,)) for i in range(readers)] + [threading.Thread(target=wr, args=(i,)) for i in range(writers)]
    for t in ts: t.start()
    time.sleep(secs); stop.set()
    for t in ts: t.join()
    return cnt["r"] / secs, cnt["w"] / secs, np.percentile(rl, 99) * 1e3
mode = "nobatch" if os.environ.get("FEATHER_AB_NOBATCH") else "batch"
for name, r, w, wk, rk in (("r8+add4", 8, 4, "add", "search"), ("r8+add16", 8, 16, "add", "search"),
                           ("r8+link4", 8, 4, "link", "search"), ("hyb8+add4", 8, 4, "add", "hybrid")):
    rs, ws, p99 = run(name, r, w, wk, rk)
    print(f"{mode:8s} {name:10s} reads {rs:7.0f}/s p99 {p99:6.2f} ms  writes {ws:7.0f}/s", flush=True)
db.close()
