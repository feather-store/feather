"""Memory of N read-only processes on one file, with and without mmap.

    python benchmarks/readers_memory.py path/to.feather --readers 10 --dim 384

Each reader opens the file read-only, runs a few searches and reports its
private physical footprint (macOS proc_pid_rusage; Linux: Private_* from
/proc/self/smaps_rollup). Mapped vectors live in the shared page cache once,
so they appear in no reader's private memory.
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r"""
import ctypes, json, os, sys, time
sys.path.insert(0, REPO)
import numpy as np, feather_db
def private():
    if sys.platform == "darwin":
        b = (ctypes.c_uint8 * 512)()
        ctypes.CDLL("/usr/lib/libSystem.B.dylib").proc_pid_rusage(os.getpid(), 2, b)
        return int.from_bytes(bytes(b[72:80]), "little")
    total = 0
    for line in open("/proc/self/smaps_rollup"):
        if line.startswith(("Private_Clean", "Private_Dirty")):
            total += int(line.split()[1]) * 1024
    return total
p0 = private(); t = time.perf_counter()
db = feather_db.DB.open(PATH, dim=DIM, read_only=True)
open_ms = (time.perf_counter() - t) * 1e3
rng = np.random.default_rng(os.getpid())
for _ in range(50):
    db.search(rng.random(DIM, dtype=np.float32), k=10)
print(json.dumps({"private": private() - p0, "open_ms": open_ms,
                  "mapped": db.storage_stats()["mapped_bytes"]}), flush=True)
sys.stdin.read()          # stay alive until every reader has reported
"""


def run(path, dim, readers, mmap):
    env = {**os.environ, "FEATHER_MMAP": "1" if mmap else "0"}
    code = CHILD.replace("REPO", repr(REPO)).replace("PATH", repr(path)).replace("DIM", str(dim))
    procs = [subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True, env=env) for _ in range(readers)]
    rows = [json.loads(p.stdout.readline()) for p in procs]     # all alive at once
    for p in procs:
        p.stdin.close(); p.wait()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--readers", type=int, default=10)
    ap.add_argument("--dim", type=int, default=384)
    a = ap.parse_args()
    print(f"{a.readers} read-only processes on {a.path} ({os.path.getsize(a.path)/1e6:.0f} MB)")
    for mmap in (False, True):
        rows = run(a.path, a.dim, a.readers, mmap)
        total = sum(r["private"] for r in rows)
        print(f"  {'mmap' if mmap else 'heap':5} private per reader {total/len(rows)/1e6:6.0f} MB   "
              f"all readers {total/1e6:6.0f} MB"
              + (f" + {rows[0]['mapped']/1e6:.0f} MB shared once" if mmap else "")
              + f"   open {min(r['open_ms'] for r in rows):5.0f} ms")


if __name__ == "__main__":
    main()
