"""A corrupt file gives an error, never a crash.

Before this, 18 of 300 single-byte corruptions of a v9 file killed the
process outright (SIGBUS), read-only opens included: the graph loader trusted
every count and link it read. The shapes that did it:

  * count x slot_size was bounds-checked before slot_size was, so the product
    could wrap under the remaining-bytes guard; the base layer was then
    allocated at the wrapped size and filled at the real one (heap overflow).
  * a link to a node that does not exist, or into a layer its target does not
    have, was followed by the next search.
  * Metadata::deserialize sized strings from 32-bit lengths before reading
    them: a 4 GB zero-filled allocation per corrupt record.
  * a WAL ADD record's dim sized a vector before anything checked it.

The fuzz is seeded, so a failure names the case that reproduces it. Scale it
up locally with FEATHER_FUZZ_CASES=2000 (and FEATHER_FUZZ_SEED to vary it).
Each case runs in a forked child, both read-only and as a writer that then
adds a record and saves; on Linux the child also runs under an address-space
limit, so an allocation bomb fails the test instead of passing slowly.
"""
import json
import os
import random
import struct
import subprocess
import sys
import zlib

import numpy as np
import pytest

import feather_db

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="probes run in forked children")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = int(os.environ.get("FEATHER_FUZZ_SEED", "20261027"))
CASES = int(os.environ.get("FEATHER_FUZZ_CASES", "60"))
DIM = 8

# ── xxh64, so v10 checksums can be recomputed without the xxhash package ──
M64 = (1 << 64) - 1
P1, P2, P3, P4, P5 = (0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
                      0x85EBCA77C2B2AE63, 0x27D4EB2F165667C5)


def _rotl(x, r):
    return ((x << r) | (x >> (64 - r))) & M64


def _round(acc, v):
    return (_rotl((acc + v * P2) & M64, 31) * P1) & M64


def xxh64(data, seed=0):
    n, i = len(data), 0
    if n >= 32:
        v = [(seed + P1 + P2) & M64, (seed + P2) & M64, seed, (seed - P1) & M64]
        for i in range(0, n - 31, 32):
            w = struct.unpack_from("<4Q", data, i)
            v = [_round(v[j], w[j]) for j in range(4)]
        i = (n // 32) * 32
        h = (_rotl(v[0], 1) + _rotl(v[1], 7) + _rotl(v[2], 12) + _rotl(v[3], 18)) & M64
        for x in v:
            h = ((h ^ _round(0, x)) * P1 + P4) & M64
    else:
        h = (seed + P5) & M64
    h = (h + n) & M64
    while i + 8 <= n:
        h = (_rotl(h ^ _round(0, struct.unpack_from("<Q", data, i)[0]), 27) * P1 + P4) & M64
        i += 8
    if i + 4 <= n:
        h = (_rotl(h ^ ((struct.unpack_from("<I", data, i)[0] * P1) & M64), 23) * P2 + P3) & M64
        i += 4
    while i < n:
        h = (_rotl(h ^ ((data[i] * P5) & M64), 11) * P1) & M64
        i += 1
    h ^= h >> 33
    h = (h * P2) & M64
    h ^= h >> 29
    h = (h * P3) & M64
    return h ^ (h >> 32)


# ── v10 / WAL layout, enough to aim corruption past the checksums ─────────

def v10_sections(raw):
    """[(offset, type, payload_len)] of the committed sections."""
    end = max(struct.unpack_from("<Q", raw, s * 4096 + 24)[0] for s in (0, 1))
    out, pos = [], 8192
    while pos < end:
        typ, _, ln = struct.unpack_from("<IIQ", raw, pos)
        out.append((pos, typ, ln))
        pos += 24 + ln
    return out


def v10_reseal(data, sec):
    pos, _, ln = sec
    struct.pack_into("<Q", data, pos + 16 + ln,
                     xxh64(bytes(data[pos + 16:pos + 16 + ln]), xxh64(bytes(data[pos:pos + 16]))))


def v10_structural(raw):
    """Byte ranges a parser interprets: section headers and every payload
    except the vector bytes (and alignment padding) of VECTORS sections."""
    spans = []
    for sec in v10_sections(raw):
        pos, typ, ln = sec
        if typ == 5:
            nl = struct.unpack_from("<H", raw, pos + 16)[0]
            spans.append((pos, pos + 16 + 2 + nl + 12, sec))
        else:
            spans.append((pos, pos + 16 + ln, sec))
    return spans


WALGEN = 0x0A     # the generation record each WAL file opens with (0.22)


def wal_records(raw, *, with_gen=False):
    """[(offset, length incl. CRC)] of a v2 WAL's records — its data records,
    unless `with_gen` (the leading WALGEN record is bookkeeping)."""
    out, pos = [], 8
    while pos + 13 <= len(raw):
        plen = struct.unpack_from("<I", raw, pos + 9)[0]
        if with_gen or raw[pos] != WALGEN:
            out.append((pos, 13 + plen + 4))
        pos += 13 + plen + 4
    return out


def wal_reseal(data, rec):
    pos, ln = rec
    struct.pack_into("<I", data, pos + ln - 4, zlib.crc32(bytes(data[pos:pos + ln - 4])))


# ── base files ────────────────────────────────────────────────────────────

def _records(db, rng, ids, with_visual=True):
    for i in ids:
        m = feather_db.Metadata()
        m.content = f"record {i} alpha beta"
        m.source = "src"
        m.namespace_id = "ns" if i % 3 else "other"
        m.entity_id = f"e{i % 5}"
        m.tags_json = '["a","b"]'
        m.set_attribute("k", f"v{i}")
        if i % 4 == 0:
            m.key = f"key-{i}"            # the reserved "\0key" attribute
        db.add(i, rng.random(DIM, dtype=np.float32), m)
        if with_visual:
            db.add(i, rng.random(4, dtype=np.float32), modality="visual")


def build(path, kind, monkeypatch):
    rng = np.random.default_rng(0)
    monkeypatch.delenv("FEATHER_FORMAT", raising=False)
    if kind.startswith("v9"):
        monkeypatch.setenv("FEATHER_FORMAT", "9")
    db = feather_db.DB.open(path, dim=DIM)
    # In-RAM int8 with refine: v10 writes FULLVEC sections (float originals),
    # the WAL holds the originals the next commit would write.
    db.set_int8_ram("visual", 1.0)
    if kind == "v9-rebuild":
        db.set_quantized("text")          # quantized modalities are rebuilt on load
    _records(db, rng, range(60))
    for i in range(1, 60):
        db.link(i, i - 1, "next", 0.5)
    db.forget(3)
    db.close()
    if kind in ("v10", "wal"):
        # Delta commits on top of the full image: LINKS-only graph records,
        # META puts and deletes, salience.
        db = feather_db.DB.open(path, dim=DIM)
        _records(db, rng, range(60, 70), with_visual=False)
        db.link(65, 2, "rel", 0.9)
        db.search(rng.random(DIM, dtype=np.float32), k=5)
        db.update_importance(10, 0.25)
        # A keyed batch: VERSIONS and IDEMPOTENCY sections, BEGIN/COMMIT in the WAL.
        b = feather_db.WriteBatch()
        b.update(11, db.get_metadata(11), expected_version=db.version(11)).link(11, 12, "cites")
        db.write(b, idempotency_key="fuzz-key", fingerprint="fp")
        db.save()
        if kind == "wal":
            _records(db, rng, range(70, 75))
            b = feather_db.WriteBatch()
            b.put(75, rng.random(DIM, dtype=np.float32), db.get_metadata(70)).link(75, 70, "cites")
            db.write(b, idempotency_key="fuzz-wal-key", fingerprint="fp")
            db.link(71, 70, "rel", 0.7)
            db.update_importance(70, 0.5)
            db.forget(72)
            db.purge("other")
            db.close(save=False)          # these live in the WAL alone
            return
        db.close()
    monkeypatch.delenv("FEATHER_FORMAT", raising=False)


# ── probing ──────────────────────────────────────────────────────────────

PROBE = r"""
import json, os, resource, signal, sys
import numpy as np, feather_db
DIM = %d
q = {"text": np.full(DIM, 0.5, dtype=np.float32), "visual": np.full(4, 0.5, dtype=np.float32)}

def limit_memory():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if sys.platform.startswith("linux"):
        vm = int(open("/proc/self/statm").read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
        cap = vm + (1 << 30)        # 1 GB beyond what is already mapped
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

def exercise(path, ro):
    db = feather_db.DB.open(path, dim=DIM, read_only=ro)
    for mod in db.modalities() if hasattr(db, "modalities") else ["text", "visual"]:
        if mod in q:
            db.search(q[mod], k=5, modality=mod)
    db.keyword_search("alpha", k=5)
    db.context_chain(q["text"], k=3, hops=2)
    for i in db.get_all_ids():
        db.get_metadata(i); db.get_edges(i); db.get_incoming(i)
    db.export_graph_json("", "")
    if not ro:
        m = feather_db.Metadata(); m.content = "after alpha"
        db.add(10_000, q["text"], m)
        db.link(10_000, 1, "rel", 0.5)
        db.save()
    db.close()

out = {}
for case in json.loads(sys.stdin.read()):
    for ro in (True, False):
        pid = os.fork()
        if pid == 0:
            signal.alarm(30)
            limit_memory()
            try:
                exercise(case, ro); code = 0
            except MemoryError:
                code = 3
            except Exception:
                code = 1
            os._exit(code)
        _, st = os.waitpid(pid, 0)
        if os.WIFSIGNALED(st):
            sig = os.WTERMSIG(st)
            r = "hang" if sig == signal.SIGALRM else "crash:" + signal.Signals(sig).name
        else:
            r = {0: "ok", 1: "error", 3: "MemoryError"}.get(os.WEXITSTATUS(st), "exit")
        out.setdefault(case, []).append(r)
print(json.dumps(out))
""" % DIM


def probe(cases):
    """{case path: [read-only outcome, writer outcome]}"""
    # Few malloc arenas: each reserves 64 MB of address space, which counts
    # against the RLIMIT_AS a probe runs under on Linux.
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MALLOC_ARENA_MAX="2")
    env.pop("FEATHER_FORMAT", None)
    r = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {REPO!r})\n" + PROBE],
                       input=json.dumps(cases), capture_output=True, text=True, env=env,
                       timeout=60 + 70 * len(cases))
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


def mutate(rng, data, lo, hi):
    """One corruption inside [lo, hi): a random byte, or a word set to a huge
    value (what a corrupt length field looks like). Returns a description."""
    kind = rng.choice(["byte", "byte", "word32", "word64"])
    if kind == "byte" or hi - lo < 8:
        off = rng.randrange(lo, hi)
        old = data[off]
        data[off] = (old ^ rng.randrange(1, 256)) & 0xFF
        return f"byte@{off} {old:#04x}->{data[off]:#04x}"
    off = rng.randrange(lo, hi - (4 if kind == "word32" else 8))
    if kind == "word32":
        val = rng.choice([0xFFFFFFFF, 0xFFFFFFF0, 0x80000000, 0x7FFFFFFF, rng.randrange(1 << 32)])
        struct.pack_into("<I", data, off, val)
    else:
        val = rng.choice([M64, 1 << 63, (1 << 62) + 1, rng.randrange(1 << 64)])
        struct.pack_into("<Q", data, off, val)
    return f"{kind}@{off}={val:#x}"


KINDS = ["v9-graph", "v9-rebuild", "v10", "v10-resealed", "wal"]


@pytest.mark.parametrize("kind", KINDS)
def test_corruption_is_an_error_not_a_crash(tmp_path, monkeypatch, kind):
    base = str(tmp_path / "base.feather")
    build(base, kind.replace("-resealed", ""), monkeypatch)
    raw = open(base, "rb").read()
    wal = open(base + ".wal", "rb").read() if kind == "wal" else None
    rng = random.Random(f"{SEED}:{kind}")

    if kind == "v10-resealed":
        spans = v10_structural(raw)
        # Self-check of the Python xxh64: resealing an intact file is a no-op.
        clean = bytearray(raw)
        for _, _, sec in spans:
            v10_reseal(clean, sec)
        assert bytes(clean) == raw, "xxh64 reimplementation disagrees with the engine"
    if kind == "wal":
        recs = wal_records(wal)
        clean = bytearray(wal)
        for rec in recs:
            wal_reseal(clean, rec)
        assert bytes(clean) == wal, "WAL record framing or CRC disagrees with the engine"

    cases, what = [], {}
    for c in range(CASES):
        d = tmp_path / f"c{c}"
        d.mkdir()
        path = str(d / "db.feather")
        if kind == "wal":
            data = bytearray(wal)
            rec = rng.choice(recs)
            desc = mutate(rng, data, rec[0], rec[0] + rec[1] - 4)
            wal_reseal(data, rec)
            open(path, "wb").write(raw)
            open(path + ".wal", "wb").write(bytes(data))
        else:
            data = bytearray(raw)
            if kind == "v10-resealed":
                lo, hi, sec = rng.choice(spans)
                desc = mutate(rng, data, lo, hi)
                v10_reseal(data, sec)
            else:
                desc = mutate(rng, data, 0, len(data))
            open(path, "wb").write(bytes(data))
        cases.append(path)
        what[path] = desc

    results = probe(cases)
    bad = [(what[p], mode, r) for p, rs in results.items()
           for mode, r in zip(("read-only", "writer"), rs) if r not in ("ok", "error")]
    assert not bad, (f"{len(bad)} of {2 * CASES} opens of a corrupt {kind} file did not fail "
                     f"cleanly (FEATHER_FUZZ_SEED={SEED}):\n" +
                     "\n".join(f"  {d:36s} {m:9s} {r}" for d, m, r in bad[:40]))
    # The fuzz must reach the parsers, not only bounce off a checksum.
    outcomes = [r for rs in results.values() for r in rs]
    if kind != "v10":
        assert outcomes.count("error") >= CASES // 10, f"{kind}: too few corruptions detected"


# ── the specific shapes, pinned ───────────────────────────────────────────
# Each opens in a child: a regression here is a crash, and it must fail this
# test rather than take the whole pytest run down with it.

def open_in_child(path, read_only):
    """(outcome, message, peak RSS in MB) of opening `path` in a fresh process."""
    r = subprocess.run([sys.executable, "-c", f"""
import resource, sys; sys.path.insert(0, {REPO!r})
import feather_db
try:
    feather_db.DB.open({path!r}, dim={DIM}, read_only={read_only}).close(); print("opened")
except Exception as e:
    print("error:", type(e).__name__, e)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(peak / (1 << 20 if sys.platform == "darwin" else 1 << 10))
"""], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return f"exit {r.returncode}", r.stderr[-500:], None
    lines = r.stdout.splitlines()
    return lines[0].split(":")[0], lines[0], float(lines[-1])


def _v9_with_markers(path, monkeypatch):
    monkeypatch.setenv("FEATHER_FORMAT", "9")
    db = feather_db.DB.open(path, dim=DIM)
    rng = np.random.default_rng(1)
    for i in range(20):
        m = feather_db.Metadata()
        m.content = "CONTENTMARK" if i == 7 else f"r{i}"
        if i == 7:
            m.set_attribute("key", "VALUEMARK")
        db.add(i, rng.random(DIM, dtype=np.float32), m)
    db.close()
    monkeypatch.delenv("FEATHER_FORMAT")
    return bytearray(open(path, "rb").read())


@pytest.mark.parametrize("marker", [b"CONTENTMARK", b"VALUEMARK"])
def test_a_huge_string_length_is_refused_without_allocating_it(tmp_path, monkeypatch, marker):
    """metadata.cpp sized the string (content / attribute value) from the
    on-disk u32 before reading it: 4 GB, zero-filled, per corrupt record."""
    p = str(tmp_path / "s.feather")
    raw = _v9_with_markers(p, monkeypatch)
    at = raw.index(marker)
    struct.pack_into("<I", raw, at - 4, 0xFFFFFFF0)
    open(p, "wb").write(bytes(raw))
    outcome, msg, peak_mb = open_in_child(p, read_only=True)
    assert outcome == "error" and "truncated" in msg, msg
    assert peak_mb < 400, f"peak RSS {peak_mb:.0f} MB: the length was allocated before it was checked"


def test_a_wrapping_graph_count_is_refused(tmp_path, monkeypatch):
    """count * slot_size wrapped below the remaining-bytes guard, then the base
    layer was allocated at the wrapped size and filled at the real one."""
    p = str(tmp_path / "g.feather")
    raw = _v9_with_markers(p, monkeypatch)
    legacy = 16 * 2 * 4 + 4 + DIM * 4 + 8            # links0 + vector + label, M=16
    at = raw.index(struct.pack("<QQ", 20, legacy))
    struct.pack_into("<Q", raw, at, (1 << 62) + 1)    # x 172 wraps to 172 mod 2^64
    open(p, "wb").write(bytes(raw))
    for ro in (True, False):
        outcome, msg, _ = open_in_child(p, read_only=ro)
        assert outcome == "error" and "corrupt" in msg, msg


def test_a_wal_add_with_an_absurd_dim_is_refused(tmp_path):
    """A CRC-valid ADD whose dim exceeds its own payload sized a vector first."""
    p = str(tmp_path / "w.feather")
    db = feather_db.DB.open(p, dim=DIM)
    db.add(1, np.ones(DIM, dtype=np.float32))
    db.close(save=False)
    data = bytearray(open(p + ".wal", "rb").read())
    rec = wal_records(data)[0]
    mod_len = struct.unpack_from("<H", data, rec[0] + 13)[0]
    struct.pack_into("<I", data, rec[0] + 13 + 2 + mod_len, 0x3FFFFFFF)
    wal_reseal(data, rec)
    open(p + ".wal", "wb").write(bytes(data))
    outcome, msg, peak_mb = open_in_child(p, read_only=True)
    assert outcome == "error" and "implausible vector dim" in msg, msg
    assert peak_mb < 400, f"peak RSS {peak_mb:.0f} MB"
    assert open(p + ".wal", "rb").read() == bytes(data), "a refused open modified the WAL"


def test_intact_files_still_open(tmp_path, monkeypatch):
    """The checks reject corruption, not anything the writers produce."""
    for kind in ("v9-graph", "v9-rebuild", "v10", "wal"):
        p = str(tmp_path / f"{kind}.feather")
        build(p, kind, monkeypatch)
        assert probe([p])[p] == ["ok", "ok"], kind
