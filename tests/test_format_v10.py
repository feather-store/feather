"""File format v10: append-only commits behind two checksummed headers.

What the format promises, and what each test here pins:

  * a save writes the change, not the DB                   -> TestSaveCost
  * a reload gives back exactly what was in memory         -> TestRoundTrip
  * a crash at any point of a commit leaves a file that
    opens to a state the WAL completes                     -> TestCrashStates
  * corruption is refused, not misread, and never
    "recovered" by discarding data that is there           -> TestCorruption
  * readers keep a consistent snapshot while it is appended
  * superseded bytes are reclaimed (vacuum)                -> TestVacuum

Crash states are rebuilt byte-for-byte from real files rather than by killing a
process at a lucky moment: the bytes a crash can leave behind are fully
determined by the commit order (sections, fsync, header, fsync, drop
<wal>.old), so each intermediate state can be constructed exactly.
"""
import os
import shutil
import struct

import numpy as np
import pytest

import feather_db
from feather_db import DB

MAGIC = 0x46454154
SLOT = 4096
DATA_START = 8192


# ── helpers ──────────────────────────────────────────────────────────────

def meta(content, ns="ns", **attrs):
    m = feather_db.Metadata()
    m.content = content
    m.namespace_id = ns
    m.timestamp = 1_700_000_000
    for k, v in attrs.items():
        m.set_attribute(k, v)
    return m


def header(raw, slot):
    magic, ver, guard, s, gen, end, start, t = struct.unpack_from("<IIIIQQQQ", raw, slot * SLOT)
    return dict(magic=magic, version=ver, guard=guard, slot=s, generation=gen, end=end, start=start)


def live(m):
    return m.source != "_forgotten" and m.get_attribute("_deleted", "") != "true"


def snapshot(db, modalities=("text",), salience=True):
    """Everything a reload must reproduce for live records."""
    recs = {}
    for i in sorted(db.all_ids()):
        m = db.get_metadata(i)
        if m is None or not live(m):
            continue
        d = m.to_dict()
        if not salience:
            d.pop("recall_count"); d.pop("last_recalled_at")
        d["edges"] = sorted((e.target_id, e.rel_type, round(e.weight, 6)) for e in db.get_edges(i))
        recs[i] = d
    vecs = {mod: {i: db.get_vector(i, mod).tobytes() for i in sorted(db.get_all_ids(mod))}
            for mod in modalities}
    return recs, vecs


def _no_versions(snap):
    recs, vecs = snap
    return {i: {k: v for k, v in d.items() if k != "version"} for i, d in recs.items()}, vecs


def build(path, n=2000, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    db = DB.open(path, dim=dim)
    db.add_batch(list(range(n)), rng.random((n, dim), dtype=np.float32),
                 [meta(f"record {i}") for i in range(n)])
    db.save()
    return db, rng


# ── layout ───────────────────────────────────────────────────────────────

class TestLayout:
    def test_two_headers_then_sections(self, tmp_path):
        p = str(tmp_path / "a.feather")
        db, _ = build(p, n=100)
        db.close()
        raw = open(p, "rb").read()
        h0, h1 = header(raw, 0), header(raw, 1)
        for h in (h0, h1):
            assert h["magic"] == MAGIC and h["version"] == 10
            assert h["start"] == DATA_START and h["end"] == len(raw)
        # a full write puts the same commit in both slots
        assert h0["generation"] == h1["generation"] == 1

    def test_legacy_guard_makes_old_builds_refuse_the_file(self, tmp_path):
        """v3-v9 readers take offset 8 as the metadata count and reject a
        count the file cannot hold, instead of parsing v10 as v9."""
        p = str(tmp_path / "g.feather")
        db, _ = build(p, n=10)
        db.close()
        raw = open(p, "rb").read()
        assert struct.unpack_from("<I", raw, 8)[0] == 0xFFFFFFFF

    def test_commits_alternate_slots_and_bump_the_generation(self, tmp_path):
        p = str(tmp_path / "alt.feather")
        db, rng = build(p, n=100, dim=8)
        for g in range(2, 6):
            db.add(1000 + g, rng.random(8, dtype=np.float32), meta(f"g{g}"))
            db.save()
            raw = open(p, "rb").read()
            newest = header(raw, g % 2)
            older = header(raw, (g + 1) % 2)
            assert newest["generation"] == g and newest["end"] == len(raw)
            assert older["generation"] == g - 1 and older["end"] < newest["end"]
        db.close()

    def test_stats_report_the_format(self, tmp_path):
        p = str(tmp_path / "s.feather")
        db = DB.open(p, dim=8)
        assert db.storage_stats()["format"] == 0          # nothing on disk yet
        db.add(1, np.ones(8, dtype=np.float32))
        db.save()
        st = db.storage_stats()
        assert st["format"] == 10 and st["generation"] == 1 and st["last_commit_full"]
        assert st["file_bytes"] == os.path.getsize(p)
        db.close()


# ── save cost ────────────────────────────────────────────────────────────

class TestSaveCost:
    def test_a_save_writes_the_change_not_the_db(self, tmp_path):
        p = str(tmp_path / "cost.feather")
        db, rng = build(p, n=3000, dim=64)
        full = os.path.getsize(p)
        for i in range(10):
            db.add(5000 + i, rng.random(64, dtype=np.float32), meta(f"new {i}"))
        db.save()
        st = db.storage_stats()
        assert not st["last_commit_full"]
        # 10 records + the ~M neighbours each one rewired, link lists only
        assert st["last_commit_bytes"] < full * 0.03, st
        assert os.path.getsize(p) == full + st["last_commit_bytes"]
        db.close()

    def test_a_save_with_nothing_to_write_writes_nothing(self, tmp_path):
        p = str(tmp_path / "noop.feather")
        db, _ = build(p, n=200)
        before = open(p, "rb").read()
        gen = db.storage_stats()["generation"]
        db.save()
        db.save()
        assert open(p, "rb").read() == before
        assert db.storage_stats()["generation"] == gen
        db.close()
        assert open(p, "rb").read() == before, "close() of an unchanged DB rewrote it"

    def test_metadata_only_change_writes_no_vectors(self, tmp_path):
        p = str(tmp_path / "m.feather")
        db, _ = build(p, n=1000, dim=256)
        db.update_importance(5, 0.25)
        db.link(1, 2, "rel", 0.5)
        db.save()
        # two metadata records; a 256-d vector alone would be 1 KB
        assert db.storage_stats()["last_commit_bytes"] < 1000
        db.close()

    def test_salience_is_written_compactly(self, tmp_path):
        p = str(tmp_path / "sal.feather")
        rng = np.random.default_rng(3)
        db = DB.open(p, dim=16)
        db.add_batch(list(range(500)), rng.random((500, 16), dtype=np.float32),
                     [meta("x" * 2000) for _ in range(500)])     # big contents
        db.save()
        for _ in range(20):
            db.search(rng.random(16, dtype=np.float32), k=10)
        counts = {i: db.get_metadata(i).recall_count for i in range(500)}
        db.save()
        touched = sum(1 for c in counts.values() if c)
        # a full META put would carry the 2 KB content of every touched record
        assert db.storage_stats()["last_commit_bytes"] < touched * 40 + 200
        db.close()
        db = DB.open(p, dim=16)
        assert {i: db.get_metadata(i).recall_count for i in range(500)} == counts
        db.close()


# ── round trip ───────────────────────────────────────────────────────────

def random_workload(db, rng, steps, dims, saves=True):
    ids = list(range(300))
    nss = ["a", "b", "c"]
    for step in range(steps):
        op = rng.random()
        i = int(rng.choice(ids))
        if op < 0.30:
            db.add(i, rng.random(dims["text"], dtype=np.float32),
                   meta(f"t{step}", ns=str(rng.choice(nss)), k=str(step % 3)))
            if rng.random() < 0.5:
                db.add(i, rng.random(dims["visual"], dtype=np.float32), meta(f"v{step}"),
                       modality="visual")
        elif op < 0.40:
            batch = [int(x) for x in rng.choice(ids, 12, replace=False)]
            db.add_batch(batch, rng.random((12, dims["text"]), dtype=np.float32),
                         [meta(f"b{step}-{j}", ns=str(rng.choice(nss))) for j in batch])
        elif op < 0.50:
            db.update_metadata(i, meta(f"u{step}", ns="b", tag="upd"))
        elif op < 0.58:
            db.update_importance(i, float(rng.random()))
        elif op < 0.70:
            db.link(i, int(rng.choice(ids)), str(rng.choice(["r1", "r2"])), float(rng.random()))
        elif op < 0.77:
            db.forget(i)
        elif op < 0.90:
            db.search(rng.random(dims["text"], dtype=np.float32), k=5)
        elif op < 0.905:
            db.purge(str(rng.choice(nss)))
        elif saves:
            db.save()


class TestRoundTrip:
    DIMS = {"text": 24, "visual": 12}

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_reload_reproduces_memory_exactly(self, tmp_path, seed):
        p = str(tmp_path / f"rt{seed}.feather")
        rng = np.random.default_rng(seed)
        db = DB.open(p, dim=24)
        db.set_int8_ram("visual", 1.0)
        random_workload(db, rng, 1500, self.DIMS)
        queries = rng.random((40, 24), dtype=np.float32)
        before = [[(r.id, r.score) for r in db.search(q, k=8)] for q in queries]
        state = snapshot(db, ("text", "visual"))
        db.close()

        db = DB.open(p, dim=24)
        assert snapshot(db, ("text", "visual")) == state
        after = [[(r.id, r.score) for r in db.search(q, k=8)] for q in queries]
        assert after == before, "graph or vectors differ after reload"
        assert db.is_int8_ram("visual")
        db.close()

    def test_wal_tail_on_top_of_a_v10_base(self, tmp_path):
        p = str(tmp_path / "wal.feather")
        rng = np.random.default_rng(9)
        db = DB.open(p, dim=24)
        random_workload(db, rng, 600, self.DIMS)
        db.save()
        random_workload(db, rng, 400, self.DIMS, saves=False)   # WAL only
        state = snapshot(db, ("text", "visual"), salience=False)
        db.close(save=False)                                     # crash

        db = DB.open(p, dim=24)
        assert snapshot(db, ("text", "visual"), salience=False) == state
        db.save()                     # the replayed tail must reach the file...
        db.close(save=False)
        assert not os.path.exists(p + ".wal") or os.path.getsize(p + ".wal") <= 8
        db = DB.open(p, dim=24)       # ...because the WAL is gone now
        assert snapshot(db, ("text", "visual"), salience=False) == state
        db.close()

    def test_quantized_modality_round_trips(self, tmp_path):
        p = str(tmp_path / "q.feather")
        rng = np.random.default_rng(4)
        db = DB.open(p, dim=32)
        db.set_quantized("text", True)
        V = rng.standard_normal((800, 32)).astype(np.float32)
        db.add_batch(list(range(800)), V, [meta(str(i)) for i in range(800)])
        db.save()
        db.add(900, V[0] * 2, meta("late"))      # appended commit, int8 encoded
        db.save()
        db.close()
        db = DB.open(p, dim=32)
        assert db.is_quantized("text") and db.size() == 801
        for i in (0, 17, 799):
            assert np.allclose(db.get_vector(i), V[i], atol=np.abs(V[i]).max() / 100)
        assert np.allclose(db.get_vector(900), V[0] * 2, atol=np.abs(V[0]).max() / 50)
        db.close()


# ── crash states ─────────────────────────────────────────────────────────

class TestCrashStates:
    """A commit is: append sections -> fsync -> rotate WAL to <wal>.old (under
    the lock, before the fsync) -> write header -> fsync -> delete <wal>.old.
    Build each state a crash can leave and check it recovers in full."""

    def _two_commits(self, tmp_path):
        p = str(tmp_path / "c.feather")
        db, rng = build(p, n=500, dim=16)
        for i in range(500, 540):
            db.add(i, rng.random(16, dtype=np.float32), meta(f"late {i}"))
        db.forget(3)
        db.link(10, 11, "rel")
        state = snapshot(db, salience=False)
        f0, w0 = open(p, "rb").read(), open(p + ".wal", "rb").read()
        db.save()
        f1 = open(p, "rb").read()
        db.close(save=False)
        assert len(f1) > len(f0) and f1[:len(f0)][SLOT * 2:] == f0[SLOT * 2:], \
            "a commit rewrote committed bytes"
        return p, f0, w0, f1, state

    def _reopen(self, p):
        db = DB.open(p, dim=16)
        try:
            return snapshot(db, salience=False), os.path.getsize(p)
        finally:
            db.close(save=False)

    def test_crash_while_appending_sections(self, tmp_path):
        p, f0, w0, f1, state = self._two_commits(tmp_path)
        half = f0 + f1[len(f0):len(f0) + (len(f1) - len(f0)) // 2]
        open(p, "wb").write(half)
        open(p + ".wal", "wb").write(w0)        # not yet rotated
        os.remove(p + ".wal.old") if os.path.exists(p + ".wal.old") else None
        got, size = self._reopen(p)
        assert got == state
        assert size == len(f0), "torn tail was not cut back to the committed end"

    def test_crash_before_the_header_switch(self, tmp_path):
        p, f0, w0, f1, state = self._two_commits(tmp_path)
        torn = bytearray(f1)
        torn[0:SLOT * 2] = f0[0:SLOT * 2]       # sections durable, header not
        open(p, "wb").write(bytes(torn))
        open(p + ".wal.old", "wb").write(w0)    # rotated
        got, size = self._reopen(p)
        assert got == state
        assert size == len(f0)

    def test_crash_during_the_header_write(self, tmp_path):
        p, f0, w0, f1, state = self._two_commits(tmp_path)
        torn = bytearray(f1)
        new_slot = header(f1, 0)["generation"] == 2 and 0 or 1
        torn[new_slot * SLOT + 20:new_slot * SLOT + 40] = b"\xAB" * 20   # half-written
        open(p, "wb").write(bytes(torn))
        open(p + ".wal.old", "wb").write(w0)
        got, _ = self._reopen(p)
        assert got == state

    def test_crash_before_dropping_the_old_wal(self, tmp_path):
        """Replay of <wal>.old over a base that already holds it: idempotent."""
        p, f0, w0, f1, state = self._two_commits(tmp_path)
        open(p + ".wal.old", "wb").write(w0)
        got, size = self._reopen(p)
        assert got == state
        assert size == len(f1)

    def test_stale_tmp_from_a_crashed_vacuum_is_removed(self, tmp_path):
        p, f0, w0, f1, state = self._two_commits(tmp_path)
        open(p + ".tmp", "wb").write(b"half a vacuum")
        got, _ = self._reopen(p)
        assert got == state
        assert not os.path.exists(p + ".tmp")


# ── corruption ───────────────────────────────────────────────────────────

class TestCorruption:
    def _file(self, tmp_path, commits=2):
        """Returns the path, each commit's end, and the WAL as it stood just
        before the last commit — what <wal>.old holds if that commit is torn."""
        p = str(tmp_path / "x.feather")
        db, rng = build(p, n=300, dim=16)
        ends, wal = [os.path.getsize(p)], None
        for c in range(commits - 1):
            db.add(1000 + c, rng.random(16, dtype=np.float32), meta(f"c{c}"))
            wal = open(p + ".wal", "rb").read()
            db.save()
            ends.append(os.path.getsize(p))
        db.close(save=False)
        return p, ends, wal

    def _tear_last_commit(self, p, ends):
        raw = bytearray(open(p, "rb").read())
        raw[ends[-2] + 40] ^= 0xFF                # inside the newest commit
        open(p, "wb").write(bytes(raw))
        return bytes(raw)

    def test_a_torn_commit_falls_back_and_the_wal_redoes_it(self, tmp_path):
        """A commit torn by a crash has <wal>.old beside it (it is deleted
        only after the header is durable), so falling back loses nothing."""
        p, ends, wal = self._file(tmp_path)
        self._tear_last_commit(p, ends)
        open(p + ".wal.old", "wb").write(wal)
        db = DB.open(p, dim=16)
        assert db.size() == 301 and db.get_metadata(1000).content == "c0"
        db.close(save=False)

    def test_damage_no_wal_can_redo_is_refused_not_dropped(self, tmp_path, monkeypatch):
        """No <wal>.old: the bytes went bad after the commit succeeded. Falling
        back would silently discard committed records — and a writer would
        then cut them off the file."""
        p, ends, _ = self._file(tmp_path)
        torn = self._tear_last_commit(p, ends)
        for ro in (False, True):
            with pytest.raises(RuntimeError, match="FEATHER_RECOVER_OLDER"):
                DB.open(p, dim=16, read_only=ro)
        assert open(p, "rb").read() == torn, "a refused open modified the file"
        monkeypatch.setenv("FEATHER_RECOVER_OLDER", "1")      # deliberate recovery
        db = DB.open(p, dim=16)
        assert db.size() == 300
        db.close(save=False)

    def test_a_writer_retires_the_header_it_fell_back_from(self, tmp_path):
        """Left in place, the abandoned header would become valid again once
        later appends reached its end, and vouch for bytes it never saw."""
        p, ends, wal = self._file(tmp_path)
        self._tear_last_commit(p, ends)
        open(p + ".wal.old", "wb").write(wal)
        db = DB.open(p, dim=16)
        db.close(save=False)
        fixed = open(p, "rb").read()
        assert len(fixed) == ends[0], "torn commit not cut"
        h0, h1 = header(fixed, 0), header(fixed, 1)
        assert h0["generation"] == h1["generation"] == 1 and h0["end"] == h1["end"] == ends[0]

    def test_a_reader_falls_back_without_touching_the_file(self, tmp_path):
        p, ends, wal = self._file(tmp_path)
        torn = self._tear_last_commit(p, ends)
        open(p + ".wal.old", "wb").write(wal)
        db = DB.open(p, dim=16, read_only=True)
        assert db.size() == 301
        db.close()
        assert open(p, "rb").read() == torn

    def test_bad_checksum_with_no_older_commit_is_refused(self, tmp_path):
        p, ends, _ = self._file(tmp_path, commits=1)
        raw = bytearray(open(p, "rb").read())
        raw[DATA_START + 100] ^= 0xFF
        open(p, "wb").write(bytes(raw))
        with pytest.raises(RuntimeError, match="corrupt"):
            DB.open(p, dim=16)
        assert open(p, "rb").read() == bytes(raw), "a failed open modified the file"

    def test_both_headers_broken_is_refused_not_treated_as_empty(self, tmp_path):
        p, _, _ = self._file(tmp_path)
        raw = bytearray(open(p, "rb").read())
        raw[16] ^= 1
        raw[SLOT + 16] ^= 1
        open(p, "wb").write(bytes(raw))
        with pytest.raises(RuntimeError, match="neither header"):
            DB.open(p, dim=16)

    def test_newer_version_is_refused(self, tmp_path):
        xxhash = pytest.importorskip("xxhash")
        p, _, _ = self._file(tmp_path, commits=1)
        raw = bytearray(open(p, "rb").read())
        for slot in (0, 1):
            o = slot * SLOT
            struct.pack_into("<I", raw, o + 4, 11)
            struct.pack_into("<Q", raw, o + 56, xxhash.xxh64_intdigest(bytes(raw[o:o + 56])))
        open(p, "wb").write(bytes(raw))
        with pytest.raises(RuntimeError, match="newer"):
            DB.open(p, dim=16)

    @pytest.mark.parametrize("optional", [False, True])
    def test_unknown_section_type(self, tmp_path, optional):
        """A mandatory unknown section means a newer writer: refuse, and do NOT
        fall back to the older header (that would discard data that is there).
        An optional one is skipped."""
        xxhash = pytest.importorskip("xxhash")
        p, _, _ = self._file(tmp_path, commits=1)
        raw = bytearray(open(p, "rb").read())
        payload = b"future"
        hdr = struct.pack("<IIQ", 99, 1 if optional else 0, len(payload))
        seed = xxhash.xxh64_intdigest(hdr)
        sec = hdr + payload + struct.pack("<Q", xxhash.xxh64_intdigest(payload, seed=seed))
        end = len(raw) + len(sec)
        raw += sec
        o = 0                                     # generation 2 -> slot 0
        struct.pack_into("<IIIIQQQQQ", raw, o, MAGIC, 10, 0xFFFFFFFF, 0, 2, end, DATA_START, 0, 0)
        struct.pack_into("<Q", raw, o + 56, xxhash.xxh64_intdigest(bytes(raw[o:o + 56])))
        open(p, "wb").write(bytes(raw))
        if optional:
            db = DB.open(p, dim=16)
            assert db.size() == 300
            db.close(save=False)
        else:
            with pytest.raises(RuntimeError, match="newer"):
                DB.open(p, dim=16)
            assert len(open(p, "rb").read()) == end, "fell back and truncated a newer commit"

    def test_truncated_file_is_refused(self, tmp_path):
        p, ends, _ = self._file(tmp_path, commits=1)
        raw = open(p, "rb").read()
        open(p, "wb").write(raw[: len(raw) * 2 // 3])
        with pytest.raises(RuntimeError):
            DB.open(p, dim=16)


# ── readers ──────────────────────────────────────────────────────────────

def test_reader_snapshot_is_stable_while_the_writer_appends(tmp_path):
    p = str(tmp_path / "r.feather")
    w, rng = build(p, n=200, dim=8)
    r = DB.open(p, dim=8, read_only=True)
    for i in range(5):
        w.add(1000 + i, rng.random(8, dtype=np.float32), meta("w"))
        w.save()
        assert r.size() == 200
        fresh = DB.open(p, dim=8, read_only=True)
        assert fresh.size() == 201 + i
        fresh.close()
    r.close()
    w.close()


# ── migration ────────────────────────────────────────────────────────────

def test_v9_file_is_upgraded_by_its_first_save(tmp_path, monkeypatch):
    p = str(tmp_path / "old.feather")
    monkeypatch.setenv("FEATHER_FORMAT", "9")
    db, rng = build(p, n=400, dim=16)
    db.forget(7)
    db.link(1, 2, "r")
    state = snapshot(db, salience=False)
    db.close()
    assert struct.unpack_from("<I", open(p, "rb").read(), 4)[0] == 9
    monkeypatch.delenv("FEATHER_FORMAT")

    db = DB.open(p, dim=16)
    assert db.storage_stats()["format"] == 9
    # v9 stores no versions: a reopen gives fresh ones, above the old.
    assert _no_versions(snapshot(db, salience=False)) == _no_versions(state)
    db.save()
    upgraded = snapshot(db, salience=False)
    assert db.storage_stats()["format"] == 10 and db.storage_stats()["last_commit_full"]
    db.close()
    db = DB.open(p, dim=16)
    assert snapshot(db, salience=False) == upgraded        # v10 keeps them, from here on
    assert _no_versions(upgraded) == _no_versions(state)
    db.close()


def test_format_9_can_still_be_written(tmp_path, monkeypatch):
    """FEATHER_FORMAT=9 is the rollback path for a fleet with pre-v10 readers."""
    p = str(tmp_path / "back.feather")
    db, rng = build(p, n=100, dim=8)
    monkeypatch.setenv("FEATHER_FORMAT", "9")
    db.add(500, rng.random(8, dtype=np.float32), meta("x"))
    db.save()
    db.close()
    assert struct.unpack_from("<I", open(p, "rb").read(), 4)[0] == 9
    db = DB.open(p, dim=8)
    assert db.size() == 101
    db.close()


# ── vacuum ───────────────────────────────────────────────────────────────

class TestVacuum:
    def test_superseded_records_trigger_a_vacuum(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FEATHER_VACUUM_MIN_BYTES", "0")
        monkeypatch.setenv("FEATHER_VACUUM_RATIO", "1.5")
        p = str(tmp_path / "v.feather")
        db, rng = build(p, n=300, dim=8)
        fulls = 0
        for round_ in range(30):
            for i in range(0, 300, 3):
                db.update_metadata(i, meta(f"rev {round_} " + "y" * 200))
            db.save()
            fulls += db.storage_stats()["last_commit_full"]
        st = db.storage_stats()
        assert fulls >= 1, "garbage never triggered a vacuum"
        # Checked at the start of each save, so the file may end one commit over.
        assert st["file_bytes"] - st["last_commit_bytes"] <= 1.5 * st["live_bytes_estimate"] + 1
        state = snapshot(db)
        db.close()
        db = DB.open(p, dim=8)
        assert snapshot(db) == state
        db.close()

    def test_explicit_vacuum_drops_superseded_bytes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FEATHER_VACUUM_RATIO", "0")        # no automatic one
        p = str(tmp_path / "e.feather")
        db, rng = build(p, n=300, dim=8)
        base = os.path.getsize(p)
        db.update_metadata(5, meta("SECRET-OLD-CONTENT"))
        db.save()
        db.update_metadata(5, meta("replacement"))
        db.forget(5)
        db.save()
        assert b"SECRET-OLD-CONTENT" in open(p, "rb").read()   # append-only keeps it
        db.vacuum()
        assert b"SECRET-OLD-CONTENT" not in open(p, "rb").read()
        assert os.path.getsize(p) <= base
        db.close()

    @pytest.mark.parametrize("action", ["purge", "compact", "set_quantized"])
    def test_space_reclaiming_operations_rewrite_on_next_save(self, tmp_path, action):
        p = str(tmp_path / f"{action}.feather")
        rng = np.random.default_rng(1)
        db = DB.open(p, dim=8)
        db.add_batch(list(range(200)), rng.random((200, 8), dtype=np.float32),
                     [meta(f"PURGED-{i}" if i < 50 else f"kept-{i}", ns="gone" if i < 50 else "kept")
                      for i in range(200)])
        db.save()
        if action == "purge":
            db.purge("gone")
        elif action == "compact":
            for i in range(60):
                db.forget(i)
            db.compact()
        else:
            db.set_quantized("text", True)
        db.save()
        assert db.storage_stats()["last_commit_full"]
        state = snapshot(db, salience=False)
        db.close()
        if action == "purge":
            assert b"PURGED-" not in open(p, "rb").read(), "purged content survived the save"
        db = DB.open(p, dim=8)
        assert set(snapshot(db, salience=False)[0]) == set(state[0])
        db.close()
