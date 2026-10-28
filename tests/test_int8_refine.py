"""In-RAM int8: a scale trained from the data, and a re-rank by the float32
originals, which are kept in the file and read back from it.

What each test pins:

  * the trained clip is the one minimising quantization error, and beats
    the old fixed default                                      -> TestTraining
  * a modality waits for INT8_TRAIN_MIN vectors, then converts in place,
    keeping its graph; the waiting state survives a reopen     -> TestTraining
  * refine ranks like float, with exact scores, on both the graph and the
    pre-filtered path                                          -> TestRefine
  * originals are never copied into RAM on the write path: they point into
    the WAL until a save puts them in the file                 -> TestRefine
  * every way a slot's vector changes, and every way the file is written,
    keeps each live record's original exact: delta commits, forget + add,
    compaction, crash + WAL replay, vacuum                      -> TestPersistence
  * a reader keeps its snapshot's originals while the writer rewrites the file
  * a GRAPH record with no FULLVEC after it (a build without refine) voids
    the stale original instead of letting it be used
"""
import os
import struct

import numpy as np
import pytest

import feather_db
from feather_db import DB, FilterBuilder
from tests.test_corrupt_files import v10_sections, xxh64

SEC_FULLVEC = 6


def unit(rng, n, dim, outliers=0):
    """Embedding-shaped rows: clustered, uneven per-dimension spread, unit norm."""
    spread = rng.lognormal(0.0, 0.5, dim).astype(np.float32)
    if outliers:
        spread[rng.choice(dim, outliers, replace=False)] *= 6.0
    centers = rng.standard_normal((max(8, n // 50), dim)).astype(np.float32) * spread
    x = centers[rng.integers(0, len(centers), n)] + \
        rng.standard_normal((n, dim)).astype(np.float32) * spread * 0.6
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def meta(ns="a"):
    m = feather_db.Metadata()
    m.namespace_id = ns
    m.timestamp = 1_700_000_000
    return m


def truth(X, Q, k):
    d = (Q ** 2).sum(1)[:, None] - 2 * Q @ X.T + (X ** 2).sum(1)[None, :]
    return np.argsort(d, axis=1)[:, :k]


def recall(db, X, Q, k=10, **kw):
    t = truth(X, Q, k)
    hits = sum(len({r.id for r in db.search(q, k=k, record_salience=False, **kw)} & set(t[i].tolist()))
               for i, q in enumerate(Q))
    return hits / (len(Q) * k)


def quant_mse(a, clip):
    s = clip / 127.0
    return float(np.mean((a - np.minimum(np.rint(a / s), 127) * s) ** 2))


def sections(path):
    return [typ for _, typ, _ in v10_sections(open(path, "rb").read())]


def assert_originals(db, expected):
    for i, v in expected.items():
        got = db.get_vector(i)
        assert np.array_equal(got, v), f"id {i}: original not returned exactly"


# ── training ─────────────────────────────────────────────────────────────

class TestTraining:
    def test_trained_clip_minimises_quantization_error(self, tmp_path):
        rng = np.random.default_rng(1)
        X = unit(rng, 3000, 64, outliers=2)
        db = DB.open(str(tmp_path / "t.feather"), dim=64)
        db.set_int8_ram("text", refine=0)
        db.add_batch(list(range(len(X))), X)
        info = db.int8_info("text")
        assert info["int8"] and not info["training"]
        clip = info["clip"]
        a = np.abs(X).ravel()
        best = min(quant_mse(a, c) for c in np.linspace(a.max() / 64, a.max(), 400))
        assert quant_mse(a, clip) <= best * 1.03, "trained clip is not near the MSE optimum"
        assert clip < a.max(), "outlier dimensions should be clamped, not fitted"
        assert quant_mse(a, clip) < quant_mse(a, 1.0)
        db.close()

    def test_trained_scale_beats_the_fixed_default(self, tmp_path):
        rng = np.random.default_rng(2)
        X, Q = unit(rng, 6000, 256), unit(rng, 80, 256)
        r = {}
        for name, clip in (("fixed", 1.0), ("trained", None)):
            db = DB.open(str(tmp_path / f"{name}.feather"), dim=256)
            db.set_int8_ram("text", clip, refine=0)
            db.add_batch(list(range(len(X))), X)
            db.set_ef(200)                      # measure the quantization, not the beam
            r[name] = recall(db, X, Q)
            db.close()
        assert r["trained"] > r["fixed"] + 0.01, r

    def test_waits_for_enough_vectors_then_converts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FEATHER_INT8_TRAIN_MIN", "50")
        rng = np.random.default_rng(3)
        X = unit(rng, 80, 32)
        db = DB.open(str(tmp_path / "w.feather"), dim=32)
        db.set_int8_ram("text")
        for i in range(49):
            db.add(i, X[i])
        info = db.int8_info("text")
        assert info["training"] and not info["int8"] and not db.is_int8_ram("text")
        db.add(49, X[49])                         # the 50th trains and converts
        info = db.int8_info("text")
        assert info["int8"] and not info["training"] and db.is_int8_ram("text")
        for i in range(50, 80):
            db.add(i, X[i])
        assert_originals(db, {i: X[i] for i in range(80)})
        db.close()

    def test_waiting_state_survives_a_reopen(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FEATHER_INT8_TRAIN_MIN", "100")
        p = str(tmp_path / "s.feather")
        rng = np.random.default_rng(4)
        X = unit(rng, 160, 32)
        db = DB.open(p, dim=32)
        db.set_int8_ram("text", refine=3)
        db.add_batch(list(range(40)), X[:40])
        db.close()
        db = DB.open(p, dim=32)
        info = db.int8_info("text")
        assert info["training"] and not info["int8"] and info["refine"] == 3
        db.add_batch(list(range(40, 160)), X[40:])
        assert db.int8_info("text")["int8"]
        db.close()
        db = DB.open(p, dim=32)
        info = db.int8_info("text")
        assert info["int8"] and info["refine"] == 3 and info["originals_on_disk"] == 160
        assert_originals(db, {i: X[i] for i in range(160)})
        db.close()

    def test_converting_a_float_modality_keeps_its_graph(self, tmp_path):
        rng = np.random.default_rng(5)
        X, Q = unit(rng, 2500, 48), unit(rng, 60, 48)
        p = str(tmp_path / "c.feather")
        db = DB.open(p, dim=48)
        db.add_batch(list(range(len(X))), X)
        db.save()
        before = recall(db, X, Q)
        db.set_int8_ram("text")                   # 2500 >= 1000: converts now
        info = db.int8_info("text")
        assert info["int8"] and info["originals_in_ram"] == len(X)   # copies until saved
        after = recall(db, X, Q)
        assert after >= before - 0.005, f"conversion cost recall: {before:.4f} -> {after:.4f}"
        db.save()
        info = db.int8_info("text")
        assert info["originals_in_ram"] == 0 and info["originals_on_disk"] == len(X)
        db.close()
        db = DB.open(p, dim=48)
        assert db.is_int8_ram("text")
        assert_originals(db, {i: X[i] for i in range(0, len(X), 7)})
        db.close()

    def test_an_int8_scale_is_fixed_once_set(self, tmp_path):
        db = DB.open(str(tmp_path / "f.feather"), dim=8)
        db.set_int8_ram("text", 1.0)
        db.add(1, np.full(8, 0.1, np.float32))
        with pytest.raises(RuntimeError, match="scale cannot change"):
            db.set_int8_ram("text", 2.0)
        db.set_int8_ram("text", refine=4)           # only refine changes
        info = db.int8_info("text")
        assert info["refine"] == 4 and abs(info["clip"] - 1.0) < 1e-6
        db.close()


# ── refine ───────────────────────────────────────────────────────────────

class TestRefine:
    def test_ranks_like_float(self, tmp_path):
        rng = np.random.default_rng(6)
        X, Q = unit(rng, 5000, 96), unit(rng, 80, 96)
        r = {}
        for name, setup in (("float", None), ("int8", 0), ("refine", 2)):
            db = DB.open(str(tmp_path / f"{name}.feather"), dim=96)
            if setup is not None:
                db.set_int8_ram("text", 1.0, refine=setup)     # the old default scale
            db.add_batch(list(range(len(X))), X)
            r[name] = recall(db, X, Q)
            db.close()
        assert r["refine"] >= r["float"] - 0.005, r
        assert r["refine"] > r["int8"], r

    def test_scores_are_exact_float_distances(self, tmp_path):
        rng = np.random.default_rng(7)
        X, Q = unit(rng, 1500, 40), unit(rng, 10, 40)
        p = str(tmp_path / "e.feather")
        db = DB.open(p, dim=40)
        db.set_int8_ram("text", 1.0)
        db.add_batch(list(range(len(X))), X)
        db.save()                                  # originals now read from the file
        for q in Q:
            for res in db.search(q, k=5, record_salience=False):
                d = float(np.sum((X[res.id].astype(np.float64) - q) ** 2))
                assert res.score == pytest.approx(1.0 / (1.0 + d), rel=1e-5)
        db.close()

    def test_context_chain_seeds_are_refined(self, tmp_path):
        rng = np.random.default_rng(19)
        X, Q = unit(rng, 1500, 40), unit(rng, 5, 40)
        db = DB.open(str(tmp_path / "cc.feather"), dim=40)
        db.set_int8_ram("text", 1.0)
        db.add_batch(list(range(len(X))), X)
        db.save()
        for q in Q:
            seeds = [n for n in db.context_chain(q, k=5, hops=0).nodes if n.hop == 0]
            assert len(seeds) == 5
            for n in seeds:
                d = float(np.sum((X[n.id].astype(np.float64) - q) ** 2))
                assert n.similarity == pytest.approx(1.0 / (1.0 + d), rel=1e-5)
        db.close()

    def test_filtered_exact_scan_is_refined(self, tmp_path):
        rng = np.random.default_rng(8)
        X, Q = unit(rng, 4000, 64), unit(rng, 40, 64)
        db = DB.open(str(tmp_path / "x.feather"), dim=64)
        db.set_int8_ram("text", 1.0)
        db.add_batch(list(range(len(X))), X, [meta("a" if i % 2 else "b") for i in range(len(X))])
        db.save()
        b = np.arange(0, len(X), 2)
        f = FilterBuilder().namespace("b").build()
        overlap = []
        for q in Q:
            d = np.sum((X[b] - q) ** 2, axis=1)
            expect = [int(b[j]) for j in np.argsort(d)[:10]]
            got = [r.id for r in db.search(q, k=10, filter=f, record_salience=False)]
            overlap.append(len(set(got) & set(expect)) / 10)
            assert got[0] == expect[0]
        assert np.mean(overlap) >= 0.99
        db.close()

    def test_written_originals_point_into_the_wal_not_ram(self, tmp_path):
        rng = np.random.default_rng(9)
        X = unit(rng, 600, 32)
        p = str(tmp_path / "w.feather")
        db = DB.open(p, dim=32)
        db.set_int8_ram("text", 1.0)
        db.add_batch(list(range(500)), X[:500])
        for i in range(500, 600):
            db.add(i, X[i])
        info = db.int8_info("text")
        assert info["originals_in_wal"] == 600 and info["originals_in_ram"] == 0
        assert_originals(db, {i: X[i] for i in range(600)})
        db.save()                                   # rotates the WAL away
        info = db.int8_info("text")
        assert info["originals_in_wal"] == 0 and info["originals_on_disk"] == 600
        assert_originals(db, {i: X[i] for i in range(600)})
        assert SEC_FULLVEC in sections(p)
        db.close()

    def test_refine_zero_keeps_no_originals(self, tmp_path):
        rng = np.random.default_rng(10)
        X = unit(rng, 300, 32)
        p = str(tmp_path / "z.feather")
        db = DB.open(p, dim=32)
        db.set_int8_ram("text", 1.0, refine=0)
        db.add_batch(list(range(300)), X)
        db.close()
        assert SEC_FULLVEC not in sections(p)
        db = DB.open(p, dim=32)
        assert not np.array_equal(db.get_vector(5), X[5])          # dequantized
        assert np.allclose(db.get_vector(5), X[5], atol=1.0 / 127)
        db.close()

    def test_turning_refine_off_drops_them_from_the_file(self, tmp_path):
        rng = np.random.default_rng(11)
        X = unit(rng, 400, 64)
        p = str(tmp_path / "o.feather")
        db = DB.open(p, dim=64)
        db.set_int8_ram("text", 1.0)
        db.add_batch(list(range(400)), X)
        db.close()
        with_originals = os.path.getsize(p)
        db = DB.open(p, dim=64)
        db.set_int8_ram("text", refine=0)
        db.close()
        assert SEC_FULLVEC not in sections(p)
        assert os.path.getsize(p) < with_originals - 400 * 64 * 4 * 0.9


# ── persistence ──────────────────────────────────────────────────────────

class TestPersistence:
    DIM = 32

    def open(self, p, **kw):
        db = DB.open(p, dim=self.DIM, **kw)
        return db

    def fresh(self, p, n, rng):
        db = self.open(p)
        db.set_int8_ram("text", 1.0)
        X = unit(rng, n, self.DIM)
        db.add_batch(list(range(n)), X)
        return db, {i: X[i] for i in range(n)}

    def test_delta_commits_update_and_forget(self, tmp_path):
        rng = np.random.default_rng(12)
        p = str(tmp_path / "d.feather")
        db, exp = self.fresh(p, 500, rng)
        db.save()
        new = unit(rng, 230, self.DIM)
        for k, i in enumerate(range(500, 600)):           # new records
            db.add(i, new[k]); exp[i] = new[k]
        for k, i in enumerate(range(0, 50)):              # vectors replaced in place
            db.add(i, new[100 + k]); exp[i] = new[100 + k]
        for i in range(100, 130):                         # forgotten...
            db.forget(i); exp.pop(i)
        for k, i in enumerate(range(700, 730)):           # ...and new ones in fresh slots
            db.add(i, new[150 + k]); exp[i] = new[150 + k]
        db.save()
        assert not db.storage_stats()["last_commit_full"], "expected an appended commit"
        assert_originals(db, exp)
        db.close()
        db = self.open(p)
        assert_originals(db, exp)
        # The forgotten slots keep their originals in the file until compact().
        assert db.int8_info("text")["originals_on_disk"] == len(exp) + 30
        db.close()

    def test_crash_replays_originals_from_the_wal(self, tmp_path):
        rng = np.random.default_rng(13)
        p = str(tmp_path / "c.feather")
        db, exp = self.fresh(p, 300, rng)
        db.save()
        new = unit(rng, 120, self.DIM)
        for k, i in enumerate(range(300, 400)):
            db.add(i, new[k]); exp[i] = new[k]
        for k, i in enumerate(range(0, 20)):
            db.add(i, new[100 + k]); exp[i] = new[100 + k]
        db.close(save=False)                               # crash
        db = self.open(p)
        info = db.int8_info("text")
        assert info["originals_in_wal"] == 120 and info["originals_in_ram"] == 0
        assert_originals(db, exp)
        db.save()
        db.close()
        db = self.open(p)
        assert_originals(db, exp)
        db.close()

    def test_compaction_keeps_originals(self, tmp_path):
        rng = np.random.default_rng(14)
        p = str(tmp_path / "k.feather")
        db, exp = self.fresh(p, 600, rng)
        db.save()                                          # most originals on disk...
        for i in range(0, 600, 2):
            db.forget(i); exp.pop(i)
        new = unit(rng, 40, self.DIM)
        for k, i in enumerate(range(1000, 1040)):          # ...some still in the WAL
            db.add(i, new[k]); exp[i] = new[k]
        assert db.compact() == 300
        assert_originals(db, exp)
        db.save()
        assert db.storage_stats()["last_commit_full"]
        db.close()
        db = self.open(p)
        assert_originals(db, exp)
        db.close()

    def test_vacuum_rewrites_them(self, tmp_path):
        rng = np.random.default_rng(15)
        p = str(tmp_path / "v.feather")
        db, exp = self.fresh(p, 300, rng)
        db.save()
        new = unit(rng, 50, self.DIM)
        for k in range(50):
            db.add(k, new[k]); exp[k] = new[k]
        db.vacuum()
        assert_originals(db, exp)
        db.close()
        db = self.open(p)
        assert_originals(db, exp)
        db.close()

    def test_writes_racing_saves_keep_the_newest_original(self, tmp_path):
        """A save publishes the offsets it wrote after releasing the data lock
        for its fsync; an update landing in between is newer and must win."""
        import threading
        rng = np.random.default_rng(18)
        p = str(tmp_path / "race.feather")
        db, exp = self.fresh(p, 3000, rng)
        db.save()
        vecs = unit(rng, 3000, self.DIM)
        done = threading.Event()

        def writer():
            # Each slot changes once: a lost update stays lost, nothing later
            # overwrites it.
            for k, v in enumerate(vecs):
                db.add(k, v)
                exp[k] = v
            done.set()

        t = threading.Thread(target=writer)
        t.start()
        saves = 0
        while not done.is_set():
            db.save()
            saves += 1
        t.join()
        assert saves > 3, "the race did not happen"
        assert_originals(db, exp)
        db.close()
        db = self.open(p)
        assert_originals(db, exp)
        db.close()

    def test_reader_keeps_its_snapshot_while_the_writer_rewrites(self, tmp_path):
        rng = np.random.default_rng(16)
        p = str(tmp_path / "r.feather")
        db, exp = self.fresh(p, 400, rng)
        db.save()
        reader = self.open(p, read_only=True)
        new = unit(rng, 50, self.DIM)
        for k in range(50):
            db.add(k, new[k])
        db.vacuum()                                        # a new inode at the path
        assert_originals(reader, {i: exp[i] for i in range(0, 400, 9)})
        assert reader.search(exp[3], k=1, record_salience=False)[0].id == 3
        reader.close()
        reader = self.open(p, read_only=True)
        assert np.array_equal(reader.get_vector(7), new[7])
        reader.close()
        db.close()

    def test_a_graph_record_without_fullvec_voids_the_stale_original(self, tmp_path):
        """A build that predates FULLVEC appends GRAPH data for a changed slot
        and nothing else. The FULLVEC an earlier commit wrote for that slot is
        then stale: it must not be used."""
        rng = np.random.default_rng(17)
        p = str(tmp_path / "g.feather")
        db, exp = self.fresh(p, 100, rng)
        db.save()
        v = np.full(self.DIM, 0.05, np.float32)
        db.add(7, v)
        db.save()                                          # GRAPH + FULLVEC for slot of 7
        db.close()
        # Turn that last FULLVEC into a section of an unknown, optional type —
        # what an older build's commit looks like to this one.
        raw = bytearray(open(p, "rb").read())
        pos, typ, ln = [s for s in v10_sections(bytes(raw)) if s[1] == SEC_FULLVEC][-1]
        struct.pack_into("<II", raw, pos, 99, 1)
        struct.pack_into("<Q", raw, pos + 16 + ln,
                         xxh64(bytes(raw[pos + 16:pos + 16 + ln]), xxh64(bytes(raw[pos:pos + 16]))))
        open(p, "wb").write(bytes(raw))
        db = self.open(p)
        got = db.get_vector(7)
        assert not np.array_equal(got, exp[7]), "stale original used"
        assert np.allclose(got, v, atol=1.0 / 127)         # the int8 value, dequantized
        assert_originals(db, {i: exp[i] for i in range(100) if i != 7})
        db.close()
