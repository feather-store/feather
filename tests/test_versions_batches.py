"""Compare-and-set versions, atomic write batches, idempotency keys.

The guarantees, each pinned down here:
  * Two writers updating one record cannot silently overwrite each other:
    put_if() / expected_version refuse a write based on a stale read.
  * A version is never reused: not after a crash, a save, a vacuum, a
    compaction, nor by a record re-created after a delete.
  * A batch is all or nothing: a failed check applies nothing, a reader never
    sees part of one, and a crash part-way through its WAL write drops it whole.
  * A retried batch with the same idempotency key is not applied twice, even
    when the retry comes after a crash.
"""
import os
import shutil
import subprocess
import sys
import threading

import numpy as np
import pytest

import feather_db
from feather_db import DB, Metadata, WriteBatch, VersionConflictError, IdempotencyKeyReusedError

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 8


def meta(content="", ns="", **attrs):
    m = Metadata()
    m.content = content
    m.namespace_id = ns
    for k, v in attrs.items():
        m.set_attribute(k, str(v))
    return m


def vec(seed):
    return np.random.default_rng(seed).random(DIM, dtype=np.float32)


@pytest.fixture
def db(tmp_path):
    d = DB.open(str(tmp_path / "v.feather"), dim=DIM)
    yield d
    d.close()


def run_child(src):
    """Run `src` in a child that SIGKILLs itself: nothing unwinds, no save."""
    code = (f"import sys, os, signal, json\nsys.path.insert(0, {REPO!r})\n"
            "import numpy as np, feather_db\n" + src +
            "\nos.kill(os.getpid(), signal.SIGKILL)\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == -9, p.stderr
    return p.stdout


# ── versions ──────────────────────────────────────────────────────────────

class TestVersions:
    def test_every_write_moves_the_version(self, db):
        assert db.version(1) == 0
        db.add(1, vec(1), meta("a"))
        seen = [db.version(1)]
        db.update_metadata(1, meta("b"))
        seen.append(db.version(1))
        db.add(2, vec(2), meta("x"))
        db.link(1, 2)
        seen.append(db.version(1))
        db.update_importance(1, 0.5)
        seen.append(db.version(1))
        db.add(1, vec(3), meta("c"))
        seen.append(db.version(1))
        assert seen == sorted(set(seen)) and seen[0] > 0
        assert db.get_metadata(1).version == seen[-1]

    def test_search_does_not_move_the_version(self, db):
        db.add(1, vec(1), meta("a"))
        v = db.version(1)
        db.search(vec(1), k=1)
        assert db.get_metadata(1).recall_count == 1
        assert db.version(1) == v

    def test_forgotten_is_zero_and_recreation_never_reuses(self, db):
        db.add(1, vec(1), meta("a"))
        old = db.version(1)
        db.forget(1)
        assert db.version(1) == 0
        db.add(1, vec(1), meta("again"))
        assert db.version(1) > old

    def test_put_if(self, db):
        assert db.put_if(1, meta("created"), 0) > 0          # 0: must not exist
        with pytest.raises(VersionConflictError) as e:
            db.put_if(1, meta("again"), 0)
        assert (e.value.id, e.value.expected, e.value.actual) == (1, 0, db.version(1))
        v = db.version(1)
        v2 = db.put_if(1, meta("edited"), v)
        assert v2 > v and db.get_metadata(1).content == "edited"
        with pytest.raises(VersionConflictError):
            db.put_if(1, meta("stale"), v)
        assert db.get_metadata(1).content == "edited"

    def test_put_if_replaces_the_vector_atomically(self, db):
        db.add(1, vec(1), meta("a"))
        v = db.version(1)
        new = vec(9)
        with pytest.raises(VersionConflictError):
            db.put_if(1, meta("b"), v + 100, vec=new)
        assert np.allclose(db.get_vector(1), vec(1)), "a refused put_if changed the vector"
        db.put_if(1, meta("b"), v, vec=new)
        assert np.allclose(db.get_vector(1), new)

    def test_concurrent_read_modify_write_loses_nothing(self, db):
        """The lost update CAS exists for: N threads increment one counter."""
        db.add(1, vec(1), meta("counter", n=0))
        threads, per = 8, 50
        conflicts = []

        def worker():
            for _ in range(per):
                while True:
                    m = db.get_metadata(1)
                    m.set_attribute("n", str(int(m.get_attribute("n")) + 1))
                    try:
                        db.put_if(1, m, m.version)
                        break
                    except VersionConflictError:
                        conflicts.append(1)

        ts = [threading.Thread(target=worker) for _ in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert int(db.get_metadata(1).get_attribute("n")) == threads * per
        assert conflicts, "no thread ever raced: the test proves nothing"

    def test_versions_survive_crash_save_vacuum_and_compaction(self, tmp_path):
        path = str(tmp_path / "c.feather")
        out = run_child(f"""
db = feather_db.DB.open({path!r}, dim={DIM})
for i in range(20):
    m = feather_db.Metadata(); m.content = f"r{{i}}"
    db.add(i, np.random.rand({DIM}).astype(np.float32), m)
db.save()
for i in range(0, 20, 3):
    db.update_importance(i, 0.5)
db.link(1, 2)
db.forget(5)
print(json.dumps({{i: db.version(i) for i in range(20)}}), flush=True)
""")
        before = {int(k): v for k, v in __import__("json").loads(out.splitlines()[-1]).items()}
        d = DB.open(path, dim=DIM)
        assert {i: d.version(i) for i in range(20)} == before, "replay changed versions"
        d.close()
        d = DB.open(path, dim=DIM)
        assert {i: d.version(i) for i in range(20)} == before, "save + reopen changed versions"
        d.vacuum()
        d.compact()
        assert {i: d.version(i) for i in range(20)} == before
        d.close()
        d = DB.open(path, dim=DIM)
        assert {i: d.version(i) for i in range(20)} == before, "vacuum/compaction changed versions"
        d.close()

    def test_clock_outlives_the_record_that_held_the_highest_version(self, tmp_path):
        path = str(tmp_path / "k.feather")
        d = DB.open(path, dim=DIM)
        d.add(1, vec(1), meta("a"))
        d.add(2, vec(2), meta("b"))
        top = d.version(2)
        d.forget(2)
        d.close()                    # the save drops record 2 entirely
        d = DB.open(path, dim=DIM)
        d.add(3, vec(3), meta("c"))
        d.add(2, vec(2), meta("b again"))
        assert d.version(3) > top and d.version(2) > top
        d.close()

    def test_a_file_without_versions_restarts_above_the_old_ones(self, tmp_path, monkeypatch):
        """FEATHER_FORMAT=9 cannot store versions: a reopen must not hand out
        a version a client may still hold for an older state."""
        path = str(tmp_path / "l.feather")
        monkeypatch.setenv("FEATHER_FORMAT", "9")
        d = DB.open(path, dim=DIM)
        d.add(1, vec(1), meta("a"))
        d.update_metadata(1, meta("b"))
        held = d.version(1)
        d.close()
        monkeypatch.delenv("FEATHER_FORMAT")
        d = DB.open(path, dim=DIM)
        assert d.version(1) > held
        d.update_metadata(1, meta("c"))
        assert d.version(1) > held
        d.close()


# ── batches ───────────────────────────────────────────────────────────────

def fact_batch(fact_id, entity_id, content="fact"):
    b = WriteBatch()
    b.put(entity_id, vec(entity_id), meta(f"entity {entity_id}"))
    b.put(fact_id, vec(fact_id), meta(content))
    b.link(fact_id, entity_id, "about")
    return b


class TestBatches:
    def test_applies_every_op_and_reports_versions(self, db):
        r = db.write(fact_batch(10, 20))
        assert r.ids == [20, 10, 10] and not r.replayed
        assert r.versions[0] < r.versions[1] < r.versions[2]
        assert db.version(10) == r.versions[2] and db.version(20) == r.versions[0]
        assert [e.target_id for e in db.get_edges(10)] == [20]
        assert [e.source_id for e in db.get_incoming(20)] == [10]

    def test_a_failed_check_applies_nothing(self, db, tmp_path):
        db.add(1, vec(1), meta("a"))
        wal = str(tmp_path / "v.feather.wal")
        size, v1 = os.path.getsize(wal), db.version(1)

        dangling = WriteBatch().put(2, vec(2), meta("b")).update(1, meta("changed")).link(2, 999)
        stale = WriteBatch().put(3, vec(3), meta("c")).update(1, meta("changed"), expected_version=v1 + 7)
        bad_dim = WriteBatch().put(4, vec(4), meta("d")).put(5, np.ones(3, np.float32), meta("e"))
        for b, err in ((dangling, ValueError), (stale, VersionConflictError), (bad_dim, RuntimeError)):
            with pytest.raises(err):
                db.write(b)
        assert [db.version(i) for i in (2, 3, 4, 5)] == [0, 0, 0, 0]
        assert db.get_metadata(1).content == "a" and db.version(1) == v1
        assert db.size() == 1
        assert os.path.getsize(wal) == size, "a refused batch reached the WAL"

    def test_expected_versions_are_checked_against_the_state_before_the_batch(self, db):
        db.add(1, vec(1), meta("a"))
        v = db.version(1)
        b = WriteBatch().update(1, meta("b"), expected_version=v).update(1, meta("c"), expected_version=v)
        r = db.write(b)
        assert db.get_metadata(1).content == "c" and db.version(1) == r.versions[1]

    def test_link_may_target_a_record_created_earlier_in_the_batch(self, db):
        db.write(WriteBatch().put(1, vec(1), meta("a")).put(2, vec(2), meta("b")).link(1, 2).link(2, 1))
        assert [e.target_id for e in db.get_edges(1)] == [2]
        with pytest.raises(ValueError):    # ...but not one the batch forgot first
            db.write(WriteBatch().forget(2).link(1, 2, "again"))
        assert db.version(2) > 0

    def test_readers_never_see_part_of_a_batch(self, db):
        """Without the batch, a fact is added and then linked: a reader in
        between sees a fact with no entity link. With it, never."""
        stop = threading.Event()
        torn = []

        def reader():
            while not stop.is_set():
                for fid in range(1000, 1200):
                    m = db.get_metadata(fid)
                    if m is not None and not m.edges:
                        torn.append(fid)

        t = threading.Thread(target=reader)
        t.start()
        try:
            for i in range(200):
                db.write(fact_batch(1000 + i, 5000 + i))
        finally:
            stop.set()
            t.join()
        assert not torn, f"a reader saw facts without their links: {torn[:5]}"

    def test_a_torn_batch_is_dropped_whole(self, tmp_path):
        """Cut the WAL at every point inside the batch's group: the batch is
        all there or not there at all, and what came before it survives."""
        path = str(tmp_path / "t.feather")
        out = run_child(f"""
db = feather_db.DB.open({path!r}, dim={DIM})
m = feather_db.Metadata(); m.content = "before"
db.add(1, np.ones({DIM}, np.float32), m)
start = os.path.getsize({path!r} + ".wal")
b = feather_db.WriteBatch()
for i in range(2, 6):
    n = feather_db.Metadata(); n.content = f"r{{i}}"
    b.put(i, np.full({DIM}, i, np.float32), n)
b.link(2, 3).link(4, 5).update(1, m)
db.write(b, idempotency_key="k", fingerprint="f")
print(start, os.path.getsize({path!r} + ".wal"), flush=True)
""")
        start, end = map(int, out.split())
        cuts = sorted(set(np.linspace(start + 1, end - 1, 12).astype(int)))
        for cut in cuts:
            d = str(tmp_path / f"cut{cut}")
            os.mkdir(d)
            p = os.path.join(d, "t.feather")
            shutil.copy(path + ".wal", p + ".wal")
            with open(p + ".wal", "r+b") as f:
                f.truncate(cut)
            db = DB.open(p, dim=DIM)
            assert db.get_metadata(1).content == "before"
            assert [db.version(i) for i in range(2, 6)] == [0] * 4, f"cut at {cut}: part of the batch applied"
            assert not db.write(WriteBatch().put(9, vec(9), meta("x")), idempotency_key="k",
                                fingerprint="g").replayed, f"cut at {cut}: the key survived its batch"
            db.close(save=False)
            assert os.path.getsize(p + ".wal") >= start
        db = DB.open(path, dim=DIM)                    # uncut: all of it
        assert [db.version(i) > 0 for i in range(2, 6)] == [True] * 4
        assert [e.target_id for e in db.get_edges(4)] == [5]
        db.close()


# ── idempotency ───────────────────────────────────────────────────────────

class TestIdempotency:
    def test_a_retry_is_not_applied_twice(self, db):
        db.add(1, vec(1), meta("a"))
        b = WriteBatch().update(1, meta("from the first attempt"))
        first = db.write(b, idempotency_key="req-1", fingerprint="h")
        db.update_metadata(1, meta("someone else, later"))
        again = db.write(b, idempotency_key="req-1", fingerprint="h")
        assert again.replayed and again.versions == first.versions and again.ids == first.ids
        assert db.get_metadata(1).content == "someone else, later", "the retry overwrote a later write"

    def test_the_note_comes_back_with_the_replay(self, db):
        assert db.idempotent_result("req", "h") is None
        b = WriteBatch().put(1, vec(1), meta("a"))
        assert db.write(b, idempotency_key="req", fingerprint="h", note='{"id": 1}').note == ""
        prior = db.idempotent_result("req", "h")
        assert prior.replayed and prior.note == '{"id": 1}' and prior.ids == [1]
        with pytest.raises(IdempotencyKeyReusedError):
            db.idempotent_result("req", "other")

    def test_same_key_different_request_is_refused(self, db):
        db.write(WriteBatch().put(1, vec(1), meta("a")), idempotency_key="req", fingerprint="A")
        with pytest.raises(IdempotencyKeyReusedError):
            db.write(WriteBatch().put(2, vec(2), meta("b")), idempotency_key="req", fingerprint="B")
        assert db.version(2) == 0

    def test_a_refused_batch_does_not_burn_its_key(self, db):
        db.add(1, vec(1), meta("a"))
        b = WriteBatch().update(1, meta("b"), expected_version=12345)
        with pytest.raises(VersionConflictError):
            db.write(b, idempotency_key="req", fingerprint="h")
        ok = WriteBatch().update(1, meta("b"), expected_version=db.version(1))
        assert not db.write(ok, idempotency_key="req", fingerprint="h").replayed

    def test_keys_survive_a_crash_and_a_save(self, tmp_path):
        path = str(tmp_path / "i.feather")
        run_child(f"""
db = feather_db.DB.open({path!r}, dim={DIM})
b = feather_db.WriteBatch().put(1, np.ones({DIM}, np.float32))
db.write(b, idempotency_key="in-wal", fingerprint="f", note="first")
""")
        d = DB.open(path, dim=DIM)
        b = WriteBatch().put(1, np.ones(DIM, np.float32))
        r = d.write(b, idempotency_key="in-wal", fingerprint="f")
        assert r.replayed and r.note == "first", "lost with the crash"
        d.write(WriteBatch().put(2, vec(2)), idempotency_key="saved", fingerprint="f", note="second")
        d.close()
        d = DB.open(path, dim=DIM)
        for key, op_id, note in (("in-wal", 1, "first"), ("saved", 2, "second")):
            r = d.write(WriteBatch().put(op_id, vec(op_id)), idempotency_key=key, fingerprint="f")
            assert r.replayed and r.ids == [op_id] and r.note == note, key
        d.vacuum()
        d.close()
        d = DB.open(path, dim=DIM)
        assert d.write(WriteBatch().put(2, vec(2)), idempotency_key="saved", fingerprint="f").replayed
        d.close()

    @pytest.mark.parametrize("env, expect_replay", [
        ({"FEATHER_IDEMPOTENCY_TTL_S": "1"}, False),
        ({"FEATHER_IDEMPOTENCY_MAX": "3"}, False),
        ({}, True),
    ])
    def test_keys_expire_and_are_bounded(self, tmp_path, env, expect_replay):
        path = str(tmp_path / "e.feather")
        src = f"""
import sys, time
sys.path.insert(0, {REPO!r})
import numpy as np, feather_db
db = feather_db.DB.open({path!r}, dim={DIM})
put = lambda i: feather_db.WriteBatch().put(i, np.ones({DIM}, np.float32))
db.write(put(0), idempotency_key="k0", fingerprint="f")
time.sleep(1.1)
for i in range(1, 6):
    db.write(put(i), idempotency_key=f"k{{i}}", fingerprint="f")
db.close()
db = feather_db.DB.open({path!r}, dim={DIM})
print(db.write(put(0), idempotency_key="k0", fingerprint="f").replayed,
      db.write(put(5), idempotency_key="k5", fingerprint="f").replayed)
db.close()
"""
        out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                             env={**os.environ, **env})
        assert out.returncode == 0, out.stderr
        first, last = out.stdout.split()
        assert first == str(expect_replay)
        assert last == "True"
