"""Inter-process file locking.

The defect this closes, measured before the change: two processes open one
`.feather`, each writes, each saves, both exit 0 — and 20 of 20 records from one
of them are gone. Each DB holds the whole dataset in RAM and `save_vectors()`
rewrites the file from that view, so the second save writes a view that never
contained the first's records. Nothing errors. Nothing logs. The data is simply
not there.

That cannot be fixed by serialising the saves — a mutex would just decide whose
records vanish. It needs either merge-on-save or a paged store, which is a format
change. So what ships is enforcement of the model that IS safe: one writer, many
readers, and a loud refusal instead of silent loss.

The lock is deliberately REENTRANT within a process, because `py::nodelete`
means `del db` never runs the destructor, so a strict lock would refuse a program
its own file. Same-process double-open stays the documented footgun it already
was; cross-process is what is now impossible.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import feather_db
from feather_db import DB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def child(code: str, **env) -> subprocess.CompletedProcess:
    """Run `code` in a separate PROCESS — the only way to test the lock."""
    src = f"import sys; sys.path.insert(0, {REPO!r})\n" + textwrap.dedent(code)
    e = {**os.environ, **{k: str(v) for k, v in env.items()}}
    return subprocess.run([sys.executable, "-c", src], capture_output=True,
                          text=True, timeout=120, env=e)


# ── the guarantee ─────────────────────────────────────────────────────────

def test_a_second_process_cannot_open_the_file_for_writing(tmp_path):
    path = str(tmp_path / "a.feather")
    holder = DB.open(path, dim=16)
    try:
        out = child(f"""
            import feather_db
            try:
                feather_db.DB.open({path!r}, dim=16)
                print("OPENED")
            except Exception as exc:
                print("REFUSED:", exc)
        """)
        assert "REFUSED" in out.stdout, out.stdout + out.stderr
        assert "another process" in out.stdout
        assert "single-writer" in out.stdout          # says what the model is
    finally:
        holder.close()


def test_the_twenty_of_twenty_data_loss_no_longer_happens(tmp_path):
    """The original report, as a test. Process A writes 20 records and holds the
    file; process B tries to write 20 more. Before the lock, B's save erased all
    of A's. Now B cannot open it, and A's records are all there."""
    path = str(tmp_path / "loss.feather")
    a = DB.open(path, dim=8)
    for i in range(20):
        meta = feather_db.Metadata()
        meta.content = f"record {i} from process A"
        a.add(id=i, vec=np.ones(8, dtype=np.float32) * (i + 1), meta=meta)
    a.save()

    out = child(f"""
        import numpy as np, feather_db
        try:
            db = feather_db.DB.open({path!r}, dim=8)
            for i in range(100, 120):
                db.add(id=i, vec=np.ones(8, dtype=np.float32))
            db.save()
            print("B WROTE")
        except Exception:
            print("B REFUSED")
    """)
    assert "B REFUSED" in out.stdout, out.stdout + out.stderr

    a.close()
    reopened = DB.open(path, dim=8)
    try:
        assert len(reopened.get_all_ids()) == 20, "process A's records were lost"
        assert reopened.get_metadata(7).content == "record 7 from process A"
    finally:
        reopened.close()


def test_closing_hands_the_file_to_the_next_process(tmp_path):
    path = str(tmp_path / "handoff.feather")
    first = DB.open(path, dim=8)
    first.add(id=1, vec=np.ones(8, dtype=np.float32))
    first.close()

    out = child(f"""
        import numpy as np, feather_db
        db = feather_db.DB.open({path!r}, dim=8)
        db.add(id=2, vec=np.ones(8, dtype=np.float32))
        db.save(); db.close()
        print("OK", len(db.get_all_ids()) if not db.is_closed() else 2)
    """)
    assert out.returncode == 0, out.stderr[-500:]
    assert "OK" in out.stdout


# ── many readers ──────────────────────────────────────────────────────────

def test_readers_are_never_blocked_by_the_writer(tmp_path):
    """One writer service, many agent processes reading the same file — the
    architecture this has to support. Readers take NO lock: a shared lock would
    conflict with the writer's exclusive one and refuse every reader while the
    writer merely held the handle open."""
    path = str(tmp_path / "shared.feather")
    writer = DB.open(path, dim=8)
    writer.add(id=1, vec=np.ones(8, dtype=np.float32))
    writer.save()                       # NOT closed — still holding the lock
    try:
        out = child(f"""
            import feather_db
            db = feather_db.DB.open({path!r}, dim=8, read_only=True)
            print("READER OK", db.size())
        """)
        assert "READER OK 1" in out.stdout, out.stdout + out.stderr

        # and several at once, still with the writer holding it
        for _ in range(3):
            out = child(f"""
                import feather_db
                db = feather_db.DB.open({path!r}, dim=8, read_only=True)
                print("OK", db.size())
            """)
            assert "OK 1" in out.stdout
    finally:
        writer.close()


def test_a_reader_sees_a_consistent_snapshot_across_a_save(tmp_path):
    """save_vectors() writes <path>.tmp and renames it over the original, so a
    reader holds the old inode and sees a complete file — just a stale one —
    until it reopens. That is what makes lock-free reads safe."""
    path = str(tmp_path / "snap.feather")
    writer = DB.open(path, dim=8)
    writer.add(id=1, vec=np.ones(8, dtype=np.float32))
    writer.save()

    reader = DB.open(path, dim=8, read_only=True)
    assert reader.size() == 1

    for i in range(2, 12):              # writer moves on underneath it
        writer.add(id=i, vec=np.ones(8, dtype=np.float32))
    writer.save()

    assert reader.size() == 1, "the reader's snapshot changed under it"
    reader.close()

    fresh = DB.open(path, dim=8, read_only=True)
    try:
        assert fresh.size() == 11       # reopening advances
    finally:
        fresh.close()
        writer.close()


def test_a_read_only_handle_does_not_clear_the_writers_wal(tmp_path):
    """A reader replays the WAL into its own memory to see uncheckpointed
    records. It must not CLEAR it — only save_vectors() does that, and a reader
    never saves. If it did, a crashed writer's unsaved records would be
    destroyed by something merely looking at the file."""
    path = str(tmp_path / "wal.feather")
    out = child(f"""
        import os, numpy as np, feather_db
        db = feather_db.DB.open({path!r}, dim=8)
        for i in range(20):
            db.add(id=i, vec=np.ones(8, dtype=np.float32))
        os._exit(0)          # crash: no checkpoint, records live only in the WAL
    """)
    assert out.returncode == 0
    assert os.path.exists(path + ".wal"), "nothing in the WAL to test"

    reader = DB.open(path, dim=8, read_only=True)
    assert reader.size() == 20          # replayed into memory
    reader.close()
    assert os.path.exists(path + ".wal"), "a read-only open destroyed the WAL"

    recovered = DB.open(path, dim=8)
    try:
        assert recovered.size() == 20   # the writer still recovers everything
    finally:
        recovered.close()


# ── read-only handles ─────────────────────────────────────────────────────

def test_a_read_only_handle_refuses_every_mutation(tmp_path):
    path = str(tmp_path / "ro.feather")
    seed = DB.open(path, dim=8)
    seed.add(id=1, vec=np.ones(8, dtype=np.float32))
    seed.close()

    ro = DB.open(path, dim=8, read_only=True)
    try:
        assert ro.is_read_only()
        assert ro.size() == 1                       # reading is fine
        with pytest.raises(RuntimeError, match="read-only"):
            ro.add(id=2, vec=np.ones(8, dtype=np.float32))
        with pytest.raises(RuntimeError, match="read-only"):
            ro.forget(1)
    finally:
        ro.close()

    # and nothing leaked through: the mutation really did not happen
    check = DB.open(path, dim=8)
    try:
        assert check.size() == 1
    finally:
        check.close()


def test_a_read_only_handle_does_not_rewrite_the_file_on_close(tmp_path):
    """A reader holds a SHARED lock, so checkpointing from it would rewrite a
    file other readers are using."""
    path = str(tmp_path / "ro2.feather")
    seed = DB.open(path, dim=8)
    seed.add(id=1, vec=np.ones(8, dtype=np.float32))
    seed.save()
    seed.close()
    before = os.stat(path).st_mtime_ns

    ro = DB.open(path, dim=8, read_only=True)
    ro.close()
    assert os.stat(path).st_mtime_ns == before, "a read-only close wrote the file"


# ── escape hatches and compatibility ──────────────────────────────────────

def test_feather_lock_0_disables_the_check(tmp_path):
    """Needed because flock is unreliable on NFS and some overlay mounts. The
    caller then owns the consequence."""
    path = str(tmp_path / "off.feather")
    holder = DB.open(path, dim=8)
    try:
        out = child(f"""
            import feather_db
            feather_db.DB.open({path!r}, dim=8)
            print("OPENED ANYWAY")
        """, FEATHER_LOCK="0")
        assert "OPENED ANYWAY" in out.stdout, out.stdout + out.stderr
    finally:
        holder.close()


def test_the_same_process_may_still_reopen_its_own_file(tmp_path):
    """Reentrancy. `del db` cannot release the lock — DB is bound py::nodelete,
    so the destructor never runs from Python. Without this, every program that
    reopened its own file would break on upgrade."""
    path = str(tmp_path / "re.feather")
    first = DB.open(path, dim=8)
    first.add(id=1, vec=np.ones(8, dtype=np.float32))
    first.save()
    del first                                   # does NOT release

    second = DB.open(path, dim=8)               # must still be allowed
    assert second.size() == 1
    second.close()


def test_the_lock_key_is_stable_across_the_files_creation(tmp_path):
    """Regression. The key was realpath(file), which only resolves once the file
    exists — and on macOS rewrites /var/... to /private/var/.... So the key
    differed before and after the first save, reentrancy missed, and the handle
    took a second flock against one this same process already held, reported as
    'another process (pid <ourselves>)'."""
    path = str(tmp_path / "subdir")
    os.makedirs(path, exist_ok=True)
    f = os.path.join(path, "k.feather")

    a = DB.open(f, dim=8)                       # file does not exist yet
    a.add(id=1, vec=np.ones(8, dtype=np.float32))
    a.save()                                    # now it does
    b = DB.open(f, dim=8)                       # same key, so admitted
    assert b.size() == 1
    a.close(); b.close()


def test_a_dead_holder_does_not_block_the_file_forever(tmp_path):
    """flock is released by the kernel when the holder dies, so a crashed writer
    must not leave the file permanently unopenable — a stale `.lock` file on disk
    is not itself a lock."""
    path = str(tmp_path / "dead.feather")
    out = child(f"""
        import numpy as np, feather_db
        db = feather_db.DB.open({path!r}, dim=8)
        db.add(id=1, vec=np.ones(8, dtype=np.float32))
        db.save()
        print("HELD")
    """)
    assert "HELD" in out.stdout, out.stdout + out.stderr
    assert os.path.exists(path + ".lock"), "no lock file was left behind to test"

    survivor = DB.open(path, dim=8)             # the holder is gone
    try:
        assert survivor.size() == 1
    finally:
        survivor.close()
