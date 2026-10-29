"""The Rust CLI and the Python binding on the same v10 files, end to end.

The CLI compiles the vendored engine (feather-cli/cpp/), so this is also the
check that the vendored copy writes and reads the same format the Python
extension does. Each CLI invocation is a separate process that opens a writer
handle and commits on exit — including `search`, whose salience touches are a
commit of their own — which makes it a natural driver for incremental commits.

Skipped unless the binary is built: `cd feather-cli && cargo build --release`.
"""
import os
import struct
import subprocess

import numpy as np
import pytest

import feather_db
from feather_db import DB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(REPO, "feather-cli", "target", "release", "feather")

pytestmark = pytest.mark.skipif(not os.path.exists(CLI), reason="Rust CLI not built")

DIM = 32


def cli(*args, check=True):
    p = subprocess.run([CLI, *map(str, args)], capture_output=True, text=True, timeout=60)
    if check:
        assert p.returncode == 0, f"feather {' '.join(map(str, args))} failed:\n{p.stdout}{p.stderr}"
    return p


def npy(tmp_path, name, vec):
    path = str(tmp_path / f"{name}.npy")
    np.save(path, np.asarray(vec, dtype=np.float32))
    return path


def header(path, slot):
    raw = open(path, "rb").read(2 * 4096)
    magic, ver, guard, s, gen, end = struct.unpack_from("<IIIIQQ", raw, slot * 4096)
    return dict(magic=magic, version=ver, guard=guard, generation=gen, end=end)


def newest(path):
    return max((header(path, s) for s in (0, 1)), key=lambda h: h["generation"])


def search_ids(out):
    return [int(line.split()[1]) for line in out.splitlines() if line.startswith("ID:")]


@pytest.fixture
def vecs():
    rng = np.random.default_rng(42)
    return rng.standard_normal((40, DIM)).astype(np.float32)


def test_cli_writes_v10_with_incremental_commits(tmp_path, vecs):
    db = str(tmp_path / "cli.feather")
    cli("new", db, "--dim", DIM)
    h = newest(db)
    assert (h["version"], h["guard"], h["generation"]) == (10, 0xFFFFFFFF, 1)

    sizes = [os.path.getsize(db)]
    for i in range(1, 21):
        cli("add", db, i, "-n", npy(tmp_path, f"v{i}", vecs[i]),
            "--content", f"cli record {i}", "--source", "rust", "--importance", 0.5,
            "--timestamp", 1_700_000_000 + i)
        sizes.append(os.path.getsize(db))
        h = newest(db)
        assert h["generation"] == i + 1, "each CLI add should be one commit"
        assert h["end"] == os.path.getsize(db)
    # Every add appended; none rewrote the file.
    growth = np.diff(sizes)
    assert (growth > 0).all()
    assert growth[-1] < 2 * (8 + DIM * 4 + 400) + 2048, growth[-1]

    cli("link", db, 1, 2)
    assert newest(db)["generation"] == 22


def test_python_reads_what_the_cli_wrote(tmp_path, vecs):
    db = str(tmp_path / "x.feather")
    cli("new", db, "--dim", DIM)
    for i in range(1, 11):
        cli("add", db, i, "-n", npy(tmp_path, f"v{i}", vecs[i]),
            "--content", f"cli record {i}", "--source", "rust", "--importance", 0.25,
            "--context-type", 2, "--timestamp", 1_700_000_000 + i)
    cli("add", db, 5, "-n", npy(tmp_path, "vis", vecs[30][:16]), "--modality", "visual")
    cli("link", db, 3, 7)

    d = DB.open(db, dim=DIM)
    try:
        assert d.size() == 10
        for i in range(1, 11):
            m = d.get_metadata(i)
            if i != 5:
                assert (m.content, m.source, m.timestamp) == (f"cli record {i}", "rust", 1_700_000_000 + i)
                assert m.importance == pytest.approx(0.25)
                assert m.type == feather_db.ContextType.EVENT
            assert np.array_equal(d.get_vector(i), vecs[i])
        assert np.array_equal(d.get_vector(5, "visual"), vecs[30][:16])
        assert [e.target_id for e in d.get_edges(3)] == [7]
        assert d.storage_stats()["format"] == 10
    finally:
        d.close()


def test_cli_reads_what_python_wrote_and_commits_salience(tmp_path, vecs):
    db = str(tmp_path / "py.feather")
    d = DB.open(db, dim=DIM)
    d.add_batch(list(range(40)), vecs, [feather_db.Metadata() for _ in range(40)])
    d.save()
    d.add(100, vecs[7] + 0.01)                 # an appended commit on top of the base
    d.save()
    q = vecs[7] + 0.001
    expect = [r.id for r in d.search(q, k=5)]
    d.close()
    gen = newest(db)["generation"]

    out = cli("search", db, "-n", npy(tmp_path, "q", q), "--k", 5).stdout
    assert search_ids(out) == expect

    # The search touched its hits; the CLI committed those counters on exit.
    assert newest(db)["generation"] == gen + 1
    d = DB.open(db, dim=DIM)
    try:
        assert all(d.get_metadata(i).recall_count >= 1 for i in expect)
    finally:
        d.close()


def test_cli_is_refused_while_python_holds_the_writer_lock(tmp_path, vecs):
    db = str(tmp_path / "lock.feather")
    d = DB.open(db, dim=DIM)
    d.add(1, vecs[1])
    d.save()
    before = open(db, "rb").read()
    try:
        p = cli("add", db, 2, "-n", npy(tmp_path, "v", vecs[2]), check=False)
        assert p.returncode != 0
        assert open(db, "rb").read() == before
    finally:
        d.close()


def test_cli_recovers_a_python_wal_and_commits_it(tmp_path, vecs):
    db = str(tmp_path / "wal.feather")
    d = DB.open(db, dim=DIM)
    d.add_batch(list(range(20)), vecs[:20], [feather_db.Metadata() for _ in range(20)])
    d.save()
    for i in range(20, 30):                    # WAL only
        m = feather_db.Metadata(); m.content = f"unsaved {i}"
        d.add(i, vecs[i], m)
    d.close(save=False)                        # crash
    assert os.path.getsize(db + ".wal") > 8

    cli("search", db, "-n", npy(tmp_path, "q", vecs[25]), "--k", 3)   # replays + commits
    assert not os.path.exists(db + ".wal") or os.path.getsize(db + ".wal") <= 8

    d = DB.open(db, dim=DIM)
    try:
        assert d.size() == 30
        assert d.get_metadata(25).content == "unsaved 25"
        assert np.array_equal(d.get_vector(29), vecs[29])
    finally:
        d.close()


def test_cli_upgrades_a_v9_file(tmp_path, vecs, monkeypatch):
    db = str(tmp_path / "old.feather")
    monkeypatch.setenv("FEATHER_FORMAT", "9")
    d = DB.open(db, dim=DIM)
    d.add_batch(list(range(10)), vecs[:10], [feather_db.Metadata() for _ in range(10)])
    d.close()
    monkeypatch.delenv("FEATHER_FORMAT")
    assert struct.unpack_from("<I", open(db, "rb").read(8), 4)[0] == 9

    cli("add", db, 50, "-n", npy(tmp_path, "v", vecs[11]))
    assert newest(db)["version"] == 10
    d = DB.open(db, dim=DIM)
    try:
        assert d.size() == 11
        assert np.array_equal(d.get_vector(3), vecs[3])
    finally:
        d.close()


def test_cli_cuts_a_torn_tail(tmp_path, vecs):
    db = str(tmp_path / "torn.feather")
    cli("new", db, "--dim", DIM)
    cli("add", db, 1, "-n", npy(tmp_path, "a", vecs[1]))
    end = os.path.getsize(db)
    with open(db, "ab") as f:
        f.write(b"\xde\xad" * 300)             # a commit that never switched the header
    cli("add", db, 2, "-n", npy(tmp_path, "b", vecs[2]))
    h = newest(db)
    assert h["end"] == os.path.getsize(db) and h["end"] > end
    assert b"\xde\xad" * 300 not in open(db, "rb").read()
    d = DB.open(db, dim=DIM)
    try:
        assert d.size() == 2
    finally:
        d.close()


def test_cli_refuses_a_corrupt_file_and_leaves_it_alone(tmp_path, vecs):
    db = str(tmp_path / "bad.feather")
    cli("new", db, "--dim", DIM)
    cli("add", db, 1, "-n", npy(tmp_path, "a", vecs[1]))
    raw = bytearray(open(db, "rb").read())
    raw[8192 + 30] ^= 0xFF          # inside the full image every header depends on
    open(db, "wb").write(bytes(raw))
    p = cli("add", db, 2, "-n", npy(tmp_path, "b", vecs[2]), check=False)
    assert p.returncode == 1, f"expected a clean failure, got rc={p.returncode}"  # not a signal
    assert "Open failed" in p.stderr
    assert open(db, "rb").read() == bytes(raw)


# ── errors cross the C ABI as errors, not as a C++ exception ──────────────

def test_engine_error_is_reported_not_aborted(tmp_path, vecs):
    """An exception unwinding through an extern "C" frame used to abort the
    process (SIGABRT). It is now a message and exit status 1."""
    db = str(tmp_path / "e.feather")
    cli("add", db, 1, "-n", npy(tmp_path, "a", vecs[0]))
    p = cli("add", db, 2, "-n", npy(tmp_path, "b", np.zeros(DIM + 3)), check=False)
    assert p.returncode == 1, (p.returncode, p.stderr)
    assert "Add failed" in p.stderr and "imension" in p.stderr, p.stderr


def test_open_refusal_names_the_reason(tmp_path, vecs):
    db = str(tmp_path / "l.feather")
    holder = DB.open(db, dim=DIM)      # lock is reentrant per process, not across
    try:
        p = cli("add", db, 1, "-n", npy(tmp_path, "a", vecs[0]), check=False)
    finally:
        holder.close()
    assert p.returncode == 1
    assert "Open failed:" in p.stderr and "lock" in p.stderr, p.stderr
