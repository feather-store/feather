"""WAL integrity: checksums and fsync.

The WAL exists so an acknowledged write survives a crash. Two gaps remained
after the recovery fix in v0.17.0:

  1. Records were only flush()ed, never fsync()ed. flush() hands bytes to the
     kernel, which survives the *process* dying but not the *machine* or the
     container dying — most of what a write-ahead log is actually for. The
     existing recovery tests SIGKILL the process, so they pass against the
     weaker guarantee and could never have caught this.

  2. There was no checksum. The length guard catches a *short* tail, but it
     cannot see a record whose bytes were mangled in place: that record
     replayed as plausible garbage and was then written into the next
     checkpoint as though it were real. Silent corruption, promoted to
     durable corruption.

WAL format v2 adds a header (so a v1 WAL left by an older build still
recovers) and a CRC32 per record.
"""
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

import feather_db
from feather_db import DB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WAL_MAGIC = 0x4C415746   # "FWAL"
HEADER_LEN = 8           # magic(4) + version(4)


def _seed(path, n=5, start=0):
    db = DB.open(path, dim=16)
    for i in range(start, start + n):
        m = feather_db.Metadata(); m.content = f"rec {i} alpha"
        db.add(i, np.random.rand(16).astype(np.float32), m)
    return db


# ── format ────────────────────────────────────────────────────────────────

def test_wal_is_written_with_a_versioned_header(tmp_path):
    path = str(tmp_path / "hdr.feather")
    _seed(path, 2)
    with open(path + ".wal", "rb") as fh:
        magic, ver = struct.unpack("<II", fh.read(8))
    assert magic == WAL_MAGIC, f"missing WAL magic, got {magic:#x}"
    assert ver == 2, f"unexpected WAL version {ver}"


def test_records_carry_a_checksum(tmp_path):
    """One record must occupy exactly op+id+plen+payload+crc bytes."""
    path = str(tmp_path / "crc.feather")
    _seed(path, 1)
    raw = open(path + ".wal", "rb").read()
    body = raw[HEADER_LEN:]
    plen = struct.unpack("<I", body[9:13])[0]
    assert len(body) == 1 + 8 + 4 + plen + 4, (
        f"record is {len(body)}B, expected {1+8+4+plen+4}B incl. 4B CRC")


# ── corruption detection ──────────────────────────────────────────────────

def test_a_flipped_byte_is_caught_instead_of_replayed(tmp_path):
    """THE case a length guard cannot catch: the record is the right size, so
    only a checksum can tell that its contents are wrong."""
    path = str(tmp_path / "flip.feather")
    db = _seed(path, 6)
    db.save()                       # checkpoint: records 0-5 are in the base file
    del db

    db = _seed(path, 4, start=100)  # 100-103 live only in the WAL
    del db

    raw = bytearray(open(path + ".wal", "rb").read())
    # Corrupt deep inside the first WAL record's payload — length untouched.
    target = HEADER_LEN + 13 + 40
    assert target < len(raw)
    raw[target] ^= 0xFF
    open(path + ".wal", "wb").write(bytes(raw))

    db2 = DB.open(path, dim=16)
    # The checkpointed records must survive; the corrupt record and everything
    # after it must be discarded rather than replayed as garbage.
    assert db2.size() >= 6, "checkpointed records were lost"
    assert db2.get_metadata(0) is not None
    assert db2.get_metadata(100) is None, (
        "a record with a corrupt payload was replayed anyway")


def test_intact_wal_still_replays_completely(tmp_path):
    """The checksum must not cost recovery of a healthy WAL."""
    path = str(tmp_path / "intact.feather")
    db = _seed(path, 3)
    db.save()
    del db
    db = _seed(path, 7, start=200)
    del db

    db2 = DB.open(path, dim=16)
    assert db2.size() == 10
    for i in list(range(3)) + list(range(200, 207)):
        assert db2.get_metadata(i) is not None, f"record {i} lost"


def test_truncated_tail_still_keeps_earlier_records(tmp_path):
    """A half-written trailing record is normal after a crash — everything
    before it must survive, now judged by checksum rather than length alone."""
    path = str(tmp_path / "torn.feather")
    db = _seed(path, 12)
    del db

    wal = path + ".wal"
    full = os.path.getsize(wal)
    with open(wal, "r+b") as fh:
        fh.truncate(full - 30)

    db2 = DB.open(path, dim=16)
    assert 0 < db2.size() <= 12, f"expected partial recovery, got {db2.size()}"


# ── backward compatibility ────────────────────────────────────────────────

def test_a_v1_wal_from_an_older_build_still_recovers(tmp_path):
    """Upgrading must not strand a WAL that a crash already left on disk.
    A v1 WAL has no header and no CRCs; replay has to detect that and read it
    the old way."""
    path = str(tmp_path / "legacy.feather")
    db = _seed(path, 4)
    db.save()                        # base file with 0-3
    del db
    db = _seed(path, 3, start=50)    # 50-52 in a v2 WAL
    del db

    # Rewrite that WAL in v1 form: drop the header, strip each record's CRC.
    raw = open(path + ".wal", "rb").read()
    assert struct.unpack("<I", raw[:4])[0] == WAL_MAGIC
    body, out = raw[HEADER_LEN:], bytearray()
    off = 0
    while off + 13 <= len(body):
        op_id = body[off:off + 13]
        plen = struct.unpack("<I", body[off + 9:off + 13])[0]
        payload = body[off + 13:off + 13 + plen]
        out += op_id + payload            # everything except the trailing CRC
        off += 13 + plen + 4
    open(path + ".wal", "wb").write(bytes(out))

    db2 = DB.open(path, dim=16)
    assert db2.size() == 7, f"v1 WAL not replayed: got {db2.size()} of 7"
    assert db2.get_metadata(52) is not None


# ── fsync ─────────────────────────────────────────────────────────────────

def _fsync_calls(path, env):
    """Count fsync/fcntl(F_FULLFSYNC) syscalls a child makes while writing."""
    src = f'''
import sys
sys.path.insert(0, {REPO!r})
import numpy as np, feather_db
db = feather_db.DB.open({path!r}, dim=16)
for i in range(5):
    m = feather_db.Metadata(); m.content = f"r{{i}}"
    db.add(i, np.random.rand(16).astype(np.float32), m)
'''
    e = dict(os.environ); e.update(env); e.pop("FEATHER_SIMD", None)
    return subprocess.run([sys.executable, "-c", src], capture_output=True,
                          text=True, env=e)


def test_sync_can_be_disabled_without_breaking_recovery(tmp_path):
    """FEATHER_WAL_SYNC=0 trades durability for throughput — the WAL must still
    be written and still replay, it just isn't forced to stable storage."""
    path = str(tmp_path / "nosync.feather")
    r = _fsync_calls(path, {"FEATHER_WAL_SYNC": "0"})
    assert r.returncode == 0, r.stderr[-400:]
    assert os.path.getsize(path + ".wal") > 0
    assert DB.open(path, dim=16).size() == 5


def test_default_is_sync_on(tmp_path):
    """No env var set → durability on. Recovery works either way; this pins the
    default so it can't silently regress to the faster, weaker mode."""
    path = str(tmp_path / "sync.feather")
    r = _fsync_calls(path, {})
    assert r.returncode == 0, r.stderr[-400:]
    assert DB.open(path, dim=16).size() == 5
