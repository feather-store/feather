"""The WAL's CRC-32 is zlib's, by every path the engine has.

A WAL written on one machine must replay on any other, so the CPU-instruction
path (ARM's CRC32 instructions, where the build targets them) and the
portable slicing-by-8 path must both compute exactly zlib.crc32. Both are
checked on every machine: `sliced=True` forces the portable path.
"""
import struct
import zlib

import numpy as np
import pytest

import feather_db
from feather_db import core

PATHS = [False, True]    # the build's own path, then slicing-by-8


@pytest.mark.parametrize("sliced", PATHS)
def test_matches_zlib_for_every_length_and_alignment(sliced):
    rng = np.random.default_rng(3)
    buf = rng.integers(0, 256, 4200, dtype=np.uint8).tobytes()
    for length in list(range(0, 70)) + [127, 128, 129, 1023, 3085, 3086, 3087, 4096]:
        for offset in range(0, 9):
            chunk = buf[offset:offset + length]
            assert core._crc32(chunk, sliced) == zlib.crc32(chunk), (sliced, length, offset)


@pytest.mark.parametrize("sliced", PATHS)
def test_known_values(sliced):
    assert core._crc32(b"", sliced) == 0
    assert core._crc32(b"123456789", sliced) == 0xCBF43926     # the CRC-32 check value
    assert core._crc32(b"\xff" * 64, sliced) == zlib.crc32(b"\xff" * 64)


def test_reports_its_path():
    assert isinstance(core._crc32_hardware(), bool)


def test_every_wal_record_the_engine_writes_carries_zlibs_crc(tmp_path):
    """Records of many sizes (vector puts, metadata of every tail length),
    read back from the WAL file and checked with zlib, not with the engine."""
    path = str(tmp_path / "c.feather")
    db = feather_db.DB.open(path, dim=24)
    rng = np.random.default_rng(5)
    for i in range(40):
        m = feather_db.Metadata()
        m.content = "x" * i                    # every payload length mod 8
        db.add(id=i, vec=rng.standard_normal(24).astype(np.float32), meta=m)
    db.close(save=False)                       # keep the WAL

    wal = path + ".wal"
    raw = open(wal, "rb").read()
    assert raw[:4] == b"FWAL"
    p, records = 8, 0
    while p < len(raw):
        plen = struct.unpack_from("<I", raw, p + 9)[0]
        body = raw[p:p + 13 + plen]
        (crc,) = struct.unpack_from("<I", raw, p + 13 + plen)
        assert crc == zlib.crc32(body), f"record {records} at byte {p}"
        p += 13 + plen + 4
        records += 1
    assert p == len(raw) and records >= 40

    db = feather_db.DB.open(path, dim=24)       # and it replays
    assert db.size() == 40
    db.close()


def test_a_wal_whose_crcs_zlib_wrote_replays(tmp_path):
    """The other direction: records checksummed by zlib (as an older build,
    or another machine, wrote them) are accepted. (A wrong CRC refused:
    tests/test_corrupt_files.py.)"""
    path = str(tmp_path / "z.feather")
    db = feather_db.DB.open(path, dim=8)
    db.add(id=1, vec=np.ones(8, np.float32))
    db.close(save=False)
    raw = bytearray(open(path + ".wal", "rb").read())
    # Rewrite every CRC with zlib's: replay must accept them all.
    p = 8
    while p < len(raw):
        plen = struct.unpack_from("<I", raw, p + 9)[0]
        struct.pack_into("<I", raw, p + 13 + plen, zlib.crc32(bytes(raw[p:p + 13 + plen])))
        p += 13 + plen + 4
    open(path + ".wal", "wb").write(bytes(raw))
    db = feather_db.DB.open(path, dim=8)
    assert db.get_vector(1).tolist() == [1.0] * 8
    db.close()
