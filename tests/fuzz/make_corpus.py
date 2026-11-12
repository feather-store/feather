"""Seed corpora for the libFuzzer targets, from the seeded fuzz's own
builders (tests/test_corrupt_files.py `build`): every file kind it corrupts,
and a WAL with every op it writes (adds, links, a keyed batch, forget, purge)
together with the base that WAL was written against.

    python tests/fuzz/make_corpus.py OUT_DIR
    -> OUT_DIR/file/*.feather   seeds for fuzz_file
       OUT_DIR/wal/*.wal        seeds for fuzz_wal
       OUT_DIR/base.feather     FEATHER_FUZZ_BASE for fuzz_wal
"""
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

from test_corrupt_files import build  # noqa: E402


class _Env:
    """The two monkeypatch calls build() makes, on os.environ."""

    def setenv(self, k, v):
        os.environ[k] = v

    def delenv(self, k, raising=True):
        os.environ.pop(k, None)


def main(out):
    for sub in ("file", "wal"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    work = tempfile.mkdtemp(prefix="feather-corpus-")
    try:
        for kind in ("v9-graph", "v9-rebuild", "v10", "wal"):
            path = os.path.join(work, f"{kind}.feather")
            build(path, kind, _Env())
            shutil.copy(path, os.path.join(out, "file", f"{kind}.feather"))
            if kind == "wal":
                shutil.copy(path, os.path.join(out, "base.feather"))
                shutil.copy(path + ".wal", os.path.join(out, "wal", "ops.wal"))
        print(f"corpus in {out}: {len(os.listdir(os.path.join(out, 'file')))} files, "
              f"WAL {os.path.getsize(os.path.join(out, 'wal', 'ops.wal'))} bytes")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
