"""Pytest wrapper around the BM25/hybrid self-running script.

The underlying _bm25_hybrid_script.py runs 49 assertions at module
import time via sys.exit(), which breaks pytest collection. We wrap it
as a subprocess so pytest can still enforce regressions in CI.
"""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent / "_bm25_hybrid_script.py"


def test_bm25_hybrid_script():
    """Runs the legacy BM25 assertions as a subprocess. Exit 0 = all pass."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        # Surface script output on failure for debugging.
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, (
        f"BM25/hybrid script exited {result.returncode}"
    )
