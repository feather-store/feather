"""SIFT1M / SIFTsmall loader.

Downloads on first run and caches to ~/.cache/feather/sift/. Parses
.fvecs / .ivecs binary format (4-byte int32 dim header + payload, per vector).

Variants:
    siftsmall : 10K base, 100 queries, gt top-100   ( 5MB compressed)
    sift1m    : 1M base,  10K queries, gt top-100   (168MB compressed)
"""
from __future__ import annotations
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
import numpy as np


CACHE_ROOT = Path(os.environ.get("FEATHER_CACHE", Path.home() / ".cache" / "feather"))
SIFT_DIR = CACHE_ROOT / "sift"

VARIANTS = {
    "siftsmall": {
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
        "subdir": "siftsmall",
        "files": {
            "base":   "siftsmall_base.fvecs",
            "query":  "siftsmall_query.fvecs",
            "gt":     "siftsmall_groundtruth.ivecs",
        },
    },
    "sift1m": {
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        "subdir": "sift",
        "files": {
            "base":   "sift_base.fvecs",
            "query":  "sift_query.fvecs",
            "gt":     "sift_groundtruth.ivecs",
        },
    },
}


def _read_fvecs(path: Path) -> np.ndarray:
    """Read TexMex .fvecs: each vec is [int32 dim][float32 * dim]."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    # Reshape: each row is (dim+1) ints, the first being the dim header
    n = raw.size // (dim + 1)
    data = raw.reshape(n, dim + 1)[:, 1:].copy().view(np.float32)
    return data


def _read_ivecs(path: Path) -> np.ndarray:
    """Read TexMex .ivecs: each vec is [int32 dim][int32 * dim]."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    n = raw.size // (dim + 1)
    return raw.reshape(n, dim + 1)[:, 1:].copy()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[sift] downloading {url} -> {dest} ...")
    subprocess.check_call([
        "curl", "-fL", "--progress-bar", "--retry", "3",
        "-o", str(dest), url,
    ])


def _ensure_extracted(variant: str) -> Path:
    """Returns the dir containing the .fvecs/.ivecs files for `variant`."""
    cfg = VARIANTS[variant]
    sub = SIFT_DIR / cfg["subdir"]
    if all((sub / f).exists() for f in cfg["files"].values()):
        return sub

    SIFT_DIR.mkdir(parents=True, exist_ok=True)
    tarball = SIFT_DIR / Path(cfg["url"]).name
    if not tarball.exists():
        _download(cfg["url"], tarball)

    print(f"[sift] extracting {tarball.name} ...")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(SIFT_DIR)

    # Sanity check
    missing = [f for f in cfg["files"].values() if not (sub / f).exists()]
    if missing:
        raise FileNotFoundError(f"after extract, missing: {missing} in {sub}")
    return sub


def load_sift(variant: str = "sift1m", n_base: int | None = None,
              n_queries: int | None = None, gt_k: int = 100):
    """Returns (base, queries, ground_truth) as float32/float32/int64 arrays.

    Args:
        variant: "sift1m" or "siftsmall"
        n_base: if set, subset to first n_base vectors of the base set.
                Ground truth references full base, so subsetting may
                introduce false negatives — recall is reported relative
                to the *intersection* of GT IDs that fall in [0, n_base).
        n_queries: subset queries (and corresponding GT rows) to first n_queries.
        gt_k: number of top-k columns to keep in ground truth (default 100,
              SIFT1M ships top-100).

    Note: HNSW max_elements = 1M in Feather DB. If using sift1m with all 1M
    base vectors, the index is exactly at capacity. Subset to e.g. 950_000
    or use a fresh DB per run.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}; must be one of {list(VARIANTS)}")
    sub = _ensure_extracted(variant)
    cfg = VARIANTS[variant]

    base = _read_fvecs(sub / cfg["files"]["base"])
    queries = _read_fvecs(sub / cfg["files"]["query"])
    gt = _read_ivecs(sub / cfg["files"]["gt"])  # int32, columns are doc indices

    if n_base is not None:
        base = base[:n_base]
        # Mask GT entries that point outside the subset
        # We do not drop them here — scenario reports recall@k vs the truncated GT
        gt = np.where(gt < n_base, gt, -1)
    if n_queries is not None:
        queries = queries[:n_queries]
        gt = gt[:n_queries]
    if gt_k < gt.shape[1]:
        gt = gt[:, :gt_k]

    return (
        np.ascontiguousarray(base, dtype=np.float32),
        np.ascontiguousarray(queries, dtype=np.float32),
        np.ascontiguousarray(gt, dtype=np.int64),
    )
