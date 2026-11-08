"""BenchRunner: orchestrates a scenario and persists the result."""
from __future__ import annotations
import json
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import feather_db

BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR / "results"
REPORTS_DIR = BENCH_DIR / "reports"


@dataclass
class BenchResult:
    scenario: str
    dataset: str
    n: int
    dim: int
    feather_version: str
    python_version: str
    platform: str
    commit: str
    started_at: float
    wall_seconds: float
    metrics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    # The exact invocation, and whether `commit` alone reproduces it:
    # a run on a dirty tree measured code that no commit contains.
    command: str = ""
    dirty: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _git_commit() -> str:
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=BENCH_DIR.parent,
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
            cwd=BENCH_DIR.parent).decode()
    except Exception:
        return False
    # Untracked code counts (a new module is code no commit has); the
    # bench's own output does not.
    return any(not line[3:].startswith(("bench/results/", "bench/reports/"))
               for line in out.splitlines() if line.strip())


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    if sys.platform == "darwin":
        return r / (1024 * 1024)
    return r / 1024


class BenchRunner:
    """Thin orchestrator — scenarios do the real work and fill in `metrics`."""

    def __init__(self, scenario: str, dataset: str):
        self.scenario = scenario
        self.dataset = dataset
        self.params: dict[str, Any] = {}
        self.notes: str = ""
        RESULTS_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)

    def run(self, fn, *, n: int, dim: int, verdict=None, **params) -> BenchResult:
        """Call `fn(runner)` which populates runner.metrics and returns a dict.

        verdict(metrics, dirty) -> reasons the result must not be reported
        (empty: reportable). Stored in the result as `reportable` and
        `not_reportable_because`, so a smoke run cannot pass for a score."""
        self.params.update(params)
        started = time.time()
        metrics = fn(self)
        elapsed = time.time() - started
        metrics.setdefault("peak_rss_mb", _rss_mb())
        dirty = _git_dirty()
        if verdict is not None:
            reasons = list(verdict(metrics, dirty))
            metrics["reportable"] = not reasons
            metrics["not_reportable_because"] = reasons

        result = BenchResult(
            scenario=self.scenario,
            dataset=self.dataset,
            n=n,
            dim=dim,
            feather_version=getattr(feather_db, "__version__", "unknown"),
            python_version=".".join(map(str, sys.version_info[:3])),
            platform=f"{platform.system()} {platform.machine()}",
            commit=_git_commit(),
            started_at=started,
            wall_seconds=elapsed,
            metrics=metrics,
            params=self.params,
            notes=self.notes,
            command=" ".join([os.path.basename(sys.executable), "-m", "bench",
                              *sys.argv[1:]]),
            dirty=dirty,
        )
        self._persist(result)
        return result

    def _persist(self, result: BenchResult) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(result.started_at))
        path = reserve(f"{result.scenario}__{result.dataset}__{stamp}", ".json")
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"[bench] wrote {path}")


_CLAIMS: set = set()


def reserve(stem: str, suffix: str, directory=None) -> Path:
    """RESULTS_DIR/<stem><suffix>, created empty so no other run can take it:
    <stem>_2<suffix>, _3, ... when it exists. Names are stamped to the
    second, and two runs started together wrote one file, the second
    overwriting the first's answers. A claim nothing was written to is
    removed by release_unused_claims() when the run ends."""
    d = Path(directory or RESULTS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        path = d / f"{stem}{'' if n == 1 else f'_{n}'}{suffix}"
        try:
            open(path, "x").close()
            _CLAIMS.add(path)
            return path
        except FileExistsError:
            n += 1


def release_unused_claims() -> int:
    """Delete this process's claims that are still empty: a run that failed
    or returned before its first record left them, and an empty results file
    reads as a run that scored nothing. Returns how many."""
    n = 0
    for path in list(_CLAIMS):
        _CLAIMS.discard(path)
        try:
            if path.stat().st_size == 0:
                path.unlink()
                n += 1
        except FileNotFoundError:
            pass
    return n


def load_results() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    out = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out
