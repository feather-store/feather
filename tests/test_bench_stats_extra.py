"""Percentile intervals and false-discovery control (bench.stats)."""
import numpy as np

from bench import stats


def test_a_p99_from_120_samples_has_no_upper_bound():
    lo, hi = stats.quantile_ci(list(range(120)), q=0.99)
    assert hi is None and lo is not None


def test_a_p99_from_3000_samples_is_bracketed_and_covers_the_truth():
    rng = np.random.default_rng(0)
    covered = 0
    for _ in range(200):
        xs = rng.exponential(1.0, 3000)
        lo, hi = stats.quantile_ci(xs, q=0.99)
        covered += lo <= np.log(100) <= hi            # the true p99 of Exp(1)
    assert covered >= 180                              # ~95% nominal


def test_benjamini_hochberg_controls_across_variants():
    assert stats.benjamini_hochberg([0.001, 0.02, 0.03, 0.5]) == [True, True, True, False]
    assert stats.benjamini_hochberg([0.04, 0.5, 0.6, 0.7]) == [False, False, False, False]
