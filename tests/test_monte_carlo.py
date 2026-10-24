"""Checks on Monte Carlo resampling. The clustering test is the one that matters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.monte_carlo import (
    block_bootstrap,
    confidence_interval,
    iid_bootstrap,
    max_drawdowns,
    path_statistics,
    probability_of_beating,
    resample_indices,
    simulate_paths,
    squared_return_autocorrelation,
    stationary_bootstrap,
)


@pytest.fixture(scope="module")
def clustered() -> pd.Series:
    """GARCH(1,1) returns: volatility clusters, as in real markets."""
    rng = np.random.default_rng(0)
    n = 4000
    r, variance = np.zeros(n), np.full(n, 1e-4)
    for t in range(1, n):
        variance[t] = 2e-6 + 0.09 * r[t - 1] ** 2 + 0.89 * variance[t - 1]
        r[t] = np.sqrt(variance[t]) * rng.standard_normal()

    return pd.Series(r + 0.0003, index=pd.bdate_range("2005-01-03", periods=n))


# --- The test that matters most ---------------------------------------------


def test_block_bootstrap_keeps_volatility_clustering_and_iid_destroys_it(clustered):
    original = squared_return_autocorrelation(clustered.to_numpy())
    block = squared_return_autocorrelation(block_bootstrap(clustered, 21, 200, seed=1))
    iid = squared_return_autocorrelation(iid_bootstrap(clustered, 200, seed=1))

    assert original > 0.1
    # Blocks keep most of it (only block seams break it); IID leaves nothing.
    assert block > 0.7 * original
    assert abs(iid) < 0.02


def test_iid_paths_understate_drawdowns(clustered):
    block = max_drawdowns(block_bootstrap(clustered, 21, 500, seed=2))
    iid = max_drawdowns(iid_bootstrap(clustered, 500, seed=2))

    # Clustering stacks bad days together; shuffling spreads them out.
    assert np.median(block) < np.median(iid)


# --- Mechanics ------------------------------------------------------------------


def test_block_size_one_matches_iid(clustered):
    np.testing.assert_array_equal(
        block_bootstrap(clustered, 1, 50, seed=3), iid_bootstrap(clustered, 50, seed=3)
    )


def test_a_fixed_seed_is_reproducible(clustered):
    for method in ("iid", "block", "stationary"):
        first = simulate_paths(clustered, method, n_runs=20, seed=4)
        second = simulate_paths(clustered, method, n_runs=20, seed=4)
        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, simulate_paths(clustered, method, n_runs=20, seed=5))


def test_resampled_mean_is_close_to_the_original(clustered):
    paths = block_bootstrap(clustered, 21, 1000, seed=6)

    # Circular resampling draws every day equally often on average.
    assert paths.mean() == pytest.approx(clustered.mean(), abs=2e-5)


def test_blocks_are_consecutive_days():
    indices = resample_indices(100, "block", 10, seed=0, block_size=5)

    steps = np.diff(indices[:, :5], axis=1) % 100

    assert (steps == 1).all()


def test_stationary_blocks_average_the_mean_length():
    indices = resample_indices(10_000, "stationary", 5, seed=1, block_size=20)
    continued = (np.diff(indices, axis=1) % 10_000) == 1

    # A block ends with probability 1/20 each day, so it continues 95% of the time.
    assert continued.mean() == pytest.approx(0.95, abs=0.01)
    assert stationary_bootstrap(np.arange(100.0), 5, 3, seed=2).shape == (3, 100)


def test_unknown_method_is_refused():
    with pytest.raises(ValueError, match="method"):
        resample_indices(10, "jackknife", 2, seed=0)


def test_several_series_are_resampled_with_the_same_days():
    dates = pd.bdate_range("2020-01-01", periods=50)
    frame = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) * 10}, index=dates)

    paths = simulate_paths(frame, "block", n_runs=7, block_size=5, seed=8)

    assert paths.shape == (7, 50, 2)
    np.testing.assert_array_equal(paths[:, :, 1], paths[:, :, 0] * 10)


def test_defaults_come_from_the_config(clustered, config):
    paths = simulate_paths(clustered.iloc[:300], "block", config)

    assert paths.shape == (config.evaluation.monte_carlo_runs, 300)


# --- Statistics -----------------------------------------------------------------


def test_path_statistics_match_hand_values():
    paths = np.array([[0.1, -0.1, 0.05], [0.0, 0.0, 0.0]])

    stats = path_statistics(paths, days=3)

    assert stats.loc[0, "total_return"] == pytest.approx(1.1 * 0.9 * 1.05 - 1)
    assert stats.loc[0, "max_drawdown"] == pytest.approx(-0.1)
    assert np.isnan(stats.loc[1, "sharpe"])
    assert stats.loc[1, "max_drawdown"] == 0


def test_probability_of_beating_paired_and_against_a_number(clustered):
    strategy = block_bootstrap(clustered + 0.0004, 21, 300, seed=9)
    benchmark = block_bootstrap(clustered, 21, 300, seed=9)

    # The same days, and the strategy earns more on every one of them.
    assert probability_of_beating(strategy, benchmark) == 1.0
    assert probability_of_beating(benchmark, strategy) == 0.0
    assert 0 < probability_of_beating(benchmark, 0.8) < 1


def test_paired_comparison_needs_matching_runs(clustered):
    with pytest.raises(ValueError, match="same number"):
        probability_of_beating(iid_bootstrap(clustered, 10), iid_bootstrap(clustered, 20))


def test_confidence_interval_is_percentile_based():
    low, high = confidence_interval(np.arange(1, 101.0), level=0.9)

    assert low == pytest.approx(np.quantile(np.arange(1, 101.0), 0.05))
    assert high == pytest.approx(np.quantile(np.arange(1, 101.0), 0.95))
    assert np.isnan(confidence_interval([np.nan])[0])
