"""Checks on fractional differentiation.

d=0 and d=1 are the two cases known by hand: the original series and its first
difference. Everything in between is trusted because the ends are exact.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import binom
from statsmodels.tsa.stattools import adfuller

from src.features.fracdiff import (
    FracDiffError,
    adf_statistic,
    critical_value,
    ffd_search,
    frac_diff,
    fracdiff_weights,
    min_ffd_order,
)


@pytest.fixture
def log_price() -> pd.Series:
    """A random walk shaped like 20 years of daily log prices."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2005-01-03", periods=5_000, freq="B")

    return pd.Series(np.cumsum(rng.normal(0.0003, 0.012, 5_000)), index=dates, name="RW")


def test_weights_d0_and_d1():
    """d=0 is the identity, d=1 is the first difference operator."""
    np.testing.assert_array_equal(fracdiff_weights(0.0), [1.0])
    np.testing.assert_array_equal(fracdiff_weights(1.0), [1.0, -1.0])


@pytest.mark.parametrize("d", [0.2, 0.45, 0.7])
def test_weights_match_binomial_expansion(d):
    """w_k = (-1)^k * C(d, k), the coefficients of (1 - B)^d."""
    weights = fracdiff_weights(d)
    expected = [(-1) ** k * binom(d, k) for k in range(len(weights))]

    np.testing.assert_allclose(weights, expected, rtol=1e-10)


@pytest.mark.parametrize("d", [0.2, 0.45, 0.7])
def test_weights_sum_to_expected_value(d):
    """The truncated sum equals (-1)^K * C(d - 1, K), the closed form for K terms."""
    weights = fracdiff_weights(d)
    last = len(weights) - 1

    assert weights.sum() == pytest.approx((-1) ** last * binom(d - 1, last), abs=1e-12)


def test_weights_are_truncated_at_threshold():
    """Every kept weight is at least the threshold in size."""
    weights = fracdiff_weights(0.4, threshold=1e-3)

    assert np.all(np.abs(weights) >= 1e-3)
    assert len(weights) < len(fracdiff_weights(0.4, threshold=1e-5))


def test_negative_order_raises():
    """A negative order would integrate, not difference."""
    with pytest.raises(FracDiffError):
        fracdiff_weights(-0.1)


def test_d0_returns_the_series_exactly(log_price):
    """No differencing gives back the input unchanged."""
    pd.testing.assert_series_equal(frac_diff(log_price, 0.0), log_price)


def test_d1_returns_the_first_difference_exactly(log_price):
    """Differencing of order one is the plain first difference."""
    pd.testing.assert_series_equal(frac_diff(log_price, 1.0), log_price.diff())


def test_frac_diff_keeps_index_and_warmup_is_nan(log_price):
    """Same index as the input, NaN (never 0) until the window is full."""
    out = frac_diff(log_price, 0.4)
    width = len(fracdiff_weights(0.4))

    assert out.index.equals(log_price.index)
    assert out.iloc[: width - 1].isna().all()
    assert out.iloc[width - 1 :].notna().all()


def test_frac_diff_uses_no_future_data(log_price):
    """Changing a future value leaves every earlier output untouched."""
    changed = log_price.copy()
    changed.iloc[3_000] += 5.0

    before = frac_diff(log_price, 0.4)
    after = frac_diff(changed, 0.4)

    pd.testing.assert_series_equal(before.iloc[:3_000], after.iloc[:3_000])
    assert before.iloc[3_000] != after.iloc[3_000]


# statsmodels warns that its return type will change; only the statistic is used.
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("lags", [0, 1, 5, 17])
def test_adf_matches_statsmodels(log_price, lags):
    """The hand rolled regression gives statsmodels' statistic at a fixed lag."""
    reference = adfuller(log_price, maxlag=lags, regression="c", autolag=None)[0]

    assert adf_statistic(log_price, lags) == pytest.approx(reference, rel=1e-8)


def test_adf_rejects_too_short_series():
    """A handful of points cannot support a regression with many lags."""
    with pytest.raises(FracDiffError, match="too short"):
        adf_statistic(np.arange(10.0), max_lags=5)


def test_critical_values():
    """The three MacKinnon values, and an error for any other level."""
    assert critical_value(0.95) == -2.86
    assert critical_value(0.99) == -3.43
    assert critical_value(0.90) == -2.57

    with pytest.raises(FracDiffError):
        critical_value(0.80)


def test_random_walk_needs_positive_d(log_price):
    """A price-like random walk is not stationary until differenced."""
    d_star = min_ffd_order(log_price, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    assert d_star > 0


def test_stationary_series_returns_zero():
    """White noise is stationary already, so no differencing is needed."""
    noise = pd.Series(np.random.default_rng(3).normal(size=3_000), name="noise")

    assert min_ffd_order(noise, [0.0, 0.5, 1.0]) == 0


def test_search_reports_memory(log_price):
    """Lower d keeps more memory: correlation with the original falls as d rises."""
    table = ffd_search(log_price, [0.0, 0.3, 0.6, 1.0])

    assert table["correlation"].iloc[0] == pytest.approx(1.0)
    assert table["correlation"].is_monotonic_decreasing


def test_min_order_uses_config_grid(config, log_price):
    """With a config, the search grid and threshold come from it."""
    d_star = min_ffd_order(log_price, config=config)

    assert d_star in config.features.fracdiff.search_grid


def test_no_passing_order_gives_nan(log_price):
    """When nothing in the grid passes, the answer is NaN rather than a guess."""
    assert math.isnan(min_ffd_order(log_price, [0.0]))
