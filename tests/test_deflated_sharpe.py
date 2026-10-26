"""Checks on the PSR, the expected maximum Sharpe and the DSR.

The worked example is the numerical example in Bailey and López de Prado
(2014), "The Deflated Sharpe Ratio", Journal of Portfolio Management 40(5):
an annualised Sharpe of 2.5 over five years of daily returns (T = 1250 at 250
a year), skew -3, raw kurtosis 10, N = 100 trials with annualised Sharpe
variance 1/2. The paper reports a threshold of 0.1132 (per day), a DSR of
about 0.90 ("a 90% chance"), 0.9505 had the search stopped at N = 46, and,
for normal returns, DSR crossing 0.95 at N = 88.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from src.evaluation.deflated_sharpe import (
    SharpeError,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    luck_threshold,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    return_moments,
    to_annual_sharpe,
    to_period_sharpe,
)

# The paper's example, converted to daily units at 250 days a year.
PAPER_SR = to_period_sharpe(2.5, 250)
PAPER_VARIANCE = 0.5 / 250
PAPER_T = 1250


# --- The published worked example --------------------------------------------------


def test_the_threshold_of_the_worked_example():
    assert expected_max_sharpe(100, PAPER_VARIANCE) == pytest.approx(0.1132, abs=5e-5)


def test_the_dsr_of_the_worked_example():
    dsr = deflated_sharpe_ratio(PAPER_SR, 100, PAPER_T, -3, 10, PAPER_VARIANCE)

    assert dsr == pytest.approx(0.9004, abs=5e-4)


def test_fewer_trials_would_have_cleared_95_percent():
    dsr = deflated_sharpe_ratio(PAPER_SR, 46, PAPER_T, -3, 10, PAPER_VARIANCE)

    assert dsr == pytest.approx(0.9505, abs=5e-4)


def test_normal_returns_cross_95_percent_at_88_trials():
    at = {n: deflated_sharpe_ratio(PAPER_SR, n, PAPER_T, 0, 3, PAPER_VARIANCE) for n in (88, 89)}

    assert at[88] >= 0.95 > at[89]


# --- Formulae --------------------------------------------------------------------------


def test_psr_matches_the_formula_by_hand():
    sr, n, skew, kurt, bench = 0.08, 500, -0.5, 6.0, 0.02
    denominator = math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    expected = stats.norm.cdf((sr - bench) * math.sqrt(n - 1) / denominator)

    assert probabilistic_sharpe_ratio(sr, n, skew, kurt, bench) == pytest.approx(expected)


def test_normal_returns_reduce_to_lo():
    sr, n = 0.05, 1000

    # Lo (2002): standard error sqrt((1 + SR^2 / 2) / (n - 1)).
    expected = stats.norm.cdf(sr / math.sqrt((1 + sr**2 / 2) / (n - 1)))

    assert probabilistic_sharpe_ratio(sr, n) == pytest.approx(expected)


def test_luck_threshold_is_the_expected_maximum():
    assert luck_threshold(42, 0.003) == expected_max_sharpe(42, 0.003)


def test_expected_maximum_matches_a_simulation():
    # The maximum of N independent zero-mean normals with variance V.
    rng = np.random.default_rng(0)
    simulated = rng.normal(0, math.sqrt(0.004), (20_000, 50)).max(axis=1).mean()

    assert expected_max_sharpe(50, 0.004) == pytest.approx(simulated, rel=0.03)


def test_minimum_track_record_is_where_psr_reaches_the_confidence():
    sr, skew, kurt, target = 0.06, -1.0, 8.0, 0.01
    length = minimum_track_record_length(sr, skew, kurt, target, 0.95)

    assert probabilistic_sharpe_ratio(sr, math.ceil(length), skew, kurt, target) >= 0.95
    assert probabilistic_sharpe_ratio(sr, math.floor(length) - 1, skew, kurt, target) < 0.95


def test_unit_conversions_round_trip():
    assert to_annual_sharpe(to_period_sharpe(0.85)) == pytest.approx(0.85)
    assert to_period_sharpe(math.sqrt(252)) == pytest.approx(1.0)


# --- Monotonicity -------------------------------------------------------------------------


def test_psr_rises_with_sample_size():
    values = [probabilistic_sharpe_ratio(0.04, n, -1, 6) for n in (100, 500, 2000, 10_000)]

    assert values == sorted(values) and values[0] < values[-1]


def test_psr_falls_with_negative_skew():
    values = [probabilistic_sharpe_ratio(0.08, 1000, skew, 6) for skew in (1.0, 0.0, -1.0, -3.0)]

    assert values == sorted(values, reverse=True) and values[0] > values[-1]


def test_psr_falls_with_fatter_tails():
    assert probabilistic_sharpe_ratio(0.08, 1000, 0, 12) < probabilistic_sharpe_ratio(
        0.08, 1000, 0, 3
    )


def test_expected_maximum_rises_with_trials():
    values = [expected_max_sharpe(n, 0.002) for n in (2, 10, 42, 100, 1000)]

    assert values == sorted(values) and values[0] < values[-1]


def test_a_sharpe_below_the_luck_threshold_gives_a_low_dsr():
    threshold = luck_threshold(100, PAPER_VARIANCE)

    at = deflated_sharpe_ratio(threshold, 100, PAPER_T, 0, 3, PAPER_VARIANCE)
    below = deflated_sharpe_ratio(threshold / 2, 100, PAPER_T, 0, 3, PAPER_VARIANCE)

    assert at == pytest.approx(0.5)
    assert below < 0.05


# --- Edge cases -----------------------------------------------------------------------------


def test_one_trial_or_no_spread_means_no_luck_to_correct():
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(50, 0.0) == 0.0
    assert deflated_sharpe_ratio(0.05, 1, 500, 0, 3, 0.01) == probabilistic_sharpe_ratio(0.05, 500)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0.05, 1), "2 observations"),
        ((0.05, 100, 0.0, 0.5), "kurtosis"),
        ((math.nan, 100), "finite"),
        ((2.0, 100, 3.0, 3.0), "not positive"),  # 1 - 6 + 2 = -3
    ],
)
def test_invalid_psr_inputs_are_refused(arguments, message):
    with pytest.raises(SharpeError, match=message):
        probabilistic_sharpe_ratio(*arguments)


def test_invalid_trial_inputs_are_refused():
    with pytest.raises(SharpeError):
        expected_max_sharpe(0, 0.01)
    with pytest.raises(SharpeError):
        expected_max_sharpe(10, -0.01)
    with pytest.raises(SharpeError):
        minimum_track_record_length(0.05, confidence=1.5)


def test_no_edge_needs_an_infinite_track_record():
    assert minimum_track_record_length(0.01, target=0.02) == math.inf


def test_return_moments_use_raw_kurtosis():
    rng = np.random.default_rng(1)
    normal = rng.normal(0.001, 0.01, 200_000)

    sharpe, n, skew, kurtosis = return_moments(normal)

    assert n == 200_000
    assert sharpe == pytest.approx(0.1, abs=0.01)
    assert skew == pytest.approx(0, abs=0.03)
    assert kurtosis == pytest.approx(3, abs=0.05)
    with pytest.raises(SharpeError):
        return_moments(np.ones(10))
