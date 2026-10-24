"""Checks on the pairs strategy: finding, hedging, scoring and trading the spread."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from src.backtest.pairs import (
    EG_CRITICAL,
    PairsError,
    engle_granger_test,
    find_pairs,
    half_life,
    hedge_ratio,
    pair_targets,
    pairs_positions,
    pairs_targets,
    spread_zscore,
)

N = 1500
DATES = pd.bdate_range("2010-01-01", periods=N)


def ar1(phi: float, n: int, rng: np.random.Generator, scale: float = 0.02) -> np.ndarray:
    """A stationary AR(1) series."""
    out = np.zeros(n)
    shocks = rng.normal(0, scale, n)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


@pytest.fixture
def cointegrated() -> tuple[pd.Series, pd.Series]:
    """B is a random walk; log A is 1.2 log B plus a mean-reverting spread."""
    rng = np.random.default_rng(0)
    log_b = np.cumsum(rng.normal(0, 0.012, N)) + np.log(50)
    log_a = 0.5 + 1.2 * log_b + ar1(0.9, N, rng)

    return pd.Series(np.exp(log_a), index=DATES, name="A"), pd.Series(
        np.exp(log_b), index=DATES, name="B"
    )


@pytest.fixture
def independent() -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(1)
    walks = np.exp(np.cumsum(rng.normal(0, 0.012, (N, 2)), axis=0)) * 50

    return pd.Series(walks[:, 0], index=DATES), pd.Series(walks[:, 1], index=DATES)


# --- Cointegration ------------------------------------------------------------


def test_a_cointegrated_pair_is_found(cointegrated):
    result = engle_granger_test(*cointegrated)

    assert result.cointegrated
    assert result.beta == pytest.approx(1.2, abs=0.05)


def test_independent_random_walks_are_rejected(independent):
    assert not engle_granger_test(*independent).cointegrated


def test_statistic_matches_statsmodels(cointegrated, independent):
    for a, b in (cointegrated, independent):
        ours = engle_granger_test(a, b, max_lags=5)
        theirs, _, _ = coint(np.log(a), np.log(b), maxlag=5, autolag=None)

        assert ours.statistic == pytest.approx(theirs, rel=1e-6)


def test_critical_values_are_stricter_than_plain_adf():
    # Fitting beta first makes residuals look more stationary than they are.
    assert EG_CRITICAL[0.05] < -2.86
    with pytest.raises(PairsError, match="significance"):
        engle_granger_test(pd.Series(range(1, 200)), pd.Series(range(1, 200)), significance=0.02)


def test_half_life_of_an_ar1_spread():
    rng = np.random.default_rng(2)
    spread = pd.Series(ar1(0.9, 20_000, rng))

    assert half_life(spread) == pytest.approx(math.log(0.5) / math.log(0.9), rel=0.1)


def test_a_random_walk_never_halves():
    walk = pd.Series(np.cumsum(np.random.default_rng(3).normal(size=5000)))

    assert half_life(walk) > 200


# --- Hedge ratio and z score ----------------------------------------------------


def test_hedge_ratio_never_uses_future_data(cointegrated):
    a, b = cointegrated
    full = hedge_ratio(a, b, 252)
    cut = hedge_ratio(a.iloc[:800], b.iloc[:800], 252)

    pd.testing.assert_series_equal(full.iloc[:800], cut)
    assert full.iloc[:251].isna().all()
    assert full.iloc[300:].mean() == pytest.approx(1.2, abs=0.15)


def test_zscore_of_a_stationary_spread_is_standardised(cointegrated):
    a, b = cointegrated
    z = spread_zscore(a, b, hedge_ratio(a, b, 252), 63, hedge_window=252).dropna()

    assert abs(z.mean()) < 0.2
    assert 0.8 < z.std() < 1.2


# --- Trading rules -----------------------------------------------------------------


def test_entry_above_2_and_exit_below_half_fire_on_a_diverging_spread():
    z = pd.Series([0.0, 1.0, 2.5, 3.0, 1.5, 0.3, 0.0, -2.2, -1.0, -0.4])

    position = pairs_positions(z, entry_z=2.0, exit_z=0.5, stop_z=4.0)

    # Short the spread when it is stretched high, long when stretched low.
    assert position.tolist() == [0, 0, -1, -1, -1, 0, 0, 1, 1, 0]


def test_a_stop_closes_the_trade_and_waits_for_the_spread_to_settle():
    z = pd.Series([2.5, 3.5, 4.5, 3.0, 2.5, 1.0, 0.2, 2.5])

    position = pairs_positions(z, 2.0, 0.5, 4.0)

    # Stopped at 4.5; no re-entry at 3.0 or 2.5 until z is back inside 0.5.
    assert position.tolist() == [-1, -1, 0, 0, 0, 0, 0, -1]


def test_crossing_through_zero_exits():
    z = pd.Series([-2.5, -1.0, 0.8])

    assert pairs_positions(z, 2.0, 0.5, 4.0).tolist() == [1, 1, 0]


def test_missing_z_means_flat():
    z = pd.Series([np.nan, 2.5, np.nan, 2.5])

    assert pairs_positions(z, 2.0, 0.5, 4.0).tolist() == [0, -1, 0, -1]


def test_leg_weights_hedge_and_respect_the_budget(cointegrated, config):
    a, b = cointegrated
    prices = pd.DataFrame({"A": a, "B": b})

    legs = pair_targets(prices, "A", "B", config).dropna()
    active = legs[legs["A"] != 0]

    assert len(active) > 0
    assert (legs.abs().sum(axis=1) <= 1 + 1e-12).all()
    # Long the spread means long A and short B, and the other way round.
    assert (np.sign(active["A"]) == -np.sign(active["B"])).all()


# --- Selection ------------------------------------------------------------------------


def test_find_pairs_selects_only_the_cointegrated_pair(cointegrated, independent, config):
    prices = pd.DataFrame(
        {
            "A": cointegrated[0],
            "B": cointegrated[1],
            "X": independent[0],
            "Y": independent[1],
        }
    )

    table = find_pairs(prices, config)

    # Only pairs that pass the correlation screen are tested at all.
    assert set(zip(table["a"], table["b"], strict=True)) == {("A", "B")}
    assert table["selected"].tolist() == [True]


def test_find_pairs_uses_only_the_given_dates(cointegrated, config):
    prices = pd.DataFrame({"A": cointegrated[0], "B": cointegrated[1]})
    early = find_pairs(prices, config, dates=DATES[:600])
    late = find_pairs(prices.iloc[:600], config)

    pd.testing.assert_frame_equal(early, late)


def test_book_of_pairs_splits_the_budget(cointegrated, config):
    a, b = cointegrated
    prices = pd.DataFrame({"A": a, "B": b, "C": a * 1.01, "D": b * 0.99})

    combined, legs = pairs_targets(prices, [("A", "B"), ("C", "D")], config)

    assert set(legs) == {"A_B", "C_D"}
    assert (combined.abs().sum(axis=1) <= 1 + 1e-12).all()
    assert combined["A"].abs().max() <= 0.5
    pd.testing.assert_series_equal(combined["A"], legs["A_B"]["A"], check_names=False)


def test_no_pair_passing_the_screen_means_no_trades(independent, config):
    prices = pd.DataFrame({"X": independent[0], "Y": independent[1]})

    table = find_pairs(prices, config)
    combined, legs = pairs_targets(prices, [], config)

    assert table.empty
    assert table["selected"].dtype == bool
    assert legs == {}
    assert (combined == 0).all().all()
