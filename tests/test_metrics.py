"""Checks on the forecast metrics against hand-computed values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.metrics import (
    cross_sectional_ic,
    cross_sectional_score,
    cross_sectional_summary,
    directional_accuracy,
    edge_over_always_up,
    hit_rate,
    ic_by_period,
    ic_t_stat,
    information_coefficient,
    mae,
    metric_table,
    r2,
    rmse,
    score,
    up_rate,
)


@pytest.fixture
def returns() -> np.ndarray:
    return np.random.default_rng(0).normal(0.0005, 0.01, 5000)


# --- Information coefficient ------------------------------------------------


def test_perfect_prediction_gives_ic_of_one(returns):
    assert information_coefficient(returns, returns) == pytest.approx(1.0)


def test_any_monotone_transform_also_gives_one(returns):
    # Rank correlation: only the ordering matters.
    assert information_coefficient(np.exp(returns * 50), returns) == pytest.approx(1.0)


def test_inverted_prediction_gives_ic_of_minus_one(returns):
    assert information_coefficient(-returns, returns) == pytest.approx(-1.0)


def test_random_prediction_gives_ic_near_zero(returns):
    noise = np.random.default_rng(1).normal(size=len(returns))

    # Under no relationship the IC has a standard error of about 1 / sqrt(n).
    assert abs(information_coefficient(noise, returns)) < 4 / np.sqrt(len(returns))


def test_ic_is_nan_for_a_constant_forecast(returns):
    assert np.isnan(information_coefficient(np.ones_like(returns), returns))


def test_missing_rows_are_ignored():
    predicted = np.array([1.0, 2.0, np.nan, 4.0])
    actual = np.array([0.1, 0.2, 0.3, np.nan])

    assert information_coefficient(predicted, actual) == pytest.approx(1.0)
    assert score(predicted, actual)["n"] == 2


def test_series_are_aligned_by_index():
    predicted = pd.Series([3.0, 2.0, 1.0], index=["c", "b", "a"])
    actual = pd.Series([0.1, 0.2, 0.3], index=["a", "b", "c"])

    assert information_coefficient(predicted, actual) == pytest.approx(1.0)


def test_ic_by_period_splits_by_year():
    dates = pd.date_range("2010-01-01", "2012-12-31", freq="B")
    actual = pd.Series(np.random.default_rng(2).normal(size=len(dates)), index=dates)
    # Right in 2010 and 2012, inverted in 2011.
    sign = np.where(dates.year == 2011, -1.0, 1.0)
    predicted = actual * sign

    table = ic_by_period(predicted, actual, "YE")

    assert list(table.index) == ["2010", "2011", "2012"]
    assert table["ic"].round(6).tolist() == [1.0, -1.0, 1.0]
    assert table["n"].sum() == len(dates)


def test_ic_t_stat():
    table = pd.DataFrame({"ic": [0.02, 0.04, 0.03, 0.05]})
    expected = 0.035 / (table["ic"].std() / 2)

    assert ic_t_stat(table) == pytest.approx(expected)


# --- Direction --------------------------------------------------------------

PREDICTED = np.array([0.5, -0.2, 0.1, -0.3, 0.0, 0.4])
ACTUAL = np.array([0.01, 0.02, -0.01, -0.03, -0.02, 0.0])
# Up?        pred: T F T F F T     actual: T T F F F F
# Matches:         T F F T T F  -> 3 of 6


def test_hit_rate_matches_a_hand_count():
    assert hit_rate(PREDICTED, ACTUAL) == pytest.approx(3 / 6)


def test_zero_counts_as_not_up_on_both_sides():
    assert hit_rate(np.array([0.0]), np.array([-0.01])) == 1.0
    assert hit_rate(np.array([0.01]), np.array([0.0])) == 0.0


def test_edge_over_always_up_subtracts_the_up_rate():
    # Up rate 2 / 6.
    assert up_rate(ACTUAL) == pytest.approx(2 / 6)
    assert edge_over_always_up(PREDICTED, ACTUAL) == pytest.approx(3 / 6 - 2 / 6)


def test_always_up_has_zero_edge(returns):
    assert edge_over_always_up(np.ones_like(returns), returns) == pytest.approx(0.0)


def test_directional_accuracy_matches_a_hand_count():
    # Up days (0, 1): 1 of 2 called up. Down days (2 to 5): 2 of 4 called down.
    assert directional_accuracy(PREDICTED, ACTUAL) == pytest.approx((1 / 2 + 2 / 4) / 2)
    # Shift one down day to a correct call and only the down-day half moves.
    better = PREDICTED.copy()
    better[2] = -0.1
    assert directional_accuracy(better, ACTUAL) == pytest.approx((1 / 2 + 3 / 4) / 2)


def test_directional_accuracy_ignores_the_drift(returns):
    # The point of the metric: leaning long earns nothing.
    assert directional_accuracy(np.ones_like(returns), returns) == pytest.approx(0.5)
    assert hit_rate(np.ones_like(returns), returns) > 0.5


def test_directions_and_returns_give_the_same_hit_rate(returns):
    forecast = np.random.default_rng(3).normal(size=len(returns))

    assert hit_rate(forecast, returns) == hit_rate(np.where(forecast > 0, 1.0, -1.0), returns)


# --- Regression view --------------------------------------------------------


def test_error_metrics_match_hand_values():
    predicted = np.array([0.01, 0.02, -0.01])
    actual = np.array([0.02, 0.0, -0.01])

    assert mae(predicted, actual) == pytest.approx(0.01)
    assert rmse(predicted, actual) == pytest.approx(np.sqrt(0.0005 / 3))
    assert r2(predicted, actual) == pytest.approx(1 - 0.0005 / 0.0005)


def test_r2_is_zero_for_a_zero_forecast_and_one_for_perfect(returns):
    assert r2(np.zeros_like(returns), returns) == pytest.approx(0.0)
    assert r2(returns, returns) == pytest.approx(1.0)


# --- Table ------------------------------------------------------------------


def test_metric_table_compares_model_and_baselines(returns):
    rng = np.random.default_rng(4)
    model = returns + rng.normal(0, 0.01, len(returns))
    table = metric_table(
        model,
        returns,
        {"always_up": np.ones_like(returns), "random": rng.choice([-1.0, 1.0], len(returns))},
    )

    assert list(table.index) == ["model", "always_up", "random"]
    assert table.loc["always_up", "edge_over_always_up"] == pytest.approx(0.0)
    assert table.loc["always_up", "hit_rate"] == pytest.approx(table.loc["model", "up_rate"])
    # Error metrics only make sense for return forecasts.
    assert np.isnan(table.loc["always_up", "rmse"])
    assert not np.isnan(table.loc["model", "rmse"])
    assert table.loc["model", "ic"] > 0.5


# --- Cross-sectional IC -----------------------------------------------------


def panel_returns(n_dates: int = 250, n_tickers: int = 11, seed: int = 5) -> pd.Series:
    """Random returns on a (date, ticker) index."""
    dates = pd.date_range("2015-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    return pd.Series(np.random.default_rng(seed).normal(0, 0.01, len(index)), index=index)


@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_cross_sectional_ic_is_one_for_perfect_and_minus_one_for_inverted(method):
    actual = panel_returns()

    assert (cross_sectional_ic(actual, actual, method)["ic"].round(10) == 1).all()
    assert (cross_sectional_ic(-actual, actual, method)["ic"].round(10) == -1).all()


def test_cross_sectional_ic_ignores_a_different_level_each_day():
    # The case the pooled IC gets wrong: perfect ranking within each day, plus
    # a daily offset unrelated to the returns.
    actual = panel_returns()
    dates = actual.index.get_level_values("date")
    offset = pd.Series(np.random.default_rng(6).normal(0, 1, dates.nunique()), index=dates.unique())
    predicted = actual + offset.reindex(dates).to_numpy()

    daily = cross_sectional_ic(predicted, actual)

    assert (daily["ic"].round(10) == 1).all()
    assert abs(information_coefficient(predicted, actual)) < 0.2


def test_cross_sectional_spearman_matches_scipy_on_one_date():
    actual = panel_returns()
    predicted = actual + np.random.default_rng(7).normal(0, 0.01, len(actual))
    date = actual.index.get_level_values("date")[0]

    expected = information_coefficient(predicted.xs(date), actual.xs(date))

    assert cross_sectional_ic(predicted, actual).loc[date, "ic"] == pytest.approx(expected)


def test_cross_sectional_pearson_and_spearman_differ_with_an_outlier():
    actual = panel_returns(n_dates=1)
    predicted = actual.copy()
    predicted.iloc[0] = 10.0  # Same rank as before if it was already the largest
    predicted.iloc[0] = -10.0  # Now an extreme value at the bottom

    spearman = cross_sectional_ic(predicted, actual, "spearman")["ic"].iloc[0]
    pearson = cross_sectional_ic(predicted, actual, "pearson")["ic"].iloc[0]

    assert spearman != pytest.approx(pearson)


def test_dates_with_too_few_tickers_are_skipped_and_counted():
    actual = panel_returns(n_dates=3)
    predicted = actual + np.random.default_rng(8).normal(0, 0.01, len(actual))
    dates = actual.index.get_level_values("date").unique()
    # Date 0: 4 valid tickers. Date 1: 5 valid. Date 2: all 11.
    predicted.loc[dates[0]] = np.r_[predicted.loc[dates[0]].to_numpy()[:4], [np.nan] * 7]
    actual.loc[dates[1]] = np.r_[actual.loc[dates[1]].to_numpy()[:5], [np.nan] * 6]

    daily = cross_sectional_ic(predicted, actual, min_tickers=5)

    assert daily["n_tickers"].tolist() == [4, 5, 11]
    assert np.isnan(daily["ic"].iloc[0])
    assert daily["ic"].iloc[1:].notna().all()


def test_a_constant_forecast_on_a_date_is_skipped():
    actual = panel_returns(n_dates=2)
    predicted = actual.copy()
    first = actual.index.get_level_values("date")[0]
    predicted.loc[first] = 1.0

    daily = cross_sectional_ic(predicted, actual)

    assert np.isnan(daily["ic"].iloc[0])
    assert daily["ic"].iloc[1] == pytest.approx(1.0)


def test_cross_sectional_ic_needs_a_ticker_level():
    series = pd.Series([1.0, 2.0], index=pd.date_range("2020-01-01", periods=2))

    with pytest.raises(ValueError, match="ticker"):
        cross_sectional_ic(series, series)
    with pytest.raises(ValueError, match="method"):
        cross_sectional_ic(panel_returns(2), panel_returns(2), method="kendall")


def test_cross_sectional_summary_matches_hand_values():
    daily = pd.DataFrame(
        {"ic": [0.1, -0.05, 0.2, np.nan], "n_tickers": [11, 10, 9, 3]},
        index=pd.date_range("2020-01-01", periods=4),
    )
    ic = pd.Series([0.1, -0.05, 0.2])

    summary = cross_sectional_summary(daily)

    assert summary["mean"] == pytest.approx(ic.mean())
    assert summary["std"] == pytest.approx(ic.std())
    assert summary["t_stat"] == pytest.approx(ic.mean() / (ic.std() / np.sqrt(3)))
    assert summary["positive_share"] == pytest.approx(2 / 3)
    assert summary["dates"] == 3
    assert summary["skipped"] == 1
    assert summary["mean_tickers"] == pytest.approx(10.0)


def test_random_forecast_has_cross_sectional_ic_near_zero():
    actual = panel_returns(n_dates=2000)
    noise = pd.Series(np.random.default_rng(9).normal(size=len(actual)), index=actual.index)

    summary = cross_sectional_summary(cross_sectional_ic(noise, actual))

    # Each daily IC has a standard error near 1 / sqrt(10); the mean over 2000 days
    # has about 1 / sqrt(20000).
    assert abs(summary["mean"]) < 4 / np.sqrt(20_000)
    assert abs(summary["t_stat"]) < 4


def test_cross_sectional_score_has_both_methods():
    actual = panel_returns()
    result = cross_sectional_score(actual, actual)

    assert result["cs_spearman_mean"] == pytest.approx(1.0)
    assert result["cs_pearson_mean"] == pytest.approx(1.0)
    assert result["cs_dates"] == 250
    assert result["cs_skipped"] == 0
    assert result["cs_mean_tickers"] == 11
