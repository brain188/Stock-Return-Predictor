"""Checks on portfolio construction, optimisation and risk decomposition."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.backtest.momentum import rebalance_schedule
from src.portfolio.construction import (
    apply_constraints,
    equal_weight,
    inverse_volatility,
    rebalance,
    risk_parity,
    risk_parity_weights,
    signal_weighted,
)
from src.portfolio.optimisation import (
    Constraints,
    OptimisationError,
    covariance_matrix,
    efficient_frontier,
    expected_returns,
    frontier_figure,
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
    out_of_sample_frontier,
    portfolio_point,
)
from src.portfolio.risk import (
    beta_to_benchmark,
    diversification_ratio,
    marginal_risk,
    portfolio_volatility,
    risk_contribution,
    rolling_correlation,
    tracking_error,
)

TICKERS = ["A", "B", "C", "D"]


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    """Four correlated assets with volatilities 0.5%, 1%, 1.5% and 2% a day."""
    rng = np.random.default_rng(0)
    n = 1500
    common = rng.normal(0, 1, n)
    vols = np.array([0.005, 0.01, 0.015, 0.02])
    shocks = 0.5 * common[:, None] + np.sqrt(0.75) * rng.normal(0, 1, (n, 4))
    data = 0.0003 + shocks * vols

    return pd.DataFrame(data, index=pd.bdate_range("2010-01-01", periods=n), columns=TICKERS)


# --- Construction -----------------------------------------------------------


def test_equal_weight_gives_one_over_n():
    weights = equal_weight(TICKERS)

    assert weights.tolist() == [0.25] * 4
    assert weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        equal_weight([])


def test_signal_weighted_sums_to_the_leverage_limit(config):
    long_short = dataclasses.replace(
        config, portfolio=dataclasses.replace(config.portfolio, long_only=False)
    )
    signals = pd.Series({"A": 2.0, "B": 1.0, "C": -1.0, "D": 0.0})

    weights = signal_weighted(signals, long_short)
    long_only = signal_weighted(signals, config)

    assert weights.abs().sum() == pytest.approx(config.backtest.max_gross_leverage)
    assert weights.tolist() == pytest.approx([0.5, 0.25, -0.25, 0.0])
    assert long_only.tolist() == pytest.approx([2 / 3, 1 / 3, 0.0, 0.0])


def test_inverse_volatility_favours_the_calm_asset(returns):
    weights = inverse_volatility(returns, 63).dropna()

    assert (weights["A"] > weights["D"]).all()
    assert np.allclose(weights.sum(axis=1), 1.0)
    # 1/vol: A is four times calmer than D, so about four times the weight.
    assert (weights["A"] / weights["D"]).median() == pytest.approx(4, rel=0.15)


def test_inverse_volatility_uses_only_the_past(returns):
    full = inverse_volatility(returns, 63)
    cut = inverse_volatility(returns.iloc[:500], 63)

    pd.testing.assert_frame_equal(full.iloc[:500], cut)


def test_risk_parity_equalises_risk_contributions(returns):
    cov = returns.cov().to_numpy()
    w = risk_parity_weights(cov)
    contributions = w * (cov @ w)

    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()
    assert np.allclose(contributions, contributions.mean(), rtol=1e-6)


def test_risk_parity_refits_only_on_schedule(returns):
    schedule = rebalance_schedule(returns.index, "monthly")

    weights = risk_parity(returns, 252, schedule)
    fitted = weights.dropna(how="all")

    assert set(fitted.index) <= set(returns.index[schedule.to_numpy()])
    assert np.allclose(fitted.sum(axis=1), 1.0)
    assert weights.iloc[:251].isna().all().all()


def test_constraints_are_never_violated(returns, config):
    capped = dataclasses.replace(
        config, backtest=dataclasses.replace(config.backtest, max_position_size=0.3)
    )
    rng = np.random.default_rng(1)
    raw = pd.DataFrame(rng.normal(0, 1, (50, 4)), columns=TICKERS)

    long_only = apply_constraints(raw, capped)
    long_short = apply_constraints(raw * 3, capped, long_only=False)

    assert (long_only >= 0).all().all()
    assert (long_only <= 0.3 + 1e-12).all().all()
    assert (long_short.abs() <= 0.3 + 1e-12).all().all()
    assert (long_short.abs().sum(axis=1) <= config.backtest.max_gross_leverage + 1e-12).all()


def test_capping_keeps_a_long_only_book_fully_invested(config):
    capped = dataclasses.replace(
        config, backtest=dataclasses.replace(config.backtest, max_position_size=0.4)
    )

    weights = apply_constraints(pd.Series({"A": 0.9, "B": 0.05, "C": 0.05}), capped)

    # A is cut to 0.4 and the excess shared between B and C; then B and C are
    # at 0.3 each, under the cap, and the book still sums to 1.
    assert weights.tolist() == pytest.approx([0.4, 0.3, 0.3])


def test_monthly_rebalancing_trades_nothing_in_between(returns, config):
    schedule = rebalance_schedule(returns.index, "monthly")
    targets = rebalance(inverse_volatility(returns, 63), schedule)

    result = run_backtest(targets, returns, config)
    traded_days = result.turnover > 0

    assert not (traded_days & ~schedule.to_numpy()).any()
    assert traded_days.sum() <= schedule.sum()


def test_rebalance_starts_from_the_current_book(returns):
    targets = rebalance(
        inverse_volatility(returns, 63), "monthly", current=pd.Series(0.25, index=TICKERS)
    )

    assert targets.iloc[0].tolist() == [0.25] * 4


def test_an_impossible_cap_leaves_the_rest_in_cash(config):
    capped = dataclasses.replace(
        config, backtest=dataclasses.replace(config.backtest, max_position_size=0.3)
    )

    weights = apply_constraints(pd.Series({"A": 0.6, "B": 0.4, "C": 0.0}), capped)

    assert weights.tolist() == pytest.approx([0.3, 0.3, 0.0])


# --- Optimisation -----------------------------------------------------------


@pytest.fixture(scope="module")
def estimates(returns):
    return expected_returns(returns), covariance_matrix(returns)


def test_minimum_variance_beats_equal_weight_in_sample(estimates):
    _, cov = estimates
    w = minimum_variance_portfolio(cov)
    ew = equal_weight(cov.index)

    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()
    assert portfolio_volatility(w, cov) < portfolio_volatility(ew, cov)


def test_the_frontier_is_monotonic(estimates):
    mu, cov = estimates
    frontier = efficient_frontier(mu, cov, 15)

    assert np.allclose(frontier[list(mu.index)].sum(axis=1), 1.0)
    assert (np.diff(frontier["target_return"]) > 0).all()
    assert (np.diff(frontier["volatility"]) >= -1e-9).all()


def test_maximum_sharpe_sits_on_the_frontier(estimates):
    mu, cov = estimates
    w = maximum_sharpe_portfolio(mu, cov)
    expected, volatility = portfolio_point(w, mu, cov)
    frontier = efficient_frontier(mu, cov, 60)

    # At its own return, the frontier's volatility is the tangency's volatility.
    on_curve = np.interp(expected, frontier["target_return"], frontier["volatility"])
    assert volatility == pytest.approx(on_curve, rel=0.01)
    assert expected / volatility >= frontier["sharpe"].max() - 1e-6


def test_shrunk_covariance_is_better_conditioned(returns):
    short = returns.iloc[:60]  # few observations per asset: a noisy sample

    sample = np.linalg.cond(covariance_matrix(short, "sample"))
    shrunk = np.linalg.cond(covariance_matrix(short, "ledoit_wolf"))

    assert shrunk < sample


def test_shrunk_means_are_pulled_together(returns):
    historical = expected_returns(returns, "historical")
    shrunk = expected_returns(returns, "shrunk")

    assert shrunk.std() < historical.std()
    with pytest.raises(OptimisationError):
        expected_returns(returns, "oracle")


def test_ewma_covariance_is_positive_definite(returns):
    assert (np.linalg.eigvalsh(covariance_matrix(returns, "ewma")) > 0).all()


def test_out_of_sample_frontier_never_touches_the_holdout(returns, config):
    result = out_of_sample_frontier(returns, config, n_points=8)

    assert pd.Timestamp(result.test_period[1]) < pd.Timestamp(config.validation.test_start)
    assert {"realised_return", "realised_volatility"} <= set(result.frontier.columns)
    assert set(result.portfolios.index) == {"equal_weight", "minimum_variance", "maximum_sharpe"}
    late = returns.copy()
    late.index = late.index + pd.DateOffset(years=10)
    with pytest.raises(OptimisationError, match="holdout"):
        out_of_sample_frontier(late, config)


def test_frontier_figure_draws(returns, config):
    result = out_of_sample_frontier(returns, config, n_points=8)

    figure = frontier_figure(result.frontier, result.assets, result.portfolios)

    assert len(figure.axes) == 1
    assert figure.axes[0].get_xlabel() == "Annual volatility (%)"


def test_an_unreachable_budget_is_refused(estimates):
    _, cov = estimates

    with pytest.raises(OptimisationError, match="unreachable"):
        minimum_variance_portfolio(cov, Constraints(max_weight=0.2))


# --- Risk decomposition ---------------------------------------------------------


def test_risk_contributions_sum_to_portfolio_volatility(estimates):
    _, cov = estimates
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=cov.index)

    contributions = risk_contribution(w, cov)

    assert contributions.sum() == pytest.approx(portfolio_volatility(w, cov))
    assert risk_contribution(w, cov, as_share=True).sum() == pytest.approx(1.0)
    # The 10% weight in the most volatile asset carries well over 10% of the risk.
    assert risk_contribution(w, cov, as_share=True)["D"] > 0.1


def test_marginal_risk_matches_a_numerical_derivative(estimates):
    _, cov = estimates
    w = np.array([0.4, 0.3, 0.2, 0.1])
    step = 1e-6
    bumped = np.array(
        [
            (portfolio_volatility(w + step * np.eye(4)[i], cov) - portfolio_volatility(w, cov))
            / step
            for i in range(4)
        ]
    )

    assert np.allclose(marginal_risk(w, cov), bumped, rtol=1e-4)


def test_a_single_asset_has_a_diversification_ratio_of_one(estimates):
    _, cov = estimates

    assert diversification_ratio(pd.Series({"B": 1.0}), cov) == pytest.approx(1.0)


def test_uncorrelated_equal_weight_diversifies_by_root_n():
    cov = np.eye(9) * 0.04

    assert diversification_ratio(np.full(9, 1 / 9), cov) == pytest.approx(3.0)


def test_rolling_correlation_rises_when_assets_move_together():
    rng = np.random.default_rng(2)
    n = 600
    common = rng.normal(0, 1, n)
    together = np.where(np.arange(n) >= 300, 0.9, 0.1)
    data = together[:, None] * common[:, None] + np.sqrt(1 - together[:, None] ** 2) * rng.normal(
        0, 1, (n, 3)
    )
    returns = pd.DataFrame(data, index=pd.bdate_range("2010-01-01", periods=n))

    average = rolling_correlation(returns, 63)

    assert average.iloc[62:300].mean() < 0.2
    assert average.iloc[400:].mean() > 0.6
    assert average.iloc[:62].isna().all()


def test_beta_and_tracking_error():
    rng = np.random.default_rng(3)
    benchmark = pd.Series(rng.normal(0, 0.01, 2000))
    strategy = 1.5 * benchmark + pd.Series(rng.normal(0, 0.002, 2000))

    assert beta_to_benchmark(strategy, benchmark) == pytest.approx(1.5, abs=0.03)
    assert tracking_error(benchmark, benchmark) == 0
    assert tracking_error(strategy, benchmark) > 0
