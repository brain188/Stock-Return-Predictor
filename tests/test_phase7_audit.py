"""Phase 7 audit: hand-calculated checks of timing, accounting and the optimisers.

Each test pins one assumption with a small example whose answer is worked out
by hand in the comments, so a regression shows up as a wrong number rather
than as a vague drift.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.backtest.engine import hold_weights, run_backtest
from src.evaluation.risk_simulation import (
    conditional_var,
    simulate_forward,
    stress_scenarios,
    value_at_risk,
)
from src.pipelines.portfolio import backtest_allocation
from src.portfolio.construction import inverse_volatility, risk_parity_weights
from src.portfolio.optimisation import (
    Constraints,
    OptimisationError,
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
    out_of_sample_frontier,
    valid_covariance,
)
from src.portfolio.risk import portfolio_volatility, risk_contribution

COV = pd.DataFrame([[0.04, 0.006], [0.006, 0.09]], index=["A", "B"], columns=["A", "B"])


# --- A. Timing --------------------------------------------------------------------


def test_weights_set_at_a_rebalance_are_first_held_after_the_next_close(config):
    dates = pd.bdate_range("2019-01-01", periods=6)
    returns = pd.DataFrame(
        {"A": [0.0, 0.01, 0.02, -0.01, 0.03, 0.00], "B": [0.0, -0.02, 0.01, 0.04, 0.00, 0.01]},
        index=dates,
    )
    weights = pd.DataFrame(np.nan, index=dates, columns=["A", "B"])
    weights.iloc[1] = [0.5, 0.5]  # decided at the close of day 1

    result = backtest_allocation("x", weights, returns, config)

    # Decided at close 1, traded at close 2 (one bar of delay), so the first
    # return earned is day 3's: 0.5(-0.01) + 0.5(0.04) = 0.015.
    assert result.weights.loc[dates[1]].tolist() == [0.0, 0.0]
    assert result.weights.loc[dates[2]].tolist() == [0.5, 0.5]
    assert result.gross.loc[dates[2]] == 0.0
    assert result.gross.loc[dates[3]] == pytest.approx(0.015)


def test_inverse_volatility_matches_a_hand_calculation():
    dates = pd.bdate_range("2019-01-01", periods=4)
    returns = pd.DataFrame(
        {"A": [0.01, -0.01, 0.01, -0.01], "B": [0.02, -0.02, 0.02, -0.02]}, index=dates
    )

    weights = inverse_volatility(returns, 4).iloc[-1]

    # B's volatility is twice A's, so A gets 2/3 and B 1/3.
    assert weights.tolist() == pytest.approx([2 / 3, 1 / 3])


# --- B. Frontier boundaries ---------------------------------------------------------


def test_the_frontier_fits_on_training_dates_only(config):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2010-01-01", periods=600)
    returns = pd.DataFrame(rng.normal(0.0004, 0.01, (600, 3)), index=dates, columns=list("ABC"))
    split = dates[300]

    base = out_of_sample_frontier(returns, config, split=split, n_points=6)
    shocked = returns.copy()
    shocked.loc[split:] *= 5  # change only the test window
    moved = out_of_sample_frontier(shocked, config, split=split, n_points=6)

    # Training: dates before the split. Test: the split date onwards.
    assert pd.Timestamp(base.train_period[1]) < split
    assert pd.Timestamp(base.test_period[0]) == split
    # Everything fitted is identical; only the realised columns change.
    fitted = ["target_return", "volatility", "sharpe", "A", "B", "C"]
    pd.testing.assert_frame_equal(base.frontier[fitted], moved.frontier[fitted])
    assert not np.allclose(
        base.frontier["realised_volatility"], moved.frontier["realised_volatility"]
    )


# --- D. Rebalancing, costs and cash ---------------------------------------------------


def test_uninvested_weight_is_cash_earning_nothing(config):
    dates = pd.bdate_range("2019-01-01", periods=3)
    returns = pd.DataFrame({"A": [0.0, 0.10, 0.0], "B": [0.0, -0.05, 0.0]}, index=dates)
    positions = pd.DataFrame({"A": 0.4, "B": 0.2}, index=dates)  # 40% in cash

    result = run_backtest(positions, returns, config)

    # 0.4(0.10) + 0.2(-0.05) = 0.03; the 40% cash adds nothing.
    assert result.gross.iloc[1] == pytest.approx(0.03)
    assert result.costs["borrow"].sum() == 0


def test_a_forced_rebalance_trades_exactly_back_from_the_drifted_weights():
    dates = pd.bdate_range("2019-01-01", periods=2)
    targets = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=dates)
    returns = pd.DataFrame({"A": [0.0, 0.10], "B": [0.0, -0.10]}, index=dates)

    _, untouched = hold_weights(targets, returns)
    weights, forced = hold_weights(targets, returns, pd.Series([False, True], index=dates))

    # After the move the book is A 0.55 and B 0.45 (portfolio return 0). Without
    # a rebalance nothing trades; with one, each leg trades 0.05 back to 0.5.
    assert untouched.iloc[1] == 0
    assert forced.iloc[1] == pytest.approx(0.10)
    assert weights.iloc[1].tolist() == [0.5, 0.5]


# --- E. Optimisation and risk -----------------------------------------------------------


def test_portfolio_volatility_and_risk_contributions_by_hand():
    w = pd.Series({"A": 0.6, "B": 0.4})

    # w'Cov w = 0.36(0.04) + 0.16(0.09) + 2(0.24)(0.006) = 0.03168.
    assert portfolio_volatility(w, COV) == pytest.approx(math.sqrt(0.03168))
    # Cov w = [0.0264, 0.0396]; w_i (Cov w)_i = [0.01584, 0.01584]: equal shares.
    contributions = risk_contribution(w, COV)
    assert contributions.tolist() == pytest.approx([0.01584 / math.sqrt(0.03168)] * 2)
    assert contributions.sum() == pytest.approx(portfolio_volatility(w, COV))


def test_minimum_variance_matches_the_two_asset_formula():
    a, b, ab = 0.04, 0.09, 0.006
    expected = (b - ab) / (a + b - 2 * ab)  # weight in A

    w = minimum_variance_portfolio(COV)

    assert w["A"] == pytest.approx(expected, abs=1e-6)
    assert w.sum() == pytest.approx(1.0)


def test_maximum_sharpe_matches_the_closed_form_when_no_constraint_binds():
    mu = pd.Series({"A": 0.08, "B": 0.12})

    w = maximum_sharpe_portfolio(mu, COV)
    closed = np.linalg.solve(COV.to_numpy(), mu.to_numpy())
    closed = closed / closed.sum()

    assert w.to_numpy() == pytest.approx(closed, abs=1e-5)


def test_maximum_sharpe_beats_a_brute_force_grid_under_constraints():
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]],
        index=list("ABC"),
        columns=list("ABC"),
    )
    mu = pd.Series({"A": 0.05, "B": 0.11, "C": 0.20})
    constraints = Constraints(long_only=True, max_weight=0.5)

    w = maximum_sharpe_portfolio(mu, cov, 0.0, constraints).to_numpy()
    best = -np.inf
    for a, b in itertools.product(np.arange(0, 0.501, 0.01), repeat=2):
        c = 1 - a - b
        if 0 <= c <= 0.5:
            x = np.array([a, b, c])
            best = max(best, x @ mu.to_numpy() / np.sqrt(x @ cov.to_numpy() @ x))

    assert (w >= -1e-9).all() and (w <= 0.5 + 1e-9).all()
    assert w @ mu.to_numpy() / np.sqrt(w @ cov.to_numpy() @ w) >= best - 1e-6


def test_maximum_sharpe_stays_a_valid_portfolio_when_every_mean_is_negative():
    mu = pd.Series({"A": -0.02, "B": -0.05})

    w = maximum_sharpe_portfolio(mu, COV)

    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()


def test_risk_parity_with_two_assets_is_inverse_volatility():
    w = risk_parity_weights(COV.to_numpy())

    vols = np.sqrt(np.diag(COV.to_numpy()))
    assert w == pytest.approx((1 / vols) / (1 / vols).sum(), abs=1e-8)


def test_an_invalid_covariance_is_refused_and_a_ragged_one_symmetrised():
    not_psd = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    ragged = COV.to_numpy() + np.array([[0, 1e-13], [0, 0]])

    with pytest.raises(OptimisationError, match="positive semidefinite"):
        valid_covariance(not_psd)
    with pytest.raises(OptimisationError, match="square"):
        valid_covariance(np.ones((2, 3)))
    fixed = valid_covariance(ragged)
    assert np.array_equal(fixed, fixed.T)


# --- F. Tail risk and crisis windows ------------------------------------------------------


def test_cvar_99_of_a_normal_matches_the_expected_shortfall_formula():
    mu, sigma, level = 0.0005, 0.01, 0.99
    one_day = np.random.default_rng(1).normal(mu, sigma, 2_000_000)
    z = stats.norm.ppf(1 - level)

    expected = -(mu - sigma * stats.norm.pdf(z) / (1 - level))

    assert conditional_var(one_day, level) == pytest.approx(expected, rel=0.01)
    assert value_at_risk(one_day, level) == pytest.approx(-(mu + sigma * z), rel=0.01)


def test_the_normal_horizon_compounds_the_daily_mean():
    returns = pd.Series(np.random.default_rng(2).normal(0.0004, 0.01, 5000))

    paths = simulate_forward(returns, 252, 20_000, "normal", seed=3)
    total = np.prod(1 + paths, axis=1) - 1

    # E[prod(1 + r)] = (1 + mean)^252 for independent days.
    assert total.mean() == pytest.approx((1 + returns.mean()) ** 252 - 1, abs=0.005)


def test_a_crisis_replay_uses_exactly_its_window():
    dates = pd.bdate_range("2011-07-01", "2011-10-31")
    returns = pd.DataFrame({"A": 0.0}, index=dates)
    window = ("2011-08-01", "2011-08-31")
    returns.loc["2011-08-01":"2011-08-31", "A"] = 0.01
    returns.loc[:"2011-07-29", "A"] = -0.5  # outside: must not count
    returns.loc["2011-09-01":, "A"] = -0.5

    table = stress_scenarios(returns, pd.Series({"A": 1.0}), {"x": window})
    days = len(returns.loc[window[0] : window[1]])

    # Both ends are inclusive; the first day's return is the move from the
    # previous close, so the book is held entering the window.
    assert table.loc["x", "days"] == days
    assert table.loc["x", "total_return"] == pytest.approx(1.01**days - 1)


def test_long_only_allocations_never_pay_borrow(config):
    no_short = dataclasses.replace(
        config, backtest=dataclasses.replace(config.backtest, borrow_rate_annual=0.5)
    )
    dates = pd.bdate_range("2019-01-01", periods=5)
    positions = pd.DataFrame({"A": 0.5, "B": 0.5}, index=dates)
    returns = pd.DataFrame(0.001, index=dates, columns=["A", "B"])

    assert run_backtest(positions, returns, no_short).costs["borrow"].sum() == 0
