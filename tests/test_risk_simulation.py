"""Checks on forward risk simulation and stress scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.evaluation.risk_simulation import (
    DEFAULT_SCENARIOS,
    RiskError,
    conditional_var,
    drawdown_distribution,
    fit_student_t,
    risk_summary,
    ruin_probability,
    simulate_forward,
    stress_scenarios,
    time_to_recovery,
    value_at_risk,
)


@pytest.fixture(scope="module")
def fat_tailed() -> pd.Series:
    """Student t returns with 4 degrees of freedom, like daily equities."""
    rng = np.random.default_rng(0)

    return pd.Series(0.0004 + 0.008 * rng.standard_t(4, 5000))


def test_var_95_of_a_normal_matches_the_analytic_value():
    rng = np.random.default_rng(1)
    mu, sigma = 0.0005, 0.01
    one_day = rng.normal(mu, sigma, 400_000)

    expected = -(mu + sigma * stats.norm.ppf(0.05))  # 1.645 sigma below the mean

    assert value_at_risk(one_day, 0.95) == pytest.approx(expected, rel=0.01)


def test_cvar_is_always_worse_than_var(fat_tailed):
    for method in ("normal", "student_t", "bootstrap"):
        paths = simulate_forward(fat_tailed, 21, 2000, method, seed=2)
        for level in (0.9, 0.95, 0.99):
            assert conditional_var(paths, level) > value_at_risk(paths, level), (method, level)


def test_student_t_has_fatter_tails_than_normal(fat_tailed):
    normal = simulate_forward(fat_tailed, 1, 200_000, "normal", seed=3).ravel()
    student = simulate_forward(fat_tailed, 1, 200_000, "student_t", seed=3).ravel()

    assert stats.kurtosis(student) > stats.kurtosis(normal) + 1
    # Further out in the tail the gap grows: the normal understates extreme days.
    assert value_at_risk(student, 0.999) > value_at_risk(normal, 0.999)


def test_the_t_fit_recovers_the_degrees_of_freedom(fat_tailed):
    df, _, _ = fit_student_t(fat_tailed)

    assert 3 < df < 5.5


def test_the_t_keeps_the_sample_volatility(fat_tailed):
    df, loc, scale = fit_student_t(fat_tailed)
    simulated = simulate_forward(fat_tailed, 1, 400_000, "student_t", seed=8).ravel()

    # Fat tails, not extra volatility: the implied variance is the sample's.
    assert scale * np.sqrt(df / (df - 2)) == pytest.approx(fat_tailed.std())
    assert loc == pytest.approx(fat_tailed.mean())
    assert simulated.std() == pytest.approx(fat_tailed.std(), rel=0.05)


def test_extremely_fat_data_is_floored_at_three_degrees_of_freedom():
    rng = np.random.default_rng(9)
    cauchy_like = pd.Series(0.01 * rng.standard_t(1.5, 20_000))

    df, _, _ = fit_student_t(cauchy_like)
    paths = simulate_forward(cauchy_like, 50, 2000, "student_t", seed=10)

    assert df == 3.0
    assert paths.min() >= -0.999


def test_a_fixed_seed_is_reproducible(fat_tailed):
    for method in ("normal", "student_t", "bootstrap"):
        np.testing.assert_array_equal(
            simulate_forward(fat_tailed, 10, 5, method, seed=4),
            simulate_forward(fat_tailed, 10, 5, method, seed=4),
        )


def test_bootstrap_paths_replay_real_days(fat_tailed):
    paths = simulate_forward(fat_tailed, 300, 4, "bootstrap", seed=5)

    assert paths.shape == (4, 300)
    assert np.isin(paths, fat_tailed.to_numpy()).all()


def test_unknown_method_is_refused(fat_tailed):
    with pytest.raises(RiskError, match="method"):
        simulate_forward(fat_tailed, 5, 5, "garch")


def test_drawdowns_recovery_and_ruin_on_known_paths():
    paths = np.array(
        [
            [-0.5, 0.0, 0.0, 0.0],  # halves and stays there: ruin at 50%
            [0.1, -0.15, 0.2, 0.0],  # dips 15% and recovers
            [0.01, 0.01, 0.01, 0.01],  # never falls
        ]
    )

    assert ruin_probability(paths, 0.5) == pytest.approx(1 / 3)
    drawdowns = drawdown_distribution(paths, thresholds=(0.1, 0.5))
    assert drawdowns["p_drawdown_50"] == pytest.approx(1 / 3)
    assert drawdowns["p_drawdown_10"] == pytest.approx(2 / 3)
    recovery = time_to_recovery(paths)
    assert recovery["share_unrecovered"] == pytest.approx(1 / 3)
    assert recovery["median_days"] == 1


def test_risk_summary_has_every_measure(fat_tailed):
    paths = simulate_forward(fat_tailed, 252, 500, "bootstrap", seed=6)

    summary = risk_summary(paths, (0.95, 0.99), 0.5)

    assert {"var_95", "cvar_95", "var_99", "cvar_99", "ruin_probability"} <= set(summary)
    assert summary["var_99"] > summary["var_95"]


# --- Stress scenarios -----------------------------------------------------------


@pytest.fixture
def asset_returns() -> pd.DataFrame:
    dates = pd.bdate_range("2008-01-01", "2019-12-31")
    rng = np.random.default_rng(7)

    return pd.DataFrame(rng.normal(0, 0.01, (len(dates), 2)), index=dates, columns=["A", "B"])


def test_a_scenario_replays_the_window_on_drifting_weights(asset_returns):
    weights = pd.Series({"A": 0.5, "B": 0.5})
    table = stress_scenarios(asset_returns, weights, {"x": ("2010-01-04", "2010-03-31")})
    window = asset_returns.loc["2010-01-04":"2010-03-31"]

    expected = 0.5 * (1 + window["A"]).prod() + 0.5 * (1 + window["B"]).prod() - 1

    assert table.loc["x", "total_return"] == pytest.approx(expected)
    assert table.loc["x", "days"] == len(window)


def test_default_scenarios_all_predate_the_holdout(config):
    for _start, end in DEFAULT_SCENARIOS.values():
        assert pd.Timestamp(end) < pd.Timestamp(config.validation.test_start)


def test_a_scenario_in_the_holdout_is_refused(asset_returns):
    weights = pd.Series({"A": 1.0})
    covid = {"2020 covid": ("2020-02-19", "2020-03-23")}

    with pytest.raises(RiskError, match="holdout"):
        stress_scenarios(asset_returns, weights, covid, holdout_start="2020-01-01")


def test_cash_is_held_for_weights_that_do_not_sum_to_one(asset_returns):
    table = stress_scenarios(
        asset_returns, pd.Series({"A": 0.0}), {"x": ("2010-01-04", "2010-02-26")}
    )

    assert table.loc["x", "total_return"] == pytest.approx(0.0)
