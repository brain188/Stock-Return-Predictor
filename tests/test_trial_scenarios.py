"""Checks on the trial-count and Sharpe-variance scenarios behind the DSR."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.deflated_sharpe import (
    _sharpe_dispersion,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    to_annual_sharpe,
)
from src.evaluation.notebook_trials import record_notebook_trials
from src.evaluation.trial_log import TrialLog
from src.evaluation.trial_scenarios import (
    ScenarioError,
    candidate_daily_returns,
    candidate_sharpes,
    check_against_log,
    deflate,
    deflation_table,
    effective_trials,
    freeze_record,
    headline,
    path_moments,
    sharpe_sampling_variance,
    trial_count_scenarios,
    trials_where_dsr_falls_below,
    variance_scenarios,
    write_freeze,
)


def log_backtest(log, name, fingerprint, sharpe, kind="backtest"):
    log.log_trial(name, kind=kind, fingerprint=fingerprint, status="ok", metrics={"sharpe": sharpe})


@pytest.fixture
def log(tmp_path):
    """Two strategies, one rerun under a corrected fingerprint, a benchmark and the notebooks."""
    log = TrialLog(tmp_path / "trials.csv")
    log_backtest(log, "ml_long_short", "old-ls", 0.80)
    log_backtest(log, "ml_long_only", "old-lo", 0.70)
    log_backtest(log, "ml_long_short", "new-ls", 0.75)
    log_backtest(log, "ml_long_only", "new-lo", 0.74)
    log_backtest(log, "momentum", "new-mo", 0.05)
    log_backtest(log, "equal_weight_hold", "new-ew", 0.86, kind="benchmark")
    log.log_trial(
        "research_random_forest_regression_fixed_horizon",
        kind="research",
        fingerprint="new-rf",
        status="ok",
        metrics={"ic_mean": 0.004},
    )
    log.log_trial("diagnostic", kind="diagnostic", fingerprint="d", status="ok", metrics={})
    record_notebook_trials(log)

    return log


@pytest.fixture
def comparison():
    """The saved comparison of the latest run."""
    return pd.DataFrame(
        {
            "kind": ["backtest", "backtest", "backtest", "benchmark"],
            "sharpe_net": [0.75, 0.74, 0.05, 0.86],
        },
        index=pd.Index(["ml_long_short", "ml_long_only", "momentum", "equal_weight_hold"]),
    )


def test_count_scenarios_run_from_upper_to_lower_bound(log):
    table = trial_count_scenarios(log)

    # 5 backtest fingerprints + 1 research + 42 notebook; benchmark and diagnostic excluded.
    assert table.loc["all_fingerprints", "n_trials"] == 48
    # Superseded versions merged: 3 strategy names + 1 research + 42.
    assert table.loc["distinct_choices", "n_trials"] == 46
    # Long/short, long only and the research model were all re-run by the pipeline.
    assert table.loc["distinct_less_overlap", "n_trials"] == 43
    assert table.loc["pipeline_only", "n_trials"] == 4
    assert table["n_trials"].is_monotonic_decreasing


def test_overlap_only_counts_counterparts_the_pipeline_has(tmp_path):
    log = TrialLog(tmp_path / "trials.csv")
    record_notebook_trials(log)
    log_backtest(log, "ml_long_only", "lo", 0.7)

    table = trial_count_scenarios(log)

    assert table.loc["distinct_less_overlap", "notebook"] == 41
    assert "notebook04_rule_long_only" in table.loc["distinct_less_overlap", "rule"]


def test_candidates_exclude_benchmarks_and_the_reference_allocation(comparison):
    allocations = pd.DataFrame(
        {"sharpe_net": [0.72, 0.79]}, index=pd.Index(["equal_weight", "minimum_variance"])
    )

    sharpes = candidate_sharpes(comparison, allocations)

    assert list(sharpes.index) == [
        "ml_long_short",
        "ml_long_only",
        "momentum",
        "allocation_minimum_variance",
    ]


def test_variance_uses_the_current_results_and_reports_the_log_wide_figure(log, comparison):
    table = variance_scenarios(comparison, None, log)

    expected = float(np.var([0.75, 0.74, 0.05], ddof=1))
    assert table.loc["strategies", "variance_annual"] == pytest.approx(expected)
    assert table.loc["strategies", "variance_period"] == pytest.approx(expected / 252)
    assert table.loc["strategies", "best"] == "ml_long_short"
    # The log-wide row also holds the superseded 0.80 and 0.70.
    logged = float(np.var([0.80, 0.70, 0.75, 0.74, 0.05], ddof=1))
    assert table.loc["every_backtest_fingerprint", "variance_annual"] == pytest.approx(logged)


def test_saved_results_that_disagree_with_the_log_are_refused(log, comparison):
    comparison.loc["momentum", "sharpe_net"] = 0.5

    with pytest.raises(ScenarioError, match="momentum"):
        check_against_log(candidate_sharpes(comparison), log)

    comparison.loc["momentum", "sharpe_net"] = 0.05
    comparison.loc["unlogged"] = ["backtest", 1.0]
    with pytest.raises(ScenarioError, match="not in the trial log"):
        variance_scenarios(comparison, None, log)


def test_paths_are_averaged_not_pooled():
    rng = np.random.default_rng(0)
    paths = [pd.Series(rng.normal(0.0005, 0.01, 1000)) for _ in range(3)]

    moments = path_moments(paths)

    assert moments["n"] == 1000
    assert moments["paths"] == 3
    sharpes = [p.mean() / p.std() for p in paths]
    assert moments["sharpe_period"] == pytest.approx(np.mean(sharpes))


def test_deflate_matches_the_formulas():
    moments = {"sharpe_period": 0.05, "n": 3500, "skew": -0.3, "kurtosis": 8.0}
    variance = 0.1 / 252

    result = deflate(moments, 50, variance)

    threshold = expected_max_sharpe(50, variance)
    assert result["luck_threshold_annual"] == pytest.approx(to_annual_sharpe(threshold))
    assert result["dsr"] == pytest.approx(
        deflated_sharpe_ratio(0.05, 50, 3500, -0.3, 8.0, variance)
    )
    assert result["psr_vs_zero"] > result["dsr"]


def test_the_crossing_is_where_the_dsr_actually_falls_below_the_level():
    moments = {"sharpe_period": 0.05, "n": 3500, "skew": 0.0, "kurtosis": 3.0}
    variance = 0.1 / 252

    crossing = trials_where_dsr_falls_below(moments, variance, 0.5)

    assert deflate(moments, crossing, variance)["dsr"] < 0.5
    assert deflate(moments, crossing - 1, variance)["dsr"] >= 0.5
    # A strategy far below every threshold is below it from the first trial.
    assert trials_where_dsr_falls_below({**moments, "sharpe_period": -0.05}, variance, 0.5) == 1


def test_deflation_table_has_one_row_per_scenario_pair(log, comparison):
    rng = np.random.default_rng(1)
    returns = {name: [pd.Series(rng.normal(0.0004, 0.01, 800))] for name in comparison.index}
    counts = trial_count_scenarios(log)
    variances = variance_scenarios(comparison, None, log)

    grid = deflation_table(counts, variances, returns.__getitem__)

    # Two variance scenarios here: the current strategies, and the log-wide one.
    assert len(grid) == len(counts) * 2
    assert set(grid["selected"]) == {"ml_long_short"}
    # More trials, higher luck threshold, lower DSR.
    for _, rows in grid.groupby("variance_scenario"):
        ordered = rows.sort_values("n_trials")
        assert ordered["dsr"].is_monotonic_decreasing
        assert not math.isnan(ordered["dsr"].iloc[0])


def test_the_sampling_variance_is_the_psr_denominator():
    moments = {"sharpe_period": 0.05, "n": 1001, "skew": -0.5, "kurtosis": 6.0}

    variance = sharpe_sampling_variance(moments)

    assert variance == pytest.approx(_sharpe_dispersion(0.05, -0.5, 6.0) / 1000)


def test_the_headline_follows_the_preregistered_rules(log, comparison):
    rng = np.random.default_rng(2)
    returns = {name: [pd.Series(rng.normal(0.0004, 0.01, 800))] for name in comparison.index}
    counts = trial_count_scenarios(log)
    variances = variance_scenarios(comparison, None, log)

    result = headline(counts, variances, returns.__getitem__)

    # N: every distinct fingerprint. V: the largest pre-registered option.
    assert result["n_trials"] == counts.loc["all_fingerprints", "n_trials"]
    options = [v for k, v in result.items() if k.startswith("variance_") and k.endswith("_annual")]
    assert result["variance_used_annual"] == pytest.approx(max(options))
    assert result["selected"] == "ml_long_short"


def test_the_selected_sampling_variance_wins_when_it_is_larger(log, comparison):
    returns = {
        name: [pd.Series(np.random.default_rng(3).normal(0, 0.01, 30))] for name in comparison.index
    }
    variances = variance_scenarios(comparison, None, log)

    # Thirty returns: the strategy's own Sharpe is far noisier than the spread across trials.
    result = headline(trial_count_scenarios(log), variances, returns.__getitem__)

    assert result["variance_used"] == "own_sampling_variance"


def test_identical_candidates_count_as_one_and_independent_ones_as_many():
    rng = np.random.default_rng(4)
    base = rng.normal(0, 0.01, 2000)
    copies = pd.DataFrame({f"c{i}": base + rng.normal(0, 1e-4, 2000) for i in range(5)})
    independent = pd.DataFrame({f"i{i}": rng.normal(0, 0.01, 2000) for i in range(5)})

    assert effective_trials(copies)["clusters"] == 1
    assert effective_trials(copies)["participation"] == pytest.approx(1, abs=0.01)
    assert effective_trials(independent)["clusters"] == 5
    assert effective_trials(independent)["participation"] == pytest.approx(5, abs=0.1)


def test_candidate_returns_average_paths_and_drop_idle_candidates():
    dates = pd.bdate_range("2010-01-01", periods=5)
    paths = {
        "a": [
            pd.Series([0.01, 0.02, 0.0, 0.01, 0.0], index=dates),
            pd.Series([0.03, 0.0, 0.0, 0.01, 0.02], index=dates),
        ],
        "idle": [pd.Series(0.0, index=dates)],
    }

    frame = candidate_daily_returns(["a", "idle"], paths.__getitem__)

    assert list(frame.columns) == ["a"]
    assert frame["a"].iloc[0] == pytest.approx(0.02)


def test_a_freeze_is_written_once_and_only_from_committed_code(log, comparison, tmp_path):
    rng = np.random.default_rng(5)
    returns = {name: [pd.Series(rng.normal(0.0004, 0.01, 800))] for name in comparison.index}
    counts = trial_count_scenarios(log)
    variances = variance_scenarios(comparison, None, log)
    record = freeze_record(
        log, counts, variances, headline(counts, variances, returns.__getitem__), "2020-01-01"
    )
    path = tmp_path / "freeze.json"

    assert record["trial_log_rows"] == log.trial_count()
    assert len(record["trials"]) == counts.loc["all_fingerprints", "n_trials"]
    with pytest.raises(ScenarioError, match="commit"):
        write_freeze({**record, "code_version": "abc1234-dirty"}, path)
    write_freeze({**record, "code_version": "abc1234"}, path)
    with pytest.raises(ScenarioError, match="never rewritten"):
        write_freeze({**record, "code_version": "abc1234"}, path)
