"""Checks on the evaluation pipeline and its command line entry point.

The pipeline runs end to end on a small synthetic panel that contains one
cointegrated pair, so every kind of strategy (ML, momentum, pairs, the
benchmarks) has something to do.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from entrypoint import evaluate as cli
from src.config import PROJECT_ROOT
from src.evaluation.trial_log import TrialLog
from src.pipelines.evaluate import (
    DEFAULT_STRATEGIES,
    SAME_CLOSE,
    build_context,
    run_evaluation,
)

TICKERS = ["SPY", "AAA", "BBB", "CCC", "DDD", "EEE"]
N = 1000


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """Six synthetic tickers; AAA and BBB are cointegrated."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2005-01-03", periods=N)
    walks = np.cumsum(rng.normal(0.0003, 0.012, (N, len(TICKERS))), axis=0)
    spread = np.zeros(N)
    for t in range(1, N):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.01)
    walks[:, 1] = 0.2 + walks[:, 2] + spread

    frames = {}
    for i, ticker in enumerate(TICKERS):
        price = 50 * np.exp(walks[:, i])
        frames[ticker] = pd.DataFrame(
            {
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "adj_close": price,
                "volume": 1e6,
                "return": pd.Series(price).pct_change().to_numpy(),
            },
            index=pd.DatetimeIndex(dates, name="date"),
        )

    return pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()


def shrink(config, log_path):
    """The synthetic universe, a small forest, 1-and-1 momentum, a given trial log."""
    forest = {**config.model.random_forest, "n_estimators": 10, "n_jobs": 1}
    momentum = dataclasses.replace(config.strategies.momentum, top_k=1, bottom_k=1)

    return dataclasses.replace(
        config,
        universe=dataclasses.replace(config.universe, tickers=TICKERS, benchmark="SPY"),
        validation=dataclasses.replace(
            config.validation, test_start=pd.Timestamp("2008-06-01").date()
        ),
        model=dataclasses.replace(config.model, random_forest=forest),
        strategies=dataclasses.replace(config.strategies, momentum=momentum),
        evaluation=dataclasses.replace(config.evaluation, trial_log=log_path),
    )


@pytest.fixture
def small_config(config, tmp_path):
    return shrink(config, tmp_path / "trials.csv")


@pytest.fixture(scope="module")
def result_and_log(panel, tmp_path_factory, config):
    """One full evaluation, shared by the tests that only read it."""
    folder = tmp_path_factory.mktemp("evaluation")
    small = shrink(config, folder / "trials.csv")
    log = TrialLog.from_config(small)

    return run_evaluation(small, panel, trial_log=log), log, folder


def test_every_strategy_produces_one_result_per_path(result_and_log):
    result, _, _ = result_and_log

    for name in (*DEFAULT_STRATEGIES, SAME_CLOSE):
        evaluation = result.evaluations[name]
        assert len(evaluation.results) == 5, name
        assert evaluation.path_table.shape[0] == 5, name


def test_every_run_appends_to_the_trial_log(result_and_log):
    result, log, _ = result_and_log
    table = log.trial_table().set_index("name")

    assert log.trial_count() == len(result.evaluations)
    assert table.loc["buy_and_hold", "kind"] == "benchmark"
    assert table.loc["equal_weight_hold", "kind"] == "benchmark"
    assert table.loc[SAME_CLOSE, "kind"] == "diagnostic"
    assert table.loc["ml_long_short", "kind"] == "backtest"
    backtests = [e for e in result.evaluations.values() if e.kind == "backtest"]
    assert log.unique_trial_count({"kind": "backtest"}) == len(backtests)
    assert table["metric_sharpe"].notna().all()


def test_the_cointegrated_pair_is_traded_and_logged_as_its_own_trial(result_and_log):
    result, log, _ = result_and_log
    selection = result.pairs_selection

    assert set(selection["split"]) == set(range(15))
    assert "pair_AAA_BBB" in result.evaluations
    assert "pair_AAA_BBB" in set(log.trial_table()["name"])
    assert result.evaluations["pairs"].summary["pairs_traded"] >= 1


def test_rule_based_paths_are_identical_and_ml_paths_are_not(result_and_log):
    result, _, _ = result_and_log

    assert result.evaluations["momentum"].summary["sharpe_net_std"] == pytest.approx(0, abs=1e-12)
    assert result.evaluations["buy_and_hold"].summary["sharpe_net_std"] == pytest.approx(
        0, abs=1e-12
    )
    assert result.evaluations["ml_long_short"].summary["sharpe_net_std"] > 0


def test_net_never_beats_gross(result_and_log):
    result, _, _ = result_and_log

    for name, evaluation in result.evaluations.items():
        for path in evaluation.results:
            assert (path.net <= path.gross + 1e-15).all(), name


def test_comparison_and_regime_tables(result_and_log):
    result, _, _ = result_and_log
    comparison = result.comparison

    assert comparison.loc["buy_and_hold", "sharpe_net_vs_buy_and_hold"] == 0
    assert comparison.loc["equal_weight_hold", "sharpe_net_vs_equal_weight_hold"] == 0
    assert comparison.loc["equal_weight_hold", "exposure_net_mean"] == pytest.approx(1.0, abs=0.01)
    assert comparison.loc["pair_AAA_BBB", "exposure_gross_mean"] > 0
    assert {"sharpe_gross", "sharpe_net", "sharpe_lost_to_costs", "breakeven_return"} <= set(
        comparison.columns
    )
    assert set(result.regimes.columns) == {"low", "normal", "high"}
    assert "days_in_regime" in result.regimes.index


def test_same_close_timing_is_the_only_difference(result_and_log):
    result, _, _ = result_and_log

    assert result.evaluations[SAME_CLOSE].parameters["execution_lag"] == 0
    assert result.evaluations["ml_long_short"].parameters["execution_lag"] == 1


def test_save_writes_the_tables_and_returns(result_and_log, tmp_path):
    result, _, _ = result_and_log
    files = result.save(tmp_path)

    assert {f.name for f in files} == {
        "comparison.csv",
        "paths.csv",
        "regimes.csv",
        "pairs_selection.csv",
        "path_returns.parquet",
    }
    returns = pd.read_parquet(tmp_path / "path_returns.parquet")
    assert "ml_long_short|path_0|net" in returns.columns


def test_n_paths_limits_the_work(panel, small_config):
    log = TrialLog.from_config(small_config)
    result = run_evaluation(
        small_config,
        panel,
        strategies=["momentum", "buy_and_hold"],
        n_paths=2,
        same_close_check=False,
        trial_log=log,
    )

    assert all(len(e.results) == 2 for e in result.evaluations.values())
    assert log.trial_count() == 2


def test_unknown_strategy_is_refused(panel, small_config):
    with pytest.raises(ValueError, match="unknown strategies"):
        run_evaluation(small_config, panel, strategies=["moonshot"])


def test_paths_cover_only_the_development_period(panel, small_config):
    context = build_context(small_config, panel, fit_models=False)

    assert context.dates.max() < pd.Timestamp("2008-06-01")
    assert sorted(context.date_group.unique()) == list(range(6))


# --- Command line -----------------------------------------------------------


def test_help_works():
    completed = subprocess.run(
        [sys.executable, "entrypoint/evaluate.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    for option in ("--config", "--strategies", "--n-paths", "--output", "--log-level"):
        assert option in completed.stdout


def test_cli_writes_results_to_the_output_path(panel, small_config, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_config", lambda path=None: small_config)
    monkeypatch.setattr("src.pipelines.evaluate.load_clean_panel", lambda config: panel)

    code = cli.main(
        ["--strategies", "momentum,buy_and_hold", "--no-same-close", "--output", str(tmp_path)]
    )

    assert code == cli.EXIT_OK
    assert (tmp_path / "comparison.csv").is_file()


def test_cli_rejects_an_unknown_strategy_cleanly(capsys):
    code = cli.main(["--strategies", "moonshot"])

    assert code == cli.EXIT_FAILED
