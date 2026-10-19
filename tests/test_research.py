"""Checks on the research pipeline and its command line entry point.

The pipeline runs end to end on a small synthetic panel, so these tests cover
the wiring: development period only, one trial log row per attempt, and
tables of the right shape. Whether the numbers are good is a question for
the real data, not for a test.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from entrypoint import research as cli
from src.config import PROJECT_ROOT, ConfigError
from src.evaluation.trial_log import TrialLog
from src.models.estimator import ReturnPredictor
from src.pipelines import research
from src.pipelines.research import (
    feature_redundancy,
    ic_across_regimes,
    importance_across_folds,
    prepare_research_data,
    run_research,
    with_overrides,
)
from tests.conftest import make_bars

TICKERS = ["SPY", "AAA", "BBB"]
TEST_START = "2008-01-01"


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """Three synthetic tickers, with the return column cleaning adds."""
    frames = {}
    for seed, ticker in enumerate(TICKERS):
        bars = make_bars(n=900, seed=seed)
        bars["return"] = bars["adj_close"].pct_change()
        frames[ticker] = bars

    return pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()


@pytest.fixture
def small_config(config, tmp_path):
    """Config for the synthetic panel, with a small forest and a temporary trial log."""
    forest = {**config.model.random_forest, "n_estimators": 10, "n_jobs": 1}

    return dataclasses.replace(
        config,
        universe=dataclasses.replace(config.universe, tickers=TICKERS),
        validation=dataclasses.replace(
            config.validation, test_start=pd.Timestamp(TEST_START).date()
        ),
        model=dataclasses.replace(config.model, random_forest=forest),
        evaluation=dataclasses.replace(config.evaluation, trial_log=tmp_path / "trials.csv"),
    )


@pytest.fixture
def log(small_config) -> TrialLog:
    return TrialLog.from_config(small_config)


# --- Pipeline ---------------------------------------------------------------


def test_pipeline_runs_end_to_end(panel, small_config, log):
    result = run_research(small_config, panel, trial_log=log)
    n_features = len(result.importance)

    assert result.path_metrics.shape[0] == 5
    assert result.predictions.shape[1] == 5
    assert result.predictions.notna().all().all()
    assert list(result.comparison.index) == [
        result.name,
        "always_up",
        "random",
        "last_sign",
        "reverse_last_sign",
    ]
    assert n_features == 19
    assert set(result.ic_by_regime.index) <= {"low", "normal", "high"}
    assert result.redundancy.shape == (n_features, n_features + 1)
    assert -1 <= result.metrics["ic_mean"] <= 1


def test_every_run_appends_a_trial(panel, small_config, log):
    run_research(small_config, panel, trial_log=log, permutation=False)
    run_research(
        with_overrides(small_config, label_method="triple_barrier"),
        panel,
        trial_log=log,
        permutation=False,
    )

    table = log.trial_table()

    assert log.trial_count() == 2
    assert list(table["label"]) == ["fixed_horizon", "triple_barrier"]
    assert table["metric_ic_mean"].notna().all()
    assert (table["metric_n_paths"] == 5).all()


def test_a_failed_run_is_still_a_trial(panel, small_config, log, monkeypatch):
    def broken(config, task=None):
        raise RuntimeError("estimator exploded")

    monkeypatch.setattr(research, "build_estimator", broken)

    with pytest.raises(RuntimeError):
        run_research(small_config, panel, trial_log=log)

    assert log.trial_count() == 1
    assert log.trial_table()["notes"].iloc[0].startswith("failed: RuntimeError")


@pytest.mark.parametrize("method", ["fixed_horizon", "triple_barrier"])
def test_research_never_touches_the_holdout(panel, small_config, method):
    config = with_overrides(small_config, label_method=method)
    data = prepare_research_data(panel, config)
    dates = data.X.index.get_level_values("date")
    first_holdout = panel.index.get_level_values("date")[
        panel.index.get_level_values("date") >= pd.Timestamp(TEST_START)
    ].min()

    assert dates.max() < pd.Timestamp(TEST_START)
    assert (data.labels["touch_time"] <= first_holdout).all()
    assert data.X.index.equals(data.labels.index)
    assert data.X.index.equals(data.weights.index)


def test_save_writes_every_table(panel, small_config, log, tmp_path):
    result = run_research(small_config, panel, trial_log=log, permutation=False)
    files = result.save(tmp_path / "out")

    assert {f.name for f in files} == {
        "metrics.csv",
        "paths.csv",
        "comparison.csv",
        "ic_by_year.csv",
        "ic_by_regime.csv",
        "redundancy.csv",
        "importance.csv",
    }


def test_overrides_are_validated(small_config):
    assert with_overrides(small_config, estimator="lightgbm").model.estimator == "lightgbm"

    with pytest.raises(ConfigError):
        with_overrides(small_config, estimator="xgboost")


# --- Building blocks --------------------------------------------------------


def test_importance_across_folds_finds_a_stable_signal():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(1500, 4)), columns=["signal", "a", "b", "c"])
    y = pd.Series(X["signal"] + rng.normal(0, 0.5, len(X)))

    models, held_out = [], []
    for fold in range(3):
        rows = np.arange(fold * 500, fold * 500 + 500)
        rest = np.setdiff1d(np.arange(len(X)), rows)
        model = ReturnPredictor("random_forest", params={"n_estimators": 20, "n_jobs": 1})
        models.append(model.fit(X.iloc[rest], y.iloc[rest]))
        held_out.append((X.iloc[rows], y.iloc[rows]))

    table = importance_across_folds(models, held_out, n_repeats=2)

    assert table.index[0] == "signal"
    assert table.loc["signal", "top_share"] == 1.0
    assert table.loc["signal", "positive_share"] == 1.0
    assert table.loc["signal", "rank_std"] == 0.0


def test_ic_across_regimes_scores_each_regime_separately():
    dates = pd.date_range("2010-01-01", periods=300, freq="B")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])
    actual = pd.Series(np.random.default_rng(1).normal(size=len(index)), index=index)
    regimes = pd.Series(np.where(np.arange(300) < 150, "low", "high"), index=dates)
    # Right in calm markets, inverted in stressed ones.
    sign = np.where(regimes.reindex(index.get_level_values("date")) == "low", 1.0, -1.0)

    table = ic_across_regimes(actual * sign, actual, regimes)

    assert list(table.index) == ["low", "high"]
    assert table.loc["low", "ic"] == pytest.approx(1.0)
    assert table.loc["high", "ic"] == pytest.approx(-1.0)
    assert table["n"].sum() == len(index)


def test_feature_redundancy_groups_near_duplicates():
    rng = np.random.default_rng(2)
    base = rng.normal(size=2000)
    features = pd.DataFrame(
        {
            "a": base,
            "a_copy": base + rng.normal(0, 0.1, 2000),
            "a_inverse": -base,
            "other": rng.normal(size=2000),
        }
    )

    table = feature_redundancy(features)
    cluster = table["cluster"]

    assert cluster["a"] == cluster["a_copy"] == cluster["a_inverse"]
    assert cluster["other"] != cluster["a"]
    assert list(table.columns[1:]) == list(table.index)


# --- Command line -----------------------------------------------------------


def test_help_works():
    completed = subprocess.run(
        [sys.executable, "entrypoint/research.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    for option in ("--config", "--refresh", "--log-level", "--label-method", "--model", "--output"):
        assert option in completed.stdout


def test_invalid_config_path_fails_cleanly(capsys, caplog):
    code = cli.main(["--config", "does/not/exist.yaml"])
    output = capsys.readouterr()

    assert code == cli.EXIT_FAILED
    assert "config file not found" in caplog.text
    assert "Traceback" not in output.out + output.err + caplog.text


def test_invalid_model_is_rejected_by_argparse():
    with pytest.raises(SystemExit) as exit_info:
        cli.parse_args(["--model", "xgboost"])

    assert exit_info.value.code == 2
