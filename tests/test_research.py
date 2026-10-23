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
    label_span_days,
    leakage_diagnostics,
    prepare_research_data,
    research_fingerprint,
    run_name,
    run_research,
    split_fields,
    split_report,
    with_overrides,
    write_split_report,
)
from src.validation import cpcv as cpcv_module
from src.validation.purged_cv import LeakageError
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
        "cs_ic_daily.csv",
        "predictions.parquet",
        "redundancy.csv",
        "splits.csv",
        "importance.csv",
    }


# --- Leakage controls -------------------------------------------------------


def test_every_split_is_recorded_with_its_row_counts(panel, small_config, log):
    result = run_research(small_config, panel, trial_log=log, permutation=False)
    splits = result.splits
    n_rows = result.metrics["n_rows"]

    assert len(splits) == 15
    assert (splits["candidates"] + splits["test_rows"] == n_rows).all()
    assert (
        splits["candidates"] == splits["purged"] + splits["embargoed"] + splits["train_rows"]
    ).all()
    # One-day labels never overlap, so nothing is purged.
    assert (splits["purged"] == 0).all()
    assert result.metrics["embargo_days"] == small_config.validation.embargo_days == 36
    assert result.metrics["label_max_span_days"] == 1
    assert result.metrics["train_rows_mean"] == pytest.approx(splits["train_rows"].mean())


@pytest.mark.parametrize(("method", "longest"), [("fixed_horizon", 1), ("triple_barrier", 5)])
def test_label_span_is_measured_in_trading_dates(panel, small_config, method, longest):
    data = prepare_research_data(panel, with_overrides(small_config, label_method=method))
    span = label_span_days(data.labels)

    assert span.max() == longest
    assert span.min() >= 1


def test_purged_overlapping_labels_raise_no_warning(panel, small_config):
    config = with_overrides(small_config, label_method="triple_barrier")
    data = prepare_research_data(panel, config)

    check = leakage_diagnostics(data.labels, config, purging=True, d=data.fracdiff_d)

    assert check.labels_overlap
    assert check.max_label_span == 5
    assert check.warnings == []


def test_overlapping_labels_without_purging_are_a_warning(panel, small_config):
    config = with_overrides(small_config, label_method="triple_barrier")
    data = prepare_research_data(panel, config)

    check = leakage_diagnostics(data.labels, config, purging=False, d=data.fracdiff_d)

    assert len(check.warnings) == 1
    assert "purging is off" in check.warnings[0]


def test_one_day_labels_are_safe_even_without_purging(panel, small_config):
    data = prepare_research_data(panel, small_config)

    check = leakage_diagnostics(data.labels, small_config, purging=False, d=data.fracdiff_d)

    assert not check.labels_overlap
    assert check.warnings == []


def test_feature_windows_longer_than_the_embargo_are_a_diagnostic_not_a_warning(
    panel, small_config
):
    data = prepare_research_data(panel, small_config)
    check = leakage_diagnostics(data.labels, small_config, purging=True, d=data.fracdiff_d)

    assert set(check.uncovered) == {
        "px_over_ma_63",
        "vol_63d",
        "mom_63d",
        "dist_high_63",
        "vol_ratio_10_63",
        "fracdiff",
    }
    assert check.warnings == []


def test_a_split_that_leaks_stops_the_run(panel, small_config, log, monkeypatch):
    # Break purging inside the splitter only; the leak check keeps its own copy.
    monkeypatch.setattr(cpcv_module, "purge_train_indices", lambda train, test, touch: train)
    config = with_overrides(small_config, label_method="triple_barrier")

    with pytest.raises(LeakageError, match="overlap test labels"):
        run_research(config, panel, trial_log=log, permutation=False)

    assert log.trial_table()["notes"].iloc[0].startswith("failed: LeakageError")


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
    for option in (
        "--config",
        "--refresh",
        "--log-level",
        "--label-method",
        "--model",
        "--output",
        "--splits-only",
    ):
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


def test_cross_sectional_ic_is_reported_alongside_pooled(panel, small_config, log):
    result = run_research(small_config, panel, trial_log=log, permutation=False)
    n_dates = result.predictions.index.get_level_values("date").nunique()

    # Three synthetic tickers is below the five-ticker minimum, so every date is
    # skipped and counted, not scored from too few points.
    assert result.metrics["cs_skipped_dates"] == n_dates
    assert np.isnan(result.metrics["cs_spearman_mean"])
    assert not np.isnan(result.metrics["ic_mean"])
    assert (result.cs_ic_daily["n_tickers"] == 3).all()
    assert {"cs_spearman_mean", "cs_pearson_mean"} <= set(result.comparison.columns)
    assert "metric_cs_skipped_dates" in log.trial_table().columns


# --- Trial identity ---------------------------------------------------------


def edited(config, **sections):
    """The config with fields replaced, section by section."""
    changes = {
        name: dataclasses.replace(getattr(config, name), **fields)
        for name, fields in sections.items()
    }

    return dataclasses.replace(config, **changes)


def test_fingerprint_ignores_settings_that_cannot_change_a_result(config, tmp_path):
    base = research_fingerprint(config)
    commented = tmp_path / "config.yaml"
    commented.write_text(config.source_file.read_text(encoding="utf-8") + "\n# note\n")
    forest = {**config.model.random_forest, "n_jobs": 1}
    lightgbm = {**config.model.lightgbm, "num_leaves": 63}

    assert research_fingerprint(dataclasses.replace(config, source_file=commented)) == base
    assert research_fingerprint(edited(config, backtest={"commission_bps": 9.0})) == base
    assert research_fingerprint(edited(config, evaluation={"monte_carlo_runs": 5000})) == base
    assert research_fingerprint(edited(config, model={"random_forest": forest})) == base
    assert research_fingerprint(edited(config, model={"lightgbm": lightgbm})) == base
    assert research_fingerprint(edited(config, validation={"n_splits": 4})) == base
    reordered = list(reversed(config.universe.tickers))
    assert research_fingerprint(edited(config, universe={"tickers": reordered})) == base


def test_fingerprint_changes_with_anything_that_can_change_a_result(config):
    base = research_fingerprint(config)
    deeper = {**config.model.random_forest, "max_depth": 8}
    barrier = dataclasses.replace(config.labelling.triple_barrier, max_holding_days=10)

    changed = [
        edited(config, validation={"embargo_days": 63}),
        edited(config, validation={"test_start": pd.Timestamp("2019-01-01").date()}),
        edited(config, model={"estimator": "lightgbm"}),
        edited(config, model={"task": "classification"}),
        edited(config, model={"random_forest": deeper}),
        edited(config, labelling={"method": "triple_barrier"}),
        edited(config, project={"random_seed": 7}),
        edited(config, features={"rsi_window": 21}),
    ]

    assert all(research_fingerprint(c) != base for c in changed)
    # The inactive labelling method's settings are not part of the trial.
    assert research_fingerprint(edited(config, labelling={"triple_barrier": barrier})) == base


def test_a_research_run_records_its_identity(panel, small_config, log):
    run_research(small_config, panel, trial_log=log, permutation=False)
    run_research(small_config, panel, trial_log=log, permutation=False)

    table = log.trial_table()

    assert list(table["kind"]) == ["research", "research"]
    assert (table["fingerprint"] == research_fingerprint(small_config)).all()
    assert log.trial_count() == 2
    assert log.unique_trial_count({"kind": "research"}) == 1
    assert log.unique_trials().iloc[0]["consistent"]


# --- Split report without a trial -------------------------------------------


def test_split_report_matches_the_run_and_logs_nothing(panel, small_config, log):
    result = run_research(small_config, panel, trial_log=log, permutation=False)
    count = log.trial_count()

    splits, leakage, data = split_report(small_config, panel)

    pd.testing.assert_frame_equal(splits, result.splits)
    assert data.X.index.equals(result.predictions.index)
    assert split_fields(splits, leakage) == {
        key: result.metrics[key] for key in split_fields(splits, leakage)
    }
    assert log.trial_count() == count


def test_write_split_report_checks_the_saved_predictions(panel, small_config, log, tmp_path):
    result = run_research(small_config, panel, trial_log=log, permutation=False)
    result.save(tmp_path)

    files = write_split_report(small_config, tmp_path, panel)
    record = pd.read_csv(files[1], index_col=0)["value"]

    assert [f.name for f in files] == ["splits.csv", "validation.csv"]
    assert record["matches_saved_predictions"] == "True"
    assert record["fingerprint"] == research_fingerprint(small_config)
    assert log.trial_count() == 1


def test_write_split_report_refuses_predictions_from_other_rows(panel, small_config, tmp_path):
    folder = tmp_path / run_name(small_config)
    folder.mkdir()
    pd.DataFrame({"path_0": [0.1]}).to_parquet(folder / "predictions.parquet")

    with pytest.raises(ValueError, match="different rows"):
        write_split_report(small_config, tmp_path, panel)
