"""Checks on the trial log.

The deflated Sharpe is only as honest as the trial count behind it, so the log
must never lose a row, never merge repeats, and survive a restart.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.trial_log import COLUMNS, TrialLog, TrialLogError, config_hash


@pytest.fixture
def log(tmp_path) -> TrialLog:
    """An empty log in a temporary directory."""
    return TrialLog(tmp_path / "results" / "trial_log.csv")


def test_empty_log_counts_zero(log):
    """A log that was never written to has no trials and no file."""
    assert log.trial_count() == 0
    assert log.trial_table().empty
    assert not log.path.exists()


def test_append_keeps_earlier_rows(log):
    """Each new trial adds a row; nothing written before disappears."""
    log.log_trial("first", metrics={"sharpe": 0.5})
    log.log_trial("second", metrics={"sharpe": 0.7})
    log.log_trial("third", metrics={"sharpe": 0.1})

    table = log.trial_table()

    assert list(table["name"]) == ["first", "second", "third"]
    assert list(table["trial_id"]) == [1, 2, 3]


def test_count_matches_row_count(log):
    """trial_count is the number of rows in the file, header excluded."""
    for i in range(5):
        log.log_trial(f"t{i}")

    with log.path.open(encoding="utf-8") as handle:
        data_lines = sum(1 for _ in handle) - 1

    assert log.trial_count() == data_lines == 5


def test_same_configuration_is_recorded_twice(log):
    """A repeat is a second trial, because it was tried a second time."""
    for _ in range(2):
        log.log_trial("rf", model_params={"max_depth": 6}, metrics={"sharpe": 0.4})

    assert log.trial_count() == 2


def test_log_survives_restart(log):
    """A fresh object on the same path, as after a restart, sees every trial."""
    log.log_trial("before_restart", metrics={"sharpe": 0.3})

    reopened = TrialLog(log.path)
    reopened.log_trial("after_restart", metrics={"sharpe": 0.2})

    assert reopened.trial_count() == 2
    assert list(TrialLog(log.path).trial_table()["trial_id"]) == [1, 2]


def test_nested_values_round_trip(log):
    """Feature order, parameters and numpy metrics come back as written."""
    features = ["ret_lag_5", "rsi_14", "ret_lag_1"]
    log.log_trial(
        "rf",
        features=features,
        model_params={"n_estimators": 300, "max_features": "sqrt"},
        label="fixed_horizon",
        metrics={"sharpe": np.float64(0.42), "ic": np.float32(0.03)},
    )

    row = log.trial_table().iloc[0]

    assert row["features"] == features
    assert row["model_params"] == {"n_estimators": 300, "max_features": "sqrt"}
    assert row["label"] == "fixed_horizon"
    assert row["metric_sharpe"] == pytest.approx(0.42)
    assert row["metric_ic"] == pytest.approx(0.03)


def test_new_metric_does_not_break_old_rows(log):
    """A metric added later is NaN for earlier trials, not an error."""
    log.log_trial("old", metrics={"sharpe": 0.5})
    log.log_trial("new", metrics={"sharpe": 0.6, "sortino": 0.9})

    table = log.trial_table()

    assert math.isnan(table.loc[0, "metric_sortino"])
    assert table.loc[1, "metric_sortino"] == pytest.approx(0.9)


def test_trial_count_with_filters(log):
    """Filters restrict the count to matching rows."""
    log.log_trial("a", label="fixed_horizon")
    log.log_trial("b", label="triple_barrier")
    log.log_trial("c", label="fixed_horizon")

    assert log.trial_count({"label": "fixed_horizon"}) == 2
    assert log.trial_count({"label": "nothing"}) == 0


def test_unknown_filter_column_raises(log):
    """A typo in a filter fails loudly instead of counting zero."""
    log.log_trial("a")

    with pytest.raises(TrialLogError, match="lable"):
        log.trial_count({"lable": "fixed_horizon"})


def test_best_trial(log):
    """The best trial is the one with the highest metric, or lowest if asked."""
    log.log_trial("low", metrics={"sharpe": 0.1, "max_drawdown": -0.5})
    log.log_trial("high", metrics={"sharpe": 0.9, "max_drawdown": -0.2})

    assert log.best_trial("sharpe")["name"] == "high"
    assert log.best_trial("metric_sharpe")["name"] == "high"
    assert log.best_trial("max_drawdown", maximise=False)["name"] == "low"


def test_best_trial_missing_metric_raises(log):
    """Asking for a metric nobody recorded is an error, not a silent None."""
    log.log_trial("a", metrics={"sharpe": 0.1})

    with pytest.raises(TrialLogError, match="calmar"):
        log.best_trial("calmar")


def test_sharpe_variance_matches_numpy(log):
    """V[SR] is the sample variance of the logged Sharpe ratios."""
    sharpes = [0.1, 0.5, -0.2, 0.8]
    for i, sharpe in enumerate(sharpes):
        log.log_trial(f"t{i}", metrics={"sharpe": sharpe})

    assert log.sharpe_variance() == pytest.approx(np.var(sharpes, ddof=1))


def test_sharpe_variance_needs_two_trials(log):
    """One Sharpe has no spread, so the variance is undefined."""
    log.log_trial("only", metrics={"sharpe": 0.5})

    assert math.isnan(log.sharpe_variance())


def test_refuses_file_with_other_layout(log):
    """Appending to a file with different columns would corrupt it."""
    log.path.parent.mkdir(parents=True)
    log.path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    with pytest.raises(TrialLogError, match="different layout"):
        log.log_trial("x")


def test_header_is_written_once(log):
    """Only the first write adds the header row."""
    log.log_trial("a")
    log.log_trial("b")

    lines = log.path.read_text(encoding="utf-8").splitlines()

    assert lines[0] == ",".join(COLUMNS)
    assert sum(line == ",".join(COLUMNS) for line in lines) == 1


def test_config_hash_is_stable(config, log):
    """The same config file always gives the same fingerprint."""
    assert config_hash(config) == config_hash(config)
    assert len(config_hash(config)) == 12
    assert config_hash(None) == ""

    log.log_trial("with_config", config=config)

    assert log.trial_table().iloc[0]["config_hash"] == config_hash(config)
