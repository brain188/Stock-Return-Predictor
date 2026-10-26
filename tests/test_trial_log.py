"""Checks on the trial log.

The deflated Sharpe is only as honest as the trial count behind it, so the log
must never lose a row, never merge repeats, and survive a restart.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from src.evaluation.trial_log import (
    COLUMNS,
    LEGACY_COLUMNS,
    TrialLog,
    TrialLogError,
    config_hash,
    trial_fingerprint,
)


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


# --- Trial identity ---------------------------------------------------------

LEGACY_HEADER = ",".join(LEGACY_COLUMNS)


def write_legacy(log: TrialLog) -> str:
    """A log in the old layout: two runs of one setup, one failure."""
    log.path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        LEGACY_HEADER + "\n"
        '1,2026-10-18T18:53:27+00:00,research_rf,fixed_horizon,db43a1442cb2,"[""a""]",'
        '"{""seed"": 42}","{""ic_mean"": 0.0043}",CPCV N=6 k=2\n'
        '2,2026-10-19T20:26:31+00:00,research_rf,fixed_horizon,db43a1442cb2,"[""a""]",'
        '"{""seed"": 42}","{""cs_mean"": 0.005, ""ic_mean"": 0.0043}",CPCV N=6 k=2\n'
        '3,2026-10-20T10:00:00+00:00,research_rf,fixed_horizon,db43a1442cb2,"[]",'
        '"{}","{}",failed: RuntimeError: boom\n'
    )
    log.path.write_text(text, encoding="utf-8")

    return text


def test_new_rows_record_every_identity(log, config):
    log.log_trial("rf", kind="research", fingerprint="abc", config=config, metrics={"ic": 0.1})

    row = log.trial_table().iloc[0]

    assert (row["kind"], row["fingerprint"], row["status"]) == ("research", "abc", "ok")
    assert row["config_hash"] == config_hash(config)
    assert row["code_version"]


def test_reruns_are_executions_not_new_trials(log):
    for _ in range(3):
        log.log_trial("rf", fingerprint="same", metrics={"sharpe": 0.4})
    log.log_trial("lgbm", fingerprint="other", metrics={"sharpe": 0.1})

    unique = log.unique_trials()

    assert log.trial_count() == 4
    assert log.unique_trial_count() == 2
    assert unique.loc[("research", "same"), "executions"] == 3
    assert unique.loc[("research", "same"), "consistent"]


def test_a_config_hash_change_alone_does_not_add_a_trial(log, config, tmp_path):
    # The same settings run from two config files that differ only in a comment.
    edited = tmp_path / "config.yaml"
    edited.write_text(config.source_file.read_text(encoding="utf-8") + "\n# a comment\n")
    other = dataclasses.replace(config, source_file=edited)

    log.log_trial("rf", fingerprint="settings", config=config, metrics={"sharpe": 0.4})
    log.log_trial("rf", fingerprint="settings", config=other, metrics={"sharpe": 0.4})

    hashes = log.trial_table()["config_hash"]

    assert hashes.nunique() == 2
    assert log.unique_trial_count() == 1


def test_failed_attempts_are_recorded_but_not_counted(log):
    log.log_trial("rf", fingerprint="a", status="failed", notes="failed: boom")
    log.log_trial("rf", fingerprint="b", metrics={"sharpe": 0.2})

    assert log.trial_count() == 2
    assert log.unique_trial_count() == 1
    assert log.unique_trial_count(include_failed=True) == 2


def test_a_failure_then_a_success_is_one_trial(log):
    log.log_trial("rf", fingerprint="a", status="failed")
    log.log_trial("rf", fingerprint="a", metrics={"sharpe": 0.2})

    assert log.unique_trial_count(include_failed=True) == 1


def test_diagnostics_are_kept_out_of_the_selection_count(log):
    log.log_trial("rf", fingerprint="a", metrics={"ic": 0.01})
    log.log_trial("embargo_0", kind="diagnostic", fingerprint="d0", metrics={"ic": 0.01})
    log.log_trial("embargo_139", kind="diagnostic", fingerprint="d1", metrics={"ic": 0.01})

    assert log.unique_trial_count({"kind": "research"}) == 1
    assert log.unique_trial_count({"kind": "diagnostic"}) == 2


def test_status_defaults_from_whether_there_were_results(log):
    log.log_trial("ran", metrics={"sharpe": 0.1})
    log.log_trial("crashed")

    assert list(log.trial_table()["status"]) == ["ok", "failed"]


def test_inconsistent_reruns_are_flagged(log):
    # Same settings, different answer: a data refresh or a code change.
    log.log_trial("rf", fingerprint="a", metrics={"sharpe": 0.4})
    log.log_trial("rf", fingerprint="a", metrics={"sharpe": 0.6})

    unique = log.unique_trials()

    assert not unique.loc[("research", "a"), "consistent"]
    assert unique.loc[("research", "a"), "metric_sharpe"] == pytest.approx(0.5)


def test_sharpe_variance_counts_each_trial_once(log):
    for sharpe in (0.1, 0.1, 0.1):
        log.log_trial("rf", fingerprint="a", metrics={"sharpe": sharpe})
    log.log_trial("lgbm", fingerprint="b", metrics={"sharpe": 0.5})

    # Unique: {0.1, 0.5}. Per execution, the three reruns would shrink it.
    assert log.sharpe_variance() == pytest.approx(np.var([0.1, 0.5], ddof=1))
    assert log.sharpe_variance(unique=False) == pytest.approx(np.var([0.1] * 3 + [0.5], ddof=1))


def test_default_fingerprint_ignores_the_config_file(log, config, tmp_path):
    edited = tmp_path / "config.yaml"
    edited.write_text(config.source_file.read_text(encoding="utf-8") + "\n# note\n")
    for source in (config, dataclasses.replace(config, source_file=edited)):
        log.log_trial("rf", config=source, model_params={"depth": 6}, metrics={"s": 1.0})

    assert log.unique_trial_count() == 1


def test_unknown_kind_or_status_is_refused(log):
    with pytest.raises(TrialLogError, match="kind"):
        log.log_trial("x", kind="experiment")
    with pytest.raises(TrialLogError, match="status"):
        log.log_trial("x", status="maybe")


def test_trial_fingerprint_is_order_independent():
    first = trial_fingerprint({"a": 1, "b": {"x": [1, 2], "y": "z"}})
    second = trial_fingerprint({"b": {"y": "z", "x": [1, 2]}, "a": 1})

    assert first == second
    assert len(first) == 16
    assert trial_fingerprint({"a": 2, "b": {"x": [1, 2], "y": "z"}}) != first


def test_config_hash_ignores_line_endings(config, tmp_path):
    text = config.source_file.read_text(encoding="utf-8")
    lf, crlf = tmp_path / "lf.yaml", tmp_path / "crlf.yaml"
    lf.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))
    crlf.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))

    assert config_hash(dataclasses.replace(config, source_file=lf)) == config_hash(
        dataclasses.replace(config, source_file=crlf)
    )


# --- Legacy files -----------------------------------------------------------


def test_a_legacy_log_is_readable_but_not_appendable(log):
    write_legacy(log)

    table = log.trial_table()

    assert log.is_legacy()
    assert list(table["kind"]) == ["research"] * 3
    assert list(table["status"]) == ["ok", "ok", "failed"]
    assert (table["fingerprint"] == "").all()
    with pytest.raises(TrialLogError, match="migrate"):
        log.log_trial("new")


def test_unidentified_legacy_rows_each_count_as_a_trial(log):
    write_legacy(log)

    # Unknown identity can only raise the count, never merge rows.
    assert log.unique_trial_count() == 2


def test_migration_keeps_every_value_and_a_backup(log):
    original = write_legacy(log)

    backup = log.migrate({1: {"fingerprint": "fp"}, 2: {"fingerprint": "fp"}})
    table = log.trial_table()

    assert backup.read_text(encoding="utf-8") == original
    assert not log.is_legacy()
    assert list(table["trial_id"]) == [1, 2, 3]
    assert list(table["fingerprint"]) == ["fp", "fp", ""]
    assert list(table["config_hash"]) == ["db43a1442cb2"] * 3
    assert table["metric_ic_mean"].tolist()[:2] == [0.0043, 0.0043]
    # Two executions of one identified trial; the failure is not counted.
    assert log.unique_trial_count() == 1
    assert log.unique_trials().iloc[0]["executions"] == 2


def test_migrated_log_accepts_new_rows(log):
    write_legacy(log)
    log.migrate()

    trial_id = log.log_trial("after", fingerprint="x", metrics={"ic": 0.0})

    assert trial_id == 4
    assert log.trial_count() == 4


def test_migration_refuses_bad_input(log):
    write_legacy(log)

    with pytest.raises(TrialLogError, match="no trial 9"):
        log.migrate({9: {"fingerprint": "x"}})
    with pytest.raises(TrialLogError, match="cannot set"):
        log.migrate({1: {"metrics": "{}"}})
    assert log.is_legacy()

    log.migrate()
    with pytest.raises(TrialLogError, match="not in the legacy layout"):
        log.migrate()


def test_benchmarks_are_recorded_but_not_selection_trials(log):
    log.log_trial("buy_and_hold", kind="benchmark", fingerprint="b", metrics={"sharpe": 0.8})
    log.log_trial("ml", kind="backtest", fingerprint="m", metrics={"sharpe": 0.3})

    assert log.unique_trial_count({"kind": "backtest"}) == 1
    assert log.unique_trial_count({"kind": "benchmark"}) == 1


def test_the_task_list_call_shape_works(log, config):
    trial_id = log.log_trial(
        config=config,
        features=["ret_1d"],
        model_params={"depth": 6},
        label="fixed_horizon",
        metrics={"sharpe": 0.3},
    )

    row = log.trial_table().iloc[0]
    assert trial_id == 1
    assert row["name"] == "fixed_horizon"
    assert row["features"] == ["ret_1d"]


def test_notebook_records_count_as_selection_trials(log):
    log.log_trial("notebook02_d_0.30", kind="notebook", status="ok", notes="count only")

    assert log.unique_trial_count({"kind": "notebook"}) == 1
    assert "notebook" in __import__("src.evaluation.trial_log", fromlist=["x"]).SELECTION_KINDS


def test_a_rewritten_history_is_detected_on_the_next_append(log):
    log.log_trial("a", metrics={"sharpe": 0.1})
    text = log.path.read_text(encoding="utf-8").replace("trial_id", "trial", 1)
    log.path.write_text(text, encoding="utf-8")

    with pytest.raises(TrialLogError, match="layout"):
        log.log_trial("b", metrics={"sharpe": 0.2})
