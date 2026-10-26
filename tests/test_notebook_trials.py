"""Checks on the 42 notebook experiments recovered for the trial log."""

from __future__ import annotations

import math

import pytest

from src.evaluation.notebook_trials import (
    EXPECTED,
    NotebookTrialError,
    historical_notebook_trials,
    record_notebook_trials,
)
from src.evaluation.trial_log import TrialLog


@pytest.fixture(scope="module")
def table():
    return historical_notebook_trials()


def test_the_count_matches_notebook_4s_own_tally(table):
    assert len(table) == 42
    assert table["experiment"].value_counts().to_dict() == EXPECTED


def test_values_come_from_the_saved_outputs(table):
    by_name = table.set_index("name")["metrics"]

    # Spot checks against the tables the notebooks printed.
    assert by_name["notebook02_fracdiff_d_0.55"]["adf_spy_2005_2024"] == -2.8857
    assert by_name["notebook03_feature_rsi_14"]["ic_spy_2005_2024"] == -0.0592
    assert by_name["notebook04_rule_long_only"]["sharpe_net_2008_2024"] == 0.7027
    assert by_name["notebook04_model_random_forest_spy"]["ic_spy_2008_2024"] == 0.0245


def test_no_notebook_metric_can_enter_the_sharpe_variance(table, tmp_path):
    log = TrialLog(tmp_path / "trials.csv")
    record_notebook_trials(log)

    # The variance reads the metric called "sharpe"; notebook Sharpe ratios
    # carry their period in the name and cover years now held out.
    assert all("sharpe" not in metrics for metrics in table["metrics"])
    assert math.isnan(log.sharpe_variance(filters={"kind": "notebook"}))


def test_records_are_appended_once_and_marked_as_partial(tmp_path):
    log = TrialLog(tmp_path / "trials.csv")

    assert record_notebook_trials(log) == 42
    rows = log.trial_table()
    assert (rows["kind"] == "notebook").all()
    assert (rows["code_version"] == "notebook, not recorded").all()
    assert rows["notes"].str.contains("not recoverable").all()
    assert log.unique_trial_count({"kind": "notebook"}) == 42
    with pytest.raises(NotebookTrialError, match="already"):
        record_notebook_trials(log)
