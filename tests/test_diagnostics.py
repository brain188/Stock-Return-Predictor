"""Checks on the embargo sensitivity diagnostic.

The diagnostic is only meaningful if the variants differ in exactly one way:
the wide embargo removes the rows near each test block, and the control
removes the same number of rows far from it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.evaluation.trial_log import TrialLog
from src.pipelines.diagnostics import control_split, embargo_sensitivity, with_embargo
from src.validation.cpcv import CombinatorialPurgedCV
from src.validation.purged_cv import date_codes, row_dates
from tests.conftest import make_bars

TICKERS = ["SPY", "AAA", "BBB"]


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    frames = {}
    for seed, ticker in enumerate(TICKERS):
        bars = make_bars(n=900, seed=seed)
        bars["return"] = bars["adj_close"].pct_change()
        frames[ticker] = bars

    return pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()


@pytest.fixture
def small_config(config, tmp_path):
    """The synthetic universe, a 5 date embargo and a temporary trial log."""
    forest = {**config.model.random_forest, "n_estimators": 10, "n_jobs": 1}

    return dataclasses.replace(
        config,
        universe=dataclasses.replace(config.universe, tickers=TICKERS),
        validation=dataclasses.replace(
            config.validation,
            embargo_days=5,
            test_start=pd.Timestamp("2008-01-01").date(),
        ),
        model=dataclasses.replace(config.model, random_forest=forest),
        evaluation=dataclasses.replace(config.evaluation, trial_log=tmp_path / "trials.csv"),
    )


@pytest.fixture
def X() -> pd.DataFrame:
    dates = pd.date_range("2010-01-04", periods=600, freq="B")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])

    return pd.DataFrame({"x": 0.0}, index=index)


def test_control_removes_as_many_dates_as_the_wide_embargo_but_far_away(X):
    codes, _ = date_codes(row_dates(X.index))
    base = list(CombinatorialPurgedCV(6, 2, embargo_days=5).iter_splits(X))
    wide = list(CombinatorialPurgedCV(6, 2, embargo_days=30).iter_splits(X))
    rng = np.random.default_rng(0)

    for b, w in zip(base, wide, strict=True):
        control = control_split(b, w, codes, 30, rng)
        removed = np.setdiff1d(codes[b.train], codes[control.train])
        test_dates = np.unique(codes[b.test])

        assert len(control.train) == len(w.train)
        assert set(control.train) <= set(b.train)
        assert np.array_equal(control.test, b.test)
        # Every removed date is more than 30 dates from every test date.
        assert all(np.abs(test_dates - date).min() > 30 for date in removed)


def test_control_fails_loudly_when_there_is_not_enough_far_data(X):
    codes, _ = date_codes(row_dates(X.index))
    base = next(CombinatorialPurgedCV(6, 2, embargo_days=0).iter_splits(X))
    wide = next(CombinatorialPurgedCV(6, 2, embargo_days=95).iter_splits(X))

    with pytest.raises(ValueError, match="far dates"):
        control_split(base, wide, codes, 400, np.random.default_rng(0))


def test_sensitivity_runs_every_variant_on_the_same_test_rows(panel, small_config):
    log = TrialLog.from_config(small_config)
    result = embargo_sensitivity(
        small_config, panel, embargoes=(0, 5, 20), control_width=20, trial_log=log
    )
    summary = result.summary

    assert list(summary.index) == ["embargo_0", "embargo_5", "embargo_20", "control_20"]
    assert summary.loc["embargo_5", "paired_diff_mean"] == 0
    # The wide embargo and its control train on the same amount of data.
    assert (
        summary.loc["control_20", "train_rows_mean"] == summary.loc["embargo_20", "train_rows_mean"]
    )
    assert (
        summary.loc["embargo_0", "train_rows_mean"] > summary.loc["embargo_20", "train_rows_mean"]
    )
    assert result.splits.shape == (15, 4)
    indexes = [p.index for p in result.predictions.values()]
    assert all(index.equals(indexes[0]) for index in indexes)


def test_sensitivity_runs_are_diagnostics_not_trials(panel, small_config):
    log = TrialLog.from_config(small_config)
    embargo_sensitivity(small_config, panel, embargoes=(0, 5, 20), control_width=20, trial_log=log)

    table = log.trial_table()

    assert log.trial_count() == 4
    assert (table["kind"] == "diagnostic").all()
    assert log.unique_trial_count({"kind": "research"}) == 0
    assert log.unique_trial_count({"kind": "diagnostic"}) == 4


def test_baseline_must_be_among_the_variants(panel, small_config):
    with pytest.raises(ValueError, match="must include"):
        embargo_sensitivity(small_config, panel, embargoes=(0, 20), control_width=20)


def test_with_embargo_changes_nothing_else(config):
    changed = with_embargo(config, 63)

    assert changed.validation.embargo_days == 63
    assert changed.validation.cpcv == config.validation.cpcv
    assert changed.model == config.model
