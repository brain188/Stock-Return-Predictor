"""Checks on combinatorial purged cross-validation and its backtest paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.cpcv import (
    CombinatorialPurgedCV,
    backtest_paths,
    combinatorial_splits,
    path_count,
    path_predictions,
)
from src.validation.purged_cv import assert_no_overlap, embargo_bars, row_dates

TICKERS = ["AAA", "BBB"]
N_DATES = 600


@pytest.fixture
def X() -> pd.DataFrame:
    dates = pd.date_range("2010-01-04", periods=N_DATES, freq="B")
    index = pd.MultiIndex.from_product([dates, TICKERS], names=["date", "ticker"])

    return pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)


@pytest.fixture
def touch(X) -> pd.Series:
    """Five day labels, so purging has something to remove."""
    dates = row_dates(X.index).unique()
    ahead = pd.Series(dates, index=dates).shift(-5)

    return pd.Series(ahead.reindex(row_dates(X.index)).to_numpy(), index=X.index)


def test_six_groups_two_at_a_time_gives_fifteen_splits(X):
    cv = CombinatorialPurgedCV(6, 2)

    assert len(combinatorial_splits(6, 2)) == 15
    assert len(list(cv.split(X))) == 15
    assert cv.get_n_splits() == 15


@pytest.mark.parametrize(("n", "k", "paths"), [(6, 2, 5), (6, 1, 1), (10, 2, 9), (8, 3, 21)])
def test_path_count(n, k, paths):
    assert path_count(n, k) == paths
    assert backtest_paths(combinatorial_splits(n, k), n).shape == (paths, n)


def test_invalid_group_counts_raise():
    with pytest.raises(ValueError):
        combinatorial_splits(6, 6)
    with pytest.raises(ValueError):
        combinatorial_splits(6, 0)


def test_every_split_is_purged_and_embargoed(X, touch):
    cv = CombinatorialPurgedCV(6, 2, embargo_pct=0.02, touch_times=touch)
    dates = row_dates(X.index)
    width = embargo_bars(N_DATES, 0.02)

    for train, test in cv.split(X):
        assert_no_overlap(train, test, width, dates=dates, touch_times=touch)


def test_purging_removes_rows_only_when_labels_overlap(X, touch):
    plain = [len(train) for train, _ in CombinatorialPurgedCV(6, 2, 0.0).split(X)]
    purged = [len(train) for train, _ in CombinatorialPurgedCV(6, 2, 0.0, touch).split(X)]

    assert all(p <= q for p, q in zip(purged, plain, strict=True))
    assert sum(purged) < sum(plain)


def test_each_path_draws_every_group_from_a_split_that_tests_it():
    splits = combinatorial_splits(6, 2)

    for path in backtest_paths(splits, 6):
        assert all(g in splits[s] for g, s in enumerate(path))


def test_every_path_covers_the_timeline_exactly_once(X):
    cv = CombinatorialPurgedCV(6, 2)
    tests = [test for _, test in cv.split(X)]
    row_group = cv.groups(X)

    for path in cv.paths:
        covered = np.concatenate([tests[s][row_group[tests[s]] == g] for g, s in enumerate(path)])
        assert np.array_equal(np.sort(covered), np.arange(len(X)))


def test_no_two_paths_are_identical():
    paths = backtest_paths(combinatorial_splits(6, 2), 6)

    assert len({tuple(p) for p in paths}) == len(paths)
    # Stronger: no two paths share the split behind any group.
    for g in range(6):
        assert len(set(paths[:, g])) == len(paths)


def test_path_predictions_fill_every_row_from_the_right_split(X):
    cv = CombinatorialPurgedCV(6, 2)
    # Each split "predicts" its own number, so the source of every value is visible.
    split_predictions = [
        pd.Series(float(i), index=X.index[test]) for i, (_, test) in enumerate(cv.split(X))
    ]
    paths = path_predictions(split_predictions, cv, X)
    row_group = cv.groups(X)

    assert paths.shape == (len(X), cv.n_paths)
    assert paths.notna().all().all()
    for p in range(cv.n_paths):
        expected = cv.paths[p][row_group].astype(float)
        assert np.array_equal(paths[f"path_{p}"].to_numpy(), expected)


def test_groups_are_contiguous_and_keep_tickers_together(X):
    row_group = CombinatorialPurgedCV(6, 2).groups(X)
    by_date = pd.Series(row_group, index=X.index).groupby(level="date").nunique()

    assert (by_date == 1).all()
    assert (np.diff(row_group) >= 0).all()
