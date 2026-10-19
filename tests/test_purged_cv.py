"""Checks that the splitters never let a training row see the test set.

Every test uses a pooled (date, ticker) panel, because splitting by row rather
than by date is the easiest way to leak: one ticker in train and another in
test on the same day.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.validation.purged_cv import (
    LeakageError,
    PurgedKFold,
    apply_embargo,
    assert_no_overlap,
    embargo_bars,
    purge_train_indices,
    row_dates,
    train_test_split_chronological,
)

TICKERS = ["AAA", "BBB", "CCC"]
N_DATES = 1000


def panel(n_dates: int = N_DATES) -> pd.DataFrame:
    """A pooled feature frame with one column."""
    dates = pd.date_range("2010-01-04", periods=n_dates, freq="B")
    index = pd.MultiIndex.from_product([dates, TICKERS], names=["date", "ticker"])

    return pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)


def touch_times(X: pd.DataFrame, horizon: int) -> pd.Series:
    """Each label ends `horizon` trading days after it starts."""
    dates = row_dates(X.index).unique()
    ahead = pd.Series(dates, index=dates).shift(-horizon)

    return pd.Series(ahead.reindex(row_dates(X.index)).to_numpy(), index=X.index)


def date_codes(X: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    """Position among the unique dates of each given row."""
    dates = row_dates(X.index)

    return dates.unique().get_indexer(dates[positions])


# --- Embargo ----------------------------------------------------------------


@pytest.mark.parametrize("pct", [0.01, 0.02, 0.05])
def test_embargo_gap_is_exactly_pct_of_the_dates(pct):
    X = panel()
    folds = list(PurgedKFold(n_splits=5, embargo_pct=pct).split(X))
    width = embargo_bars(N_DATES, pct)

    # A middle fold has training rows on both sides, so the gap is visible.
    train, test = folds[2]
    first_after = date_codes(X, train)[date_codes(X, train) > date_codes(X, test).max()].min()

    assert width == int(N_DATES * pct)
    assert first_after - date_codes(X, test).max() - 1 == width


def test_no_train_index_falls_inside_the_embargo():
    X = panel()
    width = embargo_bars(N_DATES, 0.02)

    for train, test in PurgedKFold(n_splits=6, embargo_pct=0.02).split(X):
        end = date_codes(X, test).max()
        codes = date_codes(X, train)
        assert not ((codes > end) & (codes <= end + width)).any()


def test_embargo_only_trims_after_the_test_block():
    X = panel()
    dates = row_dates(X.index)
    test = np.flatnonzero((dates >= dates.unique()[400]) & (dates <= dates.unique()[499]))
    train = np.setdiff1d(np.arange(len(X)), test)

    kept = apply_embargo(train, test, 0.01, dates)
    removed = date_codes(X, np.setdiff1d(train, kept))

    assert set(removed) == set(range(500, 510))


# --- Purging ----------------------------------------------------------------


def test_no_train_index_falls_inside_a_test_window():
    X = panel()

    for train, test in PurgedKFold(n_splits=6, embargo_pct=0.01).split(X):
        assert not np.isin(row_dates(X.index)[train], row_dates(X.index)[test]).any()


def test_one_day_labels_lose_nothing_to_purging():
    X = panel()
    touch = touch_times(X, 1)
    dates = row_dates(X.index)
    test = np.flatnonzero((dates >= dates.unique()[400]) & (dates <= dates.unique()[499]))
    train = np.setdiff1d(np.arange(len(X)), test)

    assert len(purge_train_indices(train, test, touch)) == len(train)


def test_overlapping_labels_are_purged_on_both_sides():
    X = panel()
    touch = touch_times(X, 5)
    dates = row_dates(X.index)
    test = np.flatnonzero((dates >= dates.unique()[400]) & (dates <= dates.unique()[499]))
    train = np.setdiff1d(np.arange(len(X)), test)

    removed = set(date_codes(X, np.setdiff1d(train, purge_train_indices(train, test, touch))))

    # Before: labels from 396 to 399 reach into bars 401+. After: the last
    # test labels run to 504, so training labels starting before then overlap.
    assert removed == {396, 397, 398, 399, 500, 501, 502, 503}


def test_every_fold_passes_the_leak_check():
    X = panel()
    touch = touch_times(X, 5)
    dates = row_dates(X.index)
    width = embargo_bars(N_DATES, 0.01)

    for train, test in PurgedKFold(n_splits=6, embargo_pct=0.01, touch_times=touch).split(X):
        assert_no_overlap(train, test, width, dates=dates, touch_times=touch)


def test_folds_cover_every_date_once_in_order():
    X = panel()
    tests = [test for _, test in PurgedKFold(n_splits=6).split(X)]

    assert np.array_equal(np.sort(np.concatenate(tests)), np.arange(len(X)))
    assert all(
        row_dates(X.index)[a].max() < row_dates(X.index)[b].min()
        for a, b in itertools.pairwise(tests)
    )


def test_tickers_on_the_same_date_stay_together():
    X = panel()

    for train, test in PurgedKFold(n_splits=4).split(X):
        assert len(test) % len(TICKERS) == 0
        assert len(train) % len(TICKERS) == 0


def test_sklearn_interface():
    splitter = PurgedKFold(n_splits=4)

    assert splitter.get_n_splits() == 4
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)


# --- Leak check -------------------------------------------------------------


def test_leak_check_catches_a_shared_date():
    X = panel()
    dates = row_dates(X.index)

    with pytest.raises(LeakageError, match="both train and test"):
        assert_no_overlap(np.arange(0, 30), np.arange(27, 60), 0, dates=dates)


def test_leak_check_catches_a_row_just_after_the_test_block():
    X = panel()
    dates = row_dates(X.index)
    test = np.arange(0, 30)
    train = np.arange(30, 300)

    with pytest.raises(LeakageError, match="within 5 days"):
        assert_no_overlap(train, test, 5, dates=dates)


def test_leak_check_catches_an_overlapping_label():
    X = panel()
    touch = touch_times(X, 5)
    dates = row_dates(X.index)
    test = np.flatnonzero(dates >= dates.unique()[500])
    train = np.flatnonzero(dates < dates.unique()[500])

    with pytest.raises(LeakageError, match="overlap"):
        assert_no_overlap(train, test, 0, dates=dates, touch_times=touch)


# --- Chronological split ----------------------------------------------------


def test_chronological_split_has_zero_overlap():
    X = panel()
    y = pd.Series(1.0, index=X.index)
    X_train, X_test, y_train, y_test = train_test_split_chronological(X, y, "2012-01-02")

    train_dates, test_dates = row_dates(X_train.index), row_dates(X_test.index)

    assert len(X_train) + len(X_test) == len(X)
    assert train_dates.max() < test_dates.min()
    assert test_dates.min() == pd.Timestamp("2012-01-02")
    assert y_train.index.equals(X_train.index)
    assert y_test.index.equals(X_test.index)


def test_chronological_split_purges_labels_ending_in_test():
    X = panel()
    y = pd.Series(1.0, index=X.index)
    touch = touch_times(X, 5)
    X_train, _, _, _ = train_test_split_chronological(X, y, "2012-01-02", touch)
    plain, _, _, _ = train_test_split_chronological(X, y, "2012-01-02")

    # The last four training dates have labels that reach past the first test date.
    assert len(plain) - len(X_train) == 4 * len(TICKERS)
    assert (touch.reindex(X_train.index) <= pd.Timestamp("2012-01-02")).all()
