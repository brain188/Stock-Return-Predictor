"""Cross-validation that cannot train on the answers.

Standard k-fold shuffles, so a model trains on next week and is tested on last
week. Keeping the folds in time order is the minimum, and it is still not
enough, because neighbouring rows share information in two separate ways:

- Labels can overlap. A label dated t is the return over (t, touch_time]. If a
  training label's span reaches into the test window, the model has been
  fitted on part of the test outcome. Purging removes those training rows.
- Features overlap. A 21 day volatility on the first training day after a
  test block is computed from 20 test-period returns, so that row carries test
  information into training. The embargo removes a gap of training rows after
  each test block.

For the 1 day fixed horizon label no two labels overlap, so purging removes
almost nothing and the embargo does the real work. For the 5 day triple
barrier, purging removes up to four days of training labels on each side of
a test block. The embargo cannot fully cover the fracdiff feature, whose
weights reach back further than any reasonable gap; it bounds the leak rather
than removing it.

The panel is pooled, so splits are made on dates, never on rows: every ticker
on a given date lands on the same side. Indices are positions into X, as in
scikit-learn, and work with a (date, ticker) MultiIndex or a plain date index.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


class LeakageError(AssertionError):
    """Raised when a training row shares information with the test set."""


def row_dates(index: pd.Index) -> pd.DatetimeIndex:
    """The date of each row, from a (date, ticker) MultiIndex or a date index."""
    if isinstance(index, pd.MultiIndex):
        return pd.DatetimeIndex(index.get_level_values("date"))

    return pd.DatetimeIndex(index)


def date_codes(dates: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Position of each row's date among the sorted unique dates, and those dates."""
    unique = dates.unique().sort_values()

    return unique.get_indexer(dates), unique


def _blocks(codes: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of date positions, as (first, last) pairs."""
    positions = np.unique(codes)
    if not len(positions):
        return []

    breaks = np.flatnonzero(np.diff(positions) > 1)
    firsts = positions[np.r_[0, breaks + 1]]
    lasts = positions[np.r_[breaks, len(positions) - 1]]

    return list(zip(firsts.tolist(), lasts.tolist(), strict=True))


def embargo_bars(n_dates: int, embargo_pct: float) -> int:
    """Width of the embargo in trading days: embargo_pct of the dates, rounded down."""
    return int(n_dates * embargo_pct)


def _label_ends(touch_times: pd.Series) -> np.ndarray:
    """Each label's end date; a missing touch time means the label ends where it starts."""
    starts = row_dates(touch_times.index)
    ends = pd.DatetimeIndex(touch_times)

    return np.where(ends.isna(), starts, ends).astype("datetime64[ns]")


def purge_train_indices(
    train: np.ndarray,
    test: np.ndarray,
    touch_times: pd.Series,
) -> np.ndarray:
    """Drop training rows whose label span overlaps a test block's label span.

    A label dated t covers the returns on bars (t, touch_time]. A training row
    is purged when that span intersects the span of any contiguous test
    block, from the block's first date to the latest touch time in it. The
    1 day label at t - 1 covers only bar t, which is not part of the test
    label at t, so it survives.

    Args:
        train: Positions of the candidate training rows.
        test: Positions of the test rows.
        touch_times: End date of each row's label, indexed like X. NaT means
            the label ends on its own date.

    Returns:
        The training positions that survive, in their original order.
    """
    train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
    starts = row_dates(touch_times.index).to_numpy()
    ends = _label_ends(touch_times)
    codes, unique = date_codes(row_dates(touch_times.index))

    keep = np.ones(len(train), dtype=bool)

    for first, last in _blocks(codes[test]):
        in_block = test[(codes[test] >= first) & (codes[test] <= last)]
        block_start = unique[first].to_datetime64()
        block_end = max(ends[in_block].max(), unique[last].to_datetime64())
        keep &= ~((starts[train] < block_end) & (ends[train] > block_start))

    return train[keep]


def apply_embargo(
    train: np.ndarray,
    test: np.ndarray,
    embargo_pct: float,
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    """Drop the training rows in the embargo gap after each test block.

    The gap is embargo_bars(n_dates, embargo_pct) trading days wide, where
    n_dates counts the unique dates in the sample. Only rows after a test
    block are embargoed: rows before it are in the past of every test row, so
    test features computed from them are information the model may have.

    Args:
        train: Positions of the candidate training rows.
        test: Positions of the test rows.
        embargo_pct: Gap as a share of the dates.
        dates: Date of every row of X.

    Returns:
        The training positions outside every embargo gap.
    """
    train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
    codes, unique = date_codes(pd.DatetimeIndex(dates))
    width = embargo_bars(len(unique), embargo_pct)

    keep = np.ones(len(train), dtype=bool)

    for _, last in _blocks(codes[test]):
        keep &= ~((codes[train] > last) & (codes[train] <= last + width))

    return train[keep]


def train_test_split_chronological(
    X: pd.DataFrame,
    y: pd.Series,
    test_start: str | pd.Timestamp,
    touch_times: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """The plain split the brief asks for: train before test_start, test from it.

    Nothing is shuffled. When touch times are given, training rows whose
    label ends after the first test date are purged, since their outcome is
    made of test-period returns.

    Args:
        X: Features indexed by (date, ticker) or by date.
        y: Target on the same index.
        test_start: First test date.
        touch_times: Optional label end dates on the same index.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    dates = row_dates(X.index)
    test_mask = dates >= pd.Timestamp(test_start)
    train_mask = ~test_mask

    if touch_times is not None and test_mask.any():
        first_test = dates[test_mask].min().to_datetime64()
        ends = _label_ends(touch_times.reindex(X.index))
        purged = train_mask & (ends > first_test)
        train_mask &= ~purged
        logger.info("Chronological split: purged %s training rows", int(purged.sum()))

    logger.info(
        "Chronological split: %s train rows to %s, %s test rows from %s",
        f"{train_mask.sum():,}",
        dates[train_mask].max().date() if train_mask.any() else None,
        f"{test_mask.sum():,}",
        dates[test_mask].min().date() if test_mask.any() else None,
    )

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def assert_no_overlap(
    train: np.ndarray,
    test: np.ndarray,
    lookback: int,
    *,
    dates: pd.DatetimeIndex,
    touch_times: pd.Series | None = None,
) -> None:
    """Raise if any training row could carry information about the test set.

    Checks that no date is on both sides, that no training row sits within
    `lookback` trading days after a test block (where its features would be
    built from test data), and, with touch times, that no training label
    overlaps a test label.

    Args:
        train: Positions of the training rows.
        test: Positions of the test rows.
        lookback: Trading days after each test block that must hold no
            training row. Pass the embargo width to check the embargo, or the
            longest feature window to check full feature independence.
        dates: Date of every row of X.
        touch_times: Optional label end dates, indexed like X.

    Raises:
        LeakageError: Naming the first rule broken.
    """
    train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
    codes, unique = date_codes(pd.DatetimeIndex(dates))

    shared = np.intersect1d(codes[train], codes[test])
    if len(shared):
        raise LeakageError(
            f"{len(shared)} dates are in both train and test, first {unique[shared[0]].date()}"
        )

    for _, last in _blocks(codes[test]):
        inside = (codes[train] > last) & (codes[train] <= last + lookback)
        if inside.any():
            raise LeakageError(
                f"{int(inside.sum())} training rows fall within {lookback} days after the test "
                f"block ending {unique[last].date()}"
            )

    if touch_times is not None:
        survivors = purge_train_indices(train, test, touch_times)
        if len(survivors) < len(train):
            raise LeakageError(f"{len(train) - len(survivors)} training labels overlap test labels")


class PurgedKFold:
    """K-fold over contiguous blocks of dates, purged and embargoed.

    Matches the scikit-learn splitter interface, so it can be passed as cv=
    to cross_val_score and friends. Folds are not shuffled: fold k tests on
    the k-th block of dates and trains on everything else that survives
    purging and the embargo, including data after the test block.

    Args:
        n_splits: Number of folds.
        embargo_pct: Embargo after each test block, as a share of the dates.
        touch_times: Label end dates indexed like X. None treats every label
            as ending on its own date, so nothing is purged.
    """

    def __init__(
        self,
        n_splits: int = 6,
        embargo_pct: float = 0.01,
        touch_times: pd.Series | None = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")

        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.touch_times = touch_times

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of folds, as scikit-learn expects."""
        return self.n_splits

    def _touch(self, index: pd.Index) -> pd.Series:
        """Touch times aligned to X, NaT where none was given."""
        if self.touch_times is None:
            return pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")

        return self.touch_times.reindex(index)

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        groups: object = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train, test) position arrays, one pair per fold.

        Args:
            X: Features indexed by (date, ticker) or by date.
            y: Unused; accepted for scikit-learn compatibility.
            groups: Unused; dates already define the groups.

        Yields:
            Training and test positions into X.
        """
        dates = row_dates(X.index)
        codes, unique = date_codes(dates)

        if len(unique) < self.n_splits:
            raise ValueError(f"{len(unique)} dates cannot make {self.n_splits} folds")

        touch = self._touch(X.index)
        positions = np.arange(len(X))

        for fold, block in enumerate(np.array_split(np.arange(len(unique)), self.n_splits)):
            in_test = (codes >= block[0]) & (codes <= block[-1])
            test, candidates = positions[in_test], positions[~in_test]

            train = purge_train_indices(candidates, test, touch)
            purged = len(candidates) - len(train)
            train = apply_embargo(train, test, self.embargo_pct, dates)

            logger.debug(
                "Fold %s: test %s to %s, %s train rows (%s purged, %s embargoed)",
                fold,
                unique[block[0]].date(),
                unique[block[-1]].date(),
                len(train),
                purged,
                len(candidates) - purged - len(train),
            )

            yield train, test
