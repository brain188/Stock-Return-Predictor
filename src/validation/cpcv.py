"""Combinatorial purged cross-validation: many backtest paths from one history.

A single walk forward gives one sequence of out-of-sample predictions, one
score, and no way to tell whether that score was luck. CPCV splits the dates
into N contiguous groups and tests on every combination of k of them, training
on the rest after purging and embargo. With N = 6 and k = 2 that is 15 splits.

Each group is tested in C(N - 1, k - 1) of those splits, so the predictions
can be reassembled into that many complete paths, each covering the whole
timeline exactly once with predictions from a different mix of models: 5
paths for N = 6, k = 2. The spread of a score across the paths is the honest
measure of its uncertainty, and it is what later turns the Monte Carlo
evaluation from decoration into evidence.

The paths are not independent. Every pair shares training data, so their
spread understates the true uncertainty a little; it is still far better than
the single number a walk forward gives.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from math import comb

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.validation.purged_cv import (
    apply_embargo,
    date_codes,
    purge_train_indices,
    row_dates,
)

logger = get_logger(__name__)


def combinatorial_splits(n_groups: int = 6, n_test_groups: int = 2) -> list[tuple[int, ...]]:
    """Every choice of test groups, in lexicographic order.

    Args:
        n_groups: Contiguous date groups, N.
        n_test_groups: Groups held out per split, k.

    Returns:
        C(N, k) tuples of group numbers, for example (0, 1), (0, 2), ...
    """
    if not 0 < n_test_groups < n_groups:
        raise ValueError(f"need 0 < n_test_groups < n_groups, got {n_test_groups}, {n_groups}")

    return list(itertools.combinations(range(n_groups), n_test_groups))


def path_count(n_groups: int, n_test_groups: int) -> int:
    """Complete backtest paths the configuration produces: C(N - 1, k - 1).

    There are C(N, k) splits with k test groups each, so C(N, k) * k / N
    predictions per group, and one path per prediction of a group.
    """
    return comb(n_groups - 1, n_test_groups - 1)


def backtest_paths(splits: Sequence[tuple[int, ...]], n_groups: int) -> np.ndarray:
    """Assign each (path, group) to the split whose prediction it uses.

    Group g appears in several splits; path p takes g's prediction from the
    p-th of them. Every path therefore draws each group from exactly one
    split, and no two paths use the same split for the same group.

    Args:
        splits: Output of combinatorial_splits.
        n_groups: N.

    Returns:
        An integer array of shape (n_paths, n_groups): entry [p, g] is the
        index into splits that supplies group g on path p.
    """
    uses = [[i for i, groups in enumerate(splits) if g in groups] for g in range(n_groups)]
    counts = {len(u) for u in uses}

    if len(counts) != 1:
        raise ValueError("every group must appear in the same number of splits")

    return np.array(uses, dtype=int).T


class CombinatorialPurgedCV:
    """CPCV splitter with the scikit-learn split interface.

    Args:
        n_groups: Contiguous date groups, N.
        n_test_groups: Groups held out per split, k.
        embargo_pct: Embargo after each test block, as a share of the dates.
        touch_times: Label end dates indexed like X, for purging. None treats
            every label as ending on its own date.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
        touch_times: pd.Series | None = None,
    ) -> None:
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.touch_times = touch_times
        self.splits = combinatorial_splits(n_groups, n_test_groups)
        self.paths = backtest_paths(self.splits, n_groups)

    @property
    def n_paths(self) -> int:
        """Number of complete backtest paths."""
        return path_count(self.n_groups, self.n_test_groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of splits, as scikit-learn expects."""
        return len(self.splits)

    def groups(self, X: pd.DataFrame) -> np.ndarray:
        """Group number of every row, from contiguous blocks of dates."""
        codes, unique = date_codes(row_dates(X.index))

        if len(unique) < self.n_groups:
            raise ValueError(f"{len(unique)} dates cannot make {self.n_groups} groups")

        group_of_date = np.empty(len(unique), dtype=int)
        for g, block in enumerate(np.array_split(np.arange(len(unique)), self.n_groups)):
            group_of_date[block] = g

        return group_of_date[codes]

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        groups: object = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield purged and embargoed (train, test) positions, one per split.

        Args:
            X: Features indexed by (date, ticker) or by date.
            y: Unused; accepted for scikit-learn compatibility.
            groups: Unused; the date groups are computed from X.

        Yields:
            Training and test positions into X, in the order of self.splits.
        """
        row_group = self.groups(X)
        dates = row_dates(X.index)
        positions = np.arange(len(X))
        touch = (
            pd.Series(pd.NaT, index=X.index, dtype="datetime64[ns]")
            if self.touch_times is None
            else self.touch_times.reindex(X.index)
        )

        for number, test_groups in enumerate(self.splits):
            in_test = np.isin(row_group, test_groups)
            test, candidates = positions[in_test], positions[~in_test]
            train = apply_embargo(
                purge_train_indices(candidates, test, touch), test, self.embargo_pct, dates
            )

            logger.debug(
                "Split %s, test groups %s: %s train rows, %s removed by purge and embargo",
                number,
                test_groups,
                len(train),
                len(candidates) - len(train),
            )

            yield train, test


def path_predictions(
    split_predictions: Sequence[pd.Series],
    cv: CombinatorialPurgedCV,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Reassemble out-of-sample predictions into one column per backtest path.

    Args:
        split_predictions: For each split, in cv.splits order, predictions
            indexed by the rows of its test set.
        cv: The splitter that produced the splits.
        X: The frame that was split.

    Returns:
        Indexed like X, one column per path ("path_0", ...). Every row is
        filled on every path.
    """
    row_group = cv.groups(X)
    columns = {}

    for p, sources in enumerate(cv.paths):
        path = np.full(len(X), np.nan)
        for g, split in enumerate(sources):
            rows = np.flatnonzero(row_group == g)
            path[rows] = split_predictions[split].reindex(X.index[rows]).to_numpy()
        columns[f"path_{p}"] = path

    return pd.DataFrame(columns, index=X.index)
