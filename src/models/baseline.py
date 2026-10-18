"""The forecasts a model must beat before it is interesting.

Knowing what a coin flip and a constant guess score on this data, before
looking at any model output, is the cheapest protection against fooling
myself. AlwaysUp is the hard bar: equities rise on most days, so predicting up
every day is right more than half the time for free, and it is the bar the
model in notebook 3 lost to.

Every baseline has the same interface as the real models (fit, predict,
predict_proba, name, params), so the validation and backtest code can run any
of them without knowing which it has. Predictions are directions: +1 for up
and -1 for not up, the same rule the labels use. predict_proba is the
probability of up.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class Forecaster:
    """Shared interface for every forecaster in the project.

    Subclasses implement _direction. fit returns self so calls chain, and
    sample_weight is accepted everywhere so callers never special-case a
    model that ignores it.
    """

    name = "forecaster"

    @property
    def params(self) -> dict[str, Any]:
        """Settings recorded in the trial log."""
        return {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weight: pd.Series | np.ndarray | None = None,
    ) -> Forecaster:
        """Learn from the training data. Most baselines learn nothing."""
        return self

    def _direction(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Direction per row: +1 up, -1 not up."""
        return self._direction(X).astype(float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of up per row. Baselines are certain, so 0 or 1."""
        return (self.predict(X) > 0).astype(float)


class AlwaysUp(Forecaster):
    """Predicts up every day.

    Its hit rate is exactly the share of up days, about 55% for SPY. A model
    with a lower hit rate is worse than knowing nothing but the drift.
    """

    name = "always_up"

    def _direction(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X))


class BuyAndHold(AlwaysUp):
    """The passive benchmark: long every day.

    As a forecast it is identical to AlwaysUp. It exists as its own class
    because it is the strategy every backtest is compared against, and a
    report should name it as such.
    """

    name = "buy_and_hold"


class RandomForecaster(Forecaster):
    """A coin flip, to show what luck looks like.

    The draws depend only on the seed and the number of rows, so the same
    seed gives the same predictions on every call.

    Args:
        seed: Random seed.
        p_up: Probability of predicting up.
    """

    name = "random"

    def __init__(self, seed: int = 42, p_up: float = 0.5) -> None:
        self.seed = seed
        self.p_up = p_up

    @property
    def params(self) -> dict[str, Any]:
        return {"seed": self.seed, "p_up": self.p_up}

    def _direction(self, X: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(self.seed)

        return np.where(rng.random(len(X)) < self.p_up, 1.0, -1.0)


class LastSignPersistence(Forecaster):
    """Tomorrow repeats today: the naive momentum rule.

    Reads today's return from a feature column, so it uses nothing the
    models do not also see. A zero return counts as not up.

    Args:
        column: Feature holding today's return.
        reverse: Predict the opposite of today instead, the naive reversal
            rule. Notebook 3 found reversal the better of the two, which fits
            the negative IC of the short-horizon features.
    """

    def __init__(self, column: str = "ret_1d", reverse: bool = False) -> None:
        self.column = column
        self.reverse = reverse

    @property
    def name(self) -> str:
        return "reverse_last_sign" if self.reverse else "last_sign"

    @property
    def params(self) -> dict[str, Any]:
        return {"column": self.column, "reverse": self.reverse}

    def _direction(self, X: pd.DataFrame) -> np.ndarray:
        today = X[self.column].to_numpy(dtype=float)
        direction = np.where(today > 0, 1.0, -1.0)

        return -direction if self.reverse else direction


BASELINES: dict[str, type[Forecaster]] = {
    "always_up": AlwaysUp,
    "buy_and_hold": BuyAndHold,
    "random": RandomForecaster,
    "last_sign": LastSignPersistence,
}
