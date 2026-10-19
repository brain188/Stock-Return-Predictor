"""One interface over RandomForest and LightGBM, chosen by config.

Swapping estimators changes model.estimator in config and nothing else. The
wrapper has the same fit, predict and predict_proba as the baselines, so every
model in the project can be validated and backtested by the same code.

The model is pooled: one estimator is fitted on all tickers' rows stacked
together. That assumes the relationship between features and next-day return
is shared across tickers, which buys eleven times the data at the cost of
ignoring anything ticker specific. Every feature is a ratio or a return
rather than a price level, which is what makes pooling defensible at all.

Random Forest is the primary model. With roughly 5,000 daily bars per ticker
there is too little data for a recurrent network to beat a tree, and LightGBM
is here as the stronger gradient boosting comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

from src.config import Config
from src.logger import get_logger
from src.models.baseline import Forecaster

logger = get_logger(__name__)

ESTIMATOR_CLASSES = {
    ("random_forest", "regression"): RandomForestRegressor,
    ("random_forest", "classification"): RandomForestClassifier,
    ("lightgbm", "regression"): LGBMRegressor,
    ("lightgbm", "classification"): LGBMClassifier,
}


class ModelError(Exception):
    """Raised when a model is used in a way that would give a wrong answer silently."""


class ReturnPredictor(Forecaster):
    """A tree ensemble that predicts next-day returns or their direction.

    In regression, predict returns the forecast return and predict_proba is
    unavailable, since a point forecast carries no probability. In
    classification, predict returns +1 or -1 and predict_proba the
    probability of up.

    The feature columns seen in fit are remembered, and predict refuses a
    frame whose columns differ, so a model can never silently score the
    wrong columns.

    Args:
        estimator: "random_forest" or "lightgbm".
        task: "regression" or "classification".
        params: Hyperparameters passed to the underlying estimator.
        seed: Random seed, so two fits on the same data are identical.
    """

    def __init__(
        self,
        estimator: str,
        task: str = "regression",
        params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        if (estimator, task) not in ESTIMATOR_CLASSES:
            raise ModelError(f"no estimator for ({estimator}, {task})")

        self.estimator = estimator
        self.task = task
        self.seed = seed
        self._params = dict(params or {})
        self.model = ESTIMATOR_CLASSES[estimator, task](random_state=seed, **self._params)
        self.feature_names_: list[str] = []

    @property
    def name(self) -> str:
        return f"{self.estimator}_{self.task}"

    @property
    def params(self) -> dict[str, Any]:
        return {"estimator": self.estimator, "task": self.task, "seed": self.seed, **self._params}

    def _target(self, y: pd.Series | np.ndarray) -> np.ndarray:
        """The target the estimator is fitted on: returns, or up / not up."""
        values = np.asarray(y, dtype=float)

        return np.where(values > 0, 1, -1) if self.task == "classification" else values

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weight: pd.Series | np.ndarray | None = None,
    ) -> ReturnPredictor:
        """Fit on a feature frame and a target.

        Args:
            X: Features, one column per feature_names entry.
            y: Forward returns, or labels of +1 / -1. In classification any
                positive value counts as up.
            sample_weight: Optional weight per row, from sample_weights().

        Returns:
            self.
        """
        weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.feature_names_ = list(X.columns)
        self.model.fit(X.to_numpy(dtype=float), self._target(y), sample_weight=weight)

        return self

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        """X as an array, after checking its columns match the fitted ones."""
        if not self.feature_names_:
            raise ModelError(f"{self.name} has not been fitted")

        if list(X.columns) != self.feature_names_:
            raise ModelError(
                f"{self.name} was fitted on {self.feature_names_}, got {list(X.columns)}"
            )

        return X.to_numpy(dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Forecast return (regression) or direction +1 / -1 (classification)."""
        return self.model.predict(self._matrix(X)).astype(float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of up, classification only.

        Raises:
            ModelError: In regression, which has no probability to give.
        """
        if self.task != "classification":
            raise ModelError("predict_proba needs model.task = classification")

        up = list(self.model.classes_).index(1)

        return self.model.predict_proba(self._matrix(X))[:, up]


def build_estimator(config: Config, task: str | None = None) -> Forecaster:
    """The model named by config.model.estimator, with its config parameters.

    Args:
        config: Loaded project configuration.
        task: Overrides config.model.task.

    Returns:
        An unfitted ReturnPredictor, or an LSTMPredictor for "lstm".
    """
    task = task or config.model.task
    seed = config.project.random_seed

    if config.model.estimator == "lstm":
        # Imported here so tree models never pay for loading torch.
        from src.models.sequence import LSTMPredictor

        return LSTMPredictor(config.model.lstm, task=task, seed=seed)

    return ReturnPredictor(config.model.estimator, task, config.model.params, seed)


def fit_with_weights(
    model: Forecaster,
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series | None = None,
) -> Forecaster:
    """Fit a model with its sample weights aligned row for row.

    Weights from sample_weights() are indexed by (date, ticker). Passing them
    as a bare array would trust that their order matches X; aligning on the
    index makes that a guarantee. Rows with no label or no weight are dropped,
    and the count is logged.

    Args:
        model: Any forecaster.
        X: Features indexed by (date, ticker).
        y: Target on the same index.
        weights: Sample weights on the same index, or None for equal weights.

    Returns:
        The fitted model.

    Raises:
        ModelError: If y or the weights do not cover X's index.
    """
    if not X.index.isin(y.index).all():
        raise ModelError("target does not cover every feature row")

    target = y.reindex(X.index)
    usable = target.notna()

    if weights is not None:
        if not X.index.isin(weights.index).all():
            raise ModelError("weights do not cover every feature row")
        weights = weights.reindex(X.index)
        usable &= weights.notna() & (weights > 0)

    dropped = int((~usable).sum())
    if dropped:
        logger.info("Fitting %s: dropped %s rows with no label or weight", model.name, dropped)

    model.fit(
        X[usable],
        target[usable],
        sample_weight=None if weights is None else weights[usable].to_numpy(),
    )

    return model


def feature_importance(
    model: ReturnPredictor,
    names: list[str] | None = None,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    n_repeats: int = 5,
    scoring: Any = None,
) -> pd.DataFrame:
    """Split-based and, when data is given, permutation importance.

    Split-based importance comes free with the fit but favours features with
    many distinct values. Permutation importance measures how much the score
    drops when a feature is shuffled, which is closer to the question asked,
    and should be computed on held-out data.

    Args:
        model: A fitted ReturnPredictor.
        names: Feature names. Defaults to the names seen in fit.
        X: Held-out features for permutation importance.
        y: Held-out target for permutation importance.
        n_repeats: Shuffles per feature.
        scoring: Scorer for permutation importance. None uses the
            estimator's own score (R squared or accuracy); the research
            pipeline passes an IC scorer, since R squared on daily returns is
            dominated by noise.

    Returns:
        One row per feature, sorted by split importance, with columns split
        (normalised to sum to 1) and, if data was given, permutation and
        permutation_std.

    Raises:
        ModelError: If the names do not match the fitted features.
    """
    names = names or model.feature_names_

    if list(names) != model.feature_names_:
        raise ModelError(f"names {names} do not match fitted features {model.feature_names_}")

    split = np.asarray(model.model.feature_importances_, dtype=float)
    table = pd.DataFrame({"split": split / split.sum() if split.sum() else split}, index=names)
    table.index.name = "feature"

    if X is not None and y is not None:
        result = permutation_importance(
            model.model,
            model._matrix(X),
            model._target(y),
            n_repeats=n_repeats,
            random_state=model.seed,
            scoring=scoring,
        )
        table["permutation"] = result.importances_mean
        table["permutation_std"] = result.importances_std

    return table.sort_values("split", ascending=False)
