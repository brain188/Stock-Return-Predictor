"""Feature research end to end, judged on IC and importance, never on a backtest.

The temptation is to run a backtest, see a bad Sharpe, tweak a feature and run
it again. Twenty rounds of that and the strategy is fitted to the test set
with no way to measure the damage (pitfall 2). Research here is scored only
on forecast quality: IC, hit rate against AlwaysUp, and whether the features
that matter stay the same from fold to fold and the signal survives a change
of regime. The backtest runs once, in Phase 6, after the features are settled.

Everything here uses the development period only, the dates before
validation.test_start. The holdout after it is never touched by research, so
it is still out of sample when the final backtest runs. Development labels
whose outcome reaches into the holdout are dropped at the boundary.

Validation is CPCV with purging and embargo: 15 splits and 5 complete paths
for the default N = 6, k = 2. Every metric is reported per path, and the
spread across paths is the uncertainty. Every call appends one row to the
trial log, including a run that fails, because it was still an attempt.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer

from src.config import PROJECT_ROOT, Config
from src.data.cleaning import load_clean_panel
from src.evaluation.trial_log import TrialLog
from src.features.build import build_features
from src.labelling.fixed_horizon import label_fixed_horizon
from src.labelling.sample_weights import sample_weights
from src.labelling.triple_barrier import label_triple_barrier
from src.logger import get_logger, log_frame, log_step
from src.models.baseline import AlwaysUp, Forecaster, LastSignPersistence, RandomForecaster
from src.models.estimator import (
    ReturnPredictor,
    build_estimator,
    feature_importance,
    fit_with_weights,
)
from src.reporting.figures import volatility_regimes
from src.validation.cpcv import CombinatorialPurgedCV, path_predictions
from src.validation.metrics import (
    ic_by_period,
    ic_t_stat,
    information_coefficient,
    metric_table,
    score,
)
from src.validation.purged_cv import embargo_bars, row_dates

logger = get_logger(__name__)

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "results" / "research"

# Correlation above which two features are treated as near duplicates.
REDUNDANCY_THRESHOLD = 0.7

# Rows of each held-out fold used for permutation importance, to bound its cost.
PERMUTATION_ROWS = 5_000

LABELLERS = {"fixed_horizon": label_fixed_horizon, "triple_barrier": label_triple_barrier}


def _ic_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """IC as a scikit-learn score, for permutation importance."""
    return information_coefficient(y_pred, y_true)


IC_SCORER = make_scorer(_ic_score)


@dataclass(frozen=True)
class ResearchData:
    """Development-period features, target and weights, aligned row for row."""

    X: pd.DataFrame
    labels: pd.DataFrame
    weights: pd.Series
    fracdiff_d: float

    @property
    def actual(self) -> pd.Series:
        """Realised return of each label, what every metric scores against."""
        return self.labels["return"]


@dataclass
class ResearchResult:
    """Everything one research run produced."""

    name: str
    trial_id: int
    metrics: dict[str, float]
    path_metrics: pd.DataFrame
    comparison: pd.DataFrame
    predictions: pd.DataFrame
    ic_by_year: pd.DataFrame
    ic_by_regime: pd.DataFrame
    redundancy: pd.DataFrame
    importance: pd.DataFrame | None = None
    files: list[Path] = field(default_factory=list)

    def save(self, out_dir: str | Path = DEFAULT_OUTPUT) -> list[Path]:
        """Write every table as CSV under out_dir / name.

        Returns:
            The files written.
        """
        folder = Path(out_dir) / self.name
        folder.mkdir(parents=True, exist_ok=True)

        tables = {
            "metrics": pd.Series(self.metrics, name="value").to_frame(),
            "paths": self.path_metrics,
            "comparison": self.comparison,
            "ic_by_year": self.ic_by_year,
            "ic_by_regime": self.ic_by_regime,
            "redundancy": self.redundancy,
            "importance": self.importance,
        }

        self.files = []
        for label, table in tables.items():
            if table is not None:
                path = folder / f"{label}.csv"
                table.to_csv(path)
                self.files.append(path)

        logger.info("Saved %s research tables to %s", len(self.files), folder)

        return self.files


def with_overrides(
    config: Config,
    *,
    label_method: str | None = None,
    estimator: str | None = None,
) -> Config:
    """A copy of the config with the labelling method or estimator replaced.

    The overrides go through the same validation as the YAML, so a typo on
    the command line fails the same way a typo in the file does.
    """
    if label_method:
        labelling = dataclasses.replace(config.labelling, method=label_method)
        config = dataclasses.replace(config, labelling=labelling)

    if estimator:
        config = dataclasses.replace(
            config, model=dataclasses.replace(config.model, estimator=estimator)
        )

    config.validate()

    return config


def run_name(config: Config) -> str:
    """Name of a run in the trial log and the output folder."""
    return f"research_{config.model.estimator}_{config.model.task}_{config.labelling.method}"


def prepare_research_data(panel: pd.DataFrame, config: Config) -> ResearchData:
    """Features, labels and weights for the development period.

    Rows are kept only where features, label and weight all exist, the date
    is before test_start, and the label's outcome is known by the first
    holdout date. The last of those rules matters for the 5 day triple
    barrier, whose final development labels would otherwise be made of
    holdout returns.

    Args:
        panel: Cleaned panel indexed by (date, ticker).
        config: Loaded project configuration.

    Returns:
        Aligned development data.
    """
    features = build_features(panel, config)
    labels = LABELLERS[config.labelling.method](panel, config)
    weights = sample_weights(labels, config)

    holdout = pd.Timestamp(config.validation.test_start)
    first_holdout = panel.index.get_level_values("date")[
        panel.index.get_level_values("date") >= holdout
    ].min()

    labels = labels.reindex(features.index)
    weights = weights.reindex(features.index)
    dates = row_dates(features.index)

    keep = (
        (dates < holdout)
        & labels["return"].notna().to_numpy()
        & weights.notna().to_numpy()
        & (weights > 0).to_numpy()
        & ~(labels["touch_time"] > first_holdout).to_numpy()
    )

    crossing = int(((dates < holdout) & (labels["touch_time"] > first_holdout).to_numpy()).sum())
    logger.info(
        "Development set: %s of %s rows before %s (%s dropped at the holdout boundary)",
        f"{keep.sum():,}",
        f"{len(features):,}",
        holdout.date(),
        crossing,
    )

    data = ResearchData(
        X=features[keep],
        labels=labels[keep],
        weights=weights[keep],
        fracdiff_d=float(features.attrs.get("fracdiff_d", np.nan)),
    )
    log_frame(logger, data.X, "Research features")

    return data


def forecast(model: Forecaster, X: pd.DataFrame, task: str) -> np.ndarray:
    """The score a model gives each row: forecast return, or P(up) - 0.5.

    Centring the probability at zero means the same sign rule (positive is
    up) works for both tasks in every metric.
    """
    if task == "classification":
        return model.predict_proba(X) - 0.5

    return model.predict(X)


def target_for(data: ResearchData, task: str) -> pd.Series:
    """What the model is fitted on: the return in regression, the label in classification.

    For the triple barrier the label is set by which barrier was hit, which
    can differ in sign from a return measured at a fixed time, so
    classification fits the label itself.
    """
    return data.labels["return"] if task == "regression" else data.labels["label"]


def importance_across_folds(
    models: Sequence[ReturnPredictor],
    held_out: Sequence[tuple[pd.DataFrame, pd.Series]] | None = None,
    n_repeats: int = 3,
    scoring: Any = None,
    top: int = 5,
) -> pd.DataFrame:
    """How stable each feature's importance is across the CPCV splits.

    A single importance ranking from one fit is as noisy as the fit. A
    feature worth keeping ranks high in most folds, and its permutation
    importance is positive in most of them. One that tops a single fold
    and vanishes elsewhere is fitting noise.

    Args:
        models: One fitted tree model per split.
        held_out: Each split's test features and target, for permutation
            importance. Without it, only split importance is summarised.
        n_repeats: Shuffles per feature for permutation importance.
        scoring: Permutation scorer, for example IC_SCORER.
        top: Rank cut-off for top_share.

    Returns:
        One row per feature: split_mean and split_std; with held-out data,
        perm_mean, perm_std and positive_share (folds where shuffling the
        feature hurt); and rank_mean, rank_std and top_share, ranked by
        permutation importance when available. Sorted by rank_mean.
    """
    tables = []

    for number, model in enumerate(models):
        X, y = held_out[number] if held_out else (None, None)
        if X is not None and len(X) > PERMUTATION_ROWS:
            sample = np.random.default_rng(number).choice(len(X), PERMUTATION_ROWS, replace=False)
            X, y = X.iloc[np.sort(sample)], y.iloc[np.sort(sample)]
        tables.append(feature_importance(model, X=X, y=y, n_repeats=n_repeats, scoring=scoring))

    stacked = pd.concat(tables, keys=range(len(tables)), names=["fold", "feature"])
    by_feature = stacked.groupby(level="feature")
    key = "permutation" if "permutation" in stacked else "split"
    ranks = stacked[key].groupby(level="fold").rank(ascending=False)

    summary = pd.DataFrame(
        {
            "split_mean": by_feature["split"].mean(),
            "split_std": by_feature["split"].std(),
        }
    )

    if key == "permutation":
        summary["perm_mean"] = by_feature["permutation"].mean()
        summary["perm_std"] = by_feature["permutation"].std()
        summary["positive_share"] = (stacked["permutation"] > 0).groupby(level="feature").mean()

    summary["rank_mean"] = ranks.groupby(level="feature").mean()
    summary["rank_std"] = ranks.groupby(level="feature").std()
    summary["top_share"] = (ranks <= top).groupby(level="feature").mean()

    return summary.sort_values("rank_mean")


def ic_across_regimes(
    predictions: pd.Series,
    actual: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Scores within each market regime: does the signal survive a change?

    Args:
        predictions: Forecasts indexed by (date, ticker).
        actual: Realised returns on the same index.
        regimes: A regime label per date, for example from
            volatility_regimes. Dates without a label are left out.

    Returns:
        One row per regime with ic, hit_rate, edge_over_always_up and n.
    """
    regime = regimes.reindex(row_dates(predictions.index)).to_numpy()
    rows = {}

    for name in pd.Series(regime).dropna().unique():
        mask = regime == name
        result = score(predictions[mask], actual.reindex(predictions.index)[mask])
        rows[name] = {k: result[k] for k in ("ic", "hit_rate", "edge_over_always_up", "n")}

    table = pd.DataFrame(rows).T
    table.index.name = "regime"

    return table.reindex([r for r in ("low", "normal", "high") if r in table.index] or table.index)


def feature_redundancy(
    features: pd.DataFrame, threshold: float = REDUNDANCY_THRESHOLD
) -> pd.DataFrame:
    """Rank correlation between features, clustered so near duplicates sit together.

    Redundant features split importance between them, so each looks less
    important than the information they share, and they make permutation
    importance understate both. Clustering on distance 1 - |rho| with
    average linkage and cutting at 1 - threshold groups features that are
    near copies of one another.

    Args:
        features: Feature matrix.
        threshold: |rho| above which features join a cluster.

    Returns:
        The Spearman correlation matrix with rows and columns in cluster
        order, plus a leading cluster column numbering each group.
    """
    corr = pd.DataFrame(
        spearmanr(features.to_numpy(), nan_policy="omit").statistic,
        index=features.columns,
        columns=features.columns,
    )
    distance = (1 - corr.abs()).clip(lower=0).to_numpy()
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average")

    order = corr.index[leaves_list(tree)]
    clusters = pd.Series(fcluster(tree, t=1 - threshold, criterion="distance"), index=corr.index)

    # Number clusters in the order they first appear, so the table reads top down.
    renumber = {c: i for i, c in enumerate(dict.fromkeys(clusters[order]))}
    table = corr.loc[order, order]
    table.insert(0, "cluster", clusters[order].map(renumber))
    table.index.name = "feature"

    shared = table["cluster"].value_counts()
    logger.info(
        "Feature redundancy: %s features in %s clusters at |rho| > %.2f, largest has %s",
        len(table),
        len(shared),
        threshold,
        shared.max(),
    )

    return table


def baseline_forecasts(X: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Every baseline's forecast on the development rows."""
    baselines = {
        "always_up": AlwaysUp(),
        "random": RandomForecaster(seed=seed),
        "last_sign": LastSignPersistence(),
        "reverse_last_sign": LastSignPersistence(reverse=True),
    }

    return {name: model.predict(X) for name, model in baselines.items()}


def _per_path(table_for: Any, predictions: pd.DataFrame) -> pd.DataFrame:
    """Run a per-path table builder and summarise the IC column across paths."""
    tables = {path: table_for(predictions[path]) for path in predictions}
    ic = pd.DataFrame({path: table["ic"] for path, table in tables.items()})
    first = next(iter(tables.values()))

    summary = pd.DataFrame({"ic_mean": ic.mean(axis=1), "ic_std": ic.std(axis=1)})
    summary["positive_paths"] = (ic > 0).sum(axis=1)
    summary["n"] = first["n"]

    return summary


def summarise(
    path_metrics: pd.DataFrame,
    by_year: pd.DataFrame,
    data: ResearchData,
    cv: CombinatorialPurgedCV,
) -> dict[str, float]:
    """The headline numbers recorded in the trial log."""
    ic = path_metrics["ic"]

    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "ic_min": float(ic.min()),
        "ic_max": float(ic.max()),
        "positive_ic_paths": float((ic > 0).sum()),
        "ic_year_t_stat": ic_t_stat(by_year.rename(columns={"ic_mean": "ic"})),
        "hit_rate_mean": float(path_metrics["hit_rate"].mean()),
        "edge_over_always_up_mean": float(path_metrics["edge_over_always_up"].mean()),
        "edge_over_always_up_std": float(path_metrics["edge_over_always_up"].std()),
        "directional_accuracy_mean": float(path_metrics["directional_accuracy"].mean()),
        "r2_mean": float(path_metrics["r2"].mean()),
        "n_paths": float(cv.n_paths),
        "n_splits": float(cv.get_n_splits()),
        "n_rows": float(len(data.X)),
        "fracdiff_d": data.fracdiff_d,
    }


def run_research(
    config: Config,
    panel: pd.DataFrame | None = None,
    *,
    refresh: bool = False,
    trial_log: TrialLog | None = None,
    permutation: bool = True,
) -> ResearchResult:
    """Build, label, fit under CPCV, and score one model configuration.

    Args:
        config: Configuration, with any command line overrides applied.
        panel: Cleaned panel. Loaded from disk when None.
        refresh: Rebuild the cleaned panel from the raw cache first.
        trial_log: Where to record the trial. Defaults to the config's.
        permutation: Compute permutation importance on each held-out fold,
            the slowest step. Split importance is always computed.

    Returns:
        The tables and the summary logged to the trial log.
    """
    trial_log = trial_log or TrialLog.from_config(config)
    name = run_name(config)
    task = config.model.task
    validation = config.validation
    notes = (
        f"CPCV N={validation.cpcv.n_groups} k={validation.cpcv.n_test_groups}, "
        f"embargo {validation.embargo_pct:.2%}, development period before {validation.test_start}"
    )
    features: list[str] = []
    params: dict[str, Any] = {}

    try:
        if panel is None:
            panel = load_clean_panel(config, rebuild=refresh)

        with log_step(logger, "Preparing research data"):
            data = prepare_research_data(panel, config)
        features = list(data.X.columns)

        width = embargo_bars(row_dates(data.X.index).nunique(), validation.embargo_pct)
        if width < config.features.max_lookback:
            logger.warning(
                "Embargo of %s days is shorter than the longest feature window (%s days): "
                "training rows just after a test block still see test-period prices",
                width,
                config.features.max_lookback,
            )

        cv = CombinatorialPurgedCV(
            validation.cpcv.n_groups,
            validation.cpcv.n_test_groups,
            validation.embargo_pct,
            data.labels["touch_time"],
        )
        target = target_for(data, task)

        split_predictions, models, held_out = [], [], []
        with log_step(logger, f"Fitting {cv.get_n_splits()} CPCV splits ({name})"):
            for number, (train, test) in enumerate(cv.split(data.X)):
                model = build_estimator(config)
                fit_with_weights(
                    model, data.X.iloc[train], target.iloc[train], data.weights.iloc[train]
                )
                X_test = data.X.iloc[test]
                split_predictions.append(
                    pd.Series(forecast(model, X_test, task), index=X_test.index)
                )
                models.append(model)
                held_out.append((X_test, target.iloc[test]))
                logger.info(
                    "Split %s of %s: %s train, %s test, IC %.4f",
                    number + 1,
                    cv.get_n_splits(),
                    f"{len(train):,}",
                    f"{len(test):,}",
                    information_coefficient(split_predictions[-1], data.actual.iloc[test]),
                )
        params = models[0].params

        predictions = path_predictions(split_predictions, cv, data.X)
        path_metrics = pd.DataFrame(
            {path: score(predictions[path], data.actual) for path in predictions}
        ).T
        path_metrics.index.name = "path"

        comparison = metric_table(
            predictions.iloc[:, 0],
            data.actual,
            baseline_forecasts(data.X, config.project.random_seed),
            name,
        )
        # The model row is the mean over paths, not path 0 alone.
        comparison.loc[name, path_metrics.columns] = path_metrics.mean()

        by_year = _per_path(lambda p: ic_by_period(p, data.actual, "YE"), predictions)

        # Regimes from the benchmark's volatility, labelled with past data only.
        regimes = volatility_regimes(panel.xs(config.universe.benchmark, level="ticker")["return"])
        by_regime = _per_path(lambda p: ic_across_regimes(p, data.actual, regimes), predictions)

        importance = None
        if isinstance(models[0], ReturnPredictor):
            with log_step(logger, "Importance across folds"):
                importance = importance_across_folds(
                    models,
                    held_out if permutation else None,
                    scoring=IC_SCORER if task == "regression" else None,
                )

        redundancy = feature_redundancy(data.X)
        metrics = summarise(path_metrics, by_year, data, cv)

    except Exception as exc:
        trial_log.log_trial(
            name,
            config=config,
            features=features,
            model_params=params,
            label=config.labelling.method,
            notes=f"failed: {type(exc).__name__}: {exc}",
        )
        raise

    trial_id = trial_log.log_trial(
        name,
        config=config,
        features=features,
        model_params=params,
        label=config.labelling.method,
        metrics=metrics,
        notes=notes,
    )
    logger.info(
        "Research %s: IC %.4f +/- %.4f over %s paths, edge over AlwaysUp %+.4f",
        name,
        metrics["ic_mean"],
        metrics["ic_std"],
        cv.n_paths,
        metrics["edge_over_always_up_mean"],
    )

    return ResearchResult(
        name=name,
        trial_id=trial_id,
        metrics=metrics,
        path_metrics=path_metrics,
        comparison=comparison,
        predictions=predictions,
        ic_by_year=by_year,
        ic_by_regime=by_regime,
        redundancy=redundancy,
        importance=importance,
    )
