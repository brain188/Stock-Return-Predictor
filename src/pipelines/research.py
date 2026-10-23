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
from src.evaluation.trial_log import TrialLog, trial_fingerprint
from src.features.build import build_features, feature_windows
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
from src.validation.cpcv import CombinatorialPurgedCV, Split, path_predictions
from src.validation.metrics import (
    cross_sectional_ic,
    cross_sectional_score,
    ic_by_period,
    ic_t_stat,
    information_coefficient,
    metric_table,
    score,
)
from src.validation.purged_cv import assert_no_overlap, row_dates

logger = get_logger(__name__)

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "results" / "research"

# Correlation above which two features are treated as near duplicates.
REDUNDANCY_THRESHOLD = 0.7

# Rows of each held-out fold used for permutation importance, to bound its cost.
PERMUTATION_ROWS = 5_000

LABELLERS = {"fixed_horizon": label_fixed_horizon, "triple_barrier": label_triple_barrier}

# Estimator parameters that change speed or chatter, never the fitted model.
NON_SEMANTIC_PARAMS = {"n_jobs", "verbose"}


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
    cs_ic_daily: pd.DataFrame
    redundancy: pd.DataFrame
    splits: pd.DataFrame
    importance: pd.DataFrame | None = None
    files: list[Path] = field(default_factory=list)

    def save(self, out_dir: str | Path = DEFAULT_OUTPUT) -> list[Path]:
        """Write every table as CSV, and the predictions as parquet, under out_dir / name.

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
            "cs_ic_daily": self.cs_ic_daily,
            "redundancy": self.redundancy,
            "splits": self.splits,
            "importance": self.importance,
        }

        self.files = []
        for label, table in tables.items():
            if table is not None:
                path = folder / f"{label}.csv"
                table.to_csv(path)
                self.files.append(path)

        # Out-of-sample predictions, so a new metric can be scored without refitting.
        path = folder / "predictions.parquet"
        self.predictions.to_parquet(path)
        self.files.append(path)

        logger.info("Saved %s research files to %s", len(self.files), folder)

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


def research_settings(config: Config) -> dict[str, Any]:
    """The settings that decide a research result, and nothing else.

    This is what a research trial is. Two runs with the same settings are two
    executions of one trial, whatever else differs between their config
    files: comments, paths, backtest costs, the parameters of an estimator
    that was not used, or n_jobs. Anything that can change a forecast or a
    reported score is in: the seed, universe, price column, cleaning,
    features, the active labelling method and its parameters, the
    validation design (embargo in trading dates, CPCV groups, holdout
    start) and the active estimator with its parameters.

    Not in, and stated as a limit: the data snapshot and the code. A cache
    refresh or a change to how a feature is computed, without a change of
    settings, keeps the fingerprint; the trial log records code_version per
    execution, and unique_trials() flags executions of one fingerprint
    whose metrics disagree.
    """
    labelling = config.labelling
    universe = config.universe
    validation = config.validation
    params = {
        key: value for key, value in config.model.params.items() if key not in NON_SEMANTIC_PARAMS
    }

    return {
        "pipeline": "research",
        "seed": config.project.random_seed,
        "universe": {
            "tickers": sorted(universe.tickers),
            "benchmark": universe.benchmark,
            "start": universe.start_date,
            "end": universe.end_date,
        },
        "price_column": config.data.price_column,
        "cleaning": dataclasses.asdict(config.cleaning),
        "features": dataclasses.asdict(config.features),
        "labelling": {
            "method": labelling.method,
            **dataclasses.asdict(getattr(labelling, labelling.method)),
        },
        "validation": {
            "embargo_days": validation.embargo_days,
            "n_groups": validation.cpcv.n_groups,
            "n_test_groups": validation.cpcv.n_test_groups,
            "test_start": validation.test_start,
        },
        "model": {
            "estimator": config.model.estimator,
            "task": config.model.task,
            "params": params,
        },
    }


def research_fingerprint(config: Config) -> str:
    """Identity of a research trial: trial_fingerprint of research_settings."""
    return trial_fingerprint(research_settings(config))


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


def label_span_days(labels: pd.DataFrame) -> pd.Series:
    """Trading dates from each label's date to its touch time.

    A span of 1 is the next-day label. Above 1, a label shares returns with
    the labels dated after it, which is the overlap purging exists for. The
    calendar is every date that appears as a label date or a touch time, so
    a touch on the first holdout date still counts correctly.

    Returns:
        Span per label, NaN where there is no touch time.
    """
    starts = row_dates(labels.index)
    touches = pd.DatetimeIndex(labels["touch_time"])
    calendar = starts.union(touches.dropna()).unique().sort_values()

    span = calendar.get_indexer(touches) - calendar.get_indexer(starts)

    return pd.Series(np.where(touches.isna(), np.nan, span), index=labels.index, name="span")


@dataclass(frozen=True)
class LeakageCheck:
    """What can still connect training rows to a test block, and how much.

    Attributes:
        embargo_days: Trading dates dropped after each test block.
        max_label_span: Longest label, in trading dates.
        labels_overlap: Whether any label spans more than one date.
        purging: Whether the splitter purges on touch times.
        feature_windows: Window of every feature, in trading dates.
        uncovered: Features whose window is longer than the embargo, so
            training rows just past it still hold test-period prices.
        warnings: Real leakage risks found. Empty means none.
    """

    embargo_days: int
    max_label_span: int
    labels_overlap: bool
    purging: bool
    feature_windows: dict[str, int]
    uncovered: dict[str, int]
    warnings: list[str]

    def log(self) -> None:
        """Warnings for real risks; the feature comparison as a diagnostic only."""
        for message in self.warnings:
            logger.warning("Leakage risk: %s", message)

        logger.info(
            "Labels span up to %s trading date(s) (overlap: %s); purging on touch times: %s",
            self.max_label_span,
            self.labels_overlap,
            self.purging,
        )
        covered = len(self.feature_windows) - len(self.uncovered)
        logger.info(
            "Embargo of %s trading dates covers %s of %s feature windows%s",
            self.embargo_days,
            covered,
            len(self.feature_windows),
            (
                "; longer: "
                + ", ".join(f"{name} ({days})" for name, days in self.uncovered.items())
                + ". This is shared inputs, not label leakage: purging keeps every test "
                "return out of the training targets"
            )
            if self.uncovered
            else "",
        )


def leakage_diagnostics(
    labels: pd.DataFrame,
    config: Config,
    *,
    purging: bool,
    d: float | None = None,
) -> LeakageCheck:
    """Separate the leakage risks that matter from the overlap that does not.

    A warning is raised only for a risk that can put test outcomes into
    training targets: labels that span more than one date while purging is
    off. With purging on, label overlap is removed exactly, and every split
    is also checked row by row before fitting, so no warning is needed.

    The embargo against feature windows is reported as a diagnostic, not a
    warning. A feature longer than the embargo means a training row just past
    the gap still holds some test-period prices as inputs; its target is a
    later return that is not part of any test label. That overlap is the
    same in kind as the one between each test block and the training rows
    just before it, which no embargo removes, and for the slow features
    (63 day windows, fracdiff) it outlasts any gap worth having.

    Args:
        labels: Labels with touch_time, indexed by (date, ticker).
        config: Supplies embargo_days and the feature windows.
        purging: Whether the splitter purges on touch times.
        d: Fracdiff order, for the fracdiff window.

    Returns:
        The diagnostics, with any warnings.
    """
    embargo = config.validation.embargo_days
    span = label_span_days(labels)
    max_span = int(span.max()) if span.notna().any() else 0
    overlap = max_span > 1
    windows = feature_windows(config, d)
    uncovered = {name: days for name, days in windows.items() if days > embargo}

    warnings = []
    if overlap and not purging:
        warnings.append(
            f"labels span up to {max_span} trading dates and purging is off, so training "
            f"labels within {max_span - 1} dates of a test block share its returns"
        )

    return LeakageCheck(
        embargo_days=embargo,
        max_label_span=max_span,
        labels_overlap=overlap,
        purging=purging,
        feature_windows=windows,
        uncovered=uncovered,
        warnings=warnings,
    )


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


@dataclass
class SplitFits:
    """Out-of-sample predictions and fitted models, one per CPCV split."""

    predictions: list[pd.Series]
    models: list[Forecaster]
    held_out: list[tuple[pd.DataFrame, pd.Series]]
    report: pd.DataFrame


def fit_splits(
    data: ResearchData,
    config: Config,
    splits: Sequence[Split] | Any,
    *,
    label: str = "",
) -> SplitFits:
    """Fit one model per split and predict its test rows.

    Every split is checked before its model is fitted: a training label that
    overlaps the test set, or a training row inside the embargo, stops the
    run. The check uses config.validation.embargo_days, so a caller that
    narrows the training set further (as the embargo diagnostic does) still
    passes, and one that widens it into the embargo does not.

    Args:
        data: Development data.
        config: Supplies the estimator, task and embargo.
        splits: Split objects, usually CombinatorialPurgedCV.iter_splits.
        label: Name for the log lines.

    Returns:
        The predictions, models, held-out data and a per-split row count table.
    """
    task = config.model.task
    target = target_for(data, task)
    dates = row_dates(data.X.index)
    touch = data.labels["touch_time"]
    predictions, models, held_out, rows = [], [], [], []

    for split in splits:
        train, test = split.train, split.test
        # Stops the run if a training label overlaps the test set or a
        # training row sits inside the embargo.
        assert_no_overlap(
            train, test, config.validation.embargo_days, dates=dates, touch_times=touch
        )
        rows.append(split.summary())

        model = build_estimator(config)
        fit_with_weights(model, data.X.iloc[train], target.iloc[train], data.weights.iloc[train])
        X_test = data.X.iloc[test]
        predictions.append(pd.Series(forecast(model, X_test, task), index=X_test.index))
        models.append(model)
        held_out.append((X_test, target.iloc[test]))
        logger.info(
            "%sSplit %s: %s train (%s purged, %s embargoed), %s test, IC %.4f",
            f"{label}: " if label else "",
            split.number + 1,
            f"{len(train):,}",
            split.purged,
            split.embargoed,
            f"{len(test):,}",
            information_coefficient(predictions[-1], data.actual.iloc[test]),
        )

    return SplitFits(
        predictions=predictions,
        models=models,
        held_out=held_out,
        report=pd.DataFrame(rows).set_index("split"),
    )


def split_report(
    config: Config,
    panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, LeakageCheck, ResearchData]:
    """The CPCV split table and leakage diagnostics, with no model and no trial.

    Splits depend only on the data, the labels and the validation settings,
    so this reproduces exactly the splits a research run with the same
    settings used, without fitting anything. It is safe to run at any time:
    nothing is learned and no result is seen, so it is not a trial.

    Args:
        config: Configuration to describe.
        panel: Cleaned panel. Loaded from disk when None.

    Returns:
        The split table (as saved in splits.csv), the leakage check, and the
        development data the splits were made from.
    """
    if panel is None:
        panel = load_clean_panel(config)

    data = prepare_research_data(panel, config)
    cv = CombinatorialPurgedCV.from_config(config, data.labels["touch_time"])
    leakage = leakage_diagnostics(data.labels, config, purging=True, d=data.fracdiff_d)
    dates = row_dates(data.X.index)

    rows = []
    for split in cv.iter_splits(data.X):
        assert_no_overlap(
            split.train,
            split.test,
            config.validation.embargo_days,
            dates=dates,
            touch_times=data.labels["touch_time"],
        )
        rows.append(split.summary())

    return pd.DataFrame(rows).set_index("split"), leakage, data


def write_split_report(
    config: Config,
    out_dir: str | Path = DEFAULT_OUTPUT,
    panel: pd.DataFrame | None = None,
) -> list[Path]:
    """Write splits.csv and validation.csv for a run, without fitting or logging.

    Used to fill in the validation-design record of a run made before these
    files existed. The splits are regenerated from the settings, and, when
    the run folder holds predictions.parquet, checked against it: the saved
    predictions must cover exactly the development rows the splits were made
    from, or the report would describe a different run.

    Args:
        config: The configuration of the run being described.
        out_dir: Parent of the run folder, as given to ResearchResult.save.
        panel: Cleaned panel. Loaded from disk when None.

    Returns:
        The two files written.

    Raises:
        ValueError: If saved predictions exist and cover different rows.
    """
    splits, leakage, data = split_report(config, panel)
    folder = Path(out_dir) / run_name(config)
    folder.mkdir(parents=True, exist_ok=True)

    saved = folder / "predictions.parquet"
    matches = None
    if saved.is_file():
        matches = bool(pd.read_parquet(saved).index.equals(data.X.index))
        if not matches:
            raise ValueError(f"{saved} covers different rows than these splits; not writing")

    record = {
        **split_fields(splits, leakage),
        "fingerprint": research_fingerprint(config),
        "n_rows": len(data.X),
        "matches_saved_predictions": matches,
        "generated_by": "split_report: splits regenerated from settings, no model fitted",
    }

    files = [folder / "splits.csv", folder / "validation.csv"]
    splits.to_csv(files[0])
    pd.Series(record, name="value").to_frame().to_csv(files[1])
    logger.info("Wrote the split report for %s to %s (no trial logged)", run_name(config), folder)

    return files


def split_fields(splits: pd.DataFrame, leakage: LeakageCheck) -> dict[str, float]:
    """The validation-design numbers a research run records alongside its scores."""
    return {
        "embargo_days": float(leakage.embargo_days),
        "label_max_span_days": float(leakage.max_label_span),
        "train_rows_mean": float(splits["train_rows"].mean()),
        "train_rows_min": float(splits["train_rows"].min()),
        "purged_rows_mean": float(splits["purged"].mean()),
        "embargoed_rows_mean": float(splits["embargoed"].mean()),
    }


def cross_sectional_daily(predictions: pd.DataFrame, actual: pd.Series) -> pd.DataFrame:
    """Daily cross-sectional IC on every path, with the tickers behind each date.

    Every path fills every row, so n_tickers is the same on all paths and is
    stored once.

    Returns:
        One row per date: n_tickers, then spearman_<path> and pearson_<path>
        for each path.
    """
    columns = {}

    for method in ("spearman", "pearson"):
        for path in predictions:
            daily = cross_sectional_ic(predictions[path], actual, method)
            columns.setdefault("n_tickers", daily["n_tickers"])
            columns[f"{method}_{path}"] = daily["ic"]

    return pd.DataFrame(columns)


def summarise(
    path_metrics: pd.DataFrame,
    by_year: pd.DataFrame,
    data: ResearchData,
    cv: CombinatorialPurgedCV,
    splits: pd.DataFrame,
    leakage: LeakageCheck,
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
        "cs_spearman_mean": float(path_metrics["cs_spearman_mean"].mean()),
        "cs_spearman_path_std": float(path_metrics["cs_spearman_mean"].std()),
        "cs_spearman_daily_std": float(path_metrics["cs_spearman_std"].mean()),
        "cs_spearman_t_stat": float(path_metrics["cs_spearman_t_stat"].mean()),
        "cs_spearman_positive_share": float(path_metrics["cs_spearman_positive_share"].mean()),
        "cs_pearson_mean": float(path_metrics["cs_pearson_mean"].mean()),
        "cs_pearson_daily_std": float(path_metrics["cs_pearson_std"].mean()),
        "cs_pearson_t_stat": float(path_metrics["cs_pearson_t_stat"].mean()),
        "cs_dates": float(path_metrics["cs_dates"].iloc[0]),
        "cs_skipped_dates": float(path_metrics["cs_skipped"].iloc[0]),
        "cs_mean_tickers": float(path_metrics["cs_mean_tickers"].iloc[0]),
        "n_paths": float(cv.n_paths),
        "n_splits": float(cv.get_n_splits()),
        "n_rows": float(len(data.X)),
        "fracdiff_d": data.fracdiff_d,
        **split_fields(splits, leakage),
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
    identity = research_fingerprint(config)
    task = config.model.task
    validation = config.validation
    notes = (
        f"CPCV N={validation.cpcv.n_groups} k={validation.cpcv.n_test_groups}, "
        f"embargo {validation.embargo_days} trading dates, "
        f"development period before {validation.test_start}"
    )
    features: list[str] = []
    params: dict[str, Any] = {}

    try:
        if panel is None:
            panel = load_clean_panel(config, rebuild=refresh)

        with log_step(logger, "Preparing research data"):
            data = prepare_research_data(panel, config)
        features = list(data.X.columns)

        cv = CombinatorialPurgedCV.from_config(config, data.labels["touch_time"])
        leakage = leakage_diagnostics(
            data.labels, config, purging=cv.touch_times is not None, d=data.fracdiff_d
        )
        leakage.log()

        with log_step(logger, f"Fitting {cv.get_n_splits()} CPCV splits ({name})"):
            fits = fit_splits(data, config, cv.iter_splits(data.X))
        split_predictions, models, held_out = fits.predictions, fits.models, fits.held_out
        params = models[0].params
        splits = fits.report

        predictions = path_predictions(split_predictions, cv, data.X)
        path_metrics = pd.DataFrame(
            {
                path: {
                    **score(predictions[path], data.actual),
                    **cross_sectional_score(predictions[path], data.actual),
                }
                for path in predictions
            }
        ).T
        path_metrics.index.name = "path"
        cs_daily = cross_sectional_daily(predictions, data.actual)

        baselines = baseline_forecasts(data.X, config.project.random_seed)
        comparison = metric_table(predictions.iloc[:, 0], data.actual, baselines, name)
        for label, values in baselines.items():
            cs = cross_sectional_score(pd.Series(values, index=data.X.index), data.actual)
            for key, value in cs.items():
                comparison.loc[label, key] = value
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
        metrics = summarise(path_metrics, by_year, data, cv, splits, leakage)

    except Exception as exc:
        trial_log.log_trial(
            name,
            kind="research",
            fingerprint=identity,
            status="failed",
            config=config,
            features=features,
            model_params=params,
            label=config.labelling.method,
            notes=f"failed: {type(exc).__name__}: {exc}",
        )
        raise

    trial_id = trial_log.log_trial(
        name,
        kind="research",
        fingerprint=identity,
        status="ok",
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
        cs_ic_daily=cs_daily,
        redundancy=redundancy,
        splits=splits,
        importance=importance,
    )
