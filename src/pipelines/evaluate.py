"""Every strategy through CPCV: a distribution of Sharpe ratios, gross and net.

A single backtest gives one equity curve and one Sharpe, with no way to tell
whether that Sharpe was luck. Here each strategy is run along every CPCV
backtest path (5 for N = 6, k = 2), and the spread of its Sharpe across the
paths is the uncertainty. Each path covers the whole development period
exactly once, with every date predicted by a model that never trained on it.

How each kind of strategy fills a path:

- ML strategies: each split's model forecasts its test groups; the
  forecasts are reassembled into paths first, and the sizing rule then runs
  along the path's own timeline, so a stateful rule (the threshold's hold)
  carries its state from one group into the next as it would live.
- Pairs: each split selects its pairs on its own training dates, so a path
  trades, in every group, the pairs chosen without seeing that group.
- Momentum and the benchmarks learn nothing, so every path is the same
  backtest and the spread across paths is zero. Their uncertainty has to
  come from resampling instead (Phase 7), and the comparison says so.

Only the development period (before validation.test_start) is used. The
holdout stays untouched for one final run once the strategy is chosen.

Every strategy evaluated is a trial (kind "backtest"); so is every pair a
pairs book traded. The benchmarks are logged as kind "benchmark" and the
same-close timing check as "diagnostic": recorded, not counted.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.costs import breakeven_return
from src.backtest.engine import BacktestResult, run_backtest
from src.backtest.performance import (
    exposure,
    sharpe_difference_interval,
    sharpe_ratio,
    sharpe_standard_error,
)
from src.backtest.strategy import (
    BENCHMARKS,
    ML_STRATEGIES,
    STRATEGIES,
    MeanReversionPairs,
    MLLongShort,
    Strategy,
    delay,
    enforce_limits,
)
from src.config import PROJECT_ROOT, Config
from src.data.cleaning import load_clean_panel
from src.evaluation.monte_carlo import (
    confidence_interval,
    simulate_paths,
)
from src.evaluation.monte_carlo import (
    path_statistics as resampled_statistics,
)
from src.evaluation.trial_log import TrialLog, trial_fingerprint
from src.logger import get_logger, log_frame, log_step
from src.pipelines.research import (
    ResearchData,
    fit_splits,
    prepare_research_data,
    research_settings,
)
from src.reporting.figures import volatility_regimes
from src.validation.cpcv import CombinatorialPurgedCV, Split, path_predictions
from src.validation.purged_cv import row_dates

logger = get_logger(__name__)

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "results" / "evaluation"

# Every strategy is compared with both: SPY is the market, equal weight is
# the same eleven names held passively, the fair test for anything that
# trades the whole universe.
BENCHMARK_NAMES = ("buy_and_hold", "equal_weight_hold")

# Differences in Sharpe are reported against this, with a bootstrap interval.
REFERENCE = "equal_weight_hold"
BOOTSTRAP_RUNS = 1000
BOOTSTRAP_BLOCK = 21

DEFAULT_STRATEGIES = (
    "ml_long_short",
    "ml_long_only",
    "ml_thresholded",
    "momentum",
    "pairs",
    "buy_and_hold",
    "equal_weight_hold",
)
# ML long/short traded at the signal's own close, as notebook 4 did. Not a
# candidate: it shows how much of the notebook's result was the timing.
SAME_CLOSE = "ml_long_short_same_close"


class PathError(AssertionError):
    """Raised when a backtest path duplicates, omits or leaks dates."""


@dataclass
class EvaluationContext:
    """Everything the strategies are evaluated on, built once."""

    config: Config
    data: ResearchData
    prices: pd.DataFrame
    returns: pd.DataFrame
    cv: CombinatorialPurgedCV
    splits: list[Split]
    date_group: pd.Series
    regimes: pd.Series
    forecasts: pd.DataFrame | None = None
    path_check: pd.DataFrame | None = None

    @property
    def dates(self) -> pd.DatetimeIndex:
        """The development dates every path covers."""
        return pd.DatetimeIndex(self.date_group.index)

    def split_dates(self, split: Split) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """Training and test dates of a split."""
        dates = row_dates(self.data.X.index)

        return dates[split.train].unique(), dates[split.test].unique()


@dataclass
class StrategyEvaluation:
    """One strategy along every path."""

    name: str
    kind: str
    parameters: dict[str, Any]
    results: list[BacktestResult]
    path_table: pd.DataFrame
    summary: dict[str, float]
    trial_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """The whole comparison."""

    evaluations: dict[str, StrategyEvaluation]
    comparison: pd.DataFrame
    paths: pd.DataFrame
    regimes: pd.DataFrame
    pairs_selection: pd.DataFrame
    path_check: pd.DataFrame = field(default_factory=pd.DataFrame)
    monte_carlo: pd.DataFrame | None = None
    files: list[Path] = field(default_factory=list)

    def save(self, out_dir: str | Path = DEFAULT_OUTPUT) -> list[Path]:
        """Write the tables as CSV and every path's daily returns as parquet."""
        folder = Path(out_dir)
        folder.mkdir(parents=True, exist_ok=True)

        returns = pd.concat(
            {
                (name, f"path_{p}", kind): getattr(result, kind)
                for name, evaluation in self.evaluations.items()
                for p, result in enumerate(evaluation.results)
                for kind in ("gross", "net")
            },
            axis=1,
            names=["strategy", "path", "kind"],
        )

        tables = {
            "comparison.csv": self.comparison,
            "paths.csv": self.paths,
            "regimes.csv": self.regimes,
            "pairs_selection.csv": self.pairs_selection,
            "path_check.csv": self.path_check,
        }
        if self.monte_carlo is not None:
            tables["monte_carlo.csv"] = self.monte_carlo
        self.files = []
        for name, table in tables.items():
            table.to_csv(folder / name)
            self.files.append(folder / name)

        returns.columns = ["|".join(column) for column in returns.columns]
        returns.to_parquet(folder / "path_returns.parquet")
        self.files.append(folder / "path_returns.parquet")
        logger.info("Saved %s evaluation files to %s", len(self.files), folder)

        return self.files


def build_context(
    config: Config,
    panel: pd.DataFrame | None = None,
    *,
    fit_models: bool = True,
) -> EvaluationContext:
    """Development data, CPCV splits, prices, returns and (optionally) ML forecasts.

    Args:
        config: Loaded project configuration.
        panel: Cleaned panel. Loaded from disk when None.
        fit_models: Fit the configured model on every split, for the ML
            strategies. Without it only rule-based strategies can run.

    Returns:
        The shared evaluation inputs.
    """
    panel = load_clean_panel(config) if panel is None else panel
    data = prepare_research_data(panel, config)
    cv = CombinatorialPurgedCV.from_config(config, data.labels["touch_time"])
    splits = list(cv.iter_splits(data.X))

    prices = panel[config.data.price_column].unstack("ticker").sort_index()
    returns = panel["return"].unstack("ticker").reindex(prices.index).fillna(0.0)
    log_frame(logger, prices, "Evaluation prices")

    groups = pd.Series(cv.groups(data.X), index=row_dates(data.X.index))
    date_group = groups.groupby(level=0).first().sort_index()

    benchmark = returns[config.universe.benchmark]
    context = EvaluationContext(
        config=config,
        data=data,
        prices=prices,
        returns=returns,
        cv=cv,
        splits=splits,
        date_group=date_group,
        regimes=volatility_regimes(benchmark),
    )

    if fit_models:
        with log_step(logger, f"Fitting {len(splits)} CPCV models for the ML strategies"):
            fits = fit_splits(data, config, splits)
        paths = path_predictions(fits.predictions, cv, data.X)
        context.forecasts = paths
        log_frame(logger, paths, "Path forecasts")

    context.path_check = verify_paths(context)

    return context


def verify_paths(context: EvaluationContext) -> pd.DataFrame:
    """Check every path covers each development date once, from a model that never saw it.

    For every path and every group: the split supplying the group must have
    tested all of its dates and trained on none of them. Across the groups,
    the path must cover every development date exactly once. When ML
    forecasts are present, each path must have one for every row.

    The report also shows why paths are not independent samples: each split
    fills k = 2 group slots, on one path or on two (the five splits of
    neighbouring groups put both on the same path), so models are shared
    across paths, and every path shares most of its training data with the
    others.

    Returns:
        One row per path: dates covered, duplicates, missing dates, and the
        splits it draws from.

    Raises:
        PathError: On the first broken rule.
    """
    rows = []
    dates = context.dates
    group_of = context.date_group.to_numpy()

    for p, sources in enumerate(context.cv.paths):
        covered = []
        for group, split_number in enumerate(sources):
            group_dates = dates[group_of == group]
            train, test = context.split_dates(context.splits[split_number])
            if not group_dates.isin(test).all():
                raise PathError(f"path {p} group {group}: split {split_number} did not test it")
            if group_dates.isin(train).any():
                raise PathError(f"path {p} group {group}: split {split_number} trained on it")
            covered.append(group_dates)

        allowed = pd.DatetimeIndex(np.concatenate([c.to_numpy() for c in covered]))
        duplicates = int(allowed.duplicated().sum())
        missing = len(dates.difference(allowed))
        if duplicates or missing:
            raise PathError(f"path {p}: {duplicates} duplicated and {missing} missing dates")

        if context.forecasts is not None:
            forecast = context.forecasts[f"path_{p}"]
            if not forecast.index.equals(context.data.X.index) or forecast.isna().any():
                raise PathError(f"path {p}: forecasts do not cover every development row")

        rows.append(
            {
                "path": f"path_{p}",
                "dates": len(allowed.unique()),
                "duplicates": duplicates,
                "missing": missing,
                "splits": " ".join(str(s) for s in sources),
            }
        )

    report = pd.DataFrame(rows).set_index("path")
    uses = pd.Series(np.asarray(context.cv.paths).ravel()).value_counts()
    logger.info(
        "Paths verified: %s paths x %s dates, no duplicates or gaps; each split fills %s slot(s)",
        len(report),
        len(dates),
        sorted(set(uses.tolist())),
    )

    return report


def assemble_path(
    split_frames: Sequence[pd.DataFrame],
    context: EvaluationContext,
    path: int,
) -> pd.DataFrame:
    """One path's frame: each group's dates taken from the split that tested it."""
    sources = context.cv.paths[path]
    pieces = []

    for group, split in enumerate(sources):
        dates = context.date_group.index[context.date_group.to_numpy() == group]
        pieces.append(split_frames[split].reindex(dates))

    return pd.concat(pieces).sort_index()


def path_targets(strategy: Strategy, context: EvaluationContext) -> list[pd.DataFrame]:
    """Pre-delay targets for every path, on the development dates.

    ML strategies size each path's reassembled forecasts. Strategies that
    fit (pairs) are fitted on each split's training dates and their targets
    assembled per path. The rest are computed once and shared by every path.
    """
    config = context.config
    n_paths = context.cv.n_paths
    prices = context.prices

    if strategy.uses_model:
        if context.forecasts is None:
            raise ValueError(f"{strategy.name} needs forecasts; build the context with fit_models")
        out = []
        for p in range(n_paths):
            signal = (
                context.forecasts[f"path_{p}"].unstack("ticker").reindex(columns=prices.columns)
            )
            out.append(strategy.targets(signal, prices.loc[signal.index], config))
        return out

    if isinstance(strategy, MeanReversionPairs):
        split_frames, selections = [], []
        for split in context.splits:
            train_dates, _ = context.split_dates(split)
            strategy.fit(prices, config, train_dates)
            split_frames.append(strategy.targets(None, prices, config))
            selections.append(strategy.selection_.assign(split=split.number))
        strategy.selections_ = pd.concat(selections, ignore_index=True)
        return [assemble_path(split_frames, context, p) for p in range(n_paths)]

    targets = strategy.targets(None, prices, config).reindex(context.dates)

    return [targets] * n_paths


def backtest_paths(
    name: str,
    targets: Sequence[pd.DataFrame],
    lag: int,
    context: EvaluationContext,
) -> list[BacktestResult]:
    """Delay, limit and backtest each path's targets."""
    results = []

    for p, frame in enumerate(targets):
        if not frame.index.is_unique or not frame.index.sort_values().equals(context.dates):
            raise PathError(f"{name} path {p}: positions do not cover each development date once")
        positions = enforce_limits(delay(frame.fillna(0.0), lag), context.config)
        returns = context.returns.reindex(index=positions.index, columns=positions.columns)
        results.append(run_backtest(positions, returns, context.config, name=f"{name}/path_{p}"))

    return results


def path_statistics(results: Sequence[BacktestResult], config: Config) -> pd.DataFrame:
    """One row per path: the strategy's statistics gross and net."""
    rows = {}

    for p, result in enumerate(results):
        rows[f"path_{p}"] = {
            **result.summary(),
            "breakeven_return": breakeven_return(result.turnover, config),
        }

    table = pd.DataFrame(rows).T
    table.index.name = "path"

    return table


def summarise_paths(
    results: Sequence[BacktestResult],
    table: pd.DataFrame,
    lag: int,
) -> dict[str, float]:
    """Means over paths of the statistics, plus the spread of the net Sharpe.

    Exposure is included because it explains most of what a book earns: a
    "long/short" book that is long on most days is mostly a long book.
    """
    net = table["net_sharpe"]
    exposures = [exposure(result.weights) for result in results]

    return {
        "sharpe": float(net.mean()),
        "sharpe_net_std": float(net.std()) if len(net) > 1 else 0.0,
        "sharpe_net_min": float(net.min()),
        "sharpe_net_max": float(net.max()),
        "sharpe_gross": float(table["gross_sharpe"].mean()),
        "sharpe_lost_to_costs": float((table["gross_sharpe"] - net).mean()),
        "annual_return_gross": float(table["gross_annual_return"].mean()),
        "annual_return_net": float(table["net_annual_return"].mean()),
        "annual_volatility_net": float(table["net_annual_volatility"].mean()),
        "max_drawdown_net": float(table["net_max_drawdown"].mean()),
        "turnover_annual": float(table["turnover_annual"].mean()),
        "cost_annual": float(table["cost_annual"].mean()),
        "breakeven_return": float(table["breakeven_return"].mean()),
        "exposure_net_mean": float(np.mean([e["net"].mean() for e in exposures])),
        "exposure_gross_mean": float(np.mean([e["gross"].mean() for e in exposures])),
        "days_net_short": float(np.mean([(e["net"] < 0).mean() for e in exposures])),
        "n_paths": float(len(results)),
        "execution_lag": float(lag),
    }


def evaluate_strategy(
    strategy: Strategy,
    context: EvaluationContext,
    *,
    name: str | None = None,
    kind: str = "backtest",
    n_paths: int | None = None,
) -> StrategyEvaluation:
    """One strategy across every CPCV path.

    Args:
        strategy: The strategy to run.
        context: Shared inputs from build_context.
        name: Label; defaults to strategy.name.
        kind: Trial kind it will be logged as.
        n_paths: Evaluate only the first n paths, to save time.

    Returns:
        The per-path backtests and their statistics.
    """
    name = name or strategy.name
    targets = path_targets(strategy, context)[:n_paths]
    lag = strategy.lag(context.config)
    results = backtest_paths(name, targets, lag, context)
    table = path_statistics(results, context.config)
    summary = summarise_paths(results, table, lag)

    parameters = {**strategy.parameters, "execution_lag": lag}
    extra: dict[str, Any] = {}

    if isinstance(strategy, MeanReversionPairs):
        selections = strategy.selections_
        tested = selections[["a", "b"]].drop_duplicates()
        traded = selections.loc[selections["selected"], ["a", "b"]].drop_duplicates()
        summary["pairs_tested"] = float(len(tested))
        summary["pairs_traded"] = float(len(traded))
        extra["selection"] = selections
        extra["traded"] = list(zip(traded["a"], traded["b"], strict=True))

    logger.info(
        "%s: net Sharpe %.3f (paths %.3f to %.3f), gross %.3f, turnover %.1f a year",
        name,
        summary["sharpe"],
        summary["sharpe_net_min"],
        summary["sharpe_net_max"],
        summary["sharpe_gross"],
        summary["turnover_annual"],
    )

    return StrategyEvaluation(
        name=name,
        kind=kind,
        parameters=parameters,
        results=results,
        path_table=table,
        summary=summary,
        extra=extra,
    )


def evaluate_pairs_individually(
    evaluation: StrategyEvaluation,
    context: EvaluationContext,
    n_paths: int | None = None,
) -> dict[str, StrategyEvaluation]:
    """Backtest every pair the pairs book traded on its own, one trial each.

    A pair's targets in a group come from the split that tested it, and are
    zero where that split did not select the pair.
    """
    config = context.config
    lag = int(evaluation.parameters["execution_lag"])
    selection = evaluation.extra.get("selection")
    out = {}

    for a, b in evaluation.extra.get("traded", []):
        frames = []
        for split in context.splits:
            chosen = selection[(selection["split"] == split.number) & selection["selected"]]
            frame = pd.DataFrame(0.0, index=context.prices.index, columns=context.prices.columns)
            if ((chosen["a"] == a) & (chosen["b"] == b)).any():
                strategy = MeanReversionPairs()
                strategy.selection_, strategy.pairs_ = chosen, [(a, b)]
                frame = strategy.targets(None, context.prices, config)
            frames.append(frame)

        targets = [assemble_path(frames, context, p) for p in range(context.cv.n_paths)][:n_paths]
        results = backtest_paths(f"pair_{a}_{b}", targets, lag, context)
        table = path_statistics(results, config)
        out[f"pair_{a}_{b}"] = StrategyEvaluation(
            name=f"pair_{a}_{b}",
            kind="backtest",
            parameters={"a": a, "b": b, "execution_lag": lag},
            results=results,
            path_table=table,
            summary=summarise_paths(results, table, lag),
        )

    return out


def compare_strategies(
    evaluations: dict[str, StrategyEvaluation],
    benchmarks: Sequence[str] = BENCHMARK_NAMES,
    reference: str = REFERENCE,
    seed: int = 0,
) -> pd.DataFrame:
    """The comparison table: means over paths, and paths beating each benchmark.

    Paths are compared pairwise: path p of a strategy against path p of a
    benchmark, which cover the same dates. Each benchmark present gets a
    sharpe_net_vs_<name> and a paths_beating_<name> column. Against the
    reference (equal weight) there is also the Sharpe standard error and a
    95% paired block bootstrap interval for the difference, averaged over
    paths; paths_beating counts are not evidence on their own, since the
    paths are not independent.
    """
    rows = {}
    present = [name for name in benchmarks if name in evaluations]

    for name, evaluation in evaluations.items():
        row = {"kind": evaluation.kind, **evaluation.summary}
        ours = evaluation.path_table["net_sharpe"].to_numpy()
        for benchmark in present:
            theirs = evaluations[benchmark].path_table["net_sharpe"].to_numpy()[: len(ours)]
            # A pair idle on a path has no Sharpe there; compare the paths it traded.
            row[f"sharpe_net_vs_{benchmark}"] = float(np.nanmean(ours - theirs))
            row[f"paths_beating_{benchmark}"] = float(np.sum(ours > theirs))
        uncertainty = _interval_over_paths(evaluation, evaluations.get(reference), None, seed)
        row["sharpe_se"] = uncertainty["sharpe_se"]
        for key in ("vs_reference_low", "vs_reference_high", "resamples_beating_reference"):
            if key in uncertainty:
                row[key] = uncertainty[key]
        rows[name] = row

    table = pd.DataFrame(rows).T
    table.index.name = "strategy"
    order = [
        "kind",
        "sharpe_gross",
        "sharpe",
        "sharpe_net_std",
        "sharpe_net_min",
        "sharpe_net_max",
        "sharpe_se",
        "sharpe_lost_to_costs",
        *[f"sharpe_net_vs_{b}" for b in present],
        "vs_reference_low",
        "vs_reference_high",
        "resamples_beating_reference",
        *[f"paths_beating_{b}" for b in present],
        "exposure_net_mean",
        "exposure_gross_mean",
        "annual_return_gross",
        "annual_return_net",
        "max_drawdown_net",
        "turnover_annual",
        "cost_annual",
        "breakeven_return",
    ]
    columns = [c for c in order if c in table.columns]
    columns += [c for c in table.columns if c not in columns]

    return table[columns].rename(columns={"sharpe": "sharpe_net"})


def _interval_over_paths(
    evaluation: StrategyEvaluation,
    reference: StrategyEvaluation | None,
    mask: pd.Series | None,
    seed: int,
) -> dict[str, float]:
    """Sharpe, its standard error and the difference from the reference, averaged over paths.

    Each path is measured on its own and the numbers averaged. The paths are
    not independent samples (they share models and training data and cover
    the same dates), so the average interval is the honest summary; treating
    five paths as five draws would overstate the evidence.
    """
    rows = []

    for p, result in enumerate(evaluation.results):
        net = (
            result.net
            if mask is None
            else result.net[mask.reindex(result.net.index, fill_value=False)]
        )
        row = {
            "sharpe": sharpe_ratio(net),
            "sharpe_se": sharpe_standard_error(net),
            "days": float(len(net)),
        }
        if reference is not None:
            base = reference.results[p].net.reindex(net.index)
            interval = sharpe_difference_interval(
                net, base, n_boot=BOOTSTRAP_RUNS, block=BOOTSTRAP_BLOCK, seed=seed + p
            )
            row |= {
                "vs_reference": interval["difference"],
                "vs_reference_low": interval["low"],
                "vs_reference_high": interval["high"],
                "resamples_beating_reference": interval["share_positive"],
            }
        rows.append(row)

    return pd.DataFrame(rows).mean(numeric_only=True).to_dict()


def regime_breakdown(
    evaluations: dict[str, StrategyEvaluation],
    regimes: pd.Series,
    reference: str = REFERENCE,
    seed: int = 0,
) -> pd.DataFrame:
    """Net Sharpe within each volatility regime, with its uncertainty.

    The regime label of a date is built from the benchmark's realised
    volatility up to that close, against the terciles of its own past
    (volatility_regimes). A return dated t is earned over (t - 1, t], so it
    is bucketed by the label of t - 1: the regime known when the position was
    held, not one that already includes the return being classified.

    With roughly 1,000 to 1,300 days per regime, a Sharpe ratio's standard
    error is about 0.45 to 0.5, so differences between strategies within a
    regime need the interval next to them before they mean anything.

    Returns:
        One row per (strategy, regime): sharpe, sharpe_se, days, and the
        Sharpe difference from the reference with a 95% paired block
        bootstrap interval, each averaged over paths.
    """
    known = regimes.shift(1)
    base = evaluations.get(reference)
    rows = {}

    for name, evaluation in evaluations.items():
        for regime in ("low", "normal", "high"):
            mask = known == regime
            rows[(name, regime)] = _interval_over_paths(evaluation, base, mask, seed)

    table = pd.DataFrame(rows).T
    table.index.names = ["strategy", "regime"]

    return table


def backtest_settings(config: Config, strategy: Strategy, name: str) -> dict[str, Any]:
    """The settings that decide a backtest result: the strategy's trial identity."""
    research = research_settings(config)
    backtest = config.backtest
    settings: dict[str, Any] = {
        "pipeline": "backtest",
        "strategy": name,
        "parameters": {**strategy.parameters, "execution_lag": strategy.lag(config)},
        "costs": {
            "commission_bps": backtest.commission_bps,
            "slippage_bps": backtest.slippage_bps,
            "borrow_rate_annual": backtest.borrow_rate_annual,
        },
        "limits": {
            "max_position_size": backtest.max_position_size,
            "max_gross_leverage": backtest.max_gross_leverage,
            "allow_short": backtest.allow_short,
        },
        "universe": research["universe"],
        "price_column": research["price_column"],
        "cleaning": research["cleaning"],
        "validation": research["validation"],
    }

    if strategy.uses_model:
        settings["model"] = {k: research[k] for k in ("features", "labelling", "model", "seed")}
        settings["ml_thresholded"] = config.strategies.ml_thresholded
    if strategy.name == "momentum":
        settings["momentum"] = dataclasses.asdict(config.strategies.momentum)
    if strategy.name == "pairs" or name.startswith("pair_"):
        # Selection became segment-aware (no differencing across CPCV gaps): a new
        # method, so new trials, and the earlier runs stay counted.
        settings["pairs"] = {
            **dataclasses.asdict(config.strategies.pairs),
            "selection_method": "segment_aware",
        }

    return settings


def _log(
    trial_log: TrialLog,
    evaluation: StrategyEvaluation,
    strategy: Strategy,
    config: Config,
    features: list[str],
) -> int:
    """Record one evaluated strategy in the trial log."""
    return trial_log.log_trial(
        evaluation.name,
        kind=evaluation.kind,
        fingerprint=trial_fingerprint(backtest_settings(config, strategy, evaluation.name)),
        status="ok",
        config=config,
        features=features if strategy.uses_model else [],
        model_params=evaluation.parameters,
        label=config.labelling.method if strategy.uses_model else "",
        metrics=evaluation.summary,
        notes=(
            f"CPCV backtest over {int(evaluation.summary['n_paths'])} paths of the development "
            f"period; Sharpe is the mean annualised net Sharpe across paths"
        ),
    )


def net_paths(evaluations: dict[str, StrategyEvaluation]) -> dict[str, list[pd.Series]]:
    """Each strategy's daily net returns, one series per path."""
    return {name: [result.net for result in e.results] for name, e in evaluations.items()}


def load_path_returns(folder: str | Path = DEFAULT_OUTPUT) -> dict[str, list[pd.Series]]:
    """Net path returns from a saved evaluation (path_returns.parquet), no refitting."""
    frame = pd.read_parquet(Path(folder) / "path_returns.parquet")
    out: dict[str, list[tuple[int, pd.Series]]] = {}

    for column in frame.columns:
        name, path, kind = column.split("|")
        if kind == "net":
            out.setdefault(name, []).append((int(path.removeprefix("path_")), frame[column]))

    return {name: [s for _, s in sorted(series)] for name, series in out.items()}


def monte_carlo_table(
    returns: dict[str, list[pd.Series]],
    config: Config,
    n_runs: int,
    *,
    method: str = "block",
    reference: str = REFERENCE,
    market: str = "buy_and_hold",
) -> pd.DataFrame:
    """Resampled outcomes of every strategy, paired with the benchmarks.

    On each path the strategy, equal weight and SPY are resampled together,
    with the same blocks of days (the paths' runs split n_runs between them),
    and every strategy on a path uses the same seed, so all of them are
    compared on identical resampled histories.

    Two probabilities of beating equal weight are given. The paired one
    compares Sharpe ratios on the same resampled days and is the honest
    one. The other compares each resampled path with equal weight's single
    historical Sharpe, as notebook 4 did; it ignores that the benchmark's
    own Sharpe is uncertain too.

    Returns:
        One row per strategy: median and 95% interval of the Sharpe, median
        annual return, median and 5th percentile max drawdown, and the
        probabilities of a positive Sharpe and of beating each benchmark.
    """
    if reference not in returns or market not in returns:
        raise ValueError(f"monte carlo needs '{reference}' and '{market}' among the strategies")

    n_paths = len(returns[reference])
    per_path = max(1, n_runs // n_paths)
    seed = config.project.random_seed
    rows = {}

    for name, series_by_path in returns.items():
        ours, theirs, market_runs, historical = [], [], [], []
        for p, series in enumerate(series_by_path):
            frame = pd.concat(
                [series, returns[reference][p], returns[market][p]], axis=1, join="inner"
            ).dropna()
            sims = simulate_paths(frame, method, config, n_runs=per_path, seed=seed + p)
            ours.append(resampled_statistics(sims[:, :, 0]))
            theirs.append(resampled_statistics(sims[:, :, 1])["sharpe"].to_numpy())
            market_runs.append(resampled_statistics(sims[:, :, 2])["sharpe"].to_numpy())
            historical.append(np.full(per_path, sharpe_ratio(frame.iloc[:, 1])))

        stats = pd.concat(ours, ignore_index=True)
        sharpe = stats["sharpe"].to_numpy()
        ref = np.concatenate(theirs)
        mkt = np.concatenate(market_runs)
        hist = np.concatenate(historical)
        low, high = confidence_interval(sharpe, 0.95)

        def share(a, b):
            valid = ~(np.isnan(a) | np.isnan(b))
            return float(np.mean(a[valid] > b[valid])) if valid.any() else float("nan")

        rows[name] = {
            "sharpe_median": float(np.nanmedian(sharpe)),
            "sharpe_low": low,
            "sharpe_high": high,
            "annual_return_median": float(stats["annual_return"].median()),
            "max_drawdown_median": float(stats["max_drawdown"].median()),
            "max_drawdown_p05": float(stats["max_drawdown"].quantile(0.05)),
            "p_sharpe_positive": share(sharpe, np.zeros_like(sharpe)),
            f"p_beat_{reference}": share(sharpe, ref),
            f"p_beat_{reference}_historical": share(sharpe, hist),
            f"p_beat_{market}": share(sharpe, mkt),
            "runs": float(len(sharpe)),
            # Enough to reproduce the row: method, block, and the seeds used.
            "method": method,
            "block_days": float(config.evaluation.bootstrap_block_days),
            "seed_first_path": float(seed),
        }

    # Row by row, so the numeric columns stay numeric next to the method name.
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "strategy"
    logger.info(
        "Monte Carlo: %s %s-bootstrap runs per strategy, %s days per block",
        per_path * n_paths,
        method,
        config.evaluation.bootstrap_block_days,
    )

    return table


def run_evaluation(
    config: Config,
    panel: pd.DataFrame | None = None,
    *,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    n_paths: int | None = None,
    same_close_check: bool = True,
    trial_log: TrialLog | None = None,
    monte_carlo_runs: int | None = None,
) -> EvaluationResult:
    """Run every requested strategy through CPCV and compare them.

    Args:
        config: Loaded project configuration.
        panel: Cleaned panel. Loaded from disk when None.
        strategies: Names from strategy.STRATEGIES.
        n_paths: Evaluate only the first n CPCV paths.
        same_close_check: Also run ML long/short at the signal's own close,
            as a diagnostic, to measure what notebook 4's timing was worth.
        trial_log: Where to record trials. Defaults to the config's.
        monte_carlo_runs: Block-bootstrap runs per strategy. None uses
            evaluation.monte_carlo_runs; 0 skips the resampling. It needs both
            benchmarks among the strategies.

    Returns:
        The comparison, per-path statistics, regime table, pair selections
        and, when run, the Monte Carlo table.
    """
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"unknown strategies {sorted(unknown)}; choose from {sorted(STRATEGIES)}")

    trial_log = trial_log or TrialLog.from_config(config)
    needs_model = same_close_check or any(name in ML_STRATEGIES for name in strategies)
    context = build_context(config, panel, fit_models=needs_model)
    features = list(context.data.X.columns)

    if n_paths is not None and n_paths < context.cv.n_paths:
        logger.warning("Evaluating %s of %s CPCV paths", n_paths, context.cv.n_paths)

    runs: list[tuple[str, Strategy, str]] = [
        (name, STRATEGIES[name](), "benchmark" if name in BENCHMARKS else "backtest")
        for name in strategies
    ]
    if same_close_check:
        runs.append((SAME_CLOSE, MLLongShort(execution_lag=0), "diagnostic"))

    evaluations: dict[str, StrategyEvaluation] = {}
    for name, strategy, kind in runs:
        with log_step(logger, f"Evaluating {name}"):
            evaluation = evaluate_strategy(strategy, context, name=name, kind=kind, n_paths=n_paths)
        evaluation.trial_id = _log(trial_log, evaluation, strategy, config, features)
        evaluations[name] = evaluation

        if isinstance(strategy, MeanReversionPairs):
            for pair_name, pair in evaluate_pairs_individually(
                evaluation, context, n_paths
            ).items():
                pair.trial_id = _log(trial_log, pair, strategy, config, features)
                evaluations[pair_name] = pair

    paths = pd.concat(
        {name: evaluation.path_table for name, evaluation in evaluations.items()},
        names=["strategy", "path"],
    )
    pairs = evaluations["pairs"].extra["selection"] if "pairs" in evaluations else pd.DataFrame()

    runs = config.evaluation.monte_carlo_runs if monte_carlo_runs is None else monte_carlo_runs
    monte_carlo = None
    if runs and set(BENCHMARK_NAMES) <= set(evaluations):
        with log_step(logger, "Monte Carlo resampling"):
            monte_carlo = monte_carlo_table(net_paths(evaluations), config, runs)
    elif runs:
        logger.warning("Monte Carlo skipped: it needs both benchmarks among the strategies")

    return EvaluationResult(
        evaluations=evaluations,
        comparison=compare_strategies(evaluations, seed=config.project.random_seed),
        paths=paths,
        regimes=regime_breakdown(evaluations, context.regimes, seed=config.project.random_seed),
        pairs_selection=pairs,
        path_check=context.path_check,
        monte_carlo=monte_carlo,
    )
