"""Read-only access to every saved result the dashboard shows, cached and validated.

Every function here reads a file the research pipelines wrote; none of them
fits a model, runs a backtest, logs a trial or reads anything the pipelines did
not save. Where a view needs a summary the files do not hold (a calendar-year
return, a rolling volatility), it is computed from the saved series with the
project's own tested functions, and the page says so.

Files are cached by path and modification time, so a page never reads the same
file twice, and `clear_cache()` (the header's Reload button) picks up new runs.
A missing or malformed file raises DataError with a sentence a reader can act
on; pages turn it into an empty state rather than a traceback.
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.performance import (
    annualised_return,
    max_drawdown,
    performance_stats,
    sharpe_ratio,
)
from src.config import PROJECT_ROOT, Config, load_config
from src.evaluation.trial_log import TrialLog, code_version
from src.logger import get_logger

logger = get_logger(__name__)

RESULTS = PROJECT_ROOT / "reports" / "results"
EVALUATION = RESULTS / "evaluation"
PORTFOLIO = RESULTS / "portfolio"
HOLDOUT_ATTEMPT = RESULTS / "holdout" / "attempt-1"
FREEZE = RESULTS / "holdout_freeze.json"
DECLARATION = RESULTS / "holdout_declaration.json"
MANIFEST = RESULTS / "holdout_artifacts" / "manifest.json"
STATIONARITY = RESULTS / "stationarity.csv"
RESEARCH_REPORT = PROJECT_ROOT / "reports" / "research_report.md"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

DEVELOPMENT = "development"
HOLDOUT = "holdout"
PERIODS = (DEVELOPMENT, HOLDOUT)
TRADING_DAYS = 252


class DataError(Exception):
    """A saved result is missing or unusable; the message is shown to the reader."""

    def __init__(self, title: str, detail: str) -> None:
        super().__init__(f"{title}: {detail}")
        self.title = title
        self.detail = detail


# --- Caching ----------------------------------------------------------------

_LOCK = threading.Lock()
_FILES: dict[tuple[str, int, str], Any] = {}
_DERIVED: list[Any] = []


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _cached_file[T](path: Path, reader: Callable[[Path], T], what: str) -> T:
    """Read a file once per modification, raising DataError if it is absent."""
    path = Path(path)
    if not path.is_file():
        raise DataError(f"{what} not found", f"{_relative(path)} has not been written yet.")
    key = (str(path), path.stat().st_mtime_ns, reader.__qualname__)
    with _LOCK:
        if key in _FILES:
            return _FILES[key]
    try:
        value = reader(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.exception("Could not read %s", path)
        raise DataError(f"{what} could not be read", f"{_relative(path)}: {exc}") from exc
    with _LOCK:
        _FILES[key] = value
    return value


def derived[T](function: Callable[..., T]) -> Callable[..., T]:
    """Cache a computed view; cleared with the files by clear_cache()."""
    cached = functools.lru_cache(maxsize=64)(function)
    _DERIVED.append(cached)
    return cached


def clear_cache() -> None:
    """Forget every file and computed view, so the next read sees new results."""
    with _LOCK:
        _FILES.clear()
    for function in _DERIVED:
        function.cache_clear()
    logger.info("Dashboard cache cleared")


def _csv(index_col: int | str | None = None, parse_dates: Iterable[str] = ()) -> Callable:
    def read(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, index_col=index_col, parse_dates=list(parse_dates) or False)

    read.__qualname__ = f"csv[{index_col}|{','.join(parse_dates)}]"
    return read


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def require(frame: pd.DataFrame, columns: Iterable[str], what: str) -> pd.DataFrame:
    """Refuse a table that lacks the columns a view needs."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise DataError(f"{what} has an unexpected layout", f"missing columns {missing}")
    return frame


# --- Configuration and provenance --------------------------------------------


@functools.cache
def config() -> Config:
    """The project configuration, loaded once without creating folders."""
    return load_config(create_dirs=False)


@derived
def provenance() -> dict[str, Any]:
    """Code version, last data date and when the results were last written."""
    stamps = [p.stat().st_mtime for p in RESULTS.rglob("*") if p.is_file()]
    try:
        last_bar = panel().index.get_level_values("date").max()
    except DataError:
        last_bar = None
    return {
        "code_version": code_version(),
        "last_bar": last_bar,
        "results_written": pd.Timestamp(max(stamps), unit="s") if stamps else None,
        "config_hash": freeze().get("config_hash") if FREEZE.is_file() else None,
    }


# --- Market data ------------------------------------------------------------


def panel() -> pd.DataFrame:
    """The cleaned panel, (date, ticker) by OHLCV, return and quality flags."""
    frame = _cached_file(config().data.bars_dir / "bars.parquet", _parquet, "Cleaned price panel")
    return require(frame, ["adj_close", "close", "volume", "return", "gap_flag", "filled"], "panel")


def tickers() -> list[str]:
    """The universe, in config order."""
    return list(config().universe.tickers)


@derived
def wide(column: str) -> pd.DataFrame:
    """One panel column as a date by ticker frame."""
    frame = panel()[column].unstack("ticker").sort_index()
    return frame.reindex(columns=[t for t in tickers() if t in frame.columns])


def window(frame: pd.DataFrame | pd.Series, start: Any, end: Any) -> pd.DataFrame | pd.Series:
    """Rows between two dates, inclusive; either bound may be None."""
    lo = pd.Timestamp(start) if start else None
    hi = pd.Timestamp(end) if end else None
    return frame.loc[lo:hi]


def resample_returns(returns: pd.DataFrame | pd.Series, frequency: str) -> Any:
    """Daily simple returns compounded to weekly (Friday) or month-end returns."""
    if frequency == "1D":
        return returns
    rule = {"1W": "W-FRI", "1M": "ME"}[frequency]
    # A period with no observations stays NaN rather than becoming a 0% return.
    return (1 + returns).resample(rule).prod(min_count=1) - 1


@derived
def coverage(ticker: str) -> dict[str, Any]:
    """How complete one ticker's history is, from the cleaning flags."""
    rows = panel().xs(ticker, level="ticker")
    total = len(rows)
    priced = int(rows["adj_close"].notna().sum())
    return {
        "rows": total,
        "priced": priced,
        "missing": total - priced,
        "filled": int(rows["filled"].sum()),
        "gaps_flagged": int(rows["gap_flag"].sum()),
        "first": rows.index.min(),
        "last": rows.index.max(),
        "coverage": priced / total if total else float("nan"),
    }


@derived
def rolling_volatility(ticker: str, days: int = 21) -> pd.DataFrame:
    """Annualised rolling volatility and its regime label, using only the past.

    The regime is reports.figures.volatility_regimes, the same labelling the
    research and evaluation pipelines used.
    """
    from src.reporting.figures import volatility_regimes

    returns = wide("return")[ticker].dropna()
    vol = returns.rolling(days).std() * np.sqrt(TRADING_DAYS)
    return pd.DataFrame({"volatility": vol, "regime": volatility_regimes(returns, days)})


def autocorrelation(returns: pd.Series, lags: int = 21) -> pd.DataFrame:
    """Autocorrelation of returns and of their size, with the white-noise band."""
    series = returns.dropna()
    band = 1.96 / np.sqrt(len(series)) if len(series) else float("nan")
    return pd.DataFrame(
        {
            "lag": np.arange(1, lags + 1),
            "returns": [series.autocorr(k) for k in range(1, lags + 1)],
            "absolute": [series.abs().autocorr(k) for k in range(1, lags + 1)],
            "band": band,
        }
    )


def ljung_box(returns: pd.Series, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box Q and its p-value at `lags` (statsmodels)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    series = returns.dropna()
    if len(series) <= lags + 1:
        return float("nan"), float("nan")
    table = acorr_ljungbox(series, lags=[lags])
    return float(table["lb_stat"].iloc[0]), float(table["lb_pvalue"].iloc[0])


def stationarity() -> pd.DataFrame:
    """ADF results per ticker, written by src/reporting/stationarity.py."""
    frame = _cached_file(STATIONARITY, _csv(), "Stationarity table")
    return require(
        frame, ["ticker", "adf_log_price", "adf_log_return", "critical", "d_star"], "stationarity"
    )


# --- Research ---------------------------------------------------------------


def research_dir() -> Path:
    """The saved research run for the configured model."""
    from src.pipelines.research import run_name

    return RESULTS / "research" / run_name(config())


def research_metrics() -> pd.Series:
    """The research run's headline metrics, name to value."""
    frame = _cached_file(research_dir() / "metrics.csv", _csv(index_col=0), "Research metrics")
    return frame["value"]


def forecast_comparison() -> pd.DataFrame:
    """The model against each baseline forecast, one row per forecast."""
    frame = _cached_file(research_dir() / "comparison.csv", _csv(), "Forecast comparison")
    return require(frame, ["ic", "hit_rate", "edge_over_always_up", "n"], "comparison").set_index(
        "forecast"
    )


def research_paths() -> pd.DataFrame:
    """Forecast metrics on each CPCV path."""
    return _cached_file(
        research_dir() / "paths.csv", _csv(index_col=0), "Per-path forecast metrics"
    )


def ic_by_year() -> pd.DataFrame:
    """Pooled IC per calendar year, mean and spread over paths."""
    return _cached_file(research_dir() / "ic_by_year.csv", _csv(), "IC by year")


def ic_by_regime() -> pd.DataFrame:
    """Pooled IC per volatility regime."""
    return _cached_file(research_dir() / "ic_by_regime.csv", _csv(), "IC by regime")


def importance() -> pd.DataFrame:
    """Permutation and split importance per feature, with their spread over folds."""
    frame = _cached_file(research_dir() / "importance.csv", _csv(), "Feature importance")
    return require(frame, ["feature", "perm_mean", "perm_std", "positive_share"], "importance")


def redundancy() -> pd.DataFrame:
    """Feature correlation matrix with each feature's redundancy cluster."""
    frame = _cached_file(research_dir() / "redundancy.csv", _csv(), "Feature redundancy")
    return require(frame, ["feature", "cluster"], "redundancy").set_index("feature")


def research_splits() -> pd.DataFrame:
    """Rows, purging and embargo per CPCV split."""
    return _cached_file(research_dir() / "splits.csv", _csv(), "CPCV splits")


def research_validation() -> pd.Series:
    """The validation settings and the research fingerprint."""
    return _cached_file(research_dir() / "validation.csv", _csv(index_col=0), "Validation record")[
        "value"
    ]


@derived
def daily_cross_sectional_ic() -> pd.DataFrame:
    """Each date's cross-sectional Spearman IC, averaged over the CPCV paths."""
    frame = _cached_file(
        research_dir() / "cs_ic_daily.csv", _csv(index_col=0, parse_dates=["date"]), "Daily IC"
    )
    paths = [c for c in frame.columns if c.startswith("spearman_path_")]
    if not paths:
        raise DataError("Daily IC has an unexpected layout", "no spearman_path_* columns")
    out = pd.DataFrame({"ic": frame[paths].mean(axis=1), "tickers": frame["n_tickers"]})
    out.index = (
        pd.DatetimeIndex(frame["date"]) if "date" in frame else pd.DatetimeIndex(frame.index)
    )
    return out.sort_index()


@derived
def rolling_ic(days: int = 252) -> pd.DataFrame:
    """Trailing mean of the daily cross-sectional IC with a +/- 1.96 standard-error band.

    Daily ICs overlap in information, so the band understates the uncertainty;
    the page says so.
    """
    daily = daily_cross_sectional_ic()["ic"]
    mean = daily.rolling(days, min_periods=days).mean()
    se = daily.rolling(days, min_periods=days).std() / np.sqrt(days)
    return pd.DataFrame({"ic": mean, "low": mean - 1.96 * se, "high": mean + 1.96 * se}).dropna()


@derived
def forecasts_and_outcomes() -> pd.DataFrame:
    """Each development row's forecast (mean of its 5 out-of-fold forecasts) and realised return.

    The realised return is the fixed-horizon label the model was scored on,
    built by src.labelling.fixed_horizon from the development panel.
    """
    from src.labelling.fixed_horizon import label_fixed_horizon
    from src.pipelines.research import development_panel

    predictions = _cached_file(research_dir() / "predictions.parquet", _parquet, "Saved forecasts")
    labels = label_fixed_horizon(development_panel(panel(), config()), config())
    out = pd.DataFrame(
        {
            "forecast": predictions.mean(axis=1),
            "realised": labels["return"].reindex(predictions.index),
        }
    )
    return out.dropna()


@derived
def forecast_quintiles() -> pd.DataFrame:
    """Mean realised next-day return by pooled forecast quintile, with a 95% interval."""
    frame = forecasts_and_outcomes()
    bucket = pd.qcut(frame["forecast"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    grouped = frame.groupby(bucket, observed=True)
    out = grouped.agg(
        forecast=("forecast", "mean"),
        realised=("realised", "mean"),
        std=("realised", "std"),
        n=("realised", "size"),
    )
    se = out["std"] / np.sqrt(out["n"])
    out["low"], out["high"] = out["realised"] - 1.96 * se, out["realised"] + 1.96 * se
    return out


def features() -> pd.DataFrame:
    """The saved feature matrix, (date, ticker) by feature."""
    return _cached_file(config().data.features_dir / "features.parquet", _parquet, "Feature matrix")


# --- Strategies, allocations and benchmarks ----------------------------------


@dataclass(frozen=True)
class Book:
    """One strategy, allocation or benchmark, as both periods name it."""

    key: str
    label: str
    family: str
    source: str
    development_name: str
    rebalance: str


BOOKS: dict[str, Book] = {
    b.key: b
    for b in [
        Book(
            "allocation_minimum_variance",
            "Minimum variance",
            "Allocation",
            "portfolio",
            "minimum_variance",
            "Monthly",
        ),
        Book(
            "allocation_inverse_volatility",
            "Inverse volatility",
            "Allocation",
            "portfolio",
            "inverse_volatility",
            "Monthly",
        ),
        Book(
            "allocation_risk_parity",
            "Risk parity",
            "Allocation",
            "portfolio",
            "risk_parity",
            "Monthly",
        ),
        Book(
            "allocation_maximum_sharpe",
            "Maximum Sharpe",
            "Allocation",
            "portfolio",
            "maximum_sharpe",
            "Monthly",
        ),
        Book("ml_long_short", "ML long/short", "ML signal", "evaluation", "ml_long_short", "Daily"),
        Book("ml_long_only", "ML long only", "ML signal", "evaluation", "ml_long_only", "Daily"),
        Book(
            "ml_thresholded",
            "ML thresholded",
            "ML signal",
            "evaluation",
            "ml_thresholded",
            "On signal",
        ),
        Book("momentum", "Momentum 12-1", "Rule", "evaluation", "momentum", "Monthly"),
        Book("pairs", "Pairs mean reversion", "Rule", "evaluation", "pairs", "On signal"),
        Book("buy_and_hold", "SPY buy & hold", "Benchmark", "evaluation", "buy_and_hold", "Never"),
        Book(
            "equal_weight_hold",
            "Equal weight, hold",
            "Benchmark",
            "evaluation",
            "equal_weight_hold",
            "Never",
        ),
        Book(
            "equal_weight",
            "Equal weight, monthly",
            "Benchmark",
            "portfolio",
            "equal_weight",
            "Monthly",
        ),
    ]
}
STRATEGIES = [k for k, b in BOOKS.items() if b.family != "Benchmark"]
BENCHMARKS = [k for k, b in BOOKS.items() if b.family == "Benchmark"]


def book(key: str) -> Book:
    """A book by key, refusing unknown names."""
    if key not in BOOKS:
        raise DataError("Unknown strategy", f"{key} is not a strategy the project evaluated")
    return BOOKS[key]


def strategy_comparison() -> pd.DataFrame:
    """Development CPCV statistics per strategy, means over the 5 paths."""
    frame = _cached_file(EVALUATION / "comparison.csv", _csv(), "Strategy comparison")
    return require(
        frame, ["strategy", "kind", "sharpe_gross", "sharpe_net"], "comparison"
    ).set_index("strategy")


def strategy_paths() -> pd.DataFrame:
    """Development statistics per strategy and CPCV path."""
    return _cached_file(EVALUATION / "paths.csv", _csv(), "Per-path strategy statistics")


def allocation_comparison() -> pd.DataFrame:
    """Development statistics per allocation rule."""
    frame = _cached_file(PORTFOLIO / "comparison.csv", _csv(), "Allocation comparison")
    return frame.set_index("allocation")


def monte_carlo() -> pd.DataFrame:
    """Block-bootstrap resampling of each development strategy."""
    return _cached_file(EVALUATION / "monte_carlo.csv", _csv(), "Monte Carlo table").set_index(
        "strategy"
    )


def strategy_regimes() -> pd.DataFrame:
    """Development Sharpe per volatility regime, against equal-weight hold."""
    return _cached_file(EVALUATION / "regimes.csv", _csv(), "Regime breakdown")


def _path_returns() -> pd.DataFrame:
    frame = _cached_file(EVALUATION / "path_returns.parquet", _parquet, "Development path returns")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def _allocation_returns() -> pd.DataFrame:
    frame = _cached_file(PORTFOLIO / "allocation_returns.parquet", _parquet, "Allocation returns")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def holdout_daily() -> pd.DataFrame:
    """Daily gross and net returns of every declared book over the holdout window."""
    frame = _cached_file(
        HOLDOUT_ATTEMPT / "daily_returns.csv",
        _csv(index_col=0, parse_dates=["date"]),
        "Holdout daily returns",
    )
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def holdout_summary() -> pd.DataFrame:
    """Holdout statistics per declared book."""
    return _cached_file(HOLDOUT_ATTEMPT / "summary.csv", _csv(), "Holdout summary").set_index(
        "book"
    )


def holdout_comparisons() -> pd.DataFrame:
    """Paired bootstrap intervals of each holdout book against each benchmark."""
    return _cached_file(HOLDOUT_ATTEMPT / "comparisons.csv", _csv(), "Holdout comparisons")


def holdout_evaluation() -> dict[str, Any]:
    """The holdout window, the primary's DSR and the predeclared decision."""
    return _cached_file(HOLDOUT_ATTEMPT / "evaluation.json", _json, "Holdout evaluation")


def holdout_complete() -> dict[str, Any]:
    """Output hashes and trial ids of the completed attempt."""
    return _cached_file(HOLDOUT_ATTEMPT / "COMPLETE.json", _json, "Holdout completion record")


def holdout_started() -> dict[str, Any]:
    """Preflight checks and identifiers recorded when the holdout was opened."""
    return _cached_file(HOLDOUT_ATTEMPT / "STARTED.json", _json, "Holdout start record")


def freeze() -> dict[str, Any]:
    """The frozen N, V[SR] and trial list (never rewritten)."""
    return _cached_file(FREEZE, _json, "Holdout freeze")


def declaration() -> dict[str, Any]:
    """The approved candidates, timing and decision rule."""
    return _cached_file(DECLARATION, _json, "Holdout declaration")


def manifest() -> dict[str, Any]:
    """The development-only artifacts' manifest."""
    return _cached_file(MANIFEST, _json, "Artifact manifest")


@derived
def book_returns(key: str, period: str) -> pd.DataFrame:
    """Daily gross and net returns of one book in one period.

    Development CPCV strategies are the mean over the 5 backtest paths (each
    path covers every development date once); allocations are their single
    walk-forward series; the holdout is the one-time evaluation.
    """
    b = book(key)
    if period == HOLDOUT:
        frame = holdout_daily()
        columns = {f"{key}|gross": "gross", f"{key}|net": "net"}
    elif b.source == "portfolio":
        frame = _allocation_returns()
        columns = {f"{b.development_name}|gross": "gross", f"{b.development_name}|net": "net"}
    else:
        frame = _path_returns()
        out = {}
        for kind in ("gross", "net"):
            cols = [
                c
                for c in frame.columns
                if c.startswith(f"{b.development_name}|path_") and c.endswith(f"|{kind}")
            ]
            if not cols:
                raise DataError("Returns not saved", f"no development path returns for {b.label}")
            out[kind] = frame[cols].mean(axis=1)
        return pd.DataFrame(out).dropna(how="all")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise DataError("Returns not saved", f"{b.label} has no {period} returns ({missing})")
    return frame[list(columns)].rename(columns=columns).dropna(how="all")


def _mean_over_paths(name: str) -> dict[str, float]:
    paths = strategy_paths()
    rows = paths[paths["strategy"] == name]
    if rows.empty:
        raise DataError("Statistics not saved", f"no development path statistics for {name}")
    return rows.drop(columns=["strategy", "path"]).mean(numeric_only=True).to_dict()


@derived
def book_stats(key: str, period: str) -> dict[str, Any]:
    """Gross and net statistics of one book in one period, from the saved tables.

    Values the saved tables do not hold are computed from the saved daily
    returns with src.backtest.performance, and `computed` lists them.
    """
    b = book(key)
    computed: list[str] = []
    if period == HOLDOUT:
        row = holdout_summary()
        if key not in row.index:
            raise DataError("Not evaluated", f"{b.label} was not part of the holdout")
        r = row.loc[key]
        stats = {
            f"{side}_{name}": r[f"{side}_{name}"]
            for side in ("gross", "net")
            for name in (
                "annual_return",
                "annual_volatility",
                "sharpe",
                "sortino",
                "max_drawdown",
                "calmar",
            )
        }
        stats |= {
            "turnover_annual": r["turnover_annual"],
            "cost_annual": r["cost_annual"],
            "exposure_net_mean": r["exposure_net_mean"],
            "exposure_gross_mean": r["exposure_gross_mean"],
            "sharpe_se": r["sharpe_standard_error"],
            "psr_vs_zero": r["psr_vs_zero"],
            "days": int(r["days"]),
            "source": "Holdout attempt 1 · reports/results/holdout/attempt-1/summary.csv",
        }
    elif b.source == "evaluation":
        means = _mean_over_paths(b.development_name)
        stats = {k: means.get(k, np.nan) for k in means}
        c = strategy_comparison().loc[b.development_name]
        stats |= {
            "exposure_net_mean": c["exposure_net_mean"],
            "exposure_gross_mean": c["exposure_gross_mean"],
            "sharpe_se": c["sharpe_se"],
            "sharpe_net_std": c["sharpe_net_std"],
            "days_net_short": c["days_net_short"],
            "breakeven_return": c["breakeven_return"],
            "n_paths": int(c["n_paths"]),
            "source": "Development CPCV, mean over 5 paths · reports/results/evaluation/paths.csv",
        }
    else:
        c = allocation_comparison().loc[b.development_name]
        returns = book_returns(key, period)
        gross = performance_stats(returns["gross"])
        computed = [f"gross_{k}" for k in gross if k != "sharpe"] + ["net_sortino", "net_calmar"]
        net = performance_stats(returns["net"])
        stats = {f"gross_{k}": v for k, v in gross.items()}
        stats |= {
            "gross_sharpe": c["sharpe_gross"],
            "net_sharpe": c["sharpe_net"],
            "net_annual_return": c["annual_return_net"],
            "net_annual_volatility": c["annual_volatility_net"],
            "net_max_drawdown": c["max_drawdown_net"],
            "net_sortino": net["sortino"],
            "net_calmar": net["calmar"],
            "turnover_annual": c["turnover_annual"],
            "cost_annual": c["cost_annual"],
            "exposure_net_mean": c.get("invested_mean", np.nan),
            "exposure_gross_mean": c.get("invested_mean", np.nan),
            "sharpe_se": c["sharpe_se"],
            "source": "Development walk-forward, one path · "
            "reports/results/portfolio/comparison.csv",
        }
    returns = book_returns(key, period)
    stats |= {"start": returns.index.min(), "end": returns.index.max(), "computed": computed}
    stats.setdefault("days", len(returns))
    return stats


def interval_against(key: str, benchmark: str, period: str) -> dict[str, Any] | None:
    """The saved paired interval of key's net Sharpe minus the benchmark's, if one exists."""
    if period == HOLDOUT:
        rows = holdout_comparisons()
        hit = rows[(rows["book"] == key) & (rows["benchmark"] == benchmark)]
        if hit.empty:
            return None
        r = hit.iloc[0]
        return {
            "difference": r["difference"],
            "low": r["low"],
            "high": r["high"],
            "share_positive": r["share_positive"],
            "role": r["role"],
        }
    b = book(key)
    if b.source == "evaluation" and benchmark == "equal_weight_hold" and b.family != "Benchmark":
        c = strategy_comparison().loc[b.development_name]
        return {
            "difference": c["sharpe_net_vs_equal_weight_hold"],
            "low": c["vs_reference_low"],
            "high": c["vs_reference_high"],
            "share_positive": c["resamples_beating_reference"],
            "role": "development",
        }
    if b.source == "portfolio" and benchmark == "equal_weight" and key != "equal_weight":
        c = allocation_comparison().loc[b.development_name]
        return {
            "difference": c["vs_equal_weight"],
            "low": c["vs_equal_weight_low"],
            "high": c["vs_equal_weight_high"],
            "share_positive": c["p_beat_equal_weight"],
            "role": "development",
        }
    return None


def growth(returns: pd.Series, base: float = 1.0) -> pd.Series:
    """Value of `base` invested, compounded through the returns."""
    return base * (1 + returns.fillna(0.0)).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    """Fall from the running peak of the growth curve, as a negative fraction."""
    curve = growth(returns)
    peak = np.maximum(curve.cummax(), 1.0)
    return curve / peak - 1


@derived
def calendar_years(key: str, benchmark: str, period: str) -> pd.DataFrame:
    """Per calendar year: gross, net, benchmark net, cost drag and net Sharpe.

    Year returns are compounded within the year (not annualised); a partial
    first or last year says so in its label.
    """
    returns = book_returns(key, period)
    bench = book_returns(benchmark, period)["net"].reindex(returns.index)
    rows = []
    for year, chunk in returns.groupby(returns.index.year):
        b = bench.loc[chunk.index]
        gross = float((1 + chunk["gross"]).prod() - 1)
        net = float((1 + chunk["net"]).prod() - 1)
        bench_net = float((1 + b.dropna()).prod() - 1) if b.notna().any() else float("nan")
        rows.append(
            {
                "year": int(year),
                "days": len(chunk),
                "partial": len(chunk) < 240,
                "gross": gross,
                "net": net,
                "cost_drag": gross - net,
                "benchmark": bench_net,
                "excess": net - bench_net,
                "net_sharpe": sharpe_ratio(chunk["net"]),
                "net_max_drawdown": max_drawdown(growth(chunk["net"]), start=1.0),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def whole_period(returns: pd.Series) -> dict[str, float]:
    """Annualised return, Sharpe and drawdown of a displayed series."""
    return {
        "annual_return": annualised_return(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(growth(returns), start=1.0),
    }


# --- Portfolio and risk -------------------------------------------------------


def frontier() -> pd.DataFrame:
    """The in-sample frontier (expected) with each portfolio's realised test result."""
    frame = _cached_file(PORTFOLIO / "frontier.csv", _csv(index_col=0), "Efficient frontier")
    return require(
        frame, ["volatility", "target_return", "realised_return", "realised_volatility"], "frontier"
    )


def frontier_assets() -> pd.DataFrame:
    """Each asset's expected (train) and realised (test) return and volatility."""
    return _cached_file(PORTFOLIO / "frontier_assets.csv", _csv(), "Frontier assets").set_index(
        "ticker"
    )


def frontier_portfolios() -> pd.DataFrame:
    """Equal weight, minimum variance and maximum Sharpe on the frontier."""
    return _cached_file(
        PORTFOLIO / "frontier_portfolios.csv", _csv(index_col=0), "Frontier portfolios"
    )


def portfolio_metadata() -> dict[str, Any]:
    """Periods, seeds and settings of the portfolio analysis."""
    return _cached_file(PORTFOLIO / "metadata.json", _json, "Portfolio metadata")


def risk_contributions() -> pd.DataFrame:
    """Each allocation's share of portfolio variance per asset, at its latest weights."""
    return _cached_file(
        PORTFOLIO / "risk_contributions.csv", _csv(index_col=0), "Risk contributions"
    )


def risk_summary() -> pd.DataFrame:
    """Volatility, diversification ratio and concentration per allocation."""
    return _cached_file(PORTFOLIO / "risk.csv", _csv(index_col=0), "Risk summary")


def forward_risk() -> pd.DataFrame:
    """One-year simulated VaR, CVaR and drawdown per allocation and method."""
    frame = _cached_file(PORTFOLIO / "forward_risk.csv", _csv(), "Forward risk")
    return require(
        frame, ["allocation", "method", "var_95", "cvar_95", "drawdown_median"], "forward risk"
    )


def stress() -> pd.DataFrame:
    """Each allocation through the pre-2020 crises, at its latest weights."""
    return _cached_file(PORTFOLIO / "stress.csv", _csv(), "Stress scenarios")


def rolling_correlation() -> pd.Series:
    """Average pairwise correlation across the universe over time."""
    frame = _cached_file(
        PORTFOLIO / "rolling_correlation.csv",
        _csv(index_col=0, parse_dates=["date"]),
        "Rolling correlation",
    )
    frame.index = pd.DatetimeIndex(frame.index)
    return frame["average_correlation"]


@derived
def latest_weights(rule: str) -> pd.Series:
    """One allocation rule's weights at its last development rebalance (2019-12-31).

    Recomputed with portfolio.allocation_weights, the function the portfolio
    pipeline and the holdout run used, on the development returns only. These
    are the weights risk_contributions.csv decomposes and the holdout entered.
    """
    from src.pipelines.portfolio import allocation_weights, development_returns

    returns = development_returns(config(), panel())
    settings = config().portfolio
    # Only the trailing windows matter, so a short tail gives the same last rebalance.
    tail = returns.iloc[-(max(settings.covariance_window, settings.volatility_window) + 70) :]
    weights = allocation_weights(tail, config())[rule].dropna(how="all")
    if weights.empty:
        raise DataError("Weights unavailable", f"no rebalance of {rule} in the development tail")
    last = weights.iloc[-1].fillna(0.0)
    last.name = weights.index[-1]
    return last


# --- Trials and experiments ---------------------------------------------------


@derived
def trial_table() -> pd.DataFrame:
    """Every logged execution, metrics expanded, newest last."""
    path = Path(config().evaluation.trial_log)
    if not path.is_file():
        raise DataError("Trial log not found", f"{_relative(path)} does not exist")
    return TrialLog(path).trial_table()


@derived
def backtest_trial_sharpes() -> pd.Series:
    """Net Sharpe of every distinct backtest trial, superseded versions included."""
    log = TrialLog(Path(config().evaluation.trial_log))
    trials = log.unique_trials({"kind": "backtest"})
    if "metric_sharpe" not in trials:
        return pd.Series(dtype=float)
    return trials["metric_sharpe"].dropna()


@derived
def mlflow_runs(limit: int = 50) -> pd.DataFrame:
    """Recent MLflow runs from the local SQLite store, read-only; empty if absent."""
    if not MLFLOW_DB.is_file():
        raise DataError(
            "MLflow store not found", f"{_relative(MLFLOW_DB)} is not present on this machine"
        )
    uri = f"file:{MLFLOW_DB.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            runs = pd.read_sql_query(
                "SELECT r.run_uuid, r.name, r.status, r.start_time, e.name AS experiment "
                "FROM runs r JOIN experiments e ON r.experiment_id = e.experiment_id "
                "WHERE r.lifecycle_stage = 'active' ORDER BY r.start_time DESC LIMIT ?",
                connection,
                params=(limit,),
            )
    except sqlite3.Error as exc:
        raise DataError("MLflow store unreadable", str(exc)) from exc
    runs["start_time"] = pd.to_datetime(runs["start_time"], unit="ms")
    return runs


def text_file(path: Path, what: str) -> str:
    """A saved report's text."""
    return _cached_file(path, lambda p: p.read_text(encoding="utf-8"), what)
