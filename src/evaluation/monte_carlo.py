"""Resample returns into a distribution of outcomes, not a single equity curve.

One backtest is one draw of history. Resampling the daily returns many times
asks what the same strategy would have produced on reshuffled versions of
that history, and turns "Sharpe 0.70 against 0.85" into "beat the benchmark
on 22% of paths", which is the more honest statement.

How the days are resampled decides whether the answer is honest:

- IID draws single days with replacement. It shuffles away volatility
  clustering (calm follows calm, panic follows panic), so drawdowns come
  out far too small and every strategy looks safer than it is. Kept only
  for comparison.
- Block bootstrap draws runs of 21 consecutive days, about a trading month,
  long enough to keep a volatility regime together. This is the default.
- Stationary bootstrap (Politis and Romano, 1994) draws blocks of random,
  geometrically distributed length, so the result does not hinge on one
  block size.

Notebook 4 checked the difference numerically: the autocorrelation of
squared returns was 0.062 under the block bootstrap and -0.0006 under IID.

Comparisons are paired. When a strategy is compared with a benchmark, both
are resampled with the same days, so a path that drew 2008 for the strategy
also drew 2008 for the benchmark. Resampling them independently would
compare one market with another and overstate the uncertainty.

All resampling is circular (an index past the end wraps to the start) so
every day is equally likely to be drawn.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.backtest.costs import TRADING_DAYS
from src.backtest.performance import ZERO_VOLATILITY, block_bootstrap_indices
from src.config import Config

METHODS = ("iid", "block", "stationary")
DEFAULT_BLOCK = 21


def iid_indices(n: int, n_runs: int, rng: np.random.Generator) -> np.ndarray:
    """Day indices drawn independently with replacement, shape (n_runs, n)."""
    return rng.integers(0, n, size=(n_runs, n))


def stationary_indices(
    n: int,
    mean_block: float,
    n_runs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stationary bootstrap indices: blocks of geometric length with mean mean_block.

    Each day either starts a new block at a random day (probability
    1 / mean_block) or continues the current block with the next day.
    """
    if mean_block < 1:
        raise ValueError(f"mean_block must be at least 1, got {mean_block}")

    starts = rng.integers(0, n, size=(n_runs, n))
    new_block = rng.random((n_runs, n)) < 1 / mean_block
    new_block[:, 0] = True

    indices = np.empty((n_runs, n), dtype=np.int64)
    indices[:, 0] = starts[:, 0]
    for t in range(1, n):
        indices[:, t] = np.where(new_block[:, t], starts[:, t], (indices[:, t - 1] + 1) % n)

    return indices


def resample_indices(
    n: int,
    method: str,
    n_runs: int,
    seed: int,
    block_size: float = DEFAULT_BLOCK,
) -> np.ndarray:
    """Indices for any method, shape (n_runs, n), reproducible from the seed.

    Block size 1 is the same draw as IID with the same seed, by construction.
    """
    rng = np.random.default_rng(seed)

    if method == "iid":
        return iid_indices(n, n_runs, rng)
    if method == "block":
        return block_bootstrap_indices(n, n_runs, int(block_size), rng)
    if method == "stationary":
        return stationary_indices(n, block_size, n_runs, rng)

    raise ValueError(f"method must be one of {METHODS}, got '{method}'")


def _values(returns: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
    """Returns as a float array, missing days dropped (rows with any missing value)."""
    if isinstance(returns, pd.DataFrame):
        return returns.dropna().to_numpy(dtype=float)

    values = np.asarray(returns, dtype=float)

    return values[~np.isnan(values)] if values.ndim == 1 else values


def iid_bootstrap(returns: pd.Series | np.ndarray, n_runs: int, seed: int = 0) -> np.ndarray:
    """Single days resampled with replacement, shape (n_runs, n_periods)."""
    values = _values(returns)

    return values[resample_indices(len(values), "iid", n_runs, seed)]


def block_bootstrap(
    returns: pd.Series | np.ndarray,
    block_size: int = DEFAULT_BLOCK,
    n_runs: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Blocks of consecutive days resampled with replacement, shape (n_runs, n_periods)."""
    values = _values(returns)

    return values[resample_indices(len(values), "block", n_runs, seed, block_size)]


def stationary_bootstrap(
    returns: pd.Series | np.ndarray,
    mean_block: float = DEFAULT_BLOCK,
    n_runs: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Blocks of random length resampled with replacement, shape (n_runs, n_periods)."""
    values = _values(returns)

    return values[resample_indices(len(values), "stationary", n_runs, seed, mean_block)]


def simulate_paths(
    returns: pd.Series | pd.DataFrame,
    method: str = "block",
    config: Config | None = None,
    *,
    n_runs: int | None = None,
    block_size: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Resampled return paths; several series are resampled with the same days.

    Args:
        returns: One daily return series, or several as DataFrame columns.
        method: "iid", "block" or "stationary".
        config: Supplies the defaults: evaluation.monte_carlo_runs,
            evaluation.bootstrap_block_days and project.random_seed.
        n_runs: Overrides the number of paths.
        block_size: Overrides the block size (mean block for stationary).
        seed: Overrides the seed.

    Returns:
        Shape (n_runs, n_periods) for a series, and (n_runs, n_periods,
        n_series) for a frame, where every series in a run uses the same days.
    """
    n_runs = n_runs or (config.evaluation.monte_carlo_runs if config else 1000)
    block_size = block_size or (config.evaluation.bootstrap_block_days if config else DEFAULT_BLOCK)
    seed = seed if seed is not None else (config.project.random_seed if config else 0)

    values = _values(returns)
    indices = resample_indices(len(values), method, n_runs, seed, block_size)

    return values[indices]


def max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """Worst fall from a peak on each path, as a negative fraction."""
    equity = np.cumprod(1 + paths, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(paths), 1)), equity], axis=1), axis=1)

    return (equity / peak[:, 1:] - 1).min(axis=1).clip(max=0.0)


def path_statistics(paths: np.ndarray, days: int = TRADING_DAYS) -> pd.DataFrame:
    """Sharpe, returns, volatility and drawdown of every resampled path.

    Args:
        paths: Daily returns, shape (n_runs, n_periods).
        days: Periods per year.

    Returns:
        One row per path.
    """
    paths = np.asarray(paths, dtype=float)
    std = paths.std(axis=1, ddof=1)
    total = np.prod(1 + paths, axis=1) - 1
    years = paths.shape[1] / days

    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe = np.where(std > ZERO_VOLATILITY, paths.mean(axis=1) / std, np.nan) * np.sqrt(days)
        annual = np.where(total > -1, np.power(np.maximum(1 + total, 0.0), 1 / years) - 1, -1.0)

    return pd.DataFrame(
        {
            "sharpe": sharpe,
            "annual_return": annual,
            "total_return": total,
            "annual_volatility": std * np.sqrt(days),
            "max_drawdown": max_drawdowns(paths),
        }
    )


def probability_of_beating(
    strategy_paths: np.ndarray,
    benchmark: np.ndarray | float,
    metric: str = "sharpe",
) -> float:
    """Share of paths on which the strategy beats the benchmark.

    Args:
        strategy_paths: Resampled strategy returns, shape (n_runs, n_periods).
        benchmark: Either the benchmark resampled with the same days, the
            same shape (a paired comparison, the honest one), or a single
            number to beat on every path (how notebook 4 did it: each path's
            Sharpe against the benchmark's one historical Sharpe).
        metric: A column of path_statistics.

    Returns:
        The share of paths where the strategy's metric is higher.
    """
    ours = path_statistics(strategy_paths)[metric].to_numpy()

    if np.ndim(benchmark) == 0:
        theirs = np.full(len(ours), float(benchmark))
    else:
        theirs = path_statistics(np.asarray(benchmark))[metric].to_numpy()
        if theirs.shape != ours.shape:
            raise ValueError("benchmark paths must have the same number of runs as the strategy")

    valid = ~(np.isnan(ours) | np.isnan(theirs))

    return float(np.mean(ours[valid] > theirs[valid])) if valid.any() else float("nan")


def confidence_interval(
    samples: Sequence[float] | np.ndarray, level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval holding `level` of the samples, missing values ignored."""
    values = np.asarray(samples, dtype=float)
    values = values[~np.isnan(values)]

    if not len(values):
        return float("nan"), float("nan")

    tail = (1 - level) / 2

    return float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))


def squared_return_autocorrelation(paths: np.ndarray, lag: int = 1) -> float:
    """Mean over paths of the lag autocorrelation of squared returns: volatility clustering."""
    squared = np.atleast_2d(np.asarray(paths, dtype=float)) ** 2
    a, b = squared[:, :-lag], squared[:, lag:]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))

    return float(np.nanmean((a * b).sum(axis=1) / denominator))
