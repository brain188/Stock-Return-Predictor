"""Forward-looking risk: what could happen next, not what happened.

A backtest reports the one history that occurred. This module simulates
many possible next years from the return distribution and reads off the
tail: value at risk, expected shortfall, drawdown depth, time underwater and
the chance of ruin.

Three ways to simulate, from least to most honest about the tail:

- normal: matches the mean and volatility, and nothing else. Daily equity
  returns have excess kurtosis, so a normal model understates extreme days
  badly, the most consequential modelling error in risk management. Kept
  as the reference everyone is used to.
- student_t: a Student t whose degrees of freedom are fitted by maximum
  likelihood (the tail shape) and whose scale is set so its variance equals
  the sample variance. Fitting the scale by likelihood too would describe
  the bulk of the days well but, with degrees of freedom near 2, imply a
  volatility far above the one measured (37.6% against 16.0% for one
  allocation here), so the "fat tails" would really be extra volatility.
  Days are still independent.
- bootstrap: blocks of real history, which keep both the fat tails and the
  volatility clustering, but can only replay moves that already happened.

Stress scenarios do the opposite of simulating a crash: they replay a real
one against today's weights. Every default scenario sits before the holdout
(2020 onward), and a scenario that reaches into the holdout is refused
unless the caller explicitly allows it, so risk analysis cannot quietly
spend the out-of-sample data.

Losses are reported as positive numbers: a VaR of 0.25 means losing 25%.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest.performance import block_bootstrap_indices
from src.evaluation.monte_carlo import DEFAULT_BLOCK, max_drawdowns

FORWARD_METHODS = ("normal", "student_t", "bootstrap")

# Crises before the holdout, as (first day, last day) of the selloff.
DEFAULT_SCENARIOS: dict[str, tuple[str, str]] = {
    "2008 financial crisis": ("2008-09-01", "2009-03-09"),
    "2011 euro debt crisis": ("2011-07-22", "2011-10-03"),
    "2015 China devaluation": ("2015-08-10", "2016-02-11"),
    "2018 Q4 selloff": ("2018-09-20", "2018-12-24"),
}

# The Student t needs more than 2 degrees of freedom to have a variance.
# Floor on the degrees of freedom: at 3 the variance and the skewness exist.
# Daily equity fits often land near 2.5, where the variance barely exists.
MIN_DEGREES_OF_FREEDOM = 3.0

# A long-only book cannot lose more than everything in a day.
WORST_DAY = -0.999


class RiskError(Exception):
    """Raised when a simulation or scenario cannot be run as asked."""


def fit_student_t(returns: pd.Series | np.ndarray) -> tuple[float, float, float]:
    """A Student t with the fitted tail shape and the sample's mean and variance.

    The degrees of freedom come from a maximum likelihood fit, floored at
    MIN_DEGREES_OF_FREEDOM. The location is the sample mean, and the scale is
    chosen so the t's variance, scale^2 df / (df - 2), equals the sample
    variance.

    Returns:
        Degrees of freedom, location and scale.
    """
    values = np.asarray(returns, dtype=float)
    values = values[~np.isnan(values)]
    df, _, _ = stats.t.fit(values)
    df = max(float(df), MIN_DEGREES_OF_FREEDOM)
    scale = values.std(ddof=1) * np.sqrt((df - 2) / df)

    return df, float(values.mean()), float(scale)


def simulate_forward(
    returns: pd.Series | np.ndarray,
    horizon_days: int,
    n_runs: int,
    method: str = "student_t",
    *,
    seed: int = 0,
    block_size: int = DEFAULT_BLOCK,
) -> np.ndarray:
    """Simulated daily returns for the next horizon_days, shape (n_runs, horizon_days).

    Args:
        returns: Historical daily returns the simulation is fitted to.
        horizon_days: Days ahead.
        n_runs: Paths.
        method: "normal", "student_t" or "bootstrap".
        seed: Random seed.
        block_size: Block length for the bootstrap.
    """
    values = np.asarray(returns, dtype=float)
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)

    if method == "normal":
        return rng.normal(values.mean(), values.std(ddof=1), size=(n_runs, horizon_days))

    if method == "student_t":
        df, loc, scale = fit_student_t(values)
        draws = loc + scale * rng.standard_t(df, size=(n_runs, horizon_days))
        return np.maximum(draws, WORST_DAY)

    if method == "bootstrap":
        # Long enough index rows, then cut to the horizon: blocks of real days.
        length = max(horizon_days, block_size)
        indices = block_bootstrap_indices(len(values), n_runs, block_size, rng)
        while indices.shape[1] < length:
            indices = np.concatenate(
                [indices, block_bootstrap_indices(len(values), n_runs, block_size, rng)], axis=1
            )
        return values[indices[:, :horizon_days]]

    raise RiskError(f"method must be one of {FORWARD_METHODS}, got '{method}'")


def horizon_returns(paths: np.ndarray) -> np.ndarray:
    """Compounded return over each path (one value per run); 1-D input is single periods."""
    paths = np.asarray(paths, dtype=float)

    return paths if paths.ndim == 1 else np.prod(1 + paths, axis=1) - 1


def value_at_risk(paths: np.ndarray, level: float = 0.95) -> float:
    """The loss not exceeded with probability `level`, as a positive fraction.

    Read off the distribution of horizon returns: VaR at 95% is minus the
    5th percentile.
    """
    return float(-np.quantile(horizon_returns(paths), 1 - level))


def conditional_var(paths: np.ndarray, level: float = 0.95) -> float:
    """Expected shortfall: the average loss on the paths at or beyond the VaR."""
    outcomes = horizon_returns(paths)
    cutoff = np.quantile(outcomes, 1 - level)

    return float(-outcomes[outcomes <= cutoff].mean())


def drawdown_distribution(
    paths: np.ndarray,
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
) -> dict[str, float]:
    """How deep a drawdown to expect within the horizon, and how often it gets bad.

    Returns:
        median and p95 of the worst drawdown (positive fractions), and the
        probability of a drawdown of at least each threshold.
    """
    depth = -max_drawdowns(np.asarray(paths, dtype=float))
    out = {"median": float(np.median(depth)), "p95": float(np.quantile(depth, 0.95))}

    for threshold in thresholds:
        out[f"p_drawdown_{round(threshold * 100)}"] = float(np.mean(depth >= threshold))

    return out


def time_to_recovery(paths: np.ndarray) -> dict[str, float]:
    """How long the paths spend below a previous peak.

    Returns:
        median and p95 of the longest underwater spell in days, and the
        share of paths that end the horizon still below their peak.
    """
    paths = np.asarray(paths, dtype=float)
    equity = np.cumprod(1 + paths, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(paths), 1)), equity], axis=1), axis=1)
    under = equity < peak[:, 1:] - 1e-15

    longest = np.zeros(len(paths), dtype=int)
    current = np.zeros(len(paths), dtype=int)
    for t in range(paths.shape[1]):
        current = np.where(under[:, t], current + 1, 0)
        longest = np.maximum(longest, current)

    return {
        "median_days": float(np.median(longest)),
        "p95_days": float(np.quantile(longest, 0.95)),
        "share_unrecovered": float(under[:, -1].mean()),
    }


def ruin_probability(paths: np.ndarray, threshold: float = 0.5) -> float:
    """Chance that capital falls to `threshold` of its start at any point on the path."""
    equity = np.cumprod(1 + np.asarray(paths, dtype=float), axis=1)

    return float(np.mean(equity.min(axis=1) <= threshold))


def risk_summary(
    paths: np.ndarray, levels: tuple[float, ...], ruin_floor: float
) -> dict[str, float]:
    """VaR, CVaR, drawdown, recovery and ruin of one set of simulated paths."""
    out: dict[str, float] = {}

    for level in levels:
        tag = round(level * 100)
        out[f"var_{tag}"] = value_at_risk(paths, level)
        out[f"cvar_{tag}"] = conditional_var(paths, level)

    out |= {f"drawdown_{k}": v for k, v in drawdown_distribution(paths).items()}
    out |= {f"underwater_{k}": v for k, v in time_to_recovery(paths).items()}
    out["ruin_probability"] = ruin_probability(paths, ruin_floor)

    return out


def stress_scenarios(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    scenarios: Mapping[str, tuple[str, str]] = DEFAULT_SCENARIOS,
    *,
    holdout_start: str | pd.Timestamp | None = None,
    allow_holdout: bool = False,
) -> pd.DataFrame:
    """Replay real crises against today's weights.

    The portfolio is bought at the weights on the scenario's first day and
    left to drift, as a holding would, through the crisis.

    Args:
        asset_returns: Daily returns, one column per ticker.
        weights: Today's weights, indexed by ticker; tickers missing from a
            scenario's data count as cash.
        scenarios: Name to (first day, last day).
        holdout_start: First date of the untouched holdout.
        allow_holdout: Permit scenarios that reach into the holdout.

    Returns:
        One row per scenario: days, total return, max drawdown, worst day and
        annualised volatility of the portfolio over the window.

    Raises:
        RiskError: For a scenario reaching the holdout without permission,
            or one with no data.
    """
    rows = {}
    weights = weights.astype(float)

    for name, (start, end) in scenarios.items():
        if (
            holdout_start is not None
            and pd.Timestamp(end) >= pd.Timestamp(holdout_start)
            and not allow_holdout
        ):
            raise RiskError(f"scenario '{name}' reaches the holdout starting {holdout_start}")

        window = asset_returns.loc[start:end, weights.index.intersection(asset_returns.columns)]
        if window.empty:
            raise RiskError(f"scenario '{name}' has no data between {start} and {end}")

        held = weights.reindex(window.columns).fillna(0.0)
        value = (held * (1 + window.fillna(0.0)).cumprod()).sum(axis=1) + (1 - held.sum())
        daily = value.pct_change().fillna(value.iloc[0] - 1)
        rows[name] = {
            "start": window.index[0].date(),
            "end": window.index[-1].date(),
            "days": len(window),
            "total_return": float(value.iloc[-1] - 1),
            "max_drawdown": float(
                (value / np.maximum.accumulate(np.r_[1.0, value.to_numpy()])[1:] - 1).min()
            ),
            "worst_day": float(daily.min()),
            "annual_volatility": float(daily.std() * np.sqrt(252)),
        }

    table = pd.DataFrame(rows).T
    table.index.name = "scenario"

    return table
