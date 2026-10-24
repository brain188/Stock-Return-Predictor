"""Where a portfolio's risk actually comes from.

Weights are not risk. A 10% weight in a volatile, highly correlated name can
carry 30% of the portfolio's risk, and only a decomposition shows it. The
contributions here are Euler contributions: each asset's weight times its
marginal risk, and they add up exactly to the portfolio volatility.

Rolling correlation is the crisis variable. Correlations rise toward 1 just
when diversification is needed most, which is why one correlation matrix
over the whole sample flatters a portfolio: it averages the calm years in
with the crashes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.costs import TRADING_DAYS


def _vector(
    weights: pd.Series | np.ndarray, cov: pd.DataFrame | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weights and covariance as arrays, the weights aligned to the covariance's assets."""
    if isinstance(weights, pd.Series) and isinstance(cov, pd.DataFrame):
        weights = weights.reindex(cov.index).fillna(0.0)

    return np.asarray(weights, dtype=float), np.asarray(cov, dtype=float)


def portfolio_volatility(weights: pd.Series | np.ndarray, cov: pd.DataFrame | np.ndarray) -> float:
    """sqrt(w' Cov w), in the covariance's units (annual if the covariance is annual)."""
    w, sigma = _vector(weights, cov)

    return float(np.sqrt(w @ sigma @ w))


def marginal_risk(weights: pd.Series | np.ndarray, cov: pd.DataFrame | np.ndarray) -> np.ndarray:
    """How portfolio volatility changes per unit of each weight: (Cov w) / sigma_p."""
    w, sigma = _vector(weights, cov)

    return sigma @ w / np.sqrt(w @ sigma @ w)


def risk_contribution(
    weights: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    *,
    as_share: bool = False,
) -> pd.Series | np.ndarray:
    """Each asset's part of portfolio volatility: w_i x marginal risk_i.

    The parts sum to the portfolio volatility (Euler's theorem, since
    volatility is homogeneous of degree one in the weights).

    Args:
        weights: Portfolio weights.
        cov: Covariance matrix.
        as_share: Return shares of total risk, summing to 1, instead.
    """
    w, sigma = _vector(weights, cov)
    parts = w * marginal_risk(w, sigma)

    if as_share:
        parts = parts / parts.sum()

    if isinstance(cov, pd.DataFrame):
        return pd.Series(parts, index=cov.index, name="risk_contribution")

    return parts


def diversification_ratio(weights: pd.Series | np.ndarray, cov: pd.DataFrame | np.ndarray) -> float:
    """Weighted average of the assets' volatilities over the portfolio volatility.

    1 for a single asset or perfectly correlated assets; about sqrt(n) for an
    equal-weight book of n uncorrelated assets with equal volatility. Uses
    absolute weights, so shorts count as exposure, not as offsets.
    """
    w, sigma = _vector(weights, cov)

    return float(np.abs(w) @ np.sqrt(np.diag(sigma)) / np.sqrt(w @ sigma @ w))


def rolling_correlation(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """Average pairwise correlation over a trailing window, per date.

    Uses only the window ending at each date. A rising line means the assets
    move together more, and diversification is worth less.
    """
    n = returns.shape[1]
    if n < 2:
        raise ValueError("rolling_correlation needs at least two assets")

    matrices = returns.rolling(window).corr()
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    def mean_off_diagonal(block: pd.DataFrame) -> float:
        values = block.to_numpy()
        return float(np.nanmean(values[upper])) if not np.isnan(values).all() else np.nan

    return matrices.groupby(level=0).apply(mean_off_diagonal).rename("average_correlation")


def beta_to_benchmark(returns: pd.Series, benchmark: pd.Series) -> float:
    """Sensitivity to the benchmark: Cov(r, b) / Var(b), on the dates both have."""
    joined = pd.concat([returns, benchmark], axis=1, join="inner").dropna()

    return float(joined.cov().iloc[0, 1] / joined.iloc[:, 1].var())


def tracking_error(returns: pd.Series, benchmark: pd.Series, days: int = TRADING_DAYS) -> float:
    """Annualised volatility of the difference from the benchmark."""
    joined = pd.concat([returns, benchmark], axis=1, join="inner").dropna()

    return float((joined.iloc[:, 0] - joined.iloc[:, 1]).std() * np.sqrt(days))
