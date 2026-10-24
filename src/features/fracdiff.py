"""Fractional differentiation and the ADF test that decides how much is enough.

Prices are not stationary, so a model cannot learn from their level. First
differences (returns) are stationary but keep almost none of the memory in the
price, which is why returns look like noise. Differencing by a fractional
order d between 0 and 1 sits between the two: the smallest d that passes the
ADF test gives a stationary series that keeps as much memory as possible.
This is pitfall 4 in López de Prado's list, integer differentiation.

The ADF test is hand rolled rather than imported so the regression behind the
number is visible. The tests check it against statsmodels.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)

# MacKinnon asymptotic critical values for the constant-only ADF regression,
# keyed by significance level.
ADF_CRITICAL = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}

# Safety cap on the number of weights, reached only for d very close to 0.
MAX_WEIGHTS = 10_000


class FracDiffError(Exception):
    """Raised when a series is too short or a setting has no critical value."""


@dataclass(frozen=True)
class ADFResult:
    """Outcome of one augmented Dickey-Fuller test.

    Attributes:
        statistic: t statistic on gamma. More negative means more stationary.
        lags: Lagged differences included in the regression.
        n_obs: Observations used in the regression.
        critical: Critical value at the requested significance.
        stationary: Whether the unit root is rejected.
    """

    statistic: float
    lags: int
    n_obs: int
    critical: float
    stationary: bool


def critical_value(confidence: float) -> float:
    """MacKinnon critical value for a confidence level such as 0.95.

    Raises:
        FracDiffError: If the level is not 0.90, 0.95 or 0.99.
    """
    significance = round(1 - confidence, 2)

    if significance not in ADF_CRITICAL:
        raise FracDiffError(
            f"no ADF critical value for confidence {confidence}, "
            f"use one of {sorted(1 - s for s in ADF_CRITICAL)}"
        )

    return ADF_CRITICAL[significance]


def schwert_lags(n: int) -> int:
    """Schwert's rule of thumb for the number of lagged differences."""
    return math.ceil(12 * (n / 100) ** 0.25)


def adf_statistic(
    series: pd.Series | np.ndarray,
    max_lags: int | None = None,
    *,
    constant: bool = True,
) -> float:
    """The ADF t statistic on gamma, from a hand rolled regression.

    Fits dy_t = alpha + gamma * y_{t-1} + sum_i delta_i * dy_{t-i} + e_t by
    least squares and returns gamma's t statistic. Under a unit root gamma is
    zero; a strongly negative t statistic rejects that.

    Args:
        series: The series to test. NaN values are dropped first.
        max_lags: Lagged differences to include. Defaults to Schwert's rule.
        constant: Include alpha. The Engle-Granger residual test leaves it
            out, because the residuals of a regression with an intercept
            already have mean zero.

    Returns:
        The t statistic.

    Raises:
        FracDiffError: If the series is too short for the regression.
    """
    y = pd.Series(series).dropna().to_numpy(dtype=float)
    n = len(y)
    lags = schwert_lags(n) if max_lags is None else max_lags

    if n - lags - 1 <= lags + 2:
        raise FracDiffError(f"series of {n} points is too short for {lags} lags")

    dy = np.diff(y)
    target = dy[lags:]

    # Columns: constant (optional), lagged level, then each lagged difference.
    columns = [np.ones(len(target))] if constant else []
    level = len(columns)
    columns += [y[lags:-1]]
    columns += [dy[lags - k : -k] for k in range(1, lags + 1)]
    design = np.column_stack(columns)

    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ beta
    variance = residuals @ residuals / (len(target) - design.shape[1])
    standard_error = math.sqrt(variance * np.linalg.inv(design.T @ design)[level, level])

    return float(beta[level] / standard_error)


def adf_test(
    series: pd.Series | np.ndarray,
    confidence: float = 0.95,
    max_lags: int | None = None,
) -> ADFResult:
    """Run the ADF test and compare it against the critical value.

    Args:
        series: The series to test.
        confidence: Confidence level, 0.90, 0.95 or 0.99.
        max_lags: Lagged differences to include. Defaults to Schwert's rule.

    Returns:
        The statistic, the settings used, and the verdict.
    """
    n = int(pd.Series(series).notna().sum())
    lags = schwert_lags(n) if max_lags is None else max_lags
    statistic = adf_statistic(series, lags)
    critical = critical_value(confidence)

    return ADFResult(
        statistic=statistic,
        lags=lags,
        n_obs=n - lags - 1,
        critical=critical,
        stationary=statistic < critical,
    )


def fracdiff_weights(d: float, threshold: float = 1e-4) -> np.ndarray:
    """Weights of the fractional difference operator (1 - B)^d.

    Uses the recursion w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k, and stops at
    the first weight smaller in magnitude than the threshold. For whole number
    d the recursion hits an exact zero, so d=0 gives [1] and d=1 gives [1, -1].

    Args:
        d: Differencing order, at least 0.
        threshold: Smallest weight magnitude kept.

    Returns:
        Weights ordered by lag, w_0 first.
    """
    if d < 0:
        raise FracDiffError(f"differencing order must be at least 0, got {d}")

    weights = [1.0]

    for k in range(1, MAX_WEIGHTS):
        weight = -weights[-1] * (d - k + 1) / k

        if abs(weight) < threshold:
            break

        weights.append(weight)

    return np.array(weights)


def frac_diff(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """Fractionally difference a series with a fixed width window.

    Every output value uses the same number of past observations, so the
    transform behaves identically at the start of the sample and the end. The
    value at time t uses only t and earlier, never later.

    Args:
        series: Typically a log price.
        d: Differencing order.
        threshold: Smallest weight magnitude kept, which sets the window width.

    Returns:
        A series with the input's index. The first width - 1 values, and any
        window that touches a NaN, are NaN.
    """
    weights = fracdiff_weights(d, threshold)
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)

    if len(values) >= len(weights):
        # convolve flips the weights, so out[t] = sum_k w_k * x_{t-k}.
        out[len(weights) - 1 :] = np.convolve(values, weights, mode="valid")

    return pd.Series(out, index=series.index, name=series.name)


def ffd_search(
    series: pd.Series,
    candidates: Iterable[float],
    threshold: float = 1e-4,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """ADF statistic and memory retained for every candidate order.

    Args:
        series: Typically a log price.
        candidates: Differencing orders to try.
        threshold: Weight threshold passed to frac_diff.
        confidence: ADF confidence level.

    Returns:
        One row per order that left enough data to test, with the ADF
        statistic, whether it passes, the correlation with the original series
        (the memory kept), the rows kept and the number of weights.
    """
    rows = []

    for d in sorted(candidates):
        transformed = frac_diff(series, d, threshold).dropna()

        try:
            result = adf_test(transformed, confidence)
        except FracDiffError:
            logger.debug("d=%.2f leaves %s rows, too few to test", d, len(transformed))
            continue

        original = series.loc[transformed.index]
        rows.append(
            {
                "d": d,
                "adf": result.statistic,
                "critical": result.critical,
                "stationary": result.stationary,
                "correlation": float(np.corrcoef(transformed, original)[0, 1]),
                "rows_kept": len(transformed),
                "weights": len(fracdiff_weights(d, threshold)),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["d", "adf", "critical", "stationary", "correlation", "rows_kept", "weights"],
    )


def min_ffd_order(
    series: pd.Series,
    candidates: Iterable[float] | None = None,
    config: Config | None = None,
) -> float:
    """The smallest differencing order that makes the series stationary.

    A result of 0 means the series passed the ADF test undifferenced. For a
    return series that is expected. For a price series it almost always
    means returns were passed in by mistake, or the price happened to end
    near where it started, so it is logged as a warning.

    Args:
        series: Typically a log price.
        candidates: Orders to try. Defaults to the config search grid.
        config: Supplies the grid, weight threshold and ADF confidence.

    Returns:
        The minimum passing order, or NaN if none in the grid passes.
    """
    if config is not None:
        settings = config.features.fracdiff
        grid = settings.search_grid if candidates is None else candidates
        threshold, confidence = settings.weight_threshold, settings.adf_confidence
    else:
        grid = candidates if candidates is not None else np.round(np.arange(0, 1.01, 0.1), 2)
        threshold, confidence = 1e-4, 0.95

    table = ffd_search(series, grid, threshold, confidence)
    passing = table[table["stationary"]]

    if passing.empty:
        logger.warning("%s: no order in %s passes the ADF test", series.name, list(grid))
        return math.nan

    d_star = float(passing["d"].iloc[0])

    if d_star == 0:
        logger.warning(
            "%s passes ADF undifferenced (d*=0). Expected for returns; for a price "
            "series check that returns were not passed in by mistake.",
            series.name,
        )

    return d_star
