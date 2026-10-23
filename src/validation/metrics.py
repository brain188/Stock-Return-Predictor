"""Every score that matters for a return forecast, not just accuracy.

The information coefficient and the hit rate answer different questions and
in notebook 3 they disagreed: IC +0.0279, consistent and positive, against an
edge over AlwaysUp of -0.0357, consistent and negative. IC asks whether the
forecast ranks returns in the right order. The hit rate asks whether it gets
the sign right. A contrarian forecast in a market that rises on 55% of days
can rank well and still call the sign wrong more often than predicting up
every day. Both are reported, with the conflict explained, rather than only
the flattering one.

There are two ICs. The pooled IC ranks every (date, ticker) row together, so
it mixes timing (which days are better) with selection (which stock beats
which on the same day). The cross-sectional IC ranks tickers within each date
and measures selection alone. Both are kept, since they answer different
questions: the first research run scored a cross-sectional IC of about
+0.005 (t 0.9), so whatever signal the pooled and yearly ICs show is timing,
not stock selection. With 11 tickers a single day's IC has a standard
deviation near 0.35, so only a mean above roughly 0.012 would be detectable
over 15 years.

For scale: an IC of 0.02 to 0.05 is tradeable with enough breadth, and 0.1 is
exceptional. A hit rate means nothing without the share of up days beside it.

Every function takes predicted and actual as aligned arrays or Series and
ignores rows where either is missing. Predictions can be forecast returns,
probabilities of up, or directions of +1 / -1; the direction of a forecast
return or direction is its sign, with zero counted as not up, the same rule
the labels use. Probabilities must be centred first (p - 0.5) before calling
the sign-based metrics.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.validation.purged_cv import row_dates

ArrayLike = pd.Series | np.ndarray

# Fewest tickers on a date for a cross-sectional IC to be computed. With 11
# tickers, 5 still leaves the rank correlation room to mean something.
MIN_TICKERS = 5


def _aligned(predicted: ArrayLike, actual: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """The two inputs as float arrays, with rows missing in either dropped."""
    if isinstance(predicted, pd.Series) and isinstance(actual, pd.Series):
        predicted, actual = predicted.align(actual, join="inner")

    p, a = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    if p.shape != a.shape:
        raise ValueError(f"predicted has shape {p.shape}, actual {a.shape}")

    keep = ~(np.isnan(p) | np.isnan(a))

    return p[keep], a[keep]


def _up(values: np.ndarray) -> np.ndarray:
    """True where a value counts as up: strictly positive."""
    return values > 0


def information_coefficient(predicted: ArrayLike, actual: ArrayLike) -> float:
    """Spearman rank correlation between forecast and realised return.

    Rank rather than Pearson correlation, because a strategy trades on the
    ordering of forecasts and a handful of crash days would otherwise
    dominate. Pooled over every row given; use ic_by_period to check that it
    holds up over time.

    Returns:
        IC in [-1, 1], NaN when either side is constant or empty.
    """
    p, a = _aligned(predicted, actual)

    if len(p) < 2 or np.ptp(p) == 0 or np.ptp(a) == 0:
        return float("nan")

    return float(spearmanr(p, a).statistic)


def ic_by_period(predicted: pd.Series, actual: pd.Series, freq: str = "YE") -> pd.DataFrame:
    """IC within each calendar period, to see whether the signal is stable.

    A pooled IC of 0.03 can be one good year and nine years of nothing. The
    share of periods with a positive IC, and the t statistic of their mean,
    say more than the pooled number does.

    Args:
        predicted: Forecasts indexed by (date, ticker) or by date.
        actual: Realised returns on the same index.
        freq: Pandas period alias, for example "YE" for years or "QE".

    Returns:
        One row per period with ic and n (rows scored).
    """
    predicted, actual = predicted.align(actual, join="inner")
    period = row_dates(predicted.index).to_period(freq.removesuffix("E") or freq)
    rows = []

    for label, positions in pd.Series(np.arange(len(period))).groupby(period).groups.items():
        p, a = predicted.iloc[positions], actual.iloc[positions]
        rows.append(
            {"period": str(label), "ic": information_coefficient(p, a), "n": len(_aligned(p, a)[0])}
        )

    return pd.DataFrame(rows).set_index("period")


def ic_t_stat(by_period: pd.DataFrame) -> float:
    """t statistic of the mean period IC: mean / (std / sqrt(periods))."""
    ic = by_period["ic"].dropna()

    if len(ic) < 2 or ic.std() == 0:
        return float("nan")

    return float(ic.mean() / (ic.std() / np.sqrt(len(ic))))


def cross_sectional_ic(
    predicted: pd.Series,
    actual: pd.Series,
    method: str = "spearman",
    min_tickers: int = MIN_TICKERS,
) -> pd.DataFrame:
    """IC within each date, across tickers: does the forecast rank today's stocks?

    The pooled IC ranks every (date, ticker) row against every other, so it
    mixes two questions: which stock does better today, and which day is
    better than another. A model whose average forecast drifts from year to
    year (as each CPCV split's model does, trained on a different market)
    scores badly on the second question even when it ranks stocks well on
    every single day. The cross-sectional IC answers only the first, which
    is the question a long/short book across tickers trades on.

    Only rows where both sides are present count, and the ranks are taken
    over those rows alone. A date with fewer than min_tickers such rows, or
    where either side is constant across tickers, gets NaN rather than a
    correlation from too few points: with 3 tickers a Spearman IC can only
    take 4 values, and with 2 it is always +1 or -1.

    Args:
        predicted: Forecasts indexed by (date, ticker).
        actual: Realised returns on the same index.
        method: "spearman" (rank correlation) or "pearson".
        min_tickers: Fewest valid tickers for a date to be scored.

    Returns:
        One row per date with ic (NaN when skipped) and n_tickers, the
        number of tickers with both a forecast and a return that date.

    Raises:
        ValueError: If the index has no ticker level or the method is unknown.
    """
    if method not in {"spearman", "pearson"}:
        raise ValueError(f"method must be spearman or pearson, got {method}")

    if not isinstance(predicted.index, pd.MultiIndex) or "ticker" not in predicted.index.names:
        raise ValueError("cross-sectional IC needs a (date, ticker) index")

    predicted, actual = predicted.align(actual, join="inner")
    both = predicted.notna() & actual.notna()
    p = predicted.where(both).unstack("ticker")
    a = actual.where(both).unstack("ticker")

    if method == "spearman":
        # Average ranks for ties, so Pearson on the ranks is Spearman.
        p, a = p.rank(axis=1), a.rank(axis=1)

    n_tickers = p.notna().sum(axis=1)
    p_dev = p.sub(p.mean(axis=1), axis=0)
    a_dev = a.sub(a.mean(axis=1), axis=0)
    denominator = np.sqrt((p_dev**2).sum(axis=1) * (a_dev**2).sum(axis=1))
    ic = (p_dev * a_dev).sum(axis=1) / denominator.where(denominator > 0)

    table = pd.DataFrame({"ic": ic.where(n_tickers >= min_tickers), "n_tickers": n_tickers})
    table.index.name = "date"

    return table


def cross_sectional_summary(daily: pd.DataFrame) -> dict[str, float]:
    """Mean and variability of the daily cross-sectional IC.

    The t statistic treats each date as independent, which holds for the
    1 day label. For overlapping labels (the 5 day triple barrier) the daily
    ICs are autocorrelated and the t statistic overstates the evidence.

    Args:
        daily: Output of cross_sectional_ic.

    Returns:
        mean, std, t_stat (mean / (std / sqrt(dates))), positive_share,
        dates (scored), skipped (dates left out) and mean_tickers (average
        tickers on a scored date).
    """
    ic = daily["ic"].dropna()
    n = len(ic)
    std = float(ic.std()) if n > 1 else float("nan")

    return {
        "mean": float(ic.mean()) if n else float("nan"),
        "std": std,
        "t_stat": float(ic.mean() / (std / np.sqrt(n))) if n > 1 and std > 0 else float("nan"),
        "positive_share": float((ic > 0).mean()) if n else float("nan"),
        "dates": float(n),
        "skipped": float(len(daily) - n),
        "mean_tickers": float(daily.loc[ic.index, "n_tickers"].mean()) if n else float("nan"),
    }


def cross_sectional_score(
    predicted: pd.Series,
    actual: pd.Series,
    min_tickers: int = MIN_TICKERS,
) -> dict[str, float]:
    """Spearman and Pearson cross-sectional IC summaries as one flat dict.

    Keys are cs_<method>_<statistic>, for example cs_spearman_mean, plus
    cs_dates, cs_skipped and cs_mean_tickers, which both methods share.
    """
    out: dict[str, float] = {}

    for method in ("spearman", "pearson"):
        summary = cross_sectional_summary(
            cross_sectional_ic(predicted, actual, method, min_tickers)
        )
        for key in ("mean", "std", "t_stat", "positive_share"):
            out[f"cs_{method}_{key}"] = summary[key]
        for key in ("dates", "skipped", "mean_tickers"):
            out[f"cs_{key}"] = summary[key]

    return out


def hit_rate(predicted: ArrayLike, actual: ArrayLike) -> float:
    """Share of rows where the forecast direction matches the realised one.

    Up means strictly positive on both sides. The number is only meaningful
    next to the share of up days, which edge_over_always_up subtracts.
    """
    p, a = _aligned(predicted, actual)

    if not len(p):
        return float("nan")

    return float((_up(p) == _up(a)).mean())


def up_rate(actual: ArrayLike) -> float:
    """Share of up days: the hit rate of predicting up every day."""
    a = np.asarray(actual, dtype=float)
    a = a[~np.isnan(a)]

    return float(_up(a).mean()) if len(a) else float("nan")


def edge_over_always_up(predicted: ArrayLike, actual: ArrayLike) -> float:
    """Hit rate minus the hit rate of AlwaysUp on the same rows.

    Negative means the forecast is worse at calling direction than knowing
    nothing but the drift. Notebook 3's model scored -0.0357.
    """
    p, a = _aligned(predicted, actual)

    return hit_rate(p, a) - up_rate(a)


def directional_accuracy(predicted: ArrayLike, actual: ArrayLike) -> float:
    """Balanced accuracy: the mean of the hit rate on up days and on down days.

    The plain hit rate rewards a forecast for leaning long in a market that
    mostly rises. Averaging the two classes removes that: AlwaysUp and a coin
    flip both score 0.5, so anything above 0.5 is skill at telling the two
    kinds of day apart rather than the drift.
    """
    p, a = _aligned(predicted, actual)
    up, called_up = _up(a), _up(p)

    if up.all() or not up.any():
        return float("nan")

    return float((called_up[up].mean() + (~called_up[~up]).mean()) / 2)


def mae(predicted: ArrayLike, actual: ArrayLike) -> float:
    """Mean absolute error of a return forecast."""
    p, a = _aligned(predicted, actual)

    return float(np.abs(p - a).mean()) if len(p) else float("nan")


def rmse(predicted: ArrayLike, actual: ArrayLike) -> float:
    """Root mean squared error of a return forecast."""
    p, a = _aligned(predicted, actual)

    return float(np.sqrt(((p - a) ** 2).mean())) if len(p) else float("nan")


def r2(predicted: ArrayLike, actual: ArrayLike) -> float:
    """Out-of-sample R squared against a forecast of zero.

    The textbook R squared compares against the mean of the actual values,
    which is the test set's own average return: a benchmark nobody could have
    known in advance. Against zero, as in Gu, Kelly and Xiu (2020), a
    positive number means the forecast beats saying "no change". Expect
    values just below zero for daily returns; that is normal, not a bug.
    """
    p, a = _aligned(predicted, actual)

    if not len(p) or not (a**2).sum():
        return float("nan")

    return float(1 - ((a - p) ** 2).sum() / (a**2).sum())


def is_direction_only(predicted: ArrayLike) -> bool:
    """Whether a forecast holds only +1 / -1, so error metrics do not apply."""
    p = np.asarray(predicted, dtype=float)
    p = p[~np.isnan(p)]

    return bool(len(p)) and bool(np.isin(p, [-1.0, 1.0]).all())


def score(predicted: ArrayLike, actual: ArrayLike) -> dict[str, float]:
    """Every metric for one forecast, as a flat dict.

    The error metrics (mae, rmse, r2) are NaN for a direction-only forecast,
    where an "error" of 2 for a wrong sign would be meaningless.
    """
    p, a = _aligned(predicted, actual)
    returns = not is_direction_only(p)

    return {
        "ic": information_coefficient(p, a),
        "hit_rate": hit_rate(p, a),
        "edge_over_always_up": edge_over_always_up(p, a),
        "directional_accuracy": directional_accuracy(p, a),
        "mae": mae(p, a) if returns else float("nan"),
        "rmse": rmse(p, a) if returns else float("nan"),
        "r2": r2(p, a) if returns else float("nan"),
        "n": float(len(p)),
    }


def metric_table(
    predicted: ArrayLike,
    actual: ArrayLike,
    baselines: Mapping[str, ArrayLike] | None = None,
    name: str = "model",
) -> pd.DataFrame:
    """The model and every baseline scored on the same rows, one row each.

    Args:
        predicted: The model's forecasts.
        actual: Realised returns.
        baselines: Name to forecast, for example AlwaysUp and a coin flip,
            on the same rows as predicted.
        name: Row label for the model.

    Returns:
        One row per forecast, columns as in score(), with up_rate added so
        each hit rate can be read against the drift.
    """
    rows = {name: score(predicted, actual)}

    for label, forecast in (baselines or {}).items():
        rows[label] = score(forecast, actual)

    table = pd.DataFrame(rows).T
    table["up_rate"] = up_rate(_aligned(predicted, actual)[1])
    table.index.name = "forecast"

    return table
