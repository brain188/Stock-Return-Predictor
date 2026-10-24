"""Mean reversion pairs: trade the spread of two cointegrated stocks.

The one strategy here with an economic story behind it, which makes it a
useful contrast with the ML signal. Two companies that share risk factors
(same sector, same customers) can have a price ratio that keeps returning to
an equilibrium even though each price on its own is a random walk. The spread
log(A) - beta * log(B) is then stationary: when it stretches, bet on it
snapping back.

Three warnings that decide whether this works:

1. Cointegration found in sample often breaks out of sample. Pairs are
   selected on training data only, and the selection is a multiple test:
   every pair tested counts as a trial.
2. The hedge ratio must be rolling. A static beta fitted once silently
   decays as the companies change, and the spread stops being the spread.
   hedge_ratio() uses only the trailing window, never future data.
3. The half life says whether the holding period is days or months. A
   spread that takes a quarter to halve ties up capital for a quarter and
   pays costs on every adjustment of the hedge, so slow pairs are dropped.

Selection runs on a CPCV split's training dates, which can be up to four
disconnected blocks. Nothing that differences or lags is allowed to reach
across a gap between blocks: the ADF inside Engle-Granger, the half-life fit
and the return correlations are all computed within contiguous segments
(contiguous_segments()). The rolling hedge ratio and z score never see the
gaps at all: they run on the full, unbroken trading calendar, and at each
date use only the trailing window of real consecutive prices.

The universe has few natural pairs (one energy name, one bank), so expect
few or none to pass. That is a result, not a failure.

Prices are used in logs throughout. Spread position +1 means long the
spread: long A and short beta units of B.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.features.fracdiff import adf_statistic, schwert_lags
from src.logger import get_logger

logger = get_logger(__name__)

# MacKinnon (2010) asymptotic critical values for the Engle-Granger test with
# two variables and a constant in the cointegrating regression. They are more
# negative than plain ADF values because beta was estimated to make the
# residuals look as stationary as possible.
EG_CRITICAL = {0.01: -3.8977, 0.05: -3.3367, 0.10: -3.0446}


class PairsError(Exception):
    """Raised when a pair cannot be tested or traded."""


@dataclass(frozen=True)
class EGResult:
    """Engle-Granger cointegration test of A on B."""

    alpha: float
    beta: float
    statistic: float
    critical_value: float
    lags: int

    @property
    def cointegrated(self) -> bool:
        """Whether the residuals reject a unit root at the chosen level."""
        return self.statistic < self.critical_value


def contiguous_segments(dates: pd.Index, calendar: pd.Index) -> pd.Series:
    """Label each date with the contiguous block of the calendar it belongs to.

    Two selected dates are in the same segment when they are neighbours on
    the trading calendar. A CPCV training set made of groups 0, 1 and 4, for
    example, gives two segments: groups 0-1 and group 4.

    Args:
        dates: The selected dates, a subset of the calendar.
        calendar: Every trading date, in order.

    Returns:
        A segment number per selected date, indexed by date.
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    position = pd.DatetimeIndex(calendar).get_indexer(dates)
    if (position < 0).any():
        raise PairsError("some dates are not on the calendar")

    breaks = np.r_[0, (np.diff(position) > 1).astype(int)]

    return pd.Series(np.cumsum(breaks), index=dates, name="segment")


def _critical(significance: float) -> float:
    """The Engle-Granger critical value for a supported significance level."""
    if significance not in EG_CRITICAL:
        raise PairsError(f"significance must be one of {sorted(EG_CRITICAL)}, got {significance}")

    return EG_CRITICAL[significance]


def engle_granger_test(
    a: pd.Series,
    b: pd.Series,
    significance: float = 0.05,
    max_lags: int | None = None,
    segments: pd.Series | None = None,
) -> EGResult:
    """Regress log A on log B, then test the residuals for a unit root.

    Step one is OLS: log A = alpha + beta * log B + e. Step two is an ADF
    test on e, with no constant (e already has mean zero). Cointegration
    means e is stationary.

    Args:
        a: Prices of the first stock.
        b: Prices of the second, on the same dates.
        significance: 0.01, 0.05 or 0.10.
        max_lags: Lagged differences in the ADF. Defaults to Schwert's rule.
        segments: Segment per date (contiguous_segments), so the ADF never
            differences across a gap. The levels regression needs no
            adjacency and uses every date.

    Returns:
        The fitted relation and the test.
    """
    critical = _critical(significance)
    joined = pd.concat([np.log(a), np.log(b)], axis=1).dropna()
    if len(joined) < 100:
        raise PairsError(f"only {len(joined)} common observations")

    y, x = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    (alpha, beta), *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - alpha - beta * x

    lags = schwert_lags(len(residuals)) if max_lags is None else max_lags
    labels = None if segments is None else segments.reindex(joined.index).to_numpy()
    statistic = adf_statistic(residuals, lags, constant=False, segments=labels)

    return EGResult(
        alpha=float(alpha),
        beta=float(beta),
        statistic=statistic,
        critical_value=critical,
        lags=lags,
    )


def half_life(spread: pd.Series, segments: pd.Series | None = None) -> float:
    """Trading days for a deviation of the spread to halve.

    From the AR(1) fit d(spread)_t = c + lambda * spread_{t-1}: the
    deviation shrinks by a factor (1 + lambda) a day, so it halves after
    -ln 2 / ln(1 + lambda) days. A lambda of zero or more means no pull
    back at all, and the half life is infinite.

    Args:
        spread: The spread.
        segments: Segment per date; a day-on-day change is used only when
            both days are in the same segment.
    """
    series = pd.Series(spread).dropna()
    if len(series) < 3:
        return float("nan")

    s = series.to_numpy(dtype=float)
    same = np.ones(len(s) - 1, dtype=bool)
    if segments is not None:
        labels = segments.reindex(series.index).to_numpy()
        same = labels[1:] == labels[:-1]

    lagged, change = s[:-1][same], np.diff(s)[same]
    if len(lagged) < 3:
        return float("nan")

    design = np.column_stack([np.ones(len(lagged)), lagged])
    (_, slope), *_ = np.linalg.lstsq(design, change, rcond=None)

    if slope >= 0 or slope <= -1:
        return float("inf") if slope >= 0 else 0.0

    return float(-math.log(2) / math.log(1 + slope))


def hedge_ratio(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta of log A on log B over the trailing window.

    The value at t uses the window ending at t and nothing after, so it can
    be traded on (after the execution delay) without look-ahead.
    """
    la, lb = np.log(a), np.log(b)
    covariance = la.rolling(window).cov(lb)
    variance = lb.rolling(window).var()

    return (covariance / variance).rename("hedge_ratio")


def rolling_spread(a: pd.Series, b: pd.Series, hedge: pd.Series, window: int) -> pd.Series:
    """Residual of the rolling regression of log A on log B, at each date.

    (log A_t - mean log A) - beta_t * (log B_t - mean log B), with the means
    and beta from the same trailing window. Taking the rolling intercept
    out matters: without it a small wobble in beta is multiplied by the
    whole log price (about 4 for a $50 stock) and swamps the spread itself.
    """
    la, lb = np.log(a), np.log(b)

    return ((la - la.rolling(window).mean()) - hedge * (lb - lb.rolling(window).mean())).rename(
        "spread"
    )


def spread_zscore(
    a: pd.Series,
    b: pd.Series,
    hedge: pd.Series,
    window: int,
    *,
    hedge_window: int | None = None,
) -> pd.Series:
    """How many trailing standard deviations the spread is from its trailing mean.

    The spread is rolling_spread over hedge_window (the window the hedge
    ratio was fitted on; defaults to `window`). Its mean and standard
    deviation come from the trailing `window`. Every input is known at the
    close of t.
    """
    spread = rolling_spread(a, b, hedge, hedge_window or window)
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()

    return ((spread - mean) / std.where(std > 0)).rename("zscore")


def pairs_positions(
    zscore: pd.Series,
    entry_z: float,
    exit_z: float,
    stop_z: float,
) -> pd.Series:
    """Spread position from the z score: enter wide, exit near zero, stop out far.

    Flat until |z| crosses entry_z: short the spread (-1) when z is high,
    long it (+1) when z is low. Exit when |z| comes back inside exit_z.
    If |z| reaches stop_z the relation may have broken, so close the trade
    and stay out until the spread comes back inside exit_z.

    Returns:
        -1, 0 or +1 per date, decided at that date's close.
    """
    z = zscore.to_numpy(dtype=float)
    position = np.zeros(len(z))
    state, stopped = 0.0, False

    for t, value in enumerate(z):
        if np.isnan(value):
            state = 0.0
        elif stopped:
            stopped = abs(value) >= exit_z
        elif state == 0:
            if entry_z < value < stop_z:
                state = -1.0
            elif -stop_z < value < -entry_z:
                state = 1.0
        elif abs(value) >= stop_z:
            state, stopped = 0.0, True
        elif abs(value) <= exit_z or np.sign(value) == state:
            # Reverted inside the band, or crossed through zero to the other side.
            state = 0.0
        position[t] = state

    return pd.Series(position, index=zscore.index, name="spread_position")


def pair_weights(
    spread_position: pd.Series,
    hedge: pd.Series,
    a: str,
    b: str,
) -> pd.DataFrame:
    """Leg weights for a spread position: A gets 1 and B gets -beta, scaled to gross 1."""
    scale = 1 + hedge.abs()
    leg_a = (spread_position / scale).fillna(0.0)
    leg_b = (-spread_position * hedge / scale).fillna(0.0)

    return pd.DataFrame({a: leg_a, b: leg_b})


def _wide(prices: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Prices with one column per ticker, from a wide frame or a (date, ticker) panel."""
    if isinstance(prices.index, pd.MultiIndex):
        return prices[config.data.price_column].unstack("ticker")

    return prices


def find_pairs(
    prices: pd.DataFrame,
    config: Config,
    dates: pd.Index | None = None,
) -> pd.DataFrame:
    """Screen every pair on return correlation, then test the survivors for cointegration.

    Correlations use daily log returns computed on the full series and then
    restricted to `dates`, so no return spans a gap. The Engle-Granger
    regression runs on the prices at `dates`; when those dates are not
    contiguous (CPCV training groups) the residual series has jumps at the
    gaps, a small and stated approximation.

    Args:
        prices: Wide prices, or the cleaned (date, ticker) panel.
        config: Supplies the screen, significance and half-life limit.
        dates: Dates to select on, normally the training dates. All when None.

    Returns:
        One row per pair tested for cointegration, with correlation, beta,
        statistic, cointegrated, half_life and selected. Pairs that failed
        the correlation screen are not rows: they were not tested.
    """
    settings = config.strategies.pairs
    wide = _wide(prices, config)
    calendar = wide.index
    log_returns = np.log(wide).diff()

    if dates is None:
        segments = pd.Series(0, index=calendar)
    else:
        segments = contiguous_segments(calendar[calendar.isin(dates)], calendar)
        wide = wide.loc[segments.index]
        log_returns = log_returns.loc[segments.index]
        # The first return of a later segment starts from a close outside the
        # selection (a test-block date), so it is dropped.
        starts = segments.ne(segments.shift()) & (segments.index != segments.index[0])
        log_returns[starts.to_numpy()] = np.nan

    correlation = log_returns.corr()
    rows = []

    for a, b in itertools.combinations(sorted(wide.columns), 2):
        if correlation.loc[a, b] < settings.min_correlation:
            continue
        test = engle_granger_test(wide[a], wide[b], settings.significance, segments=segments)
        spread = np.log(wide[a]) - test.beta * np.log(wide[b])
        life = half_life(spread, segments)
        rows.append(
            {
                "a": a,
                "b": b,
                "correlation": float(correlation.loc[a, b]),
                "beta": test.beta,
                "statistic": test.statistic,
                "critical_value": test.critical_value,
                "cointegrated": test.cointegrated,
                "half_life": life,
                "selected": test.cointegrated and 0 < life <= settings.max_half_life,
            }
        )

    table = pd.DataFrame(
        rows,
        columns=[
            "a",
            "b",
            "correlation",
            "beta",
            "statistic",
            "critical_value",
            "cointegrated",
            "half_life",
            "selected",
        ],
    ).astype({"cointegrated": bool, "selected": bool})
    logger.info(
        "Pairs on %s segment(s): %s of %s passed the correlation screen, %s cointegrated, "
        "%s selected",
        segments.nunique(),
        len(table),
        math.comb(wide.shape[1], 2),
        int(table["cointegrated"].sum()),
        int(table["selected"].sum()),
    )

    return table


def pair_targets(prices: pd.DataFrame, a: str, b: str, config: Config) -> pd.DataFrame:
    """Leg weights for one pair over the whole price history, before any delay."""
    settings = config.strategies.pairs
    wide = _wide(prices, config)
    # Rolling windows must run on the unbroken calendar, never on a subset of
    # dates, or they would treat disconnected dates as consecutive.
    if not (wide.index.is_monotonic_increasing and wide.index.is_unique):
        raise PairsError("pair targets need prices on a sorted, unique trading calendar")
    hedge = hedge_ratio(wide[a], wide[b], settings.hedge_window)
    z = spread_zscore(
        wide[a], wide[b], hedge, settings.zscore_window, hedge_window=settings.hedge_window
    )
    position = pairs_positions(z, settings.entry_z, settings.exit_z, settings.stop_z)

    return pair_weights(position, hedge, a, b)


def pairs_targets(
    prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    config: Config,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The book of every selected pair, each with an equal share of the budget.

    Returns:
        The combined weights (one column per ticker in the price frame) and
        each pair's own legs, keyed "A_B", scaled to the same share.
    """
    wide = _wide(prices, config)
    combined = pd.DataFrame(0.0, index=wide.index, columns=wide.columns)
    legs = {}
    share = config.backtest.max_gross_leverage / max(len(pairs), 1)

    for a, b in pairs:
        weights = pair_targets(wide, a, b, config) * share
        legs[f"{a}_{b}"] = weights
        combined[[a, b]] += weights[[a, b]].to_numpy()

    return combined, legs
