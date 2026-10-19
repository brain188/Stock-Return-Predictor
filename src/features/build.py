"""Assemble the feature matrix for the whole universe, with no look-ahead.

Features are computed per ticker, from that ticker's own history, and stacked
into one matrix indexed by (date, ticker). Pooling tickers this way assumes the
signal is shared across them; that assumption belongs to the model, but the
matrix makes it possible.

The leak check from notebook 3 runs inside the build as an assertion: the
matrix is rebuilt on a panel cut off at an earlier date, and every row the two
share must be identical. A feature that used data from after its own date
would differ, and the build stops before the leak can reach a model.

The fractional differencing order is a single number for the whole universe,
the largest of the per-ticker minimum orders, so no ticker is left sitting on
the stationarity threshold. It is chosen on data before validation.test_start
only, so the test period never picks its own d. Data before that date is
still evaluated in cross-validation later, which is a small and stated leak
of one scalar.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import Config
from src.features import technical
from src.features.fracdiff import frac_diff, fracdiff_weights, min_ffd_order
from src.logger import get_logger, log_frame

logger = get_logger(__name__)

# Rows compared in the look-ahead check must agree to this tolerance.
LOOKAHEAD_TOLERANCE = 1e-10


class LookaheadError(AssertionError):
    """Raised when a feature changes after data later than its date is removed."""


def feature_names(config: Config) -> list[str]:
    """Every feature column, in the order the matrix holds them.

    This is the single source of truth for column order. Models, saved
    bundles and importance tables all take their order from here.
    """
    f = config.features
    short, long = f.vol_ratio_windows

    names = [f"ret_{lag}d" for lag in f.lagged_returns]
    names += [f"px_over_ma_{w}" for w in f.moving_averages]
    names += [f"vol_{w}d" for w in f.volatility_windows]
    names += [f"rsi_{f.rsi_window}"]
    names += [f"mom_{w}d" for w in f.momentum_windows]
    names += [f"dist_high_{f.high_window}", f"vol_ratio_{short}_{long}"]
    names += [f"volume_ratio_{f.volume_window}"]

    if f.fracdiff.enabled:
        names.append("fracdiff")

    return names


def choose_fracdiff_order(panel: pd.DataFrame, config: Config) -> float:
    """The differencing order used for every ticker.

    Args:
        panel: Cleaned panel indexed by (date, ticker).
        config: Supplies the search grid, threshold, confidence and test start.

    Returns:
        The largest per-ticker minimum order on pre-test data. Falls back to
        1.0 (plain log returns) if any ticker has no passing order.
    """
    prices = panel[config.data.price_column].unstack("ticker")
    # Strictly before test_start: .loc slicing would include the first test day.
    history = np.log(prices[prices.index < pd.Timestamp(config.validation.test_start)])

    orders = {ticker: min_ffd_order(history[ticker].dropna(), config=config) for ticker in history}

    if any(math.isnan(d) for d in orders.values()):
        logger.warning("No passing fracdiff order for %s, using d=1.0", orders)
        return 1.0

    d = max(orders.values())
    logger.info(
        "Fracdiff order d=%.2f (max of per-ticker d* before %s)", d, history.index[-1].date()
    )

    return d


def warmup_rows(config: Config, d: float | None = None) -> int:
    """Rows at the start of each ticker before every feature has a value.

    Args:
        config: Supplies the feature windows.
        d: Fracdiff order, whose weight count can exceed the longest window.

    Returns:
        The number of leading rows to drop per ticker.
    """
    rows = config.features.max_lookback

    if config.features.fracdiff.enabled and d is not None:
        width = len(fracdiff_weights(d, config.features.fracdiff.weight_threshold))
        rows = max(rows, width - 1)

    return rows


def ticker_features(bars: pd.DataFrame, config: Config, d: float) -> pd.DataFrame:
    """Every feature for one ticker.

    Args:
        bars: One ticker's cleaned bars, indexed by date.
        config: Supplies the windows and the price column.
        d: Fracdiff order.

    Returns:
        Features indexed by date, columns in feature_names order.
    """
    f = config.features
    close = bars[config.data.price_column]
    returns = close.pct_change(fill_method=None)
    short, long = f.vol_ratio_windows

    columns: dict[str, pd.Series] = {}
    columns |= technical.lagged_returns(close, f.lagged_returns).to_dict("series")
    columns |= {f"px_over_ma_{w}": technical.ma_ratio(close, w) for w in f.moving_averages}
    columns |= {f"vol_{w}d": technical.rolling_volatility(returns, w) for w in f.volatility_windows}
    columns[f"rsi_{f.rsi_window}"] = technical.rsi(close, f.rsi_window)
    columns |= {f"mom_{w}d": technical.momentum(close, w) for w in f.momentum_windows}
    columns[f"dist_high_{f.high_window}"] = technical.distance_from_high(close, f.high_window)
    columns[f"vol_ratio_{short}_{long}"] = technical.volatility_ratio(returns, short, long)
    columns[f"volume_ratio_{f.volume_window}"] = technical.volume_ratio(
        bars["volume"], f.volume_window
    )

    if f.fracdiff.enabled:
        columns["fracdiff"] = frac_diff(np.log(close), d, f.fracdiff.weight_threshold)

    return pd.DataFrame(columns, index=bars.index)[feature_names(config)]


def _compute(panel: pd.DataFrame, config: Config, d: float) -> pd.DataFrame:
    """Features for every ticker, stacked on (date, ticker)."""
    frames = {
        ticker: ticker_features(panel.xs(ticker, level="ticker"), config, d)
        for ticker in panel.index.unique("ticker")
    }

    return pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()


def assert_no_lookahead(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    config: Config,
    d: float,
    cut_fraction: float = 0.8,
) -> None:
    """Rebuild on a panel cut short and require the shared rows to match.

    If any feature at time t used data after t, removing the later data
    would change it.

    Args:
        features: The matrix built on the full panel.
        panel: The full cleaned panel.
        config: Loaded project configuration.
        d: Fracdiff order used for the full build.
        cut_fraction: Share of dates kept in the shortened panel.

    Raises:
        LookaheadError: If any shared value differs, naming the columns.
    """
    dates = panel.index.unique("date").sort_values()
    cut = dates[int(len(dates) * cut_fraction)]

    truncated = _compute(panel.loc[panel.index.get_level_values("date") <= cut], config, d)
    full = features.loc[truncated.index, truncated.columns]

    both_nan = full.isna() & truncated.isna()
    difference = (full - truncated).abs().where(~both_nan, 0.0).fillna(np.inf)
    worst = difference.max()
    leaking = list(worst.index[worst > LOOKAHEAD_TOLERANCE])

    if leaking:
        raise LookaheadError(f"features change when data after {cut.date()} is removed: {leaking}")

    logger.info("Look-ahead check passed: %s rows identical up to %s", len(full), cut.date())


def drop_warmup(frame: pd.DataFrame, config: Config, d: float | None = None) -> pd.DataFrame:
    """Drop each ticker's first warm-up rows, counted per ticker, not globally.

    Args:
        frame: Features indexed by (date, ticker).
        config: Supplies the feature windows.
        d: Fracdiff order, which may lengthen the warm-up.

    Returns:
        The frame without each ticker's first warmup_rows(config, d) rows.
    """
    rows = warmup_rows(config, d)
    position = frame.groupby(level="ticker").cumcount()

    return frame[position.to_numpy() >= rows]


def build_features(
    panel: pd.DataFrame,
    config: Config,
    d: float | None = None,
    *,
    check: bool = True,
) -> pd.DataFrame:
    """The full feature matrix, warm-up dropped and checked for look-ahead.

    Args:
        panel: Cleaned panel indexed by (date, ticker).
        config: Loaded project configuration.
        d: Fracdiff order. Chosen from pre-test data when None.
        check: Run assert_no_lookahead. Only tests should switch it off.

    Returns:
        Features indexed by (date, ticker), columns in feature_names order.
        The order used is stored in frame.attrs["fracdiff_d"].

    Raises:
        LookaheadError: If the look-ahead check fails.
    """
    if config.features.fracdiff.enabled and d is None:
        d = choose_fracdiff_order(panel, config)
    d = 1.0 if d is None else d

    features = _compute(panel, config, d)

    if check:
        assert_no_lookahead(features, panel, config, d)

    features = drop_warmup(features, config, d)
    features.attrs["fracdiff_d"] = d

    missing = int(features.isna().any(axis=1).sum())
    logger.info(
        "Features: %s columns, %s rows dropped as warm-up per ticker, %s rows with a gap",
        features.shape[1],
        warmup_rows(config, d),
        missing,
    )
    log_frame(logger, features, "Feature matrix")

    return features


def save_features(features: pd.DataFrame, config: Config) -> None:
    """Write the matrix to the features directory as parquet."""
    path = config.data.features_dir / "features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path)
    logger.info("Saved feature matrix to %s", path)
