"""Checks on price data loading.

Two sources with different column layouts have to end up identical, and the
cache has to be a true round trip. A silent difference between them would put
a subtle bug into every feature built afterwards.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from src.data.loaders import (
    OHLCV_COLUMNS,
    DataSourceError,
    _normalise,
    fetch_ticker,
    to_panel,
)
from tests.conftest import make_bars


@pytest.fixture
def stooq_shaped() -> pd.DataFrame:
    """Bars as Stooq returns them: title case, descending, no adjusted column."""
    bars = make_bars()
    frame = bars[["open", "high", "low", "close", "volume"]].copy()
    frame.columns = ["Open", "High", "Low", "Close", "Volume"]

    return frame.iloc[::-1]


@pytest.fixture
def yfinance_shaped() -> pd.DataFrame:
    """Bars as yfinance returns them: an Adj Close column, ascending."""
    bars = make_bars()
    frame = bars[["open", "high", "low", "close", "adj_close", "volume"]].copy()
    frame.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    return frame


def test_normalise_produces_one_layout(stooq_shaped, yfinance_shaped):
    """Both sources end up with the same columns in the same order."""
    assert list(_normalise(stooq_shaped).columns) == OHLCV_COLUMNS
    assert list(_normalise(yfinance_shaped).columns) == OHLCV_COLUMNS


def test_normalise_sorts_ascending(stooq_shaped):
    """Descending input comes back oldest first, or every rolling window is wrong."""
    assert _normalise(stooq_shaped).index.is_monotonic_increasing


def test_missing_adjusted_close_is_filled(stooq_shaped):
    """Stooq has no adjusted column; its close is already adjusted."""
    frame = _normalise(stooq_shaped)

    assert frame["adj_close"].equals(frame["close"])


def test_existing_adjusted_close_is_kept(yfinance_shaped):
    """When a source supplies an adjusted column it is not overwritten."""
    adjusted = yfinance_shaped.copy()
    adjusted["Adj Close"] = adjusted["Close"] * 0.9

    frame = _normalise(adjusted)

    assert not frame["adj_close"].equals(frame["close"])


def test_all_price_columns_are_numeric(stooq_shaped):
    """Strings from a CSV source would break every downstream calculation."""
    frame = _normalise(stooq_shaped.astype(str))

    assert all(pd.api.types.is_numeric_dtype(frame[column]) for column in OHLCV_COLUMNS)


def test_cache_round_trip(config, tmp_path, bars):
    """A cached ticker reloads without touching the network."""
    patched = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, cache_dir=tmp_path),
        universe=dataclasses.replace(config.universe, tickers=["TEST"], benchmark="TEST"),
    )
    bars.to_parquet(tmp_path / "TEST.parquet")

    result = fetch_ticker("TEST", patched)

    assert result.from_cache is True
    assert result.source == "cache"
    assert len(result.frame) == len(bars)


def test_panel_combines_tickers(bars):
    """One column per ticker, aligned on a shared date index."""
    panel = to_panel({"A": bars, "B": bars}, "adj_close")

    assert list(panel.columns) == ["A", "B"]
    assert len(panel) == len(bars)


def test_panel_rejects_a_missing_column(bars):
    """Asking for a column no ticker has fails loudly."""
    with pytest.raises(DataSourceError, match="vwap"):
        to_panel({"A": bars}, "vwap")


def test_unfetchable_ticker_raises(config, tmp_path):
    """With no cache and no network, the failure names both sources tried."""
    patched = dataclasses.replace(config, data=dataclasses.replace(config.data, cache_dir=tmp_path))

    with pytest.raises(DataSourceError, match="no source returned data"):
        fetch_ticker("NOTAREALTICKER", patched)
