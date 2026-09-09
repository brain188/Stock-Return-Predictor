"""Fetching daily price data, with a cache and a fallback source.

Stooq is the primary source: it needs no API key and serves plain CSV. yfinance
scrapes unofficial endpoints and breaks whenever Yahoo changes them, so it is
only used when Stooq has nothing.

Everything is cached to parquet on first pull. After that the network is never
touched, so a source outage cannot stop work already in progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import os

from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)

# Column names every loader returns, whatever the source called them.
OHLCV_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]

# Stooq wants a market suffix on US tickers.
STOOQ_SUFFIX = ".US"


class DataSourceError(Exception):
    """Raised when no source could supply usable data for a ticker."""


@dataclass(frozen=True)
class FetchResult:
    """What a fetch produced, and where it came from."""

    ticker: str
    frame: pd.DataFrame
    source: str
    from_cache: bool

    @property
    def n_rows(self) -> int:
        """Number of daily bars returned."""
        return len(self.frame)

    @property
    def span(self) -> str:
        """First and last date in the series."""
        if self.frame.empty:
            return "empty"

        return f"{self.frame.index[0].date()} to {self.frame.index[-1].date()}"


def _cache_path(ticker: str, config: Config) -> Path:
    """Return where a ticker's raw data is cached."""
    return config.data.cache_dir / f"{ticker.upper()}.parquet"


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Put any source's output into one column layout and dtype.

    Args:
        frame: Raw frame from a data source.

    Returns:
        A frame indexed by date with the OHLCV columns, sorted ascending.
    """
    out = frame.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]

    # Stooq has no adjusted column; its close is already adjusted.
    if "adj_close" not in out.columns:
        out["adj_close"] = out["close"]

    for column in OHLCV_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA

    out = out[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"

    return out.sort_index()


def _fetch_massive(
    ticker: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Pull daily bars from Massive."""

    api_key = os.getenv("MASSIVE_API_KEY")

    if not api_key:
        raise DataSourceError(
            "MASSIVE_API_KEY is not configured"
        )

    try:
        from massive import RESTClient

        client = RESTClient(api_key)

        rows = []

        for bar in client.list_aggs(
            ticker,
            1,
            "day",
            start.isoformat(),
            end.isoformat(),
            adjusted=True,
            sort="asc",
            limit=50000,
        ):
            rows.append(
                {
                    "date": pd.to_datetime(bar.timestamp, unit="ms"),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            )

    except Exception as exc:
        raise DataSourceError(
            f"massive failed for {ticker}: {exc}"
        ) from exc

    if not rows:
        raise DataSourceError(
            f"massive returned no rows for {ticker}"
        )

    frame = pd.DataFrame(rows).set_index("date")

    return _normalise(frame)

def _fetch_yfinance(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Pull daily bars from Yahoo via yfinance.

    Args:
        ticker: Symbol to request.
        start: First date to request.
        end: Last date to request.

    Returns:
        Normalised daily bars.

    Raises:
        DataSourceError: If the request fails or returns nothing.
    """
    try:
        import yfinance as yf

        frame = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )
    except Exception as exc:
        raise DataSourceError(f"yfinance failed for {ticker}: {exc}") from exc

    if frame is None or frame.empty:
        raise DataSourceError(f"yfinance returned no rows for {ticker}")

    return _normalise(frame)


SOURCES = {"massive": _fetch_massive, "yfinance": _fetch_yfinance}


def fetch_ticker(ticker: str, config: Config, *, refresh: bool = False) -> FetchResult:
    """Return daily bars for one ticker, from cache when possible.

    Args:
        ticker: Symbol to fetch.
        config: Loaded project configuration.
        refresh: Ignore the cache and pull again.

    Returns:
        The bars, with the source they came from.

    Raises:
        DataSourceError: If neither the primary nor the fallback source worked.
    """
    path = _cache_path(ticker, config)

    if config.data.use_cache and not refresh and path.is_file():
        frame = pd.read_parquet(path)
        return FetchResult(ticker, frame, source="cache", from_cache=True)

    start, end = config.universe.start_date, config.universe.end_date
    order = [config.data.primary_source, config.data.fallback_source]
    failures = []

    for name in order:
        try:
            frame = SOURCES[name](ticker, start, end)
        except DataSourceError as exc:
            failures.append(str(exc))
            logger.debug("%s: %s", ticker, exc)
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)

        return FetchResult(ticker, frame, source=name, from_cache=False)

    raise DataSourceError(f"no source returned data for {ticker}: {'; '.join(failures)}")


def fetch_universe(
    config: Config,
    *,
    refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Fetch every ticker in the configured universe.

    A ticker that cannot be fetched is reported rather than raised, so one dead
    symbol does not stop the other ten from loading.

    Args:
        config: Loaded project configuration.
        refresh: Ignore the cache and pull again.

    Returns:
        The bars per ticker, and the tickers that could not be fetched.
    """
    frames: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in config.universe.tickers:
        try:
            result = fetch_ticker(ticker, config, refresh=refresh)
        except DataSourceError as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
            failed.append(ticker)
            continue

        if result.n_rows < config.data.min_history_days:
            logger.warning(
                "Skipping %s: only %s bars, need %s",
                ticker,
                result.n_rows,
                config.data.min_history_days,
            )
            failed.append(ticker)
            continue

        frames[ticker] = result.frame
        logger.info(
            "%-6s %5s bars  %s  via %s",
            ticker,
            f"{result.n_rows:,}",
            result.span,
            result.source,
        )

    logger.info("Fetched %s of %s tickers", len(frames), len(config.universe.tickers))

    return frames, failed


def to_panel(frames: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """Combine per-ticker frames into one wide frame of a single column.

    Args:
        frames: Bars per ticker.
        column: Which OHLCV column to extract.

    Returns:
        A frame indexed by date with one column per ticker.

    Raises:
        DataSourceError: If the column is missing from any ticker.
    """
    missing = [t for t, f in frames.items() if column not in f.columns]

    if missing:
        raise DataSourceError(f"'{column}' not available for {missing}")

    panel = pd.DataFrame({ticker: frame[column] for ticker, frame in frames.items()})

    return panel.sort_index()


def load_panel(config: Config, column: str | None = None) -> pd.DataFrame:
    """Fetch the universe and return one wide price frame.

    Args:
        config: Loaded project configuration.
        column: Which column to extract. Defaults to the configured price column.

    Returns:
        Prices indexed by date, one column per ticker.

    Raises:
        DataSourceError: If no ticker could be fetched.
    """
    frames, failed = fetch_universe(config)

    if not frames:
        raise DataSourceError(f"no tickers could be fetched, all {len(failed)} failed")

    return to_panel(frames, column or config.data.price_column)