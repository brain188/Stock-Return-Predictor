"""Which tickers the project trades, from when, and what that choice costs.

The universe is fixed as of the start date and held for the whole sample.
Every name in it is a large company that still existed in 2024, which is the
definition of survivorship bias: the companies that failed or were delisted
between 2005 and 2024 were never candidates. Removing that bias needs a point
in time constituent list, which this project does not have, so the bias is
stated in code and printed wherever results are reported.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)

# Published estimates of how much survivorship inflates annual equity returns.
SURVIVORSHIP_BIAS_RANGE = (0.01, 0.04)


class UniverseError(Exception):
    """Raised when the universe is undefined or the data does not cover it."""


# A panel is either bars per ticker (as fetch_universe returns them) or one
# wide frame with a column per ticker (as to_panel returns it).
Panel = Mapping[str, pd.DataFrame] | pd.DataFrame


@dataclass(frozen=True)
class UniverseReport:
    """How well the loaded data covers the configured universe.

    Attributes:
        table: One row per ticker with bars, first and last date, expected
            bars, coverage, and the reasons it was flagged, if any.
        expected_bars: Bar count of the benchmark, which every ticker should match.
    """

    table: pd.DataFrame
    expected_bars: int

    @property
    def flagged(self) -> list[str]:
        """Tickers with at least one coverage problem."""
        return list(self.table.index[self.table["reasons"] != ""])

    @property
    def ok(self) -> bool:
        """Whether every ticker has the full expected history."""
        return not self.flagged


def resolve_universe(config: Config) -> list[str]:
    """Return the configured tickers, upper case, in config order.

    Args:
        config: Loaded project configuration.

    Returns:
        The ticker list.

    Raises:
        UniverseError: If a ticker appears twice. A duplicate would be pooled
            twice in the model and double counted in every portfolio.
    """
    tickers = [ticker.strip().upper() for ticker in config.universe.tickers]
    duplicates = sorted({t for t in tickers if tickers.count(t) > 1})

    if duplicates:
        raise UniverseError(f"universe.tickers lists {duplicates} more than once")

    return tickers


def _first_valid_dates(panel: Panel) -> dict[str, pd.Timestamp | None]:
    """First date with data for each ticker in the panel."""
    if isinstance(panel, pd.DataFrame):
        return {str(t): panel[t].first_valid_index() for t in panel.columns}

    return {
        str(t): frame.dropna(how="all").index.min() if len(frame) else None
        for t, frame in panel.items()
    }


def _bar_counts(panel: Panel) -> dict[str, int]:
    """Number of bars with data for each ticker in the panel."""
    if isinstance(panel, pd.DataFrame):
        return {str(t): int(panel[t].notna().sum()) for t in panel.columns}

    return {str(t): len(frame.dropna(how="all")) for t, frame in panel.items()}


def _span(panel: Panel, ticker: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """First and last date with data for one ticker."""
    if isinstance(panel, pd.DataFrame):
        series = panel[ticker]
        return series.first_valid_index(), series.last_valid_index()

    index = panel[ticker].dropna(how="all").index

    return (index.min(), index.max()) if len(index) else (None, None)


def eligible_from(ticker: str, config: Config, panel: Panel | None = None) -> pd.Timestamp:
    """The first date a ticker may be traded or used for training.

    Without data this is the universe start date, because the universe was
    fixed then. With a panel it is the later of that date and the ticker's
    first bar, so a name that listed mid-sample is not traded on dates before
    it existed.

    Args:
        ticker: Symbol to look up.
        config: Loaded project configuration.
        panel: Optional loaded data, to account for a late first bar.

    Returns:
        The first eligible date.

    Raises:
        UniverseError: If the ticker is not in the universe or has no data.
    """
    ticker = ticker.upper()

    if ticker not in resolve_universe(config):
        raise UniverseError(f"{ticker} is not in the configured universe")

    start = pd.Timestamp(config.universe.start_date)

    if panel is None:
        return start

    first = _first_valid_dates(panel).get(ticker)

    if first is None or pd.isna(first):
        raise UniverseError(f"{ticker} has no data in the panel")

    return max(start, pd.Timestamp(first))


def survivorship_note(config: Config) -> str:
    """The survivorship bias statement printed by the report and the dashboard.

    Args:
        config: Loaded project configuration.

    Returns:
        A plain sentence stating the size and direction of the bias.

    Raises:
        UniverseError: If config.universe.survivorship_bias is not
            "acknowledged". Results are not reported until it is.
    """
    if config.universe.survivorship_bias != "acknowledged":
        raise UniverseError(
            "universe.survivorship_bias must be 'acknowledged' before results are reported, "
            f"got '{config.universe.survivorship_bias}'"
        )

    low, high = SURVIVORSHIP_BIAS_RANGE
    tickers = resolve_universe(config)

    return (
        f"Survivorship bias: the {len(tickers)} tickers were chosen as of "
        f"{config.universe.start_date} and every one of them survived to "
        f"{config.universe.end_date}. Companies that failed or were delisted in between "
        f"were never candidates, which inflates returns, typically by {low:.0%} to "
        f"{high:.0%} a year. Without a point in time constituent list this bias cannot be "
        "removed, so every return in this project should be read as an upper bound."
    )


def validate_universe(panel: Panel, config: Config) -> UniverseReport:
    """Check the loaded data covers every ticker with a full history.

    A missing ticker raises, because downstream code would silently trade a
    smaller universe than the one configured. A short history is flagged and
    left for the caller to decide, because it can be legitimate (a late
    listing) and is visible in the report.

    Args:
        panel: Bars per ticker, or a wide frame with a column per ticker.
        config: Loaded project configuration.

    Returns:
        The coverage report.

    Raises:
        UniverseError: If any configured ticker is absent from the panel.
    """
    tickers = resolve_universe(config)
    counts = _bar_counts(panel)

    missing = [t for t in tickers if t not in counts]
    if missing:
        raise UniverseError(f"tickers missing from the panel: {missing}")

    benchmark = config.universe.benchmark.upper()
    expected = counts[benchmark]
    minimum = config.data.min_history_days

    rows = {}
    for ticker in tickers:
        bars = counts[ticker]
        first, last = _span(panel, ticker)

        reasons = []
        if bars < minimum:
            reasons.append(f"fewer than {minimum} bars")
        if bars < expected:
            reasons.append(f"{expected - bars} bars short of {benchmark}")

        rows[ticker] = {
            "bars": bars,
            "first": first,
            "last": last,
            "expected": expected,
            "coverage": bars / expected if expected else 0.0,
            "reasons": "; ".join(reasons),
        }

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "ticker"
    report = UniverseReport(table=table, expected_bars=expected)

    logger.info(
        "Universe: %s tickers, %s expected bars each, %s flagged",
        len(tickers),
        f"{expected:,}",
        len(report.flagged),
    )
    for ticker in report.flagged:
        logger.warning("%s flagged: %s", ticker, table.loc[ticker, "reasons"])

    return report
