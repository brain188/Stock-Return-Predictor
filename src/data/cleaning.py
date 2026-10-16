"""Turning raw pulls into bars the rest of the pipeline can trust.

Each rule here is something notebook 1 checked by hand. Cleaning bugs do not
announce themselves, they quietly become signal, so every rule reports how
many rows it touched and clean_panel refuses to return a panel whose row
counts do not reconcile with its input.

Every rule uses only the current and earlier bars. Gap detection compares a
bar with the one before it, filling only carries values forward, and the
winsorisation bounds are fixed numbers from config rather than quantiles of
the full sample, which would leak the future distribution into the past.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.logger import get_logger, log_frame

logger = get_logger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]

# Returns are computed from the adjusted close, which includes dividends and
# splits. The raw close drops dividends and understates total return.
RETURN_PRICE_COLUMN = "adj_close"

# Columns the cleaned panel carries on top of OHLCV.
FLAG_COLUMNS = ["return", "gap_flag", "filled"]


class CleaningError(Exception):
    """Raised when the input cannot be cleaned or the row counts do not reconcile."""


@dataclass(frozen=True)
class CleaningReport:
    """How many rows each cleaning rule touched, per ticker.

    Two identities must hold for every ticker, and clean_panel checks both:

        input_rows  = kept + zero_volume_dropped + off_calendar_dropped
        output_rows = kept + forward_filled + left_missing

    Attributes:
        table: One row per ticker with the counts above, plus gaps_flagged and
            returns_winsorised.
        gaps: Every flagged move, with its date, ticker and return, for
            inspection by hand.
    """

    table: pd.DataFrame
    gaps: pd.DataFrame

    def reconciles(self) -> bool:
        """Whether both row count identities hold for every ticker."""
        t = self.table
        removed = t["kept"] + t["zero_volume_dropped"] + t["off_calendar_dropped"]
        added = t["kept"] + t["forward_filled"] + t["left_missing"]

        return bool((t["input_rows"] == removed).all() and (t["output_rows"] == added).all())

    @property
    def totals(self) -> pd.Series:
        """Each count summed across the universe."""
        return self.table.sum(numeric_only=True)


def _move_bounds(config: Config) -> tuple[float, float]:
    """Lowest and highest simple return that counts as a plausible daily move.

    The threshold is applied symmetrically in log space: a rise of +50% and a
    fall of -33.3% are the same size of move. Applied to simple returns, a 2
    for 1 split would show as exactly -50% and sit on the boundary, where noise
    decides whether it is caught.
    """
    upper = config.cleaning.max_abs_daily_return

    return 1 / (1 + upper) - 1, upper


def drop_zero_volume(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Remove days on which nothing traded.

    Zero volume is a holiday row or a trading halt. Either way the price on it
    was not traded, and a return computed through it is not a real return.

    Args:
        frame: Bars for one ticker.
        config: Loaded project configuration.

    Returns:
        The bars without zero-volume days, or unchanged if the rule is off.
    """
    if not config.cleaning.drop_zero_volume:
        return frame

    return frame[frame["volume"] != 0]


def flag_price_gaps(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Flag bar to bar moves too large to be real, usually unadjusted splits.

    A move is flagged when either the close or the adjusted close moves beyond
    the configured bound. Flagged rows are kept, so they can be inspected; the
    caller decides what to do with them.

    Args:
        frame: Bars for one ticker, sorted by date.
        config: Loaded project configuration.

    Returns:
        A copy of the bars with a boolean gap_flag column.
    """
    limit = math.log1p(config.cleaning.max_abs_daily_return)
    out = frame.copy()
    flag = pd.Series(False, index=out.index)

    for column in ("close", "adj_close"):
        if column in out.columns:
            log_move = np.log(out[column] / out[column].shift(1))
            flag |= log_move.abs() > limit

    out["gap_flag"] = flag

    return out


def winsorise_returns(returns: pd.Series | pd.DataFrame, config: Config) -> pd.Series | pd.DataFrame:
    """Cap extreme returns at fixed bounds without deleting the row.

    The bounds come from config, not from quantiles of the data, so capping a
    return in 2008 never depends on returns from 2020.

    Args:
        returns: Simple returns.
        config: Loaded project configuration.

    Returns:
        The returns clipped to the plausible daily range. NaN stays NaN.
    """
    lower, upper = _move_bounds(config)

    return returns.clip(lower=lower, upper=upper)


def align_calendar(
    panel: Mapping[str, pd.DataFrame],
    benchmark: str,
    max_fill_days: int = 0,
) -> dict[str, pd.DataFrame]:
    """Put every ticker on the benchmark's trading days.

    Dates the benchmark did not trade are dropped. Dates the benchmark traded
    and a ticker did not are forward filled for up to max_fill_days, carrying
    the last traded price, and marked filled=True with volume left as NaN
    because no volume was observed. Beyond the limit, or before a ticker's
    first bar, the row stays NaN.

    Args:
        panel: Bars per ticker, including the benchmark.
        benchmark: Ticker whose dates define the calendar.
        max_fill_days: Longest run of missing days to forward fill.

    Returns:
        Bars per ticker, every one indexed exactly like the benchmark.

    Raises:
        CleaningError: If the benchmark is not in the panel.
    """
    if benchmark not in panel:
        raise CleaningError(f"benchmark {benchmark} is not in the panel")

    calendar = panel[benchmark].index
    aligned = {}

    for ticker, frame in panel.items():
        present = pd.Series(True, index=frame.index).reindex(calendar, fill_value=False)
        out = frame.reindex(calendar)

        prices = [c for c in PRICE_COLUMNS if c in out.columns]
        if max_fill_days:
            out[prices] = out[prices].ffill(limit=max_fill_days)

        out["filled"] = ~present & out[prices].notna().all(axis=1)
        if "gap_flag" in out.columns:
            out["gap_flag"] = out["gap_flag"].astype("boolean").fillna(False).astype(bool)

        aligned[ticker] = out

    return aligned


def clean_panel(
    panel: Mapping[str, pd.DataFrame],
    config: Config,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply every cleaning rule to the universe, in a fixed order.

    The order is: drop zero-volume days, flag gaps on what remains, align to
    the benchmark calendar, then compute and winsorise returns. Gaps are
    flagged before alignment so a move is always measured between two real
    traded bars.

    Args:
        panel: Raw bars per ticker, as fetch_universe returns them.
        config: Loaded project configuration.

    Returns:
        The cleaned panel, indexed by (date, ticker), with OHLCV plus return,
        gap_flag and filled columns, and the report of what each rule did.

    Raises:
        CleaningError: If the benchmark is missing, a ticker has duplicate
            dates, or the row counts fail to reconcile.
    """
    benchmark = config.universe.benchmark
    if benchmark not in panel:
        raise CleaningError(f"benchmark {benchmark} is not in the panel")

    counts: dict[str, dict[str, int]] = {}
    stage: dict[str, pd.DataFrame] = {}

    for ticker, raw in panel.items():
        if raw.index.has_duplicates:
            raise CleaningError(f"{ticker} has duplicate dates")

        frame = raw.sort_index()
        traded = drop_zero_volume(frame, config)
        stage[ticker] = flag_price_gaps(traded, config)

        counts[ticker] = {
            "input_rows": len(frame),
            "zero_volume_dropped": len(frame) - len(traded),
        }

    calendar = stage[benchmark].index
    aligned = align_calendar(stage, benchmark, config.cleaning.max_forward_fill_days)

    gap_rows = []
    for ticker, frame in aligned.items():
        returns = frame[RETURN_PRICE_COLUMN].pct_change(fill_method=None)
        capped = winsorise_returns(returns, config)
        frame["return"] = capped

        kept = int(stage[ticker].index.isin(calendar).sum())
        filled = int(frame["filled"].sum())
        counts[ticker] |= {
            "off_calendar_dropped": len(stage[ticker]) - kept,
            "kept": kept,
            "forward_filled": filled,
            "left_missing": len(frame) - kept - filled,
            "output_rows": len(frame),
            "gaps_flagged": int(frame["gap_flag"].sum()),
            "returns_winsorised": int(((capped != returns) & returns.notna()).sum()),
        }

        for date in frame.index[frame["gap_flag"]]:
            gap_rows.append({"date": date, "ticker": ticker, "return": returns.loc[date]})

    table = pd.DataFrame.from_dict(counts, orient="index")
    table.index.name = "ticker"
    gaps = pd.DataFrame(gap_rows, columns=["date", "ticker", "return"])
    report = CleaningReport(table=table, gaps=gaps)

    if not report.reconciles():
        raise CleaningError(f"row counts do not reconcile:\n{table}")

    cleaned = pd.concat(aligned, names=["ticker", "date"]).swaplevel().sort_index()
    cleaned = cleaned[[c for c in [*PRICE_COLUMNS, "volume", *FLAG_COLUMNS] if c in cleaned]]

    _log_report(report)
    log_frame(logger, cleaned, "Cleaned panel")

    return cleaned, report


def _log_report(report: CleaningReport) -> None:
    """Write the totals of each cleaning rule to the log."""
    totals = report.totals
    logger.info(
        "Cleaning: %s rows in, %s zero-volume dropped, %s off-calendar dropped, "
        "%s forward filled, %s left missing, %s gaps flagged, %s returns winsorised",
        f"{int(totals['input_rows']):,}",
        int(totals["zero_volume_dropped"]),
        int(totals["off_calendar_dropped"]),
        int(totals["forward_filled"]),
        int(totals["left_missing"]),
        int(totals["gaps_flagged"]),
        int(totals["returns_winsorised"]),
    )
    for _, row in report.gaps.iterrows():
        logger.warning(
            "Gap flagged: %s on %s, return %.2f%%",
            row["ticker"],
            row["date"].date(),
            row["return"] * 100,
        )


def save_clean_panel(cleaned: pd.DataFrame, config: Config) -> Path:
    """Write the cleaned panel to the bars directory as parquet.

    Args:
        cleaned: Output of clean_panel.
        config: Loaded project configuration.

    Returns:
        The path written.
    """
    path = config.data.bars_dir / "bars.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(path)
    logger.info("Saved cleaned panel to %s", path)

    return path
