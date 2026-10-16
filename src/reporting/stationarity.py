"""The stationarity table: ADF results and the minimum fracdiff order per ticker.

Notebook 2 established, by hand, that log prices fail the ADF test, log
returns pass it, and a fractional order well below 1 is enough to pass while
keeping most of the price's memory. This module recomputes that table from the
cleaned panel so the report and the dashboard read it from disk.

Run it with:

    python -m src.reporting.stationarity
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, Config, load_config
from src.features.fracdiff import adf_test, ffd_search
from src.logger import get_logger, log_step

logger = get_logger(__name__)

RESULTS_DIR = PROJECT_ROOT / "reports" / "results"


def log_prices(cleaned: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Wide frame of log prices, one column per ticker.

    Args:
        cleaned: Output of clean_panel, indexed by (date, ticker).
        config: Supplies the price column.

    Returns:
        Log prices indexed by date.
    """
    prices = cleaned[config.data.price_column].unstack("ticker")

    return np.log(prices)


def ticker_stationarity(series: pd.Series, config: Config) -> dict[str, float | bool]:
    """ADF on the level and the return, and the fracdiff search, for one series.

    Args:
        series: Log price of one ticker.
        config: Supplies the search grid, weight threshold and confidence.

    Returns:
        One row of the stationarity table.
    """
    settings = config.features.fracdiff
    confidence = settings.adf_confidence
    series = series.dropna()

    level = adf_test(series, confidence)
    returns = adf_test(series.diff().dropna(), confidence)

    search = ffd_search(series, settings.search_grid, settings.weight_threshold, confidence)
    passing = search[search["stationary"]]
    at_one = search[search["d"] == 1.0]

    row: dict[str, float | bool] = {
        "adf_log_price": level.statistic,
        "adf_log_return": returns.statistic,
        "critical": level.critical,
        "price_stationary": level.stationary,
        "return_stationary": returns.stationary,
        "d_star": np.nan,
        "adf_at_d_star": np.nan,
        "corr_at_d_star": np.nan,
        "corr_at_d_one": float(at_one["correlation"].iloc[0]) if len(at_one) else np.nan,
        "rows_lost": np.nan,
        # A price that passes undifferenced is usually a random walk that ended
        # near where it started, not a stationary process.
        "suspicious": level.stationary,
    }

    if not passing.empty:
        best = passing.iloc[0]
        row |= {
            "d_star": float(best["d"]),
            "adf_at_d_star": float(best["adf"]),
            "corr_at_d_star": float(best["correlation"]),
            "rows_lost": int(len(series) - best["rows_kept"]),
        }

    return row


def stationarity_table(cleaned: pd.DataFrame, config: Config) -> pd.DataFrame:
    """The full stationarity table across the universe.

    Args:
        cleaned: Output of clean_panel.
        config: Loaded project configuration.

    Returns:
        One row per ticker, sorted by d_star, with memory_gained being how
        much more correlation with the price fracdiff keeps than plain returns.
    """
    prices = log_prices(cleaned, config)
    rows = {ticker: ticker_stationarity(prices[ticker], config) for ticker in prices.columns}

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "ticker"
    table["memory_gained"] = table["corr_at_d_star"] - table["corr_at_d_one"]

    for ticker in table.index[table["suspicious"]]:
        logger.warning("%s log price passes ADF undifferenced, treat with suspicion", ticker)

    logger.info(
        "Stationarity: %s of %s log prices stationary, %s of %s returns, median d* %.2f",
        int(table["price_stationary"].sum()),
        len(table),
        int(table["return_stationary"].sum()),
        len(table),
        table["d_star"].median(),
    )

    return table.sort_values("d_star")


def write_stationarity_table(table: pd.DataFrame, out_dir: Path | None = None) -> Path:
    """Save the table as CSV for the report and the dashboard.

    Args:
        table: Output of stationarity_table.
        out_dir: Directory to write to. Defaults to reports/results.

    Returns:
        The path written.
    """
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stationarity.csv"
    table.to_csv(path, float_format="%.6f")
    logger.info("Saved stationarity table to %s", path)

    return path


def main() -> None:
    """Rebuild the stationarity table from the cleaned panel."""
    from src.data.cleaning import load_clean_panel

    config = load_config()

    with log_step(logger, "Stationarity table"):
        table = stationarity_table(load_clean_panel(config), config)
        write_stationarity_table(table)

    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
