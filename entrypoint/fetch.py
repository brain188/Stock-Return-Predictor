"""Pull and cache the price data.

Run once before anything else:

    python entrypoint/fetch.py
    python entrypoint/fetch.py --refresh --log-level DEBUG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ConfigError, load_config
from src.data.loaders import DataSourceError, fetch_universe, to_panel
from src.logger import get_logger, log_config, log_step, setup_logging

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
RULE = "-" * 70


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Pull and cache daily price data.")
    parser.add_argument("--config", default=None, help="config file (default: config/config.yaml)")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and pull again")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console detail (default: INFO)",
    )
    return parser.parse_args()


def main() -> int:
    """Fetch every ticker in the universe and report what landed."""
    args = parse_args()
    setup_logging(level=args.log_level)

    try:
        config = load_config(args.config)
        log_config(logger, config)

        with log_step(logger, "Fetching the universe"):
            frames, failed = fetch_universe(config, refresh=args.refresh)

        if not frames:
            logger.error("No tickers could be fetched")
            return EXIT_FAILED

        panel = to_panel(frames, config.data.price_column)

    except (ConfigError, DataSourceError) as exc:
        logger.error("%s", exc)
        return EXIT_FAILED
    except Exception:
        logger.exception("Fetch failed")
        return EXIT_FAILED

    print(f"\nCached\n{RULE}")
    print(f"tickers     {len(frames)} of {len(config.universe.tickers)}")
    print(f"panel       {panel.shape[0]:,} rows x {panel.shape[1]} columns")
    print(f"period      {panel.index[0].date()} to {panel.index[-1].date()}")
    print(f"missing     {int(panel.isna().sum().sum()):,} values")
    print(f"location    {config.data.cache_dir}")

    if failed:
        print(f"\nfailed      {', '.join(failed)}")

    print("\nNext: notebooks/01_data_quality.ipynb")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())