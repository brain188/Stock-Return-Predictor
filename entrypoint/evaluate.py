"""Backtest every strategy through CPCV on the development period.

    python entrypoint/evaluate.py
    python entrypoint/evaluate.py --strategies ml_long_short,momentum,buy_and_hold
    python entrypoint/evaluate.py --n-paths 2 --output reports/results/evaluation

Every strategy evaluated is appended to the trial log. The holdout period is
not touched.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ConfigError, load_config
from src.logger import get_logger, log_config, setup_logging
from src.tracking.mlflow_tracking import get_tracker

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
RULE = "-" * 70

SHOWN = [
    "kind",
    "sharpe_gross",
    "sharpe_net",
    "sharpe_net_std",
    "sharpe_lost_to_costs",
    "paths_beating_buy_and_hold",
    "paths_beating_equal_weight_hold",
    "exposure_net_mean",
    "annual_return_net",
    "max_drawdown_net",
    "turnover_annual",
    "breakeven_return",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(
        description="Backtest the strategies along every CPCV path, gross and net of costs."
    )
    parser.add_argument("--config", default=None, help="config file (default: config/config.yaml)")
    parser.add_argument(
        "--strategies",
        default=None,
        help="comma separated names (default: every strategy and both benchmarks)",
    )
    parser.add_argument(
        "--n-paths",
        type=int,
        default=None,
        help="evaluate only the first n CPCV paths (default: all)",
    )
    parser.add_argument(
        "--monte-carlo-runs",
        type=int,
        default=None,
        help="block-bootstrap runs per strategy (default: evaluation.monte_carlo_runs; 0 skips)",
    )
    parser.add_argument(
        "--from-saved",
        action="store_true",
        help="only recompute the Monte Carlo table from saved path returns: no refit, no trial",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="do not log runs to MLflow (sqlite:///mlflow.db by default)",
    )
    parser.add_argument(
        "--no-same-close",
        action="store_true",
        help="skip the diagnostic run of ML long/short at the signal's own close",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="folder for the result files (default: reports/results/evaluation)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console detail (default: INFO)",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation and print the comparison."""
    args = parse_args(argv)
    setup_logging(level=args.log_level)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        return EXIT_FAILED

    # Imported after the config loads, so a bad path fails fast and cleanly.
    from src.pipelines.evaluate import (
        DEFAULT_OUTPUT,
        DEFAULT_STRATEGIES,
        load_path_returns,
        monte_carlo_table,
        run_evaluation,
    )

    output = Path(args.output) if args.output else DEFAULT_OUTPUT

    if args.from_saved:
        try:
            runs = args.monte_carlo_runs or config.evaluation.monte_carlo_runs
            table = monte_carlo_table(load_path_returns(output), config, runs)
        except (OSError, ValueError) as exc:
            logger.error("%s", exc)
            return EXIT_FAILED
        table.to_csv(output / "monte_carlo.csv")
        print(f"\nMonte Carlo from saved paths, no refit, no trial\n{RULE}")
        print(table.round(3).to_string())
        return EXIT_OK

    strategies = (
        [name.strip() for name in args.strategies.split(",") if name.strip()]
        if args.strategies
        else list(DEFAULT_STRATEGIES)
    )

    try:
        log_config(logger, config)
        result = run_evaluation(
            config,
            strategies=strategies,
            n_paths=args.n_paths,
            same_close_check=not args.no_same_close,
            monte_carlo_runs=args.monte_carlo_runs,
            tracker=None if args.no_tracking else get_tracker(),
        )
        files = result.save(output)
    except (ConfigError, ValueError) as exc:
        logger.error("%s", exc)
        return EXIT_FAILED
    except Exception:
        logger.exception("Evaluation failed")
        return EXIT_FAILED

    shown = [c for c in SHOWN if c in result.comparison.columns]
    print(f"\nStrategies over the CPCV paths, development period only\n{RULE}")
    print(result.comparison[shown].round(3).to_string())
    if result.monte_carlo is not None:
        print(f"\nMonte Carlo, block bootstrap\n{RULE}")
        print(result.monte_carlo.round(3).to_string())
    print(f"\nFiles in {files[0].parent}")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
