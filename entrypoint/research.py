"""Run feature research: CPCV scores, importance and IC, no backtest.

    python entrypoint/research.py
    python entrypoint/research.py --model lightgbm --label-method triple_barrier
    python entrypoint/research.py --refresh --log-level DEBUG --output reports/results/research

Every run is appended to the trial log, so each invocation counts as a trial.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ESTIMATORS, LABELLING_METHODS, ConfigError, load_config
from src.logger import get_logger, log_config, setup_logging

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
RULE = "-" * 70


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(
        description="Score a model configuration with CPCV on the development period."
    )
    parser.add_argument("--config", default=None, help="config file (default: config/config.yaml)")
    parser.add_argument(
        "--refresh", action="store_true", help="rebuild the cleaned panel from the raw cache"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console detail (default: INFO)",
    )
    parser.add_argument(
        "--label-method",
        choices=sorted(LABELLING_METHODS),
        default=None,
        help="override labelling.method",
    )
    parser.add_argument(
        "--model", choices=sorted(ESTIMATORS), default=None, help="override model.estimator"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="folder for the result tables (default: reports/results/research)",
    )
    parser.add_argument(
        "--no-permutation",
        action="store_true",
        help="skip permutation importance, the slowest step",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one research configuration and print its headline numbers."""
    args = parse_args(argv)
    setup_logging(level=args.log_level)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        return EXIT_FAILED

    # Imported after the config loads, so a bad path fails fast and cleanly.
    from src.pipelines.research import DEFAULT_OUTPUT, run_research, with_overrides

    try:
        config = with_overrides(config, label_method=args.label_method, estimator=args.model)
        log_config(logger, config)
        result = run_research(config, refresh=args.refresh, permutation=not args.no_permutation)
        files = result.save(args.output or DEFAULT_OUTPUT)
    except ConfigError as exc:
        logger.error("%s", exc)
        return EXIT_FAILED
    except Exception:
        logger.exception("Research failed")
        return EXIT_FAILED

    m = result.metrics
    print(f"\nResearch: {result.name} (trial {result.trial_id})\n{RULE}")
    print(f"rows         {m['n_rows']:,.0f} development rows, {m['n_paths']:.0f} CPCV paths")
    print(f"IC           {m['ic_mean']:+.4f} +/- {m['ic_std']:.4f} (min {m['ic_min']:+.4f})")
    print(f"IC > 0       {m['positive_ic_paths']:.0f} of {m['n_paths']:.0f} paths")
    print(f"hit rate     {m['hit_rate_mean']:.4f}")
    print(f"vs AlwaysUp  {m['edge_over_always_up_mean']:+.4f}")
    print(f"balanced     {m['directional_accuracy_mean']:.4f}")
    print(f"\n{result.comparison.round(4).to_string()}")
    print(f"\nTables in    {files[0].parent}")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
