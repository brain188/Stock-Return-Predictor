"""The one-time 2020-2024 holdout: declare, check, prepare, run.

    python entrypoint/holdout.py check                     # preflight only, reads no price
    python entrypoint/holdout.py declare --approved-by NAME --approved-on YYYY-MM-DD
    python entrypoint/holdout.py declare --replace REASON --approved-by NAME --approved-on ...
    python entrypoint/holdout.py prepare --authorised-by NAME   # fits the final model
    python entrypoint/holdout.py run                       # opens the holdout, once
    python entrypoint/holdout.py run --new-attempt REASON  # after a failed attempt

`check` runs every preflight check and stops: it never calls the price loader.
`prepare` and `run` are separate, authorised steps (holdout_protocol.md). None
of these is part of `make all` or CI.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ConfigError, load_config
from src.logger import get_logger, setup_logging

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
RULE = "-" * 70


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="config file (default: config/config.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="preflight only; reads no holdout data")
    check.add_argument("--new-attempt", default=None, help="check as for a further attempt")

    declare = commands.add_parser("declare", help="write the declaration, once")
    declare.add_argument("--approved-by", required=True, help="who approved the protocol")
    declare.add_argument("--approved-on", required=True, help="approval date, YYYY-MM-DD")
    declare.add_argument("--replace", default=None, help="reason for a replacement")

    prepare = commands.add_parser("prepare", help="fit the final model and select pairs")
    prepare.add_argument("--authorised-by", required=True, help="who authorised this step")

    run = commands.add_parser("run", help="open the holdout: one attempt")
    run.add_argument("--new-attempt", default=None, help="reason for an attempt after another")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the command; every refusal is an exit code of 1 with the reason."""
    args = parse_args(argv)
    setup_logging()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        return EXIT_FAILED

    # Imported after the config loads, so a bad path fails fast and cleanly.
    from src.pipelines.holdout import (
        HoldoutError,
        PreflightError,
        preflight,
        prepare_artifacts,
        run_holdout,
        write_declaration,
    )

    try:
        if args.command == "check":
            result = preflight(config, stage="run", new_attempt=args.new_attempt)
            print(f"\nPreflight passed, no holdout data read\n{RULE}")
            for name, detail in result.record["checks"].items():
                print(f"{name}: {detail}")
        elif args.command == "declare":
            path = write_declaration(
                config,
                approved_on=args.approved_on,
                approved_by=args.approved_by,
                replace=args.replace,
            )
            print(f"Declaration written: {path}")
        elif args.command == "prepare":
            print(
                f"Artifacts prepared: {prepare_artifacts(config, authorised_by=args.authorised_by)}"
            )
        else:
            print(f"Holdout attempt written: {run_holdout(config, new_attempt=args.new_attempt)}")
    except PreflightError as exc:
        print(f"\nPreflight failed, nothing opened\n{RULE}")
        for failure in exc.failures:
            print(f"- {failure}")
        return EXIT_FAILED
    except (HoldoutError, FileExistsError) as exc:
        logger.error("%s", exc)
        return EXIT_FAILED

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
