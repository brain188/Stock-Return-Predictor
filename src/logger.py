"""Logging setup for the pipeline.

Call setup_logging() once in an entry point, then get_logger() everywhere else.
Console output stays readable; the file keeps the detail.
"""

from __future__ import annotations

import logging
import logging.config
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.config import PROJECT_ROOT

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_LOGGING_PATH = PROJECT_ROOT / "config" / "logging.yaml"

_FALLBACK_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
_FALLBACK_DATEFMT = "%H:%M:%S"

_is_configured = False


def setup_logging(
    path: str | Path | None = None,
    *,
    level: str | None = None,
    force: bool = False,
) -> None:
    """Configure logging from config/logging.yaml.

    Args:
        path: Logging config file. Defaults to config/logging.yaml.
        level: Overrides the console level, for example "DEBUG".
        force: Reapply the configuration even if it was already set up.
    """
    global _is_configured

    if _is_configured and not force:
        return

    config_path = Path(path).resolve() if path else DEFAULT_LOGGING_PATH

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        _apply_fallback(level)
        logging.getLogger(__name__).warning(
            "Could not load %s (%s). Using basic console logging.", config_path, exc
        )
        _is_configured = True
        return

    # Handlers use relative paths, so anchor them before dictConfig opens them.
    for handler in config.get("handlers", {}).values():
        if filename := handler.get("filename"):
            resolved = (PROJECT_ROOT / filename).resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            handler["filename"] = str(resolved)

    if level:
        config["handlers"]["console"]["level"] = level.upper()

    logging.config.dictConfig(config)
    _is_configured = True


def _apply_fallback(level: str | None) -> None:
    """Set up plain console logging when the YAML config is unavailable."""
    logging.basicConfig(
        level=getattr(logging, (level or "INFO").upper(), logging.INFO),
        format=_FALLBACK_FORMAT,
        datefmt=_FALLBACK_DATEFMT,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring logging first if nobody has yet.

    Args:
        name: Logger name. Pass __name__ from the calling module.

    Returns:
        A logger ready to use.
    """
    if not _is_configured:
        setup_logging()

    return logging.getLogger(name)


@contextmanager
def log_step(logger: logging.Logger, description: str) -> Generator[None, None, None]:
    """Log the start and end of a step, with how long it took.

    Args:
        logger: Logger to write to.
        description: What the step does.

    Yields:
        None. Run the step inside the with block.
    """
    logger.info("START  %s", description)
    started = time.perf_counter()

    try:
        yield
    except Exception:
        logger.exception("FAILED %s after %.2fs", description, time.perf_counter() - started)
        raise
    else:
        logger.info("DONE   %s (%.2fs)", description, time.perf_counter() - started)


def log_frame(logger: logging.Logger, frame: pd.DataFrame, label: str) -> None:
    """Log the shape, date range, and missing values of a dataframe.

    Args:
        logger: Logger to write to.
        frame: Dataframe to describe.
        label: Name for the dataframe in the log line.
    """
    missing = int(frame.isna().sum().sum())

    if isinstance(frame.index, type(frame.index)) and len(frame):
        span = f"{frame.index[0]} to {frame.index[-1]}"
    else:
        span = "empty"

    logger.info(
        "%s: %s rows x %s columns, %s missing, %s",
        label,
        f"{len(frame):,}",
        frame.shape[1],
        f"{missing:,}",
        span,
    )


def log_config(logger: logging.Logger, config: Any) -> None:
    """Log the settings that decide how a run behaves.

    Args:
        logger: Logger to write to.
        config: A loaded Config object.
    """
    logger.info("Config       : %s", config.source_file)
    logger.info(
        "Universe     : %s tickers, %s to %s",
        len(config.universe.tickers),
        config.universe.start_date,
        config.universe.end_date,
    )
    logger.info(
        "Labelling    : %s (labels overlap: %s)",
        config.labelling.method,
        config.labelling.labels_overlap,
    )
    logger.info("Estimator    : %s", config.model.estimator)
    logger.info("Embargo      : %.1f%% of the sample", config.validation.embargo_pct * 100)
    logger.info("Round trip   : %.3f%% per trade", config.backtest.round_trip_cost * 100)
