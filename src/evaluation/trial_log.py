"""A permanent record of every configuration tried.

The deflated Sharpe ratio asks how good the best of N trials would look by luck
alone, so it needs N, and it needs the spread of the Sharpe ratios across those
trials. Neither can be reconstructed after the fact. This log is append only:
a row is written for every attempt, including the failures and the repeats,
and nothing ever rewrites or deletes an earlier row.

Storage is a plain CSV so the log can be read without this module, diffed in
git, and opened in a spreadsheet. Nested values (feature lists, model
parameters, metrics) are stored as JSON strings so the column layout never
changes when a new metric is added.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.logger import get_logger

if TYPE_CHECKING:
    from src.config import Config

logger = get_logger(__name__)

# Fixed on-disk layout. Changing it would orphan every log already written.
COLUMNS = [
    "trial_id",
    "timestamp",
    "name",
    "label",
    "config_hash",
    "features",
    "model_params",
    "metrics",
    "notes",
]

# Columns stored as JSON text, decoded by trial_table().
JSON_COLUMNS = ["features", "model_params", "metrics"]

# Prefix given to metric columns when the metrics JSON is expanded.
METRIC_PREFIX = "metric_"


class TrialLogError(Exception):
    """Raised when the log file cannot be trusted or a query cannot be answered."""


def _to_jsonable(value: Any) -> Any:
    """Convert numpy scalars, arrays and paths into plain JSON types."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path | pd.Timestamp):
        return str(value)

    raise TypeError(f"cannot store {type(value).__name__} in the trial log")


def _dumps(value: Any) -> str:
    """Serialise a value to JSON with a stable key order."""
    return json.dumps(value, sort_keys=True, default=_to_jsonable)


def config_hash(config: Config | None) -> str:
    """Fingerprint the config file a trial ran under.

    Hashes the file's bytes rather than the loaded object, because the loaded
    object holds absolute paths that differ between machines.

    Args:
        config: Loaded project configuration, or None for an ad hoc trial.

    Returns:
        The first 12 hex characters of the SHA-256, or an empty string.
    """
    if config is None or not config.source_file.is_file():
        return ""

    return hashlib.sha256(config.source_file.read_bytes()).hexdigest()[:12]


class TrialLog:
    """Append-only log of experiments, backed by a CSV file.

    Every query reads the file afresh, so two TrialLog objects on the same path,
    or a new process after a restart, always see the same trials.

    Args:
        path: CSV file to append to. Created, with its parent directory, on the
            first write.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def from_config(cls, config: Config) -> TrialLog:
        """Open the log at the path set in config.evaluation.trial_log."""
        return cls(config.evaluation.trial_log)

    def _check_header(self) -> bool:
        """Confirm an existing file has the expected layout.

        Returns:
            True if the file exists and already has a header row.

        Raises:
            TrialLogError: If the header does not match COLUMNS, since appending
                to a file with a different layout would corrupt it.
        """
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return False

        with self.path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])

        if header != COLUMNS:
            raise TrialLogError(
                f"{self.path} has columns {header}, expected {COLUMNS}. "
                "Refusing to append to a file with a different layout."
            )

        return True

    def _read_raw(self) -> pd.DataFrame:
        """Read the file as stored, with JSON columns still as text."""
        if not self._check_header():
            return pd.DataFrame(columns=COLUMNS)

        # Text columns stay text: "" must not turn into NaN, nor "1" into 1.
        frame = pd.read_csv(
            self.path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )
        frame["trial_id"] = frame["trial_id"].astype(int)

        return frame

    def log_trial(
        self,
        name: str,
        *,
        config: Config | None = None,
        features: Sequence[str] = (),
        model_params: Mapping[str, Any] | None = None,
        label: str = "",
        metrics: Mapping[str, Any] | None = None,
        notes: str = "",
    ) -> int:
        """Append one trial. Called for every attempt, whatever its outcome.

        The same configuration logged twice is recorded twice, on purpose: it
        was tried twice, and both attempts count towards the trial total.

        Args:
            name: What was tried, for example "rf_fixed_horizon" or "pair_XOM_JPM".
            config: The configuration the trial ran under, fingerprinted by hash.
            features: Feature columns used, in order.
            model_params: Estimator or strategy parameters.
            label: Labelling method, for example "fixed_horizon".
            metrics: Scores the trial produced. Keep units consistent across
                trials (daily or annualised Sharpe, not a mix), since
                sharpe_variance() compares them directly.
            notes: Free text, for example why a run failed.

        Returns:
            The trial_id given to the new row.
        """
        has_header = self._check_header()
        trial_id = self.trial_count() + 1

        row = {
            "trial_id": trial_id,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "name": name,
            "label": label,
            "config_hash": config_hash(config),
            "features": _dumps(list(features)),
            "model_params": _dumps(dict(model_params or {})),
            "metrics": _dumps(dict(metrics or {})),
            "notes": notes,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Mode "a" only ever appends, so earlier rows cannot be lost.
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            if not has_header:
                writer.writeheader()
            writer.writerow(row)

        logger.info("Trial %s logged: %s", trial_id, name)

        return trial_id

    def trial_table(self) -> pd.DataFrame:
        """Every trial, with metrics expanded into metric_<name> columns.

        Returns:
            One row per trial. features and model_params are decoded to Python
            objects; each metric becomes its own float column.
        """
        frame = self._read_raw()

        for column in JSON_COLUMNS:
            frame[column] = [json.loads(text) if text else None for text in frame[column]]

        if frame.empty:
            return frame

        metrics = pd.json_normalize([m or {} for m in frame["metrics"]]).add_prefix(METRIC_PREFIX)
        metrics.index = frame.index

        return pd.concat([frame, metrics.apply(pd.to_numeric, errors="coerce")], axis=1)

    def _filtered(self, filters: Mapping[str, Any] | None) -> pd.DataFrame:
        """The trial table restricted to rows matching every filter."""
        table = self.trial_table()

        for column, value in (filters or {}).items():
            if column not in table.columns:
                raise TrialLogError(f"cannot filter on '{column}', not a trial log column")
            table = table[table[column] == value]

        return table

    def trial_count(self, filters: Mapping[str, Any] | None = None) -> int:
        """How many trials have been run, the N in the deflated Sharpe.

        Args:
            filters: Column to value pairs every counted row must match, for
                example {"label": "fixed_horizon"}. None counts everything.

        Returns:
            The number of matching rows.
        """
        if not filters:
            return len(self._read_raw())

        return len(self._filtered(filters))

    def best_trial(self, metric: str, *, maximise: bool = True) -> pd.Series:
        """The trial with the best value of a metric.

        Args:
            metric: Metric name, with or without the metric_ prefix.
            maximise: Whether higher is better.

        Returns:
            The winning row.

        Raises:
            TrialLogError: If no trial recorded the metric.
        """
        column = metric if metric.startswith(METRIC_PREFIX) else METRIC_PREFIX + metric
        table = self.trial_table()

        if column not in table.columns or table[column].isna().all():
            raise TrialLogError(f"no trial has recorded '{metric}'")

        values = table[column]
        position = values.idxmax() if maximise else values.idxmin()

        return table.loc[position]

    def sharpe_variance(
        self,
        metric: str = "sharpe",
        filters: Mapping[str, Any] | None = None,
    ) -> float:
        """Variance of the Sharpe ratio across trials, the V[SR] in the DSR.

        Uses the sample variance (ddof=1), because the trials are a sample of
        what could have been tried rather than the whole population.

        Args:
            metric: Metric holding the Sharpe ratio, with or without the prefix.
            filters: Optional restriction, as in trial_count().

        Returns:
            The variance, or NaN when fewer than two trials recorded the metric.
        """
        column = metric if metric.startswith(METRIC_PREFIX) else METRIC_PREFIX + metric
        table = self._filtered(filters)

        if column not in table.columns:
            return math.nan

        values = table[column].dropna()

        if len(values) < 2:
            return math.nan

        return float(values.var(ddof=1))
