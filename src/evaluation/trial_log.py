"""A permanent record of every configuration tried, and an honest count of them.

The deflated Sharpe ratio asks how good the best of N trials would look by luck
alone, so it needs N, and it needs the spread of the Sharpe ratios across those
trials. Neither can be reconstructed after the fact. This log is append only:
a row is written for every attempt, including the failures and the repeats,
and nothing rewrites or deletes an earlier row. The one exception is migrate(),
an explicit, backed-up schema upgrade that adds columns and changes no value.

Three identities are kept apart, because mixing them miscounts N:

- Execution: trial_id, timestamp and code_version. One row per attempt. A
  rerun of the same thing is a new execution and gets a new row.
- Config file: config_hash, a hash of the YAML's text. Provenance only. It
  changes when a comment, a path or an unrelated section changes, so it must
  never be used to count trials.
- Trial: (kind, fingerprint). The fingerprint hashes the resolved settings
  that decide a result (the pipeline supplies them), so two executions with
  the same fingerprint are the same trial however their config files differ.

The deflated Sharpe uses unique_trial_count() and sharpe_variance(), which
count each (kind, fingerprint) once and leave out attempts that failed before
producing a result: a configuration whose result was never seen cannot have
been selected. trial_count() counts executions and is for auditing only.

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
import shutil
import subprocess
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Fixed on-disk layout. Changing it needs a migration, never a silent rewrite.
COLUMNS = [
    "trial_id",
    "timestamp",
    "kind",
    "name",
    "label",
    "status",
    "fingerprint",
    "config_hash",
    "code_version",
    "features",
    "model_params",
    "metrics",
    "notes",
]

# The layout before trial identity was separated from the config file.
LEGACY_COLUMNS = [
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

# What a trial is. Only research and backtest trials are candidates for
# selection. Benchmarks are reference points every strategy is compared with,
# and diagnostics check the method; both are recorded and counted apart.
KINDS = {"research", "backtest", "benchmark", "diagnostic", "other"}
SELECTION_KINDS = ("research", "backtest")
STATUSES = {"ok", "failed"}

# Columns stored as JSON text, decoded by trial_table().
JSON_COLUMNS = ["features", "model_params", "metrics"]

# Prefix given to metric columns when the metrics JSON is expanded.
METRIC_PREFIX = "metric_"

# Two executions of one trial agree when every shared metric is this close.
CONSISTENCY_TOLERANCE = 1e-9

FINGERPRINT_LENGTH = 16


class TrialLogError(Exception):
    """Raised when the log file cannot be trusted or a query cannot be answered."""


def _to_jsonable(value: Any) -> Any:
    """Convert numpy scalars, arrays, paths and dates into plain JSON types."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path | pd.Timestamp) or hasattr(value, "isoformat"):
        return str(value)

    raise TypeError(f"cannot store {type(value).__name__} in the trial log")


def _dumps(value: Any) -> str:
    """Serialise a value to JSON with a stable key order."""
    return json.dumps(value, sort_keys=True, default=_to_jsonable)


def trial_fingerprint(settings: Mapping[str, Any]) -> str:
    """Identity of a trial: a hash of the settings that decide its result.

    The settings are serialised as canonical JSON (sorted keys, dates as ISO
    text), so the same settings give the same fingerprint on any machine and
    from any config file that resolves to them.

    Args:
        settings: Plain data describing the trial, built by the pipeline.

    Returns:
        The first 16 hex characters of the SHA-256.
    """
    return hashlib.sha256(_dumps(dict(settings)).encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


def config_hash(config: Config | None) -> str:
    """Fingerprint the config file a trial ran under, for provenance only.

    Hashes the file's text rather than the loaded object, because the loaded
    object holds absolute paths that differ between machines. Line endings
    are normalised first: git can check the same file out with CRLF on one
    machine and LF on another, and that alone used to change the hash.
    Rows logged before this normalisation hashed the raw bytes.

    Args:
        config: Loaded project configuration, or None for an ad hoc trial.

    Returns:
        The first 12 hex characters of the SHA-256, or an empty string.
    """
    if config is None or not config.source_file.is_file():
        return ""

    text = config.source_file.read_bytes().replace(b"\r\n", b"\n")

    return hashlib.sha256(text).hexdigest()[:12]


def git_state() -> dict[str, Any]:
    """The current commit and whether tracked files had uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}

    return {"commit": commit, "dirty": bool(status.strip())}


def code_version() -> str:
    """Short commit, with -dirty when tracked files had uncommitted changes."""
    state = git_state()

    if state["commit"] == "unknown":
        return "unknown"

    return state["commit"][:7] + ("-dirty" if state["dirty"] else "")


def _legacy_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill the identity columns a legacy row never had, as far as it allows.

    kind comes from the name prefix and status from the notes, which is how
    the old pipeline recorded them. The fingerprint stays empty: it cannot be
    inferred from the row, and an empty fingerprint is counted as its own
    trial rather than merged with anything.
    """
    frame = frame.copy()
    frame["kind"] = [
        "research" if name.startswith("research_") else "other" for name in frame["name"]
    ]
    frame["status"] = ["failed" if note.startswith("failed") else "ok" for note in frame["notes"]]
    frame["fingerprint"] = ""
    frame["code_version"] = "unrecorded"

    return frame[COLUMNS]


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

    def _header(self) -> list[str] | None:
        """The file's header row, or None when there is no file yet."""
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return None

        with self.path.open("r", encoding="utf-8", newline="") as handle:
            return next(csv.reader(handle), [])

    def is_legacy(self) -> bool:
        """Whether the file still has the layout from before trial identity."""
        return self._header() == LEGACY_COLUMNS

    def _check_header(self) -> bool:
        """Confirm an existing file can be appended to.

        Returns:
            True if the file exists and already has a header row.

        Raises:
            TrialLogError: If the header is the legacy layout (migrate first)
                or anything else unexpected, since appending to a file with a
                different layout would corrupt it.
        """
        header = self._header()

        if header is None:
            return False

        if header == LEGACY_COLUMNS:
            raise TrialLogError(
                f"{self.path} uses the layout from before trial identity was recorded. "
                "Run TrialLog(path).migrate() once; it adds the new columns, changes no "
                "existing value and keeps a backup."
            )

        if header != COLUMNS:
            raise TrialLogError(
                f"{self.path} has columns {header}, expected {COLUMNS}. "
                "Refusing to append to a file with a different layout."
            )

        return True

    def _read_raw(self) -> pd.DataFrame:
        """Read the file as stored, with JSON columns still as text.

        A legacy file is readable, with identity columns filled by
        _legacy_defaults, so queries work before it is migrated.
        """
        header = self._header()

        if header is None:
            return pd.DataFrame(columns=COLUMNS)

        if header not in (COLUMNS, LEGACY_COLUMNS):
            self._check_header()

        # Text columns stay text: "" must not turn into NaN, nor "1" into 1.
        frame = pd.read_csv(self.path, dtype=str, keep_default_na=False, encoding="utf-8")

        if header == LEGACY_COLUMNS:
            frame = _legacy_defaults(frame)

        frame["trial_id"] = frame["trial_id"].astype(int)

        return frame

    def migrate(
        self,
        identities: Mapping[int, Mapping[str, str]] | None = None,
    ) -> Path:
        """Upgrade a legacy file to the current layout, keeping a backup.

        Every existing value is kept exactly; only the identity columns are
        added. They take their defaults from _legacy_defaults unless
        `identities` supplies them for a trial: {trial_id: {"fingerprint":
        ..., "kind": ..., "code_version": ...}}. Supply a fingerprint only when
        the settings behind that row are known for certain, for example from
        the committed config whose hash the row recorded.

        Args:
            identities: Known identity values per trial_id.

        Returns:
            The path of the backup copy of the legacy file.

        Raises:
            TrialLogError: If the file is not in the legacy layout, an
                identity names an unknown trial or column, or the rewritten
                file would not reproduce every original value.
        """
        if not self.is_legacy():
            raise TrialLogError(f"{self.path} is not in the legacy layout, nothing to migrate")

        original = pd.read_csv(self.path, dtype=str, keep_default_na=False, encoding="utf-8")
        upgraded = _legacy_defaults(original)
        allowed = {"fingerprint", "kind", "code_version"}

        for trial_id, values in (identities or {}).items():
            rows = upgraded["trial_id"] == str(trial_id)
            if not rows.any():
                raise TrialLogError(f"no trial {trial_id} in {self.path}")
            if unknown := set(values) - allowed:
                raise TrialLogError(f"cannot set {sorted(unknown)} during migration")
            for column, value in values.items():
                upgraded.loc[rows, column] = value

        if bad := set(upgraded["kind"]) - KINDS:
            raise TrialLogError(f"unknown kind {sorted(bad)}")

        backup = self.path.with_name(self.path.stem + ".legacy-backup" + self.path.suffix)
        shutil.copy2(self.path, backup)

        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(upgraded.to_dict("records"))

        reread = pd.read_csv(self.path, dtype=str, keep_default_na=False, encoding="utf-8")
        if not reread[LEGACY_COLUMNS].equals(original[LEGACY_COLUMNS]):
            shutil.copy2(backup, self.path)
            raise TrialLogError("migration changed an existing value; the original was restored")

        logger.info(
            "Migrated %s (%s trials) to the current layout; backup at %s",
            self.path,
            len(upgraded),
            backup,
        )

        return backup

    def log_trial(
        self,
        name: str,
        *,
        kind: str = "research",
        fingerprint: str | None = None,
        status: str | None = None,
        config: Config | None = None,
        features: Sequence[str] = (),
        model_params: Mapping[str, Any] | None = None,
        label: str = "",
        metrics: Mapping[str, Any] | None = None,
        notes: str = "",
    ) -> int:
        """Append one execution. Called for every attempt, whatever its outcome.

        The same configuration logged twice is recorded twice, on purpose: it
        was executed twice. Whether it is one trial or two is decided by the
        fingerprint, not by the number of rows.

        Args:
            name: What was tried, for example "rf_fixed_horizon" or "pair_XOM_JPM".
            kind: One of KINDS.
            fingerprint: Identity of the trial, from trial_fingerprint(). When
                omitted, it is derived from the name, label, features and
                model parameters, which never depends on the config file.
            status: "ok" or "failed". Defaults to "ok" when metrics were
                given and "failed" otherwise.
            config: The configuration the trial ran under, hashed for provenance.
            features: Feature columns used, in order.
            model_params: Estimator or strategy parameters.
            label: Labelling method, for example "fixed_horizon".
            metrics: Scores the trial produced. Keep units consistent across
                trials (daily or annualised Sharpe, not a mix), since
                sharpe_variance() compares them directly.
            notes: Free text, for example why a run failed.

        Returns:
            The trial_id given to the new row.

        Raises:
            TrialLogError: For an unknown kind or status, or a file that
                cannot be appended to.
        """
        status = status or ("ok" if metrics else "failed")

        if kind not in KINDS:
            raise TrialLogError(f"kind must be one of {sorted(KINDS)}, got '{kind}'")
        if status not in STATUSES:
            raise TrialLogError(f"status must be one of {sorted(STATUSES)}, got '{status}'")

        if fingerprint is None:
            fingerprint = trial_fingerprint(
                {
                    "name": name,
                    "label": label,
                    "features": list(features),
                    "model_params": dict(model_params or {}),
                }
            )

        has_header = self._check_header()
        trial_id = self.trial_count() + 1

        row = {
            "trial_id": trial_id,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "kind": kind,
            "name": name,
            "label": label,
            "status": status,
            "fingerprint": fingerprint,
            "config_hash": config_hash(config),
            "code_version": code_version(),
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

        logger.info("Trial %s logged: %s (%s, %s)", trial_id, name, kind, fingerprint)

        return trial_id

    def trial_table(self) -> pd.DataFrame:
        """Every execution, with metrics expanded into metric_<name> columns.

        Returns:
            One row per execution. features and model_params are decoded to
            Python objects; each metric becomes its own float column.
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
        """How many executions were logged, for auditing.

        This is not the N in the deflated Sharpe: a rerun of the same
        settings is a second execution of one trial. Use unique_trial_count.

        Args:
            filters: Column to value pairs every counted row must match, for
                example {"label": "fixed_horizon"}. None counts everything.

        Returns:
            The number of matching rows.
        """
        if not filters:
            return len(self._read_raw())

        return len(self._filtered(filters))

    def unique_trials(
        self,
        filters: Mapping[str, Any] | None = None,
        *,
        include_failed: bool = False,
    ) -> pd.DataFrame:
        """One row per distinct trial, (kind, fingerprint), with its executions.

        Rows with no fingerprint (legacy rows nobody could identify) are kept
        as separate trials, so an unknown identity can only raise the count.

        Args:
            filters: Restriction, as in trial_count().
            include_failed: Also count trials whose every execution failed.

        Returns:
            Indexed by (kind, fingerprint), with name, label, executions,
            first_trial_id, last_trial_id, consistent (whether every metric
            shared by its executions agrees) and the mean of each metric.
        """
        table = self._filtered(filters)

        if not include_failed:
            table = table[table["status"] == "ok"]

        if table.empty:
            return pd.DataFrame(
                columns=["name", "label", "executions", "first_trial_id", "last_trial_id"]
            )

        table = table.copy()
        table["fingerprint"] = [
            f or f"unidentified-{trial_id}"
            for f, trial_id in zip(table["fingerprint"], table["trial_id"], strict=True)
        ]
        metric_columns = [c for c in table.columns if c.startswith(METRIC_PREFIX)]
        groups = table.groupby(["kind", "fingerprint"], sort=False)

        summary = groups.agg(
            name=("name", "first"),
            label=("label", "first"),
            executions=("trial_id", "size"),
            first_trial_id=("trial_id", "min"),
            last_trial_id=("trial_id", "max"),
        )
        summary["consistent"] = groups[metric_columns].agg(_consistent).all(axis=1)

        return pd.concat([summary, groups[metric_columns].mean()], axis=1)

    def unique_trial_count(
        self,
        filters: Mapping[str, Any] | None = None,
        *,
        include_failed: bool = False,
    ) -> int:
        """The N in the deflated Sharpe: distinct trials that produced a result.

        Args:
            filters: Restriction, as in trial_count(), for example
                {"kind": "backtest"}.
            include_failed: Also count trials whose every execution failed.

        Returns:
            The number of distinct (kind, fingerprint) pairs.
        """
        return len(self.unique_trials(filters, include_failed=include_failed))

    def best_trial(self, metric: str, *, maximise: bool = True) -> pd.Series:
        """The execution with the best value of a metric.

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
        *,
        unique: bool = True,
    ) -> float:
        """Variance of the Sharpe ratio across trials, the V[SR] in the DSR.

        Uses the sample variance (ddof=1), because the trials are a sample of
        what could have been tried rather than the whole population. Each
        distinct trial contributes once, at the mean over its executions, so a
        rerun neither adds a point nor shrinks the variance.

        Args:
            metric: Metric holding the Sharpe ratio, with or without the prefix.
            filters: Optional restriction, as in trial_count().
            unique: One value per distinct trial. False uses every execution,
                for auditing only.

        Returns:
            The variance, or NaN when fewer than two trials recorded the metric.
        """
        column = metric if metric.startswith(METRIC_PREFIX) else METRIC_PREFIX + metric
        table = self.unique_trials(filters) if unique else self._filtered(filters)

        if column not in table.columns:
            return math.nan

        values = table[column].dropna()

        if len(values) < 2:
            return math.nan

        return float(values.var(ddof=1))


def _consistent(values: pd.Series) -> bool:
    """Whether every non-missing value agrees, within CONSISTENCY_TOLERANCE."""
    present = values.dropna()

    if len(present) < 2:
        return True

    scale = max(1.0, float(present.abs().max()))

    return bool(present.max() - present.min() <= CONSISTENCY_TOLERANCE * scale)
