"""Experiment tracking with MLflow, which never takes the pipeline down with it.

Two lessons shape this module. MLflow 3 removed the plain file backend, so a
tracking URI of file:./mlruns fails; the default here is a local SQLite
database, sqlite:///mlflow.db at the project root (view it with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`). And a tracker that
crashes the research pipeline when the server is down is worse than no
tracker at all.

So every call checks is_available first, and only failures that mean
"tracking is unavailable" are absorbed: MLflow not installed, a connection
or file error, or an MlflowException. Those switch the tracker off with a
warning that keeps the reason. Anything else (a TypeError from bad
arguments, say) is a bug and is raised as usual.

The trial log, not MLflow, is the record that counts trials; MLflow is a
convenience for browsing runs.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, Config
from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
DEFAULT_EXPERIMENT = "stock-return-predictor"
DEFAULT_UI_URL = "http://127.0.0.1:5000"

# MLflow rejects parameter values longer than this.
MAX_PARAM_LENGTH = 6000

# Fail fast against an unreachable HTTP server instead of retrying for minutes.
HTTP_DEFAULTS = {"MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0", "MLFLOW_HTTP_REQUEST_TIMEOUT": "5"}


def _expected_errors() -> tuple[type[BaseException], ...]:
    """Exceptions that mean tracking is unavailable rather than that the code is wrong."""
    errors: list[type[BaseException]] = [ImportError, OSError]

    try:
        from mlflow.exceptions import MlflowException

        errors.append(MlflowException)
    except ImportError:
        pass

    try:
        from sqlalchemy.exc import SQLAlchemyError

        errors.append(SQLAlchemyError)
    except ImportError:
        pass

    return tuple(errors)


def flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested mappings as one level with dotted keys: {"a": {"b": 1}} -> {"a.b": 1}."""
    out: dict[str, Any] = {}

    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out |= flatten(value, f"{name}.")
        else:
            out[name] = value

    return out


class MLflowTracker:
    """A thin, fault-tolerant wrapper around MLflow tracking.

    Args:
        tracking_uri: Where runs are stored. Defaults to MLFLOW_TRACKING_URI,
            then to sqlite:///mlflow.db at the project root.
        experiment: Experiment name.
        ui_url: Address of an MLflow UI serving this store, for run_url().
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment: str = DEFAULT_EXPERIMENT,
        ui_url: str = DEFAULT_UI_URL,
    ) -> None:
        self.tracking_uri = (
            tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
        )
        self.experiment = experiment
        self.ui_url = ui_url.rstrip("/")
        self.is_available = False
        self.reason = ""
        self.experiment_id: str | None = None
        self._mlflow: Any = None

        if self.tracking_uri.startswith("file:"):
            self._disable("the file backend was removed in MLflow 3; use sqlite:///mlflow.db")
            return

        try:
            if self.tracking_uri.startswith(("http://", "https://")):
                for key, value in HTTP_DEFAULTS.items():
                    os.environ.setdefault(key, value)
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            self.experiment_id = mlflow.set_experiment(experiment).experiment_id
            self._mlflow = mlflow
            self.is_available = True
            logger.info("MLflow tracking to %s (experiment %s)", self.tracking_uri, experiment)
        except _expected_errors() as exc:
            self._disable(f"{type(exc).__name__}: {exc}")

    def _disable(self, reason: str) -> None:
        """Switch tracking off for the rest of the process, keeping the reason."""
        self.is_available = False
        self.reason = reason
        logger.warning("MLflow tracking unavailable, continuing without it: %s", reason)

    def _call(self, action: str, function, *args, **kwargs) -> Any:
        """Run an MLflow call; an expected failure disables tracking instead of raising."""
        if not self.is_available:
            return None

        try:
            return function(*args, **kwargs)
        except _expected_errors() as exc:
            self._disable(f"{action} failed: {type(exc).__name__}: {exc}")
            return None

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> Generator[str | None, None, None]:
        """A run as a context manager, yielding its id (None when tracking is off).

        The body always runs, tracked or not, so callers never branch on
        is_available themselves.
        """
        run = self._call(
            "start_run",
            lambda: self._mlflow.start_run(
                experiment_id=self.experiment_id, run_name=run_name, tags=dict(tags or {})
            ),
        )
        try:
            yield run.info.run_id if run is not None else None
        finally:
            if run is not None:
                self._call("end_run", self._mlflow.end_run)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log parameters, nested ones with dotted names, values as text."""
        if not isinstance(params, Mapping):
            raise TypeError(f"params must be a mapping, got {type(params).__name__}")

        flat = {k: str(v)[:MAX_PARAM_LENGTH] for k, v in flatten(params).items()}
        if flat:
            self._call("log_params", lambda: self._mlflow.log_params(flat))

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        """Log the numeric, finite metrics; anything else is skipped, not invented."""
        if not isinstance(metrics, Mapping):
            raise TypeError(f"metrics must be a mapping, got {type(metrics).__name__}")

        numeric = {
            k: float(v)
            for k, v in flatten(metrics).items()
            if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(float(v))
        }
        if numeric:
            self._call("log_metrics", lambda: self._mlflow.log_metrics(numeric, step=step))

    def log_artifact(self, path: str | Path) -> None:
        """Attach a file to the active run."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)

        self._call("log_artifact", lambda: self._mlflow.log_artifact(str(path)))

    def log_figure(self, figure: Any, name: str) -> None:
        """Attach a matplotlib figure to the active run as name.png."""
        self._call("log_figure", lambda: self._mlflow.log_figure(figure, f"{name}.png"))

    def log_backtest(self, result: Any, config: Config) -> str | None:
        """One backtest as one run: its settings, its statistics and its path table.

        Accepts an engine BacktestResult (statistics from result.summary()) or
        an evaluation StrategyEvaluation (result.summary, per-path table).

        Returns:
            The run id, or None when tracking is off.
        """
        name = getattr(result, "name", "backtest")
        summary = result.summary() if callable(getattr(result, "summary", None)) else result.summary
        parameters = getattr(result, "parameters", None) or getattr(result, "metadata", {})
        backtest = config.backtest

        with self.start_run(
            run_name=name, tags={"kind": getattr(result, "kind", "backtest")}
        ) as run_id:
            self.log_params(
                {
                    "strategy": name,
                    "parameters": dict(parameters),
                    "costs": {
                        "commission_bps": backtest.commission_bps,
                        "slippage_bps": backtest.slippage_bps,
                        "borrow_rate_annual": backtest.borrow_rate_annual,
                    },
                    "execution_lag": backtest.execution_lag,
                    "test_start": str(config.validation.test_start),
                }
            )
            self.log_metrics(summary)

            table = getattr(result, "path_table", None)
            if run_id is not None and isinstance(table, pd.DataFrame):
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / f"{name}_paths.csv"
                    table.to_csv(path)
                    self.log_artifact(path)

        return run_id

    def recent_runs(self, max_results: int = 20) -> pd.DataFrame:
        """The latest runs of the experiment, newest first; empty when tracking is off."""
        runs = self._call(
            "search_runs",
            lambda: self._mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                max_results=max_results,
                order_by=["attributes.start_time DESC"],
            ),
        )

        return runs if isinstance(runs, pd.DataFrame) else pd.DataFrame()

    def run_url(self, run_id: str) -> str | None:
        """Link to a run in the MLflow UI, or None when tracking is off."""
        if not self.is_available or not run_id:
            return None

        return f"{self.ui_url}/#/experiments/{self.experiment_id}/runs/{run_id}"


_TRACKER: MLflowTracker | None = None


def get_tracker() -> MLflowTracker:
    """The process-wide tracker, created on first use."""
    global _TRACKER

    if _TRACKER is None:
        _TRACKER = MLflowTracker()

    return _TRACKER
