"""Save and load a model with everything needed to reproduce its predictions.

Every prediction must be traceable to the run that produced it: which rows it
was trained on, which features in which order, which library versions, which
commit, which config. A bundle is a directory holding the pickled model and a
metadata.json a person can read without Python.

The feature list check on load is the guard against the quiet failure where a
feature is added to config and an old model happily predicts on the wrong
columns.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import PROJECT_ROOT, Config
from src.evaluation.trial_log import config_hash
from src.features.build import feature_names
from src.logger import get_logger
from src.models.baseline import Forecaster

logger = get_logger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"

LIBRARIES = ["numpy", "pandas", "scikit-learn", "lightgbm", "torch"]


class PersistenceError(Exception):
    """Raised when a bundle is missing, incomplete, or does not fit the config."""


@dataclass(frozen=True)
class ModelBundle:
    """A loaded model and the record of how it was made."""

    model: Forecaster
    metadata: dict[str, Any]

    @property
    def feature_names(self) -> list[str]:
        """Feature columns the model was trained on, in order."""
        return list(self.metadata["features"])


def _git_commit() -> dict[str, Any]:
    """The current commit and whether the working tree had uncommitted changes."""
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


def _library_versions() -> dict[str, str]:
    """Installed versions of the libraries a prediction depends on."""
    versions = {"python": platform.python_version()}

    for name in LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"

    return versions


def model_metadata(model: Forecaster, config: Config, data_window: pd.Index) -> dict[str, Any]:
    """Everything needed to trace a prediction back to its run.

    Args:
        model: A fitted model.
        config: The configuration it was trained under.
        data_window: Index of the training rows, (date, ticker) or dates.

    Returns:
        A JSON-serialisable dict.
    """
    dates = (
        data_window.get_level_values("date")
        if isinstance(data_window, pd.MultiIndex)
        else data_window
    )

    return {
        "name": model.name,
        "class": type(model).__name__,
        "params": model.params,
        "features": list(getattr(model, "feature_names_", [])),
        "training_window": {
            "start": str(dates.min().date()) if len(dates) else None,
            "end": str(dates.max().date()) if len(dates) else None,
            "rows": len(data_window),
        },
        "labelling": config.labelling.method,
        "config_hash": config_hash(config),
        "libraries": _library_versions(),
        "git": _git_commit(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def save_model(
    model: Forecaster,
    config: Config,
    metrics: dict[str, Any],
    path: str | Path,
    data_window: pd.Index,
) -> Path:
    """Write a model bundle: the pickled model and its metadata.

    Args:
        model: A fitted model.
        config: The configuration it was trained under.
        metrics: Scores to record alongside it.
        path: Bundle directory, created if needed.
        data_window: Index of the training rows.

    Returns:
        The bundle directory.

    Raises:
        PersistenceError: If the model's features differ from the config's,
            which would make the bundle unloadable later.
    """
    metadata = model_metadata(model, config, data_window)
    metadata["metrics"] = metrics

    expected = feature_names(config)
    if metadata["features"] != expected:
        raise PersistenceError(
            f"model features {metadata['features']} differ from config features {expected}"
        )

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / MODEL_FILE)
    (path / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    logger.info("Saved %s to %s", model.name, path)

    return path


def load_model(path: str | Path, config: Config) -> ModelBundle:
    """Load a bundle, refusing one trained on a different feature list.

    Args:
        path: Bundle directory.
        config: The current configuration.

    Returns:
        The model and its metadata.

    Raises:
        PersistenceError: If a file is missing or the feature list differs
            from feature_names(config).
    """
    path = Path(path)
    model_file, metadata_file = path / MODEL_FILE, path / METADATA_FILE

    if not model_file.is_file() or not metadata_file.is_file():
        raise PersistenceError(f"{path} is not a complete model bundle")

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    expected = feature_names(config)

    if metadata["features"] != expected:
        added = sorted(set(expected) - set(metadata["features"]))
        removed = sorted(set(metadata["features"]) - set(expected))
        raise PersistenceError(
            f"{path} was trained on different features (config adds {added}, removes "
            f"{removed}, or reorders them). Retrain rather than predict on the wrong columns."
        )

    model = joblib.load(model_file)
    logger.info("Loaded %s from %s (commit %s)", metadata["name"], path, metadata["git"]["commit"])

    return ModelBundle(model=model, metadata=metadata)
