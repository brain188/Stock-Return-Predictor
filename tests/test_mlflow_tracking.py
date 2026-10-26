"""Checks on MLflow tracking: it degrades instead of crashing, and round-trips when up."""

from __future__ import annotations

import builtins
import importlib.util

import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.backtest.engine import run_backtest
from src.tracking import mlflow_tracking
from src.tracking.mlflow_tracking import MLflowTracker, flatten, get_tracker

needs_mlflow = pytest.mark.skipif(
    importlib.util.find_spec("mlflow") is None, reason="mlflow is not installed"
)


@pytest.fixture
def no_mlflow(monkeypatch):
    """Make `import mlflow` fail, as if it were not installed."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("No module named 'mlflow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


@pytest.fixture
def local(tmp_path):
    """A tracker on a throwaway SQLite store."""
    return MLflowTracker(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}", "tests")


# --- Unavailable ----------------------------------------------------------------


def test_missing_mlflow_turns_tracking_off_without_raising(no_mlflow, tmp_path):
    tracker = MLflowTracker(f"sqlite:///{(tmp_path / 'x.db').as_posix()}")

    assert tracker.is_available is False
    assert "mlflow" in tracker.reason
    with tracker.start_run("nothing") as run_id:
        tracker.log_params({"a": 1})
        tracker.log_metrics({"sharpe": 0.5})
    assert run_id is None
    assert tracker.recent_runs().empty
    assert tracker.run_url("abc") is None


@needs_mlflow
def test_an_unreachable_server_turns_tracking_off(monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")

    tracker = MLflowTracker("http://127.0.0.1:9")  # nothing listens on port 9

    assert tracker.is_available is False
    with tracker.start_run("offline") as run_id:
        tracker.log_metrics({"sharpe": 1.0})
    assert run_id is None


def test_the_removed_file_backend_is_refused_not_crashed():
    tracker = MLflowTracker("file:./mlruns")

    assert tracker.is_available is False
    assert "sqlite" in tracker.reason


def test_the_default_backend_is_sqlite_at_the_project_root():
    assert mlflow_tracking.DEFAULT_TRACKING_URI.startswith("sqlite:///")
    assert mlflow_tracking.DEFAULT_TRACKING_URI.endswith("/mlflow.db")


def test_programming_errors_are_not_swallowed(no_mlflow):
    tracker = MLflowTracker("sqlite:///unused.db")

    with pytest.raises(TypeError):
        tracker.log_params(["not", "a", "mapping"])
    with pytest.raises(TypeError):
        tracker.log_metrics(0.5)


def test_flatten_uses_dotted_names():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_get_tracker_is_a_singleton(monkeypatch, no_mlflow):
    monkeypatch.setattr(mlflow_tracking, "_TRACKER", None)

    assert get_tracker() is get_tracker()


# --- Available --------------------------------------------------------------------


@needs_mlflow
def test_parameters_and_metrics_round_trip(local):
    with local.start_run("round trip") as run_id:
        local.log_params({"estimator": "random_forest", "model": {"depth": 6}})
        local.log_metrics({"sharpe": 0.75, "ic": 0.004, "skipped": float("nan"), "label": "x"})

    runs = local.recent_runs()
    row = runs.set_index("run_id").loc[run_id]

    assert local.is_available
    assert row["params.estimator"] == "random_forest"
    assert row["params.model.depth"] == "6"
    assert row["metrics.sharpe"] == pytest.approx(0.75)
    assert row["metrics.ic"] == pytest.approx(0.004)
    assert "metrics.skipped" not in runs.columns  # NaN is not logged, not invented
    assert local.run_url(run_id).endswith(f"/runs/{run_id}")


@needs_mlflow
def test_a_run_with_no_metrics_does_not_crash(local):
    with local.start_run("empty") as run_id:
        local.log_metrics({})

    assert run_id is not None
    assert len(local.recent_runs()) == 1


@needs_mlflow
def test_a_backtest_is_logged_as_one_run(local, config, tmp_path):
    dates = pd.bdate_range("2019-01-01", periods=30)
    returns = pd.DataFrame({"A": 0.001}, index=dates)
    result = run_backtest(pd.DataFrame({"A": 1.0}, index=dates), returns, config, name="hold")

    run_id = local.log_backtest(result, config)
    row = local.recent_runs().set_index("run_id").loc[run_id]

    assert row["tags.mlflow.runName"] == "hold"
    assert row["params.costs.commission_bps"] == str(config.backtest.commission_bps)
    assert "metrics.net_annual_return" in row.index


@needs_mlflow
def test_a_figure_and_an_artifact_are_attached(local, tmp_path):
    figure = Figure()
    figure.add_subplot().plot([1, 2, 3])
    note = tmp_path / "note.txt"
    note.write_text("hello", encoding="utf-8")

    with local.start_run("files"):
        local.log_figure(figure, "curve")
        local.log_artifact(note)

    assert local.is_available
