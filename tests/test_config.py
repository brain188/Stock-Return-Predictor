"""Checks on the configuration loader.

Config errors should surface at load time with a message that says what is
wrong, not three stages later as a confusing failure.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest
import yaml

from src.config import DEFAULT_CONFIG_PATH, ConfigError, load_config


@pytest.fixture
def raw() -> dict:
    """The config file as a plain dictionary."""
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def write(raw: dict, tmp_path) -> str:
    """Write a modified config to a temporary file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_default_config_loads(config):
    """The committed config is valid."""
    assert config.universe.tickers
    assert config.model.estimator


def test_derived_values_are_computed(config):
    """Properties derived from the config match what it declares."""
    assert config.features.max_lookback == max(
        [
            *config.features.lagged_returns,
            *config.features.moving_averages,
            *config.features.volatility_windows,
            config.features.rsi_window,
            *config.features.momentum_windows,
            config.features.high_window,
            config.features.volume_window,
            *config.features.vol_ratio_windows,
        ]
    )
    assert config.backtest.round_trip_cost > 0
    assert config.model.params == getattr(config.model, config.model.estimator)


def test_fixed_horizon_of_one_day_does_not_overlap(config):
    """Whether labels overlap decides if purging is mandatory."""
    assert config.labelling.labels_overlap is (
        config.labelling.method == "triple_barrier"
        or config.labelling.fixed_horizon.horizon_days > 1
    )


def test_config_is_immutable(config):
    """A typo assigning to config should raise, not silently succeed."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.project.name = "changed"


def test_missing_file_is_reported(tmp_path):
    """A missing config file fails with a clear message."""
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


@pytest.mark.parametrize(
    "section,key,value,message",
    [
        ("universe", "benchmark", "TSLA", "benchmark"),
        ("universe", "tickers", [], "empty"),
        ("universe", "start_date", "2030-01-01", "before"),
        ("validation", "embargo_days", -1, "embargo_days"),
        ("validation", "embargo_days", 1.5, "embargo_days"),
        ("validation", "embargo_days", True, "embargo_days"),
        ("validation", "test_start", "2001-01-01", "inside"),
        ("labelling", "method", "magic", "labelling.method"),
        ("model", "estimator", "svm", "model.estimator"),
        ("backtest", "commission_bps", -1, "negative"),
        ("cleaning", "max_abs_daily_return", 5.0, "max_abs_daily_return"),
    ],
)
def test_invalid_values_are_rejected(raw, tmp_path, section, key, value, message):
    """Each guard catches its own class of mistake."""
    broken = copy.deepcopy(raw)
    broken[section][key] = value

    with pytest.raises(ConfigError, match=message):
        load_config(write(broken, tmp_path), create_dirs=False)


def test_missing_section_is_rejected(raw, tmp_path):
    """An incomplete config fails rather than defaulting silently."""
    broken = copy.deepcopy(raw)
    del broken["backtest"]

    with pytest.raises(ConfigError, match="missing sections"):
        load_config(write(broken, tmp_path), create_dirs=False)


def test_cpcv_groups_must_produce_paths(raw, tmp_path):
    """A test set as large as the whole sample yields no backtest paths."""
    broken = copy.deepcopy(raw)
    broken["validation"]["cpcv"]["n_test_groups"] = broken["validation"]["cpcv"]["n_groups"]

    with pytest.raises(ConfigError, match="n_test_groups"):
        load_config(write(broken, tmp_path), create_dirs=False)
