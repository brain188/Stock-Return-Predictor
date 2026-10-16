"""Checks on the universe definition.

A ticker that silently drops out of the data, or a short history accepted
without comment, changes what every backtest is measuring.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from src.data.universe import (
    UniverseError,
    eligible_from,
    resolve_universe,
    survivorship_note,
    validate_universe,
)
from tests.conftest import make_bars


@pytest.fixture
def small_config(config):
    """The project config cut down to three tickers."""
    universe = dataclasses.replace(config.universe, tickers=["SPY", "AAA", "BBB"], benchmark="SPY")

    return dataclasses.replace(config, universe=universe)


@pytest.fixture
def frames() -> dict[str, pd.DataFrame]:
    """Full-length bars for the three tickers in small_config."""
    return {t: make_bars(n=600, seed=i) for i, t in enumerate(["SPY", "AAA", "BBB"])}


def test_resolve_universe_matches_config(config):
    """The resolved list is the configured list, in order."""
    assert resolve_universe(config) == [t.upper() for t in config.universe.tickers]


def test_duplicate_ticker_raises(config):
    """A ticker listed twice would be double counted everywhere."""
    universe = dataclasses.replace(config.universe, tickers=["SPY", "AAPL", "aapl"])

    with pytest.raises(UniverseError, match="AAPL"):
        resolve_universe(dataclasses.replace(config, universe=universe))


def test_missing_ticker_raises(small_config, frames):
    """A configured ticker absent from the data is an error, not a smaller universe."""
    del frames["BBB"]

    with pytest.raises(UniverseError, match="BBB"):
        validate_universe(frames, small_config)


def test_full_history_passes(small_config, frames):
    """Every ticker with the benchmark's bar count passes."""
    report = validate_universe(frames, small_config)

    assert report.ok
    assert report.expected_bars == 600
    assert (report.table["coverage"] == 1.0).all()


def test_short_history_is_flagged_not_accepted(small_config, frames):
    """A ticker below the minimum history is flagged, and the report says why."""
    frames["AAA"] = frames["AAA"].iloc[-300:]

    report = validate_universe(frames, small_config)

    assert not report.ok
    assert report.flagged == ["AAA"]
    assert "fewer than" in report.table.loc["AAA", "reasons"]


def test_bars_short_of_benchmark_are_flagged(small_config, frames):
    """Above the minimum but short of the benchmark still gets flagged."""
    frames["BBB"] = frames["BBB"].iloc[10:]

    report = validate_universe(frames, small_config)

    assert report.flagged == ["BBB"]
    assert "10 bars short of SPY" in report.table.loc["BBB", "reasons"]


def test_wide_panel_is_accepted(small_config, frames):
    """A wide price frame, as to_panel returns, is validated the same way."""
    wide = pd.DataFrame({t: f["adj_close"] for t, f in frames.items()})
    wide.iloc[:200, 1] = float("nan")

    report = validate_universe(wide, small_config)

    assert report.flagged == ["AAA"]
    assert report.table.loc["AAA", "bars"] == 400


def test_survivorship_note_is_not_empty(config):
    """The note exists and states the bias range."""
    note = survivorship_note(config)

    assert note
    assert "1% to 4%" in note
    assert str(len(config.universe.tickers)) in note


def test_survivorship_note_requires_acknowledgement(config):
    """Results cannot be reported until the bias is acknowledged in config."""
    universe = dataclasses.replace(config.universe, survivorship_bias="ignored")

    with pytest.raises(UniverseError, match="acknowledged"):
        survivorship_note(dataclasses.replace(config, universe=universe))


def test_eligible_from_defaults_to_start_date(config):
    """With no data, every ticker is eligible from the universe start date."""
    assert eligible_from("AAPL", config) == pd.Timestamp(config.universe.start_date)


def test_eligible_from_respects_a_late_listing(small_config, frames):
    """A ticker whose data starts late is not eligible before its first bar."""
    frames["AAA"] = frames["AAA"].iloc[100:]

    assert eligible_from("AAA", small_config, frames) == frames["AAA"].index[0]


def test_eligible_from_unknown_ticker_raises(config):
    """Only universe members have an eligibility date."""
    with pytest.raises(UniverseError, match="TSLA"):
        eligible_from("TSLA", config)
