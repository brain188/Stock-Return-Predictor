"""Checks on feature matrix assembly.

Notebook 3 caught leaks with three checks. The truncation check is the one
that catches a feature using later data, and it runs inside the build; these
tests also prove it fires when a leak is planted.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.features import build
from src.features.build import (
    LookaheadError,
    assert_no_lookahead,
    build_features,
    drop_warmup,
    feature_names,
    feature_windows,
    warmup_rows,
)
from src.features.fracdiff import fracdiff_weights
from src.labelling.fixed_horizon import label_fixed_horizon
from tests.conftest import make_bars

TICKERS = ["SPY", "AAA", "BBB"]
D = 0.5


def make_panel(n: int = 600, offset: int = 0) -> pd.DataFrame:
    """A cleaned-panel-shaped frame for three synthetic tickers.

    Args:
        n: Bars per ticker.
        offset: Business days to move every date forward, for the shift test.
    """
    frames = {}
    for seed, ticker in enumerate(TICKERS):
        bars = make_bars(n=n, seed=seed)
        bars.index = bars.index + pd.offsets.BDay(offset)
        frames[ticker] = bars

    return pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()


@pytest.fixture
def panel() -> pd.DataFrame:
    """Three synthetic tickers on one calendar."""
    return make_panel()


@pytest.fixture
def small_config(config):
    """Config for the synthetic panel, reading adjusted prices."""
    return dataclasses.replace(
        config,
        universe=dataclasses.replace(config.universe, tickers=TICKERS, benchmark="SPY"),
        data=dataclasses.replace(config.data, price_column="adj_close"),
    )


def test_columns_follow_feature_names(panel, small_config):
    """The matrix columns are feature_names, in that order, every time."""
    first = build_features(panel, small_config, d=D)
    second = build_features(panel, small_config, d=D)

    assert list(first.columns) == feature_names(small_config)
    assert list(first.columns) == list(second.columns)


def test_feature_names_cover_config(small_config):
    """One column per configured window, plus RSI, ratios and fracdiff."""
    names = feature_names(small_config)
    f = small_config.features

    assert len(names) == len(set(names))
    assert "fracdiff" in names
    assert len([n for n in names if n.startswith("ret_")]) == len(f.lagged_returns)


def test_index_is_date_ticker(panel, small_config):
    """Indexed by (date, ticker), sorted, with every ticker present."""
    features = build_features(panel, small_config, d=D)

    assert features.index.names == ["date", "ticker"]
    assert features.index.is_monotonic_increasing
    assert sorted(features.index.unique("ticker")) == sorted(TICKERS)


def test_warmup_dropped_per_ticker_not_globally(small_config):
    """A ticker starting late loses its own first rows, not the others' rows."""
    panel = make_panel()
    late = panel.xs("BBB", level="ticker", drop_level=False).index[100:]
    panel = panel[(panel.index.get_level_values("ticker") != "BBB") | panel.index.isin(late)]

    features = build_features(panel, small_config, d=D)
    rows = warmup_rows(small_config, D)
    counts = features.groupby(level="ticker").size()

    assert counts["SPY"] == 600 - rows
    assert counts["BBB"] == 500 - rows


def test_no_nan_after_warmup(panel, small_config):
    """On clean data every feature has a value once the warm-up is gone."""
    features = build_features(panel, small_config, d=D)

    assert not features.isna().any().any()


def test_warmup_covers_fracdiff_width(small_config):
    """A small d needs many weights, which can outlast the longest window."""
    assert warmup_rows(small_config, 0.1) > small_config.features.max_lookback
    assert warmup_rows(small_config, 1.0) == small_config.features.max_lookback


def test_shift_by_one_day_shifts_output_by_one_day(small_config):
    """Moving every date one day later moves every feature one day later.

    This pins the alignment between dates and rows. The truncation check
    below is what catches a feature reading future data.
    """
    original = build_features(make_panel(), small_config, d=D)
    shifted = build_features(make_panel(offset=1), small_config, d=D)

    moved = original.copy()
    moved.index = moved.index.set_levels(moved.index.levels[0] + pd.offsets.BDay(1), level="date")

    pd.testing.assert_frame_equal(moved, shifted)


def test_truncation_check_passes_on_honest_features(panel, small_config):
    """The built-in check accepts the real feature set."""
    features = build._compute(panel, small_config, D)

    assert_no_lookahead(features, panel, small_config, D)


def test_truncation_check_catches_a_planted_leak(panel, small_config, monkeypatch):
    """A feature that peeks one day ahead makes the build fail loudly."""
    honest = build.ticker_features

    def leaky(bars, config, d):
        out = honest(bars, config, d)
        out["ret_1d"] = out["ret_1d"].shift(-1)
        return out

    monkeypatch.setattr(build, "ticker_features", leaky)

    with pytest.raises(LookaheadError, match="ret_1d"):
        build_features(panel, small_config, d=D)


def test_no_feature_nearly_equals_the_target(panel, small_config):
    """A feature correlated above 0.99 with tomorrow's return is a leak."""
    features = build_features(panel, small_config, d=D)
    target = label_fixed_horizon(panel, small_config)["return"].reindex(features.index)

    correlation = features.corrwith(target).abs()

    assert (correlation < 0.99).all()


def test_drop_warmup_counts_rows_per_ticker(panel, small_config):
    """Exactly warmup_rows rows go from each ticker."""
    features = build._compute(panel, small_config, D)

    trimmed = drop_warmup(features, small_config, D)

    assert len(trimmed) == len(features) - len(TICKERS) * warmup_rows(small_config, D)


def test_fracdiff_order_is_recorded(panel, small_config):
    """The order used travels with the matrix."""
    assert build_features(panel, small_config, d=D).attrs["fracdiff_d"] == D


def test_fracdiff_order_uses_pre_test_data_only(small_config):
    """Wrecking prices after test_start leaves the chosen order unchanged."""
    panel = make_panel(n=4_000)
    test_start = pd.Timestamp(small_config.validation.test_start)

    damaged = panel.copy()
    after = damaged.index.get_level_values("date") > test_start
    damaged.loc[after, "adj_close"] = np.linspace(1, 1_000, after.sum())

    assert build.choose_fracdiff_order(panel, small_config) == build.choose_fracdiff_order(
        damaged, small_config
    )


def test_feature_windows_follow_feature_names(small_config):
    """One window per feature, in matrix order, read from the config."""
    windows = feature_windows(small_config, d=D)
    width = len(fracdiff_weights(D, small_config.features.fracdiff.weight_threshold)) - 1

    assert list(windows) == feature_names(small_config)
    assert windows["ret_21d"] == 21
    assert windows["px_over_ma_63"] == 63
    assert windows["vol_ratio_10_63"] == 63
    assert windows["rsi_14"] == 14
    assert windows["volume_ratio_21"] == 21
    assert windows["fracdiff"] == width
    # Without d, fracdiff falls back to the longest configured window.
    assert feature_windows(small_config)["fracdiff"] == small_config.features.max_lookback
