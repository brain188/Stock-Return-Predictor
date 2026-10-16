"""Checks on cleaning.

Every rule must touch exactly the rows it claims to, and the counts must add
up. Notebook 1 found a masking bug that inflated an extreme move count from 0
to 55,341, which is the kind of error these tests exist to catch.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import (
    CleaningError,
    align_calendar,
    clean_panel,
    drop_zero_volume,
    flag_price_gaps,
    winsorise_returns,
)
from tests.conftest import make_bars


@pytest.fixture
def small_config(config):
    """The project config with a benchmark that exists in the synthetic panel."""
    universe = dataclasses.replace(config.universe, tickers=["SPY", "AAA"], benchmark="SPY")

    return dataclasses.replace(config, universe=universe)


@pytest.fixture
def panel() -> dict[str, pd.DataFrame]:
    """Two clean synthetic tickers on the same calendar."""
    return {"SPY": make_bars(n=300, seed=1), "AAA": make_bars(n=300, seed=2)}


def split_2_for_1(bars: pd.DataFrame, at: int) -> pd.DataFrame:
    """Halve every price from row `at` onwards, as an unadjusted split would."""
    out = bars.copy()
    out.iloc[at:, out.columns.get_indexer(["open", "high", "low", "close", "adj_close"])] /= 2

    return out


def test_split_is_flagged(config, bars):
    """A 2 for 1 split, a -50% move, is flagged on the day it happens and no other."""
    flagged = flag_price_gaps(split_2_for_1(bars, at=200), config)

    assert flagged["gap_flag"].sum() == 1
    assert flagged["gap_flag"].iloc[200]


def test_reverse_split_is_flagged(config, bars):
    """A 1 for 2 reverse split, a +100% move, is flagged too."""
    reverse = bars.copy()
    reverse.iloc[150:, :5] *= 2

    assert flag_price_gaps(reverse, config)["gap_flag"].iloc[150]


def test_normal_moves_are_not_flagged(config, bars):
    """Ordinary 1% daily noise never trips the gap rule."""
    assert not flag_price_gaps(bars, config)["gap_flag"].any()


def test_zero_volume_day_is_removed(config, bars):
    """Exactly the zero-volume rows disappear."""
    bars.iloc[[10, 20], bars.columns.get_loc("volume")] = 0

    out = drop_zero_volume(bars, config)

    assert len(out) == len(bars) - 2
    assert bars.index[10] not in out.index
    assert (out["volume"] != 0).all()


def test_zero_volume_rule_can_be_switched_off(config, bars):
    """With the rule off, zero-volume rows are kept."""
    off = dataclasses.replace(
        config, cleaning=dataclasses.replace(config.cleaning, drop_zero_volume=False)
    )
    bars.iloc[10, bars.columns.get_loc("volume")] = 0

    assert len(drop_zero_volume(bars, off)) == len(bars)


def test_winsorise_caps_without_dropping(config):
    """Extremes are capped at the bounds, every row survives, NaN stays NaN."""
    returns = pd.Series([np.nan, 0.01, 0.9, -0.6, -0.02])

    capped = winsorise_returns(returns, config)

    assert len(capped) == len(returns)
    assert np.isnan(capped.iloc[0])
    assert capped.iloc[2] == pytest.approx(0.5)
    assert capped.iloc[3] == pytest.approx(1 / 1.5 - 1)
    assert capped.iloc[[1, 4]].tolist() == [0.01, -0.02]


def test_align_calendar_uses_benchmark_index(panel):
    """Every ticker ends up on the benchmark's dates, no more and no fewer."""
    panel["AAA"] = panel["AAA"].drop(panel["AAA"].index[[5, 6]])
    extra = panel["AAA"].iloc[[0]].copy()
    extra.index = pd.DatetimeIndex([pd.Timestamp("2005-01-01")], name="date")
    panel["AAA"] = pd.concat([extra, panel["AAA"]])

    aligned = align_calendar(panel, "SPY", max_fill_days=3)

    for frame in aligned.values():
        assert frame.index.equals(panel["SPY"].index)


def test_forward_fill_respects_the_limit(panel):
    """A five day hole with a limit of three: three filled, two left NaN."""
    hole = panel["AAA"].index[100:105]
    panel["AAA"] = panel["AAA"].drop(hole)

    aligned = align_calendar(panel, "SPY", max_fill_days=3)["AAA"]

    assert aligned.loc[hole, "filled"].tolist() == [True, True, True, False, False]
    assert aligned.loc[hole[:3], "adj_close"].eq(aligned["adj_close"].iloc[99]).all()
    assert aligned.loc[hole[3:], "adj_close"].isna().all()
    assert aligned.loc[hole, "volume"].isna().all()


def test_cleaned_panel_has_benchmark_index(small_config, panel):
    """For every ticker, the cleaned dates are exactly the benchmark's dates."""
    panel["AAA"] = panel["AAA"].drop(panel["AAA"].index[50])

    cleaned, _ = clean_panel(panel, small_config)

    for ticker in panel:
        dates = cleaned.xs(ticker, level="ticker").index
        assert dates.equals(panel["SPY"].index)


def test_report_counts_add_up(small_config, panel):
    """Every input row is accounted for, and so is every output row."""
    panel["AAA"].iloc[[30, 31], panel["AAA"].columns.get_loc("volume")] = 0
    panel["AAA"] = panel["AAA"].drop(panel["AAA"].index[200:210])
    off_calendar = panel["AAA"].iloc[[0]].copy()
    off_calendar.index = pd.DatetimeIndex([pd.Timestamp("2005-01-01")], name="date")
    panel["AAA"] = pd.concat([off_calendar, panel["AAA"]])

    _, report = clean_panel(panel, small_config)
    row = report.table.loc["AAA"]

    assert report.reconciles()
    assert row["input_rows"] == 291
    assert row["zero_volume_dropped"] == 2
    assert row["off_calendar_dropped"] == 1
    assert row["kept"] == 288
    assert row["forward_filled"] == 2 + 3
    assert row["left_missing"] == 7
    assert row["output_rows"] == 300


def test_clean_panel_reports_gaps(small_config, panel):
    """A split in the raw data appears in the report with its date and ticker."""
    panel["AAA"] = split_2_for_1(panel["AAA"], at=120)

    cleaned, report = clean_panel(panel, small_config)

    assert report.table.loc["AAA", "gaps_flagged"] == 1
    assert report.gaps.iloc[0]["ticker"] == "AAA"
    assert report.gaps.iloc[0]["date"] == panel["AAA"].index[120]
    assert report.table.loc["AAA", "returns_winsorised"] == 1
    assert cleaned.loc[(panel["AAA"].index[120], "AAA"), "gap_flag"]


def test_clean_panel_layout(small_config, panel):
    """Indexed by (date, ticker), with OHLCV and the three added columns."""
    cleaned, _ = clean_panel(panel, small_config)

    assert cleaned.index.names == ["date", "ticker"]
    assert cleaned.index.is_monotonic_increasing
    assert list(cleaned.columns) == [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "return",
        "gap_flag",
        "filled",
    ]


def test_returns_use_only_the_past(small_config, panel):
    """The return on day t is adj_close[t] / adj_close[t-1] - 1, nothing later."""
    cleaned, _ = clean_panel(panel, small_config)
    aaa = cleaned.xs("AAA", level="ticker")

    expected = panel["AAA"]["adj_close"].pct_change()

    pd.testing.assert_series_equal(aaa["return"], expected, check_names=False)


def test_missing_benchmark_raises(small_config, panel):
    """No benchmark means no calendar to align to."""
    del panel["SPY"]

    with pytest.raises(CleaningError, match="SPY"):
        clean_panel(panel, small_config)


def test_duplicate_dates_raise(small_config, panel):
    """A duplicated date would break every row count, so it is refused."""
    panel["AAA"] = pd.concat([panel["AAA"], panel["AAA"].iloc[[5]]])

    with pytest.raises(CleaningError, match="duplicate"):
        clean_panel(panel, small_config)
