"""Checks on the stationarity report.

Three known cases pin the ADF implementation: a random walk has a unit root,
white noise does not, and neither does an AR(1) with coefficient 0.5. If these
three are right, the numbers on real data can be trusted.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.features.fracdiff import adf_test
from src.reporting.stationarity import stationarity_table, write_stationarity_table

N = 3_000


def ar_one(phi: float, seed: int) -> np.ndarray:
    """An AR(1) process x_t = phi * x_{t-1} + e_t."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=N)
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = phi * x[t - 1] + shocks[t]

    return x


def test_random_walk_fails():
    """A random walk has a unit root, so the test must not reject it."""
    walk = np.cumsum(np.random.default_rng(1).normal(size=N))

    assert not adf_test(walk).stationary


def test_white_noise_passes():
    """White noise is the textbook stationary series."""
    assert adf_test(np.random.default_rng(2).normal(size=N)).stationary


def test_ar_one_half_passes():
    """An AR(1) with phi = 0.5 reverts quickly and is stationary."""
    assert adf_test(ar_one(0.5, seed=3)).stationary


@pytest.fixture
def cleaned_panel() -> pd.DataFrame:
    """A cleaned-panel-shaped frame: two random walk tickers in log space."""
    dates = pd.date_range("2005-01-03", periods=N, freq="B", name="date")
    rng = np.random.default_rng(4)
    frames = {
        ticker: pd.DataFrame(
            {"adj_close": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, N)))},
            index=dates,
        )
        for ticker in ["SPY", "AAA"]
    }

    return pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()


@pytest.fixture
def adj_config(config):
    """Config reading prices from adj_close, whatever the checked-in value is."""
    return dataclasses.replace(
        config, data=dataclasses.replace(config.data, price_column="adj_close")
    )


def test_table_has_one_row_per_ticker(cleaned_panel, adj_config):
    """Every ticker gets a row, and prices fail while returns pass."""
    table = stationarity_table(cleaned_panel, adj_config)

    assert sorted(table.index) == ["AAA", "SPY"]
    assert not table["price_stationary"].any()
    assert table["return_stationary"].all()


def test_d_star_keeps_more_memory_than_returns(cleaned_panel, adj_config):
    """A passing d* below 1 keeps more correlation with the price than returns do."""
    table = stationarity_table(cleaned_panel, adj_config)

    assert (table["d_star"] > 0).all()
    assert (table["d_star"] <= 1).all()
    assert (table["memory_gained"] >= 0).all()


def test_table_is_written(cleaned_panel, adj_config, tmp_path):
    """The table lands on disk and reads back with the same tickers."""
    path = write_stationarity_table(stationarity_table(cleaned_panel, adj_config), tmp_path)

    assert path.is_file()
    assert sorted(pd.read_csv(path, index_col="ticker").index) == ["AAA", "SPY"]
