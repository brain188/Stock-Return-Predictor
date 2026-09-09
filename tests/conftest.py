"""Shared fixtures.

Tests run on synthetic price series so they are fast and need no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config, load_config


@pytest.fixture(scope="session")
def config() -> Config:
    """The project configuration, loaded once per session."""
    return load_config(create_dirs=False)


def make_bars(n: int = 600, seed: int = 0, start: str = "2005-01-03") -> pd.DataFrame:
    """Build a synthetic OHLCV series shaped like a real daily bar file.

    Args:
        n: Number of business days.
        seed: Random seed, so tests are repeatable.
        start: First date.

    Returns:
        Daily bars indexed by date.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="B")
    price = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))

    return pd.DataFrame(
        {
            "open": price * 0.995,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "adj_close": price,
            "volume": rng.integers(1_000_000, 9_000_000, n).astype(float),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


@pytest.fixture
def bars() -> pd.DataFrame:
    """A clean synthetic price series."""
    return make_bars()