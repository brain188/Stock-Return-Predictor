"""The global controls: their ids, defaults, and how pages read them.

The header owns these components; pages take them as callback inputs through
INPUTS, in this order, and turn the values into a dict with `selection()`.
"""

from __future__ import annotations

from typing import Any

from dash import Input, State

from dashboard import data

CONTROLS = ("asset", "dates", "period", "freq", "model", "strategy", "benchmark")
MODEL_KEY = "research_random_forest_regression_fixed_horizon"
MODEL_LABELS = {
    MODEL_KEY: "Random forest (regression)",
    "always_up": "Baseline: always up",
    "random": "Baseline: random",
    "last_sign": "Baseline: last sign",
    "reverse_last_sign": "Baseline: reverse last sign",
}

PROPERTIES = (
    ("ctl-asset", "value"),
    ("ctl-dates", "start_date"),
    ("ctl-dates", "end_date"),
    ("ctl-period", "value"),
    ("ctl-freq", "value"),
    ("ctl-model", "value"),
    ("ctl-strategy", "value"),
    ("ctl-benchmark", "value"),
)
STATES = [State(i, p) for i, p in PROPERTIES]
INPUTS = [Input(i, p) for i, p in PROPERTIES]
RELOAD = Input("reload-token", "data")


def defaults() -> dict[str, Any]:
    """The values the dashboard starts with and Reset returns to."""
    first = last = None
    try:
        dates = data.panel().index.get_level_values("date")
        first, last = str(dates.min().date()), str(dates.max().date())
    except data.DataError:
        pass
    return {
        "asset": "SPY",
        "start": first,
        "end": last,
        "period": data.HOLDOUT,
        "freq": "1D",
        "model": MODEL_KEY,
        "strategy": "allocation_minimum_variance",
        "benchmark": "equal_weight_hold",
    }


def selection(*values: Any) -> dict[str, Any]:
    """Control values, in PROPERTIES order, as a dict with defaults for blanks."""
    keys = ("asset", "start", "end", "period", "freq", "model", "strategy", "benchmark")
    chosen = dict(zip(keys, values, strict=False))
    base = defaults()
    return {k: chosen.get(k) or base[k] for k in keys}
