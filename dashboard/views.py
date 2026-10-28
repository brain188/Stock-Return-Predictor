"""View helpers shared by several pages: labels, windows and the standard lines.

Nothing here reads a file directly; it composes data.py results into the
small pieces several pages need, so the wording and the colours stay the same.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from dash import html

from dashboard import charts, data
from dashboard import formatting as fmt
from dashboard import theme as t

PERIOD_LABELS = {
    data.DEVELOPMENT: "Development",
    data.HOLDOUT: "Holdout",
}


def holdout_window() -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """First and last row of the holdout evaluation window, as the run recorded them."""
    try:
        window = data.holdout_evaluation()["window"]
    except (data.DataError, KeyError):
        return None
    return pd.Timestamp(window["first_row_and_trade"]), pd.Timestamp(window["last_date"])


def period_caption(key: str, period: str) -> str:
    """Where a book's numbers in a period come from, in one line."""
    b = data.book(key)
    if period == data.HOLDOUT:
        window = holdout_window()
        span = fmt.date_range(*window) if window else "2020–2024"
        return f"Holdout {span} · one-time evaluation, attempt 1"
    if b.source == "portfolio":
        return "Development 2006-01-31 → 2019-12-31 · walk-forward, one path"
    return "Development 2005-07-22 → 2019-12-30 · CPCV, mean of 5 out-of-sample paths"


def trials_note() -> str:
    """The trial count every Sharpe is shown with."""
    try:
        return f"N = {data.freeze()['headline']['n_trials']} trials"
    except (data.DataError, KeyError):
        return "trial count unavailable"


def growth_lines(key: str, benchmark: str, period: str) -> list[charts.Line]:
    """Gross, net and benchmark net growth of 100 for one book in one period."""
    returns = data.book_returns(key, period)
    bench = data.book_returns(benchmark, period)["net"].reindex(returns.index)
    return [
        charts.Line(
            f"{data.BOOKS[benchmark].label} (net)", 100 * data.growth(bench), t.BENCHMARK, width=1.3
        ),
        charts.Line(
            f"{data.BOOKS[key].label} gross",
            100 * data.growth(returns["gross"]),
            t.GROSS,
            "dash",
            1.3,
        ),
        charts.Line(
            f"{data.BOOKS[key].label} net", 100 * data.growth(returns["net"]), t.NET, width=2
        ),
    ]


def joined_growth(key: str, benchmark: str) -> tuple[list[charts.Line], pd.Timestamp | None]:
    """Development then holdout growth on one axis, chained at the boundary.

    The holdout book starts from cash, so chaining multiplies the two periods'
    growth; the shading marks where the evaluation changes.
    """
    lines = []
    window = holdout_window()
    # The benchmark is cut to the strategy's own dates, so both start from 100 together.
    dates = []
    for period in data.PERIODS:
        try:
            dates.append(data.book_returns(key, period).index)
        except data.DataError:
            continue
    common = dates[0].append(dates[1:]) if dates else pd.DatetimeIndex([])
    for label, colour, dash, width, column, name in (
        (f"{data.BOOKS[benchmark].label} (net)", t.BENCHMARK, "solid", 1.3, "net", benchmark),
        (f"{data.BOOKS[key].label} gross", t.GROSS, "dash", 1.3, "gross", key),
        (f"{data.BOOKS[key].label} net", t.NET, "solid", 2.0, "net", key),
    ):
        pieces = []
        for period in data.PERIODS:
            try:
                pieces.append(data.book_returns(name, period)[column])
            except data.DataError:
                continue
        if not pieces:
            continue
        series = pd.concat(pieces).sort_index()
        series = series[~series.index.duplicated()].reindex(common).dropna()
        lines.append(charts.Line(label, 100 * data.growth(series), colour, dash, width))
    return lines, (window[0] if window else None)


def tone_vs(value: float, reference: float) -> str:
    """pos when value beats the reference, neg when it trails it."""
    if fmt.is_missing(value) or fmt.is_missing(reference):
        return "neutral"
    return "pos" if value > reference else "neg"


def interval_text(interval: dict[str, Any] | None) -> str:
    """A paired interval as "−0.26 [−0.74, +0.14]"."""
    if not interval:
        return fmt.MISSING
    return (
        f"{fmt.sharpe(interval['difference'], signed=True)} "
        f"[{fmt.sharpe(interval['low'], signed=True)}, {fmt.sharpe(interval['high'], signed=True)}]"
    )


def interval_verdict(interval: dict[str, Any] | None) -> tuple[str, str]:
    """Whether a paired interval excludes zero, in words and tone."""
    if not interval or fmt.is_missing(interval.get("low")):
        return "not computed", "neutral"
    if interval["low"] > 0:
        return "distinguishably better", "pos"
    if interval["high"] < 0:
        return "distinguishably worse", "neg"
    return "not distinguishable", "neutral"


def period_tabs(current: str) -> html.Div:
    """Development / holdout toggle bound to the global period control."""
    return html.Div(
        [
            html.Button(
                PERIOD_LABELS[p],
                id={"type": "pick-period", "key": p},
                className="tab active" if p == current else "tab",
                **{"aria-pressed": "true" if p == current else "false"},
            )
            for p in data.PERIODS
        ],
        className="tabs",
        role="group",
        **{"aria-label": "Evaluation period"},
    )
