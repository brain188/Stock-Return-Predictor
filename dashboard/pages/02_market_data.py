"""Market data and EDA: prices, returns, volatility, memory, correlation and stationarity.

Follows the asset, date window and sampling controls. Every statistic is
computed on exactly the window and frequency the chart shows, except where a
panel says otherwise (the ADF table is the saved full-sample result, and the
volatility reference lines use the asset's whole history).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from dash import Output, callback, html
from scipy import stats

from dashboard import charts, data
from dashboard import formatting as fmt
from dashboard import theme as t
from dashboard.controls import INPUTS, RELOAD, selection

PATH = "/market"
TITLE = "Market Data & EDA"
SUBTITLE = "Q1: stationarity & volatility"
ICON = "candlestick_chart"
USES = {"asset", "dates", "freq"}
FREQ_LABELS = {"1D": "daily", "1W": "weekly", "1M": "monthly"}


def layout() -> html.Div:
    """The page skeleton; one callback fills every block from the controls."""
    return html.Div(
        [
            html.Div(id="md-strip"),
            html.Div(id="md-kpis"),
            html.Div(
                [
                    html.Div(id="md-price", className="c-8 stack-md"),
                    html.Div(id="md-density", className="c-4 stack-md"),
                ],
                className="grid",
            ),
            html.Div(
                [html.Div(id="md-vol", className="c-6"), html.Div(id="md-acf", className="c-6")],
                className="grid",
            ),
            html.Div(
                [
                    html.Div(id="md-corr", className="c-5 stack-md"),
                    html.Div(id="md-adf", className="c-7 stack-md"),
                ],
                className="grid",
            ),
        ],
        className="page",
    )


def _window(c: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Daily rows of the asset inside the window, and its returns at the chosen frequency."""
    rows = data.panel().xs(c["asset"], level="ticker")
    rows = data.window(rows, c["start"], c["end"])
    if len(rows) < 30:
        raise data.DataError(
            "Window too short", "Choose a date window of at least 30 trading days."
        )
    returns = data.resample_returns(rows["return"].dropna(), c["freq"]).dropna()
    return rows, returns


def _num(text: str, tone: str = "") -> html.Span:
    return html.Span(text, className=f"num {tone}".strip())


def _p(p: float) -> str:
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"


def strip(c: dict[str, Any], rows: pd.DataFrame) -> html.Header:
    """Which asset, window and sampling every panel below uses."""
    return t.page_header(
        "Market data and exploratory analysis",
        "Prices, returns, volatility, memory and stationarity. Every statistic uses the asset, "
        "window and sampling selected above unless its panel says otherwise.",
        eyebrow=f"Market data · {c['asset']}",
        meta=[
            ("Asset", c["asset"]),
            ("Window", fmt.date_range(rows.index.min(), rows.index.max())),
            ("Bars", fmt.integer(len(rows))),
            ("Sampling", FREQ_LABELS[c["freq"]]),
            ("Universe", ", ".join(data.tickers())),
        ],
    )


def coverage(c: dict[str, Any], rows: pd.DataFrame) -> html.Section:
    """Completeness and provenance of the price data, in one strip."""
    whole = data.coverage(c["asset"])
    priced = int(rows["adj_close"].notna().sum())
    flagged = int(rows["gap_flag"].sum())
    return t.panel(
        t.stat_grid(
            [
                t.stat(
                    "Coverage",
                    fmt.pct(priced / len(rows), 2),
                    f"{fmt.integer(priced)} of {fmt.integer(len(rows))} bars priced",
                    "" if priced == len(rows) else "warn",
                ),
                t.stat(
                    "Missing · forward-filled",
                    f"{len(rows) - priced} · {int(rows['filled'].sum())}",
                    "Bars in the window",
                ),
                t.stat(
                    "Flagged moves",
                    fmt.integer(flagged),
                    "Daily moves beyond ±50%",
                    "warn" if flagged else "",
                ),
                t.stat(
                    "Price series", "Adjusted close", "Dividends and splits, Yahoo back-adjusted"
                ),
                t.stat(
                    "History",
                    fmt.date(whole["first"]),
                    f"First bar · {fmt.integer(whole['rows'])} rows",
                ),
                t.stat("Last bar", fmt.date(whole["last"]), "SPY trading calendar"),
            ]
        ),
        title="Data coverage",
        subtitle=f"{c['asset']} in the selected window; history and last bar cover the whole file",
    )


def price_panel(c: dict[str, Any], rows: pd.DataFrame) -> html.Section:
    """Adjusted close, volume and daily returns over the window."""
    figure = charts.price_panel(rows["adj_close"], rows["volume"], rows["return"], c["asset"])
    last = rows["adj_close"].dropna()
    vol = data.rolling_volatility(c["asset"])["volatility"].reindex(rows.index).dropna()
    change = last.iloc[-1] / last.iloc[0] - 1
    return t.panel(
        t.graph(figure, 380, label=f"{c['asset']} adjusted close, volume and daily returns"),
        title=f"{c['asset']}: adjusted close, volume and daily returns",
        subtitle="Daily bars · USD, adjusted for dividends and splits",
        legend=t.legend(
            [
                ("Adjusted close", t.PRIMARY, "line"),
                ("Up day", t.POS, "box"),
                ("Down day", t.NEG, "box"),
            ]
        ),
        footer=[
            html.Span(["Last close ", _num(f"${last.iloc[-1]:,.2f}")]),
            html.Span(["Change in window ", _num(fmt.pct(change, 1, True), fmt.tone(change))]),
            html.Span(
                [
                    "21-day volatility at end ",
                    _num(fmt.pct(vol.iloc[-1], 1) if len(vol) else fmt.MISSING),
                ]
            ),
        ],
    )


def density_panel(c: dict[str, Any], returns: pd.Series) -> html.Section:
    """The return distribution at the chosen frequency against the normal."""
    label = FREQ_LABELS[c["freq"]]
    skew = float(stats.skew(returns, bias=False)) if len(returns) > 2 else np.nan
    excess = float(stats.kurtosis(returns, fisher=True, bias=False)) if len(returns) > 3 else np.nan
    jb = stats.jarque_bera(returns) if len(returns) > 2 else None
    return t.panel(
        [
            t.graph(
                charts.distribution(returns, label), 236, label=f"Distribution of {label} returns"
            ),
            html.Div(
                [
                    t.kv(
                        "Excess kurtosis", fmt.number(excess, 2), "warn" if excess > 1 else "strong"
                    ),
                    t.kv("Skewness", fmt.number(skew, 2), "strong"),
                    t.kv(
                        "Jarque-Bera",
                        f"{jb.statistic:,.1f} ({_p(jb.pvalue)})" if jb else fmt.MISSING,
                        "strong",
                    ),
                    t.kv(
                        "Mean · s.d.",
                        f"{fmt.pct(returns.mean(), 2)} · {fmt.pct(returns.std(), 2)}",
                        "strong",
                    ),
                ],
                className="list-rows",
                style={"marginTop": "12px"},
            ),
        ],
        title="Return distribution",
        subtitle=f"{label.capitalize()} returns in the window against a normal with the same mean "
        f"and spread · n = {fmt.integer(len(returns))}",
        actions=t.tag("Fat tails", "warn") if excess > 1 else None,
        legend=t.legend([("Empirical", t.PRIMARY_CONTAINER, "box"), ("Normal", t.OUTLINE, "dash")]),
    )


def volatility_panel(c: dict[str, Any], rows: pd.DataFrame) -> html.Section:
    """Rolling 21-day volatility in the window, against the whole history's median and 95th pct."""
    full = data.rolling_volatility(c["asset"])
    history = full["volatility"].dropna()
    shown = full.reindex(rows.index).dropna(subset=["volatility"])
    if shown.empty:
        raise data.DataError("Window too short", "Rolling volatility needs 21 days of returns.")
    median, high = float(history.median()), float(history.quantile(0.95))
    regime = shown["regime"].dropna()
    return t.panel(
        t.graph(
            charts.volatility(shown, median, high),
            240,
            label="Rolling 21-day annualised volatility",
        ),
        title="Rolling 21-day realised volatility",
        subtitle="Annualised from daily returns · reference lines from the whole 2005–2024 history",
        legend=t.legend(
            [
                ("21-day", t.PRIMARY, "line"),
                ("Median", t.OUTLINE, "dash"),
                ("95th percentile", t.NEG, "dash"),
            ]
        ),
        footer=[
            html.Span(
                ["Regime at end ", html.B(str(regime.iloc[-1]) if len(regime) else fmt.MISSING)]
            ),
            html.Span(
                [
                    "Days above 95th percentile ",
                    _num(fmt.integer((shown["volatility"] > high).sum())),
                ]
            ),
            html.Span("Regimes use only past volatility."),
        ],
    )


def acf_panel(c: dict[str, Any], returns: pd.Series) -> html.Section:
    """Autocorrelation of returns and of their size, with Ljung-Box."""
    acf = data.autocorrelation(returns, 21 if c["freq"] == "1D" else 12)
    q, p = data.ljung_box(returns, 10)
    lag1 = acf["returns"].iloc[0]
    significant = p < 0.05
    return t.panel(
        t.graph(charts.autocorrelation(acf), 240, label="Autocorrelation of returns"),
        title=f"Autocorrelation of {FREQ_LABELS[c['freq']]} returns",
        subtitle="Shaded band: 95% range for white noise",
        legend=t.legend([("Returns", t.PRIMARY, "box"), ("Absolute returns", t.WARN, "box")]),
        footer=[
            html.Span(["Lag-1 ", _num(fmt.number(lag1, 4, signed=True))]),
            html.Span(["Ljung-Box Q(10) ", _num(f"{fmt.number(q, 2)}, {_p(p)}")]),
            html.Span(
                "Autocorrelation is significant at 5%."
                if significant
                else "No significant autocorrelation at 5%."
            ),
        ],
    )


def correlation_panel(c: dict[str, Any]) -> html.Section:
    """Pearson correlation of the universe's returns over the same window and sampling."""
    returns = data.window(data.wide("return"), c["start"], c["end"])
    returns = data.resample_returns(returns, c["freq"]).dropna(how="all")
    matrix = returns.corr()
    eig = np.linalg.eigvalsh(matrix.fillna(0).to_numpy())
    upper = matrix.where(np.triu(np.ones(matrix.shape, dtype=bool), k=1)).stack()
    return t.panel(
        t.graph(
            charts.heatmap(matrix, "Pearson ρ", show_text=True),
            360,
            label="Correlation matrix of returns",
        ),
        title="Cross-asset correlation",
        subtitle=f"Pearson ρ of {FREQ_LABELS[c['freq']]} returns in the window · "
        f"n = {fmt.integer(len(returns))}",
        footer=[
            html.Span(["Mean pairwise ρ ", _num(fmt.number(upper.mean(), 3))]),
            html.Span(
                [
                    "First eigenvalue ",
                    _num(f"{eig[-1]:.2f}"),
                    f" ({eig[-1] / eig.sum():.0%} of variance)",
                ]
            ),
        ],
    )


def adf_panel(c: dict[str, Any]) -> html.Section:
    """The saved ADF table: log prices, log returns and the fractional differencing order."""
    table = data.stationarity()
    rows = table.to_dict("records")
    crit = float(table["critical"].iloc[0])
    for row in rows:
        row["verdict"] = "Unit root" if row["adf_log_price"] > row["critical"] else "Stationary"
    columns = [
        t.Col("Ticker", "ticker", str, numeric=False),
        t.Col("ADF log price", "adf_log_price", lambda v: fmt.number(v, 2)),
        t.Col("ADF log return", "adf_log_return", lambda v: fmt.number(v, 2)),
        t.Col(
            "d*",
            "d_star",
            lambda v: fmt.number(v, 1),
            title="Smallest fractional differencing order passing ADF",
        ),
        t.Col("ADF at d*", "adf_at_d_star", lambda v: fmt.number(v, 2)),
        t.Col(
            "Memory kept",
            "corr_at_d_star",
            lambda v: fmt.number(v, 2),
            title="Correlation of the d* series with the log price",
        ),
        # A unit root in log prices is expected, not a defect: no warning colour.
        t.Col("Log price", "verdict", str, numeric=False),
    ]
    unit_roots = int((table["adf_log_price"] > table["critical"]).sum())
    return t.panel(
        t.table(rows, columns, compact=True, highlight=lambda r: r["ticker"] == c["asset"]),
        title="Stationarity: augmented Dickey-Fuller",
        subtitle=f"H0: unit root · 5% critical value {crit:.2f} · saved full-sample result "
        "(2005–2024), "
        "not the window above",
        footer=[
            html.Span(
                f"{unit_roots} of {len(table)} log prices keep a unit root; "
                "every log return rejects it."
            ),
            html.Span("The pipeline uses d = 0.60, the largest d* on development data only."),
        ],
    )


@callback(
    Output("md-strip", "children"),
    Output("md-kpis", "children"),
    Output("md-price", "children"),
    Output("md-density", "children"),
    Output("md-vol", "children"),
    Output("md-acf", "children"),
    Output("md-corr", "children"),
    Output("md-adf", "children"),
    *INPUTS,
    RELOAD,
)
def update(*values: Any) -> tuple:
    """Rebuild every panel for the asset, window and sampling."""
    c = selection(*values[:-1])
    try:
        rows, returns = _window(c)
    except data.DataError as exc:
        missing = t.empty(exc.title, exc.detail, "invalid")
        return (
            missing,
            None,
            None,
            None,
            None,
            None,
            None,
            t.guarded(lambda: adf_panel(c), "the ADF table"),
        )
    return (
        t.guarded(lambda: strip(c, rows), "the page header"),
        t.guarded(lambda: coverage(c, rows), "the coverage strip"),
        t.guarded(lambda: price_panel(c, rows), "the price chart"),
        t.guarded(lambda: density_panel(c, returns), "the return distribution"),
        t.guarded(lambda: volatility_panel(c, rows), "the volatility chart"),
        t.guarded(lambda: acf_panel(c, returns), "the autocorrelation chart"),
        t.guarded(lambda: correlation_panel(c), "the correlation matrix"),
        t.guarded(lambda: adf_panel(c), "the ADF table"),
    )


def export(controls: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    """The asset's daily bars in the window, as the price chart shows them."""
    rows = data.window(
        data.panel().xs(controls["asset"], level="ticker"), controls["start"], controls["end"]
    )
    if rows.empty:
        raise data.DataError("No rows", "the date window holds no bars")
    return f"market_{controls['asset']}", rows
