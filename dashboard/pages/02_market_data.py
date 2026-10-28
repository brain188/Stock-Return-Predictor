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
            html.Div(id="md-kpis", className="kpis", style={"--n": 6}),
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


def strip(c: dict[str, Any], rows: pd.DataFrame) -> html.Div:
    """Which asset, window and sampling every panel below uses."""
    universe = ", ".join(data.tickers())
    return html.Div(
        [
            html.Div(
                [
                    t.icon("analytics", "lg"),
                    html.Span(f"EDA / {c['asset']}", className="h-sm primary mono"),
                    html.Span(
                        ["Universe: ", html.Span(universe, className="strong")],
                        className="mono-sm muted",
                    ),
                ],
                className="strip-main",
                style={"flexWrap": "wrap"},
            ),
            html.Div(
                [
                    html.Span(
                        [
                            "Window: ",
                            html.Span(
                                fmt.date_range(rows.index.min(), rows.index.max()),
                                className="strong",
                            ),
                        ]
                    ),
                    html.Span(["Bars: ", html.Span(fmt.integer(len(rows)), className="strong")]),
                    html.Span(
                        ["Sampling: ", html.Span(FREQ_LABELS[c["freq"]], className="primary")]
                    ),
                ],
                className="meta mono-sm",
            ),
        ],
        className="strip",
    )


def kpis(c: dict[str, Any], rows: pd.DataFrame) -> list[Any]:
    """Coverage and provenance of the price data in the window."""
    whole = data.coverage(c["asset"])
    priced = int(rows["adj_close"].notna().sum())
    return [
        t.kpi(
            "Coverage ratio",
            fmt.pct(priced / len(rows), 2),
            f"({fmt.integer(priced)}/{fmt.integer(len(rows))})",
            tone="pos" if priced == len(rows) else "warn",
        ),
        t.kpi(
            "Missing · filled",
            f"{len(rows) - priced} · {int(rows['filled'].sum())}",
            "bars in window",
        ),
        t.kpi(
            "Gaps flagged",
            fmt.integer(rows["gap_flag"].sum()),
            "moves > 50%",
            tone="pos" if not rows["gap_flag"].any() else "warn",
        ),
        t.kpi(
            "Adjustment",
            "Adj. close",
            "dividends & splits",
            rows=[("Source", "Yahoo, back-adjusted", "soft")],
        ),
        t.kpi(
            "Asset history",
            fmt.date(whole["first"]),
            "first bar",
            rows=[("Rows", fmt.integer(whole["rows"]), "soft")],
        ),
        t.kpi(
            "Last bar",
            fmt.date(whole["last"]),
            "",
            tone="primary",
            rows=[("Calendar", "SPY trading days", "soft")],
        ),
    ]


def price_panel(c: dict[str, Any], rows: pd.DataFrame) -> html.Section:
    """Adjusted close, volume and daily returns over the window."""
    figure = charts.price_panel(rows["adj_close"], rows["volume"], rows["return"], c["asset"])
    last = rows["adj_close"].dropna()
    vol = data.rolling_volatility(c["asset"])["volatility"].reindex(rows.index).dropna()
    return t.panel(
        t.graph(figure, 360, label=f"{c['asset']} adjusted close, volume and daily returns"),
        title=f"{c['asset']}: adjusted close, volume and daily returns",
        subtitle="Daily bars · prices in USD, adjusted for dividends and splits",
        actions=t.legend(
            [
                ("Adj. close", t.PRIMARY, "line"),
                ("Up day", t.POS, "box"),
                ("Down day", t.NEG, "box"),
            ]
        ),
        footer=[
            html.Span(["Last: ", html.Span(f"${last.iloc[-1]:,.2f}", className="strong")]),
            html.Span(
                [
                    "Change in window: ",
                    html.Span(
                        fmt.pct(last.iloc[-1] / last.iloc[0] - 1, 1, True),
                        className=fmt.tone(last.iloc[-1] / last.iloc[0] - 1),
                    ),
                ]
            ),
            html.Span(
                [
                    "21-day vol at end: ",
                    html.Span(
                        fmt.pct(vol.iloc[-1], 1) if len(vol) else fmt.MISSING, className="strong"
                    ),
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
        t.graph(charts.distribution(returns, label), 220, label=f"Distribution of {label} returns"),
        title="Return density: empirical vs normal",
        subtitle=f"{label.capitalize()} returns in the window · n = {fmt.integer(len(returns))}",
        actions=t.tag("fat tails" if excess > 1 else "near normal", "warn" if excess > 1 else ""),
        footer=html.Div(
            [
                t.kv("Excess kurtosis", fmt.number(excess, 2), "warn" if excess > 1 else "strong"),
                t.kv("Skewness", fmt.number(skew, 2), "strong"),
                t.kv(
                    "Jarque-Bera",
                    (
                        f"{jb.statistic:,.1f} (p "
                        f"{'< 0.0001' if jb.pvalue < 1e-4 else f'= {jb.pvalue:.4f}'})"
                    )
                    if jb
                    else fmt.MISSING,
                    "strong",
                ),
                t.kv(
                    "Mean · s.d.",
                    f"{fmt.pct(returns.mean(), 2)} · {fmt.pct(returns.std(), 2)}",
                    "strong",
                ),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "2px", "width": "100%"},
        ),
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
            220,
            label="Rolling 21-day annualised volatility",
        ),
        title="Rolling 21-day realised volatility",
        subtitle="Annualised from daily returns · reference lines from the whole 2005–2024 history",
        actions=t.legend(
            [
                ("21-day vol", t.PRIMARY, "line"),
                ("Median", t.OUTLINE, "dash"),
                ("95th pct", t.NEG, "dash"),
            ]
        ),
        footer=[
            html.Span(
                [
                    "Current regime: ",
                    html.Span(
                        str(regime.iloc[-1]) if len(regime) else fmt.MISSING, className="primary"
                    ),
                ]
            ),
            html.Span(
                [
                    "Days above 95th pct: ",
                    html.Span(fmt.integer((shown["volatility"] > high).sum()), className="strong"),
                ]
            ),
            html.Span("Regimes use only past volatility (no look-ahead)", className="muted"),
        ],
    )


def acf_panel(c: dict[str, Any], returns: pd.Series) -> html.Section:
    """Autocorrelation of returns and of their size, with Ljung-Box."""
    acf = data.autocorrelation(returns, 21 if c["freq"] == "1D" else 12)
    q, p = data.ljung_box(returns, 10)
    lag1 = acf["returns"].iloc[0]
    return t.panel(
        t.graph(charts.autocorrelation(acf), 220, label="Autocorrelation of returns"),
        title=f"Autocorrelation: {FREQ_LABELS[c['freq']]} returns",
        subtitle="Bars: returns · diamonds: absolute returns · shaded: 95% white-noise band",
        actions=t.legend([("Returns", t.PRIMARY, "box"), ("|Returns|", t.WARN, "box")]),
        footer=[
            html.Span(["Lag-1: ", html.Span(fmt.number(lag1, 4, signed=True), className="strong")]),
            html.Span(
                [
                    "Ljung-Box Q(10): ",
                    html.Span(
                        f"{fmt.number(q, 2)} (p = {fmt.number(p, 4)})",
                        className="neg" if p < 0.05 else "pos",
                    ),
                ]
            ),
            html.Span(
                ["Band ±", html.Span(fmt.number(acf["band"].iloc[0], 4), className="strong")]
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
            330,
            label="Correlation matrix of returns",
        ),
        title=f"Cross-asset correlation ({len(matrix)}×{len(matrix)})",
        subtitle=(
            f"Pearson ρ of {FREQ_LABELS[c['freq']]} returns in the window · n = "
            f"{fmt.integer(len(returns))}"
        ),
        footer=[
            html.Span(
                ["Mean pairwise ρ: ", html.Span(fmt.number(upper.mean(), 3), className="strong")]
            ),
            html.Span(
                [
                    "First eigenvalue: ",
                    html.Span(
                        f"{eig[-1]:.2f} ({eig[-1] / eig.sum():.0%} of variance)", className="strong"
                    ),
                ]
            ),
        ],
    )


def adf_panel(c: dict[str, Any]) -> html.Section:
    """The saved ADF table: log prices, log returns and the fractional differencing order."""
    table = data.stationarity()
    rows = table.to_dict("records")
    crit = float(table["critical"].iloc[0])

    def verdict(row: dict) -> str:
        return "Non-stationary I(1)" if row["adf_log_price"] > row["critical"] else "Stationary"

    for row in rows:
        row["verdict"] = verdict(row)
        row["returns_verdict"] = (
            "Stationary" if row["adf_log_return"] < row["critical"] else "Unit root"
        )
    columns = [
        t.Col("Ticker", "ticker", str, numeric=False),
        t.Col(
            "ADF log price",
            "adf_log_price",
            lambda v: fmt.number(v, 2),
            tone=lambda v, r: "neg" if v > r["critical"] else "pos",
        ),
        t.Col(
            "ADF log return",
            "adf_log_return",
            lambda v: fmt.number(v, 2),
            tone=lambda v, r: "pos" if v < r["critical"] else "neg",
        ),
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
        t.Col(
            "Price", "verdict", str, numeric=False, tone=lambda v, r: "neg" if "Non" in v else "pos"
        ),
    ]
    return t.panel(
        t.table(rows, columns, compact=True, highlight=lambda r: r["ticker"] == c["asset"]),
        title="Stationarity: augmented Dickey-Fuller",
        subtitle="H0: unit root · saved full-sample result (2005–2024), not the date window above",
        actions=t.tag(f"5% critical {crit:.2f}"),
        footer=[
            html.Span(
                f"{int((table['adf_log_price'] > table['critical']).sum())} of {len(table)} "
                "log prices keep a unit root; every log return rejects it."
            ),
            html.Span(
                "The pipeline uses d = 0.60, the largest d* on development data only.",
                className="muted",
            ),
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
            [],
            None,
            None,
            None,
            None,
            None,
            t.guarded(lambda: adf_panel(c), "the ADF table"),
        )
    return (
        t.guarded(lambda: strip(c, rows), "the strip"),
        t.guarded(lambda: kpis(c, rows), "the coverage cards"),
        t.guarded(lambda: price_panel(c, rows), "the price chart"),
        t.guarded(lambda: density_panel(c, returns), "the return density"),
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
