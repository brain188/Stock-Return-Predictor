"""Backtesting: the selected strategy's saved results, gross beside net, against its benchmark.

Read-only by design. Choosing a strategy or a period shows a backtest the
pipelines already ran and saved; nothing is rerun, refitted or logged, so the
page offers no "run" action. Gross and net are always side by side, and the
trial count sits next to every Sharpe.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from dash import Input, Output, callback, dcc, html

from dashboard import charts, data, views
from dashboard import formatting as fmt
from dashboard import theme as t
from dashboard.controls import INPUTS, RELOAD, STATES, selection

PATH = "/backtest"
TITLE = "Backtesting"
SUBTITLE = "Q4: gross vs net PnL"
ICON = "query_stats"
USES = {"strategy", "benchmark", "period"}


def layout() -> html.Div:
    """The page skeleton."""
    return html.Div(
        [
            dcc.Download(id="bt-download"),
            html.Div(id="bt-tabs"),
            html.Div(id="bt-costs"),
            html.Div(id="bt-kpis", className="kpis", style={"--n": 5}),
            html.Div(
                [
                    html.Div(id="bt-equity", className="c-8 stack-md"),
                    html.Div(id="bt-drawdown", className="c-4 stack-md"),
                ],
                className="grid",
            ),
            html.Div(
                [
                    html.Div(id="bt-years", className="c-7 stack-md"),
                    html.Div(id="bt-rigor", className="c-5 stack-md"),
                ],
                className="grid",
            ),
        ],
        className="page",
    )


def tabs(c: dict[str, Any]) -> html.Div:
    """Strategy tabs bound to the global strategy, and the engine's fixed settings."""
    cfg = data.config().backtest
    b = data.book(c["strategy"])
    stats = data.book_stats(c["strategy"], c["period"])
    roles = {}
    try:
        decl = data.declaration()["candidates"]
        roles = {decl["primary"]: "primary"} | dict.fromkeys(decl["secondary"], "secondary")
    except data.DataError:
        pass
    buttons = [
        html.Button(
            [data.BOOKS[k].label, t.tag("primary") if roles.get(k) == "primary" else None],
            id={"type": "pick-strategy", "key": k},
            className="tab active" if k == c["strategy"] else "tab",
            title=f"{data.BOOKS[k].family} · {roles.get(k, 'development only')}",
            **{"aria-pressed": "true" if k == c["strategy"] else "false"},
        )
        for k in data.STRATEGIES
    ]
    chips = [
        (
            "Execution",
            f"{cfg.execution_lag}-bar lag, close to close",
            "trade at the close after the signal",
        ),
        ("Max gross leverage", f"{cfg.max_gross_leverage:.1f}×", "long + short"),
        ("Rebalance", b.rebalance, b.family),
        ("Universe", f"{len(data.tickers())} tickers", "incl. SPY"),
        (
            "Backtest window",
            fmt.date_range(stats["start"], stats["end"]),
            f"{fmt.integer(stats['days'])} trading days",
        ),
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Span("Active strategy:", className="caps muted"),
                    html.Div(buttons, className="tabs"),
                ],
                style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    html.Span(f"Source: {stats['source']}", className="mono-sm muted"),
                    t.tag(
                        views.PERIOD_LABELS[c["period"]],
                        "pos" if c["period"] == data.HOLDOUT else "info",
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "8px",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                    "margin": "8px 0",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(label, className="caps muted"),
                            html.Span(main, className="mono-data strong"),
                            html.Span(sub, className="mono-sm muted"),
                        ],
                        className="kpi",
                        style={"background": "var(--surface-low)"},
                    )
                    for label, main, sub in chips
                ],
                className="kpis",
                style={"--n": 5},
            ),
        ],
        className="panel low",
    )


def costs(c: dict[str, Any]) -> html.Section:
    """The cost assumptions in basis points, and what they cost this strategy."""
    cfg = data.config().backtest
    s = data.book_stats(c["strategy"], c["period"])
    turnover, drag = s.get("turnover_annual"), s.get("cost_annual")

    def cell(
        label: str, main: str, unit: str, note: str, share: float, tone: str = "strong"
    ) -> html.Div:
        return html.Div(
            [
                html.Span(label, className="caps muted"),
                html.Div(
                    [
                        html.Span(main, className=f"mono-display {tone}"),
                        html.Span(unit, className="mono-sm muted"),
                    ],
                    className="kpi-value",
                ),
                html.Span(note, className="mono-sm muted"),
                html.Div(
                    html.Div(
                        className="bar-fill",
                        style={"width": f"{max(2, min(100, share * 100)):.0f}%"},
                    ),
                    className="bar-track",
                ),
            ],
            className="kpi",
            style={"background": "var(--surface-low)"},
        )

    return t.panel(
        html.Div(
            [
                cell(
                    "Commission",
                    f"{cfg.commission_bps:.1f}",
                    "bps / side",
                    "per unit traded",
                    cfg.commission_bps / 10,
                ),
                cell(
                    "Slippage",
                    f"{cfg.slippage_bps:.1f}",
                    "bps / side",
                    "per unit traded",
                    cfg.slippage_bps / 10,
                ),
                cell(
                    "Short borrow",
                    f"{cfg.borrow_rate_annual * 1e4:.0f}",
                    "bps / yr",
                    "on short exposure held",
                    cfg.borrow_rate_annual * 20,
                ),
                cell(
                    "Turnover",
                    fmt.number(turnover, 1),
                    "× / yr",
                    "traded weight, one side",
                    min(1.0, (turnover or 0) / 40),
                ),
                cell(
                    "Annual cost drag",
                    fmt.number(-(drag or 0) * 1e4, 1),
                    "bps / yr",
                    f"{fmt.pct(-(drag or 0), 2)} p.a. of capital",
                    min(1.0, (drag or 0) * 20),
                    "neg",
                ),
            ],
            className="kpis",
            style={"--n": 5},
        ),
        title_icon="tune",
        kicker="Transaction cost & slippage protocol",
        title="Costs applied at every trade, on the traded weight",
        actions=html.Span(
            [
                "Round trip: ",
                html.Span(f"{cfg.round_trip_cost * 1e4:.1f} bps", className="primary mono-data"),
            ],
            className="mono-sm muted",
        ),
    )


def kpis(c: dict[str, Any]) -> list[Any]:
    """Gross against net for return, volatility, Sharpe, drawdown and Calmar."""
    s = data.book_stats(c["strategy"], c["period"])
    b = data.book_stats(c["benchmark"], c["period"])
    bl = data.BOOKS[c["benchmark"]].label

    def pair(key: str, f: Any) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
        gross = s[f"gross_{key}"]
        return ("Gross", f(gross), "pos" if fmt.tone(gross) == "pos" else "strong"), (
            "Net",
            f(s[f"net_{key}"]),
            "primary",
        )

    computed = s.get("computed", [])
    note = "gross from saved daily returns" if computed else None
    return [
        t.kpi_pair(
            "Annualised return",
            *pair("annual_return", lambda v: fmt.pct(v, 1, True)),
            rows=[
                ("Net drag", fmt.pp(s["net_annual_return"] - s["gross_annual_return"]), "neg"),
                (bl, fmt.pct(b["net_annual_return"], 1, True), "soft"),
            ],
        ),
        t.kpi_pair(
            "Annualised volatility",
            *pair("annual_volatility", lambda v: fmt.pct(v, 1)),
            rows=[(bl, fmt.pct(b["net_annual_volatility"], 1), "soft")],
        ),
        t.kpi_pair(
            "Sharpe ratio (rf = 0)",
            *pair("sharpe", fmt.sharpe),
            rows=[
                (bl, fmt.sharpe(b["net_sharpe"]), views.tone_vs(b["net_sharpe"], s["net_sharpe"])),
                ("Trials searched", views.trials_note(), "soft"),
            ],
            note="below benchmark" if s["net_sharpe"] < b["net_sharpe"] else None,
            note_tone="neg",
        ),
        t.kpi_pair(
            "Max drawdown",
            *pair("max_drawdown", lambda v: fmt.pct(v, 1)),
            rows=[(bl, fmt.pct(b["net_max_drawdown"], 1), "soft")],
        ),
        t.kpi_pair(
            "Calmar ratio",
            *pair("calmar", lambda v: fmt.number(v, 2)),
            rows=[("Return ÷ max drawdown", note or "saved statistics", "soft")],
        ),
    ]


def equity(c: dict[str, Any]) -> html.Section:
    """Growth of 100, gross, net and the benchmark, in the selected period."""
    lines = views.growth_lines(c["strategy"], c["benchmark"], c["period"])
    totals = [(line.name, line.series.iloc[-1] / 100 - 1) for line in lines]
    return t.panel(
        t.graph(charts.growth(lines, log_y=True), 300, label="Growth of 100"),
        title="Cumulative growth (base = 100)",
        subtitle=views.period_caption(c["strategy"], c["period"]),
        actions=[t.tag("log scale"), views.period_tabs(c["period"])],
        footer=[
            html.Span([f"{name}: ", html.Span(fmt.pct(v, 1, True), className="strong")])
            for name, v in reversed(totals)
        ],
    )


def drawdown_panel(c: dict[str, Any]) -> html.Section:
    """Underwater curve of the net strategy against the benchmark."""
    returns = data.book_returns(c["strategy"], c["period"])
    bench = data.book_returns(c["benchmark"], c["period"])["net"].reindex(returns.index)
    s, b = data.drawdown(returns["net"]), data.drawdown(bench)
    lines = [
        charts.Line(data.BOOKS[c["benchmark"]].label, b, t.BENCHMARK, width=1.2),
        charts.Line(f"{data.BOOKS[c['strategy']].label} net", s, t.NEG, width=1.6, fill="tozeroy"),
    ]
    return t.panel(
        t.graph(charts.drawdowns(lines), 300, label="Drawdown from peak"),
        title="Underwater drawdown",
        subtitle="Net strategy (filled) against the benchmark, from each running peak",
        actions=t.tag("peak-to-trough"),
        footer=[
            html.Span(
                [
                    "Max: ",
                    html.Span(fmt.pct(s.min(), 1), className="neg"),
                    f" ({fmt.date(s.idxmin())})",
                ]
            ),
            html.Span(
                [
                    "Benchmark: ",
                    html.Span(fmt.pct(b.min(), 1), className="strong"),
                    f" ({fmt.date(b.idxmin())})",
                ]
            ),
        ],
    )


def years(c: dict[str, Any]) -> html.Section:
    """Calendar-year decomposition: gross, net, benchmark and excess."""
    frame = data.calendar_years(c["strategy"], c["benchmark"], c["period"])
    rows = [{"year": f"{y}{'*' if r['partial'] else ''}", **r} for y, r in frame.iterrows()]
    whole_s = data.whole_period(data.book_returns(c["strategy"], c["period"])["net"])
    whole_g = data.whole_period(data.book_returns(c["strategy"], c["period"])["gross"])
    whole_b = data.whole_period(data.book_returns(c["benchmark"], c["period"])["net"])
    rows.append(
        {
            "year": "Annualised",
            "gross": whole_g["annual_return"],
            "net": whole_s["annual_return"],
            "benchmark": whole_b["annual_return"],
            "excess": whole_s["annual_return"] - whole_b["annual_return"],
            "net_sharpe": whole_s["sharpe"],
            "total": True,
        }
    )
    columns = [
        t.Col("Year", "year", str, numeric=False),
        t.Col("Gross", "gross", lambda v: fmt.pct(v, 1, True), tone=lambda v, r: fmt.tone(v)),
        t.Col("Net", "net", lambda v: fmt.pct(v, 1, True), tone=lambda v, r: fmt.tone(v)),
        t.Col(data.BOOKS[c["benchmark"]].label, "benchmark", lambda v: fmt.pct(v, 1, True)),
        t.Col("Net excess", "excess", lambda v: fmt.pct(v, 1, True), tone=lambda v, r: fmt.tone(v)),
        t.Col("Net Sharpe", "net_sharpe", fmt.sharpe),
    ]
    return t.panel(
        html.Div(
            t.table(rows, columns, compact=True, highlight=lambda r: bool(r.get("total"))),
            className="scroll-y",
        ),
        title="Annual return decomposition",
        subtitle=f"{views.PERIOD_LABELS[c['period']]} · compounded within each year · * partial",
        actions=t.tag("Units: %"),
    )


def rigor(c: dict[str, Any]) -> html.Section:
    """Execution statistics, the paired interval and the resampling evidence, in plain words."""
    s = data.book_stats(c["strategy"], c["period"])
    interval = data.interval_against(c["strategy"], c["benchmark"], c["period"])
    verdict, tone = views.interval_verdict(interval)
    rows = [
        t.kv(
            "Turnover", f"{fmt.number(s.get('turnover_annual'), 1)}× a year", "strong", "mono-data"
        ),
        t.kv(
            "Mean exposure (net · gross)",
            f"{fmt.pct(s.get('exposure_net_mean'), 0)} · "
            f"{fmt.pct(s.get('exposure_gross_mean'), 0)}",
            "strong",
            "mono-data",
        ),
        t.kv("Sharpe standard error", fmt.sharpe(s.get("sharpe_se")), "strong", "mono-data"),
    ]
    if c["period"] == data.DEVELOPMENT and data.book(c["strategy"]).source == "evaluation":
        mc = data.monte_carlo()
        name = data.book(c["strategy"]).development_name
        if name in mc.index:
            rows.append(
                t.kv(
                    "Resampled paths beating EW hold",
                    fmt.pct(mc.loc[name, "p_beat_equal_weight_hold"], 1),
                    "strong",
                    "mono-data",
                )
            )
        rows.append(
            t.kv("Breakeven cost", fmt.bps(s.get("breakeven_return"), 1), "strong", "mono-data")
        )
    if c["period"] == data.HOLDOUT:
        rows.append(
            t.kv("PSR against 0", fmt.number(s.get("psr_vs_zero"), 3), "strong", "mono-data")
        )
    label = data.BOOKS[c["benchmark"]].label
    if interval:
        conclusion = (
            f"Net Sharpe minus {label}: {views.interval_text(interval)} "
            "(95%, paired block bootstrap). "
            f"The strategy is {verdict} from the benchmark"
            + (
                " — this is the predeclared decision comparison."
                if interval.get("role") == "decision"
                else "."
            )
        )
    else:
        conclusion = (
            f"No paired interval against {label} was saved for this period; "
            "compare the Sharpe ratios only descriptively."
        )
    return t.panel(
        [
            html.Div(rows, className="list-rows"),
            html.Div(
                [
                    html.Span("Conclusion: ", className="caps warn"),
                    html.Span(conclusion, className="mono-sm soft"),
                ],
                style={
                    "marginTop": "8px",
                    "padding": "8px",
                    "background": "var(--surface-low)",
                    "borderRadius": "4px",
                },
            ),
        ],
        title="Execution & statistical rigour",
        actions=[
            t.tag(verdict, tone),
            html.Button(
                [t.icon("download", "sm"), "Daily returns CSV"],
                id="bt-csv",
                className="btn small",
                title="Download this strategy's daily gross and net returns with the benchmark",
            ),
        ],
        footer=[
            html.Span(
                "Trade-level fills were not saved: turnover and exposure come from the "
                "daily weights.",
                className="muted",
            )
        ],
    )


@callback(
    Output("bt-tabs", "children"),
    Output("bt-costs", "children"),
    Output("bt-kpis", "children"),
    Output("bt-equity", "children"),
    Output("bt-drawdown", "children"),
    Output("bt-years", "children"),
    Output("bt-rigor", "children"),
    *INPUTS,
    RELOAD,
)
def update(*values: Any) -> tuple:
    """Show the saved backtest of the selected strategy, benchmark and period."""
    c = selection(*values[:-1])
    return (
        t.guarded(lambda: tabs(c), "the strategy tabs"),
        t.guarded(lambda: costs(c), "the cost panel"),
        t.guarded(lambda: kpis(c), "the gross and net cards"),
        t.guarded(lambda: equity(c), "the growth chart"),
        t.guarded(lambda: drawdown_panel(c), "the drawdown chart"),
        t.guarded(lambda: years(c), "the annual table"),
        t.guarded(lambda: rigor(c), "the rigour panel"),
    )


def daily_frame(c: dict[str, Any]) -> pd.DataFrame:
    """Daily gross and net returns of the strategy with the benchmark's net, as charted."""
    returns = data.book_returns(c["strategy"], c["period"]).add_prefix(f"{c['strategy']}_")
    bench = data.book_returns(c["benchmark"], c["period"])["net"].rename(f"{c['benchmark']}_net")
    return returns.join(bench, how="left")


@callback(
    Output("bt-download", "data"), Input("bt-csv", "n_clicks"), *STATES, prevent_initial_call=True
)
def download_daily(_clicks: int, *values: Any) -> Any:
    """The charted daily returns as CSV."""
    c = selection(*values)
    return dcc.send_data_frame(
        daily_frame(c).to_csv, f"backtest_{c['strategy']}_{c['period']}_daily.csv"
    )


def export(controls: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    """The calendar-year decomposition of the selected strategy and period."""
    frame = data.calendar_years(controls["strategy"], controls["benchmark"], controls["period"])
    return f"backtest_{controls['strategy']}_{controls['period']}_years", frame
