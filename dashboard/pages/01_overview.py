"""Overview: the research question, the headline numbers and the honest verdict.

The verdict panel is built from the holdout's own record (evaluation.json and
the declaration), so it states the predeclared conclusion exactly, whatever it
is. KPI cards follow the global model, strategy, benchmark and period.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from dash import Input, Output, callback, dcc, html

from dashboard import charts, data, views
from dashboard import formatting as fmt
from dashboard import theme as t
from dashboard.controls import INPUTS, MODEL_KEY, MODEL_LABELS, RELOAD, selection

PATH = "/"
TITLE = "Overview"
SUBTITLE = "Executive synthesis"
ICON = "space_dashboard"
USES = {"model", "strategy", "benchmark", "period"}


def layout() -> html.Div:
    """The page skeleton; callbacks fill each block."""
    return html.Div(
        [
            dcc.Download(id="ov-download"),
            html.Div(id="ov-strip"),
            html.Div(id="ov-kpis", className="kpis", style={"--n": 5}),
            html.Div(
                [
                    html.Div(id="ov-verdict", className="c-8 stack-md"),
                    html.Div(id="ov-quality", className="c-4 stack-md"),
                ],
                className="grid",
            ),
            html.Div(
                [
                    html.Div(id="ov-quintile", className="c-6"),
                    html.Div(id="ov-growth", className="c-6"),
                ],
                className="grid",
            ),
            html.Div(id="ov-years"),
            html.Div(id="ov-diag", className="grid"),
        ],
        className="page",
    )


# --- Blocks ------------------------------------------------------------------


def strip() -> html.Div:
    """The research question, its scope and the holdout verdict badge."""
    evaluation = data.holdout_evaluation()
    primary = evaluation["primary"]
    edge = bool(primary["evidence_of_edge"])
    return html.Div(
        [
            html.Div(
                [
                    t.tag("Research question", "info"),
                    html.Div(
                        [
                            html.H1(
                                (
                                    "Do tree-based forecasts of next-day returns beat passive "
                                    "holdings after costs?"
                                ),
                                className="h-sm strong",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        [
                                            "Target: ",
                                            html.Span(
                                                "next-day return (1D horizon)", className="strong"
                                            ),
                                        ]
                                    ),
                                    html.Span(
                                        [
                                            "Universe: ",
                                            html.Span(
                                                f"{len(data.tickers())} tickers incl. SPY",
                                                className="strong",
                                            ),
                                        ]
                                    ),
                                    html.Span(
                                        [
                                            "Trials: ",
                                            html.Span(views.trials_note(), className="strong"),
                                        ]
                                    ),
                                ],
                                className="meta mono-sm",
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "minWidth": 0},
                    ),
                ],
                className="strip-main",
            ),
            html.Div(
                [
                    t.badge(
                        f"Holdout verdict: {primary['conclusion']}",
                        "pos" if edge else "neg",
                        "check_circle" if edge else "gavel",
                    ),
                    html.Div(className="divider-v"),
                    html.Button(
                        [t.icon("description", "sm"), "Holdout report"],
                        id="ov-report",
                        className="btn small",
                        title="Download the holdout attempt's report (Markdown)",
                    ),
                    dcc.Link(
                        [t.icon("rule", "sm"), "Trials audit"],
                        href="/experiments",
                        className="btn small",
                    ),
                ],
                className="panel-actions",
            ),
        ],
        className="strip",
    )


def model_cards(model: str) -> list[Any]:
    """Directional accuracy and IC of the selected forecast, development CPCV."""
    table = data.forecast_comparison()
    if model not in table.index:
        raise data.DataError("Model not evaluated", f"{model} has no saved forecast metrics")
    row = table.loc[model]
    label = MODEL_LABELS.get(model, model)
    accuracy = t.kpi(
        "Directional accuracy",
        fmt.pct(row["hit_rate"], 2),
        "hit rate",
        "radar",
        rows=[
            ("Always-up rate", fmt.pct(row["up_rate"], 2), "soft"),
            (
                "Edge over always-up",
                fmt.pp(row["edge_over_always_up"]),
                fmt.tone(row["edge_over_always_up"]),
            ),
            ("Rows (dev CPCV)", fmt.integer(row["n"]), "soft"),
        ],
        title=f"{label}: share of next-day directions called correctly, development out-of-fold",
    )
    ic = row["ic"]
    information = t.kpi(
        "Information coefficient",
        fmt.number(ic, 4, signed=True),
        "pooled Spearman",
        "insights",
        tone=fmt.tone(ic) if not fmt.is_missing(ic) else "muted",
        rows=[
            ("Cross-sectional IC", fmt.number(row.get("cs_spearman_mean"), 4, signed=True), "soft"),
            ("t-stat (daily IC)", fmt.number(row.get("cs_spearman_t_stat"), 2), "soft"),
            ("Model", label, "soft"),
        ],
        title="Rank correlation of forecast and realised next-day return, development out-of-fold",
    )
    return [accuracy, information]


def strategy_cards(key: str, benchmark: str, period: str) -> list[Any]:
    """Net Sharpe (with the trial count beside it) and drawdown of the selected strategy."""
    s = data.book_stats(key, period)
    b = data.book_stats(benchmark, period)
    label = data.BOOKS[key].label
    sharpe = t.kpi(
        "Net Sharpe ratio",
        fmt.sharpe(s["net_sharpe"]),
        f"ann. net · {views.PERIOD_LABELS[period].lower()}",
        "trending_up",
        tone="primary",
        rows=[
            ("Gross Sharpe", fmt.sharpe(s["gross_sharpe"]), "soft"),
            (
                f"{data.BOOKS[benchmark].label}",
                fmt.sharpe(b["net_sharpe"]),
                views.tone_vs(b["net_sharpe"], s["net_sharpe"]),
            ),
            ("Trials searched", views.trials_note(), "soft"),
        ],
        title=f"{label}: {views.period_caption(key, period)}",
    )
    drawdown = t.kpi(
        "Max drawdown",
        fmt.pct(s["net_max_drawdown"], 1),
        f"net · {label}",
        "waterfall_chart",
        tone="neg",
        rows=[
            ("Gross drawdown", fmt.pct(s["gross_max_drawdown"], 1), "soft"),
            (f"{data.BOOKS[benchmark].label}", fmt.pct(b["net_max_drawdown"], 1), "soft"),
            ("Window", fmt.date_range(s["start"], s["end"]), "soft"),
        ],
        title=views.period_caption(key, period),
    )
    return [sharpe, drawdown]


def dsr_card() -> Any:
    """The primary's deflated Sharpe ratio, holdout and development, against 0.95."""
    evaluation = data.holdout_evaluation()["primary"]
    frozen = data.freeze()["headline"]
    level = data.declaration()["statistics"]["decision_rule"]["dsr_above"]
    passed = bool(evaluation["dsr_above_level"])
    return t.kpi(
        "Deflated Sharpe (primary)",
        fmt.number(evaluation["dsr"], 3),
        "pass" if passed else "fail",
        "rule",
        tone="pos" if passed else "neg",
        rows=[
            ("Required", f"> {level:.2f}", "soft"),
            ("Development DSR", fmt.number(frozen["dsr"], 3), "soft"),
            ("Deflated by", f"N = {evaluation['n_trials']} trials", "soft"),
        ],
        title=f"Primary candidate {data.BOOKS[data.declaration()['candidates']['primary']].label}, "
        "holdout DSR with the frozen N and V[SR]",
    )


def verdict() -> html.Section:
    """The predeclared decision, in sentences built from the holdout record."""
    evaluation = data.holdout_evaluation()
    declaration = data.declaration()
    primary_key = declaration["candidates"]["primary"]
    reference = declaration["references"]["decision"]
    p = evaluation["primary"]
    stats = data.book_stats(primary_key, data.HOLDOUT)
    bench = data.book_stats(reference, data.HOLDOUT)
    window = evaluation["window"]
    comparison = data.forecast_comparison().loc[MODEL_KEY]
    dev = data.book_stats(primary_key, data.DEVELOPMENT)
    sentences = [
        "In the one-time holdout (",
        _num(f"{window['first_row_and_trade']} → {window['last_date']}"),
        "), the predeclared primary, ",
        html.Span(data.BOOKS[primary_key].label, className="strong"),
        ", earned a net Sharpe of ",
        _num(fmt.sharpe(stats["net_sharpe"])),
        (
            f" against {fmt.sharpe(bench['net_sharpe'])} for "
            f"{data.BOOKS[reference].label.lower()} (difference "
        ),
        _num(views.interval_text(p["interval"])),
        "). Its deflated Sharpe ratio was ",
        html.Span(
            fmt.number(p["dsr"], 3), className="num neg" if not p["dsr_above_level"] else "num pos"
        ),
        (
            f" against the {declaration['statistics']['decision_rule']['dsr_above']:.2f} "
            "required, deflated by the frozen "
        ),
        _num(f"N = {p['n_trials']}"),
        " trials. In development it had scored ",
        _num(fmt.sharpe(dev["net_sharpe"])),
        ". The model behind the ML strategies had a pooled IC of ",
        _num(fmt.number(comparison["ic"], 4, signed=True)),
        " and a hit rate of ",
        _num(fmt.pct(comparison["hit_rate"], 2)),
        " against ",
        _num(fmt.pct(comparison["up_rate"], 2)),
        " for always predicting up.",
    ]
    edge = bool(p["evidence_of_edge"])
    started = data.holdout_started()
    return t.panel(
        [
            html.Div(
                [
                    t.icon("gavel", "lg"),
                    html.Span(
                        f"Research verdict: {p['conclusion']}",
                        className=f"caps {'pos' if edge else 'neg'}",
                    ),
                ],
                className="callout-title",
                style={"marginBottom": "8px"},
            ),
            html.P(sentences, className="prose"),
        ],
        cls="high",
        footer=[
            html.Span(
                [
                    "Rule fixed before opening: DSR > 0.95 ",
                    html.Span("and", className="strong"),
                    " interval vs equal-weight hold above 0 · both failed"
                    if not edge
                    else " interval vs equal-weight hold above 0 · both passed",
                ]
            ),
            html.Span(
                [
                    "Runner ",
                    html.Span(started["runner_commit"][:7], className="primary"),
                    " · frozen code ",
                    html.Span(started["frozen_code_version"], className="strong"),
                ]
            ),
        ],
    )


def _num(text: str) -> html.Span:
    return html.Span(text, className="num")


def quality() -> html.Section:
    """Data coverage and the biases the design cannot remove, stated plainly."""
    panel_data = data.panel()
    cells = len(panel_data)
    priced = int(panel_data["adj_close"].notna().sum())
    filled = int(panel_data["filled"].sum())
    gaps = int(panel_data["gap_flag"].sum())
    cfg = data.config()
    p = data.provenance()
    rows = [
        t.kv(
            "Bar coverage",
            f"{fmt.pct(priced / cells, 2)} ({fmt.integer(priced)}/{fmt.integer(cells)})",
            "strong",
        ),
        t.kv(
            "Forward-filled · gaps flagged",
            f"{fmt.integer(filled)} · {fmt.integer(gaps)}",
            "strong",
        ),
        t.kv("Survivorship bias", "Not controlled: hindsight universe", "warn"),
        t.kv("Look-ahead guard", f"{cfg.backtest.execution_lag}-bar execution lag", "pos"),
        t.kv(
            "Cost model",
            f"{cfg.backtest.commission_bps:g} + {cfg.backtest.slippage_bps:g} bps/side, "
            f"{cfg.backtest.borrow_rate_annual:.1%} borrow",
            "strong",
        ),
    ]
    return t.panel(
        [
            html.Div(
                [
                    t.icon("verified", "lg"),
                    html.Span("Data quality & known biases", className="caps pos"),
                ],
                className="callout-title",
                style={"marginBottom": "8px"},
            ),
            html.Div(rows, className="list-rows"),
        ],
        cls="low",
        footer=[
            html.Span("Last price bar"),
            html.Span(fmt.date(p["last_bar"]), className="strong"),
        ],
    )


def quintile_panel(model: str) -> html.Section:
    """Predicted against realised next-day return, by forecast quintile."""
    if model != MODEL_KEY:
        body = t.empty(
            "No continuous forecast for this baseline",
            f"{MODEL_LABELS.get(model, model)} predicts a direction only, so it has no quintiles. "
            "Select the random forest to see the spread.",
        )
        foot = None
    else:
        q = data.forecast_quintiles()
        body = t.graph(charts.quintiles(q), 250, label="Mean realised return by forecast quintile")
        spread = q["realised"].iloc[-1] - q["realised"].iloc[0]
        foot = [
            html.Span(
                [
                    "Q5 − Q1 realised: ",
                    html.Span(fmt.bps(spread, 2, signed=True), className="strong"),
                ]
            ),
            html.Span(
                ["Rows per quintile: ", html.Span(fmt.integer(q["n"].iloc[0]), className="strong")]
            ),
            html.Span("Holdout forecasts were not recorded by the holdout run", className="muted"),
        ]
    return t.panel(
        body,
        title="Forecast quintiles and realised next-day returns",
        subtitle=(
            "Development 2005-07 → 2019-12 · mean of 5 CPCV out-of-fold forecasts per row · 95% CI"
        ),
        actions=t.legend([("Realised", t.PRIMARY, "line"), ("Mean forecast", t.OUTLINE, "dash")]),
        footer=foot,
    )


def growth_panel(key: str, benchmark: str) -> html.Section:
    """Growth of 100 through development and the holdout, gross, net and benchmark."""
    lines, start = views.joined_growth(key, benchmark)
    window = views.holdout_window()
    figure = charts.growth(lines, (start, window[1]) if window else None, log_y=True)
    foot = []
    for period in data.PERIODS:
        try:
            s = data.book_returns(key, period)
            b = data.book_returns(benchmark, period)["net"]
        except data.DataError:
            continue
        net = (1 + s["net"]).prod() - 1
        bench = (1 + b).prod() - 1
        foot.append(
            html.Span(
                [
                    f"{views.PERIOD_LABELS[period]} net ",
                    html.Span(fmt.pct(net, 1, signed=True), className=views.tone_vs(net, bench)),
                    f" vs {fmt.pct(bench, 1, signed=True)}",
                ]
            )
        )
    return t.panel(
        t.graph(figure, 250, label="Growth of 100, development then holdout"),
        title="Growth of 100: gross vs net vs benchmark",
        subtitle=f"{data.BOOKS[key].label} · log scale · development then holdout (shaded)",
        actions=t.legend(
            [
                ("Gross", t.GROSS, "dash"),
                ("Net", t.NET, "line"),
                (data.BOOKS[benchmark].label, t.BENCHMARK, "line"),
            ]
        ),
        footer=foot,
    )


def years_table(key: str, benchmark: str) -> html.Section:
    """Calendar-year returns of the selected strategy in both periods."""
    frame = years_frame(key, benchmark)
    rows = frame.reset_index().to_dict("records")
    columns = [
        t.Col("Year", "label", str, numeric=False),
        t.Col("Period", "period", str, numeric=False),
        t.Col(
            "Gross return", "gross", lambda v: fmt.pct(v, 1, True), tone=lambda v, r: fmt.tone(v)
        ),
        t.Col("Net return", "net", lambda v: fmt.pct(v, 1, True), tone=lambda v, r: fmt.tone(v)),
        t.Col(
            "Cost drag",
            "cost_drag",
            lambda v: fmt.pp(-v),
            tone=lambda v, r: "neg" if v > 0.001 else "muted",
        ),
        t.Col(f"{data.BOOKS[benchmark].label}", "benchmark", lambda v: fmt.pct(v, 1, True)),
        t.Col("Net excess", "excess", lambda v: fmt.pct(v, 1, True), tone=lambda v, r: fmt.tone(v)),
        t.Col("Net Sharpe", "net_sharpe", fmt.sharpe),
        t.Col("Max DD", "net_max_drawdown", lambda v: fmt.pct(v, 1)),
    ]
    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2(
                                "Calendar-year performance and cost erosion",
                                className="h-sm strong",
                            ),
                            t.tag("Yearly granularity"),
                        ],
                        style={"display": "flex", "gap": "8px", "alignItems": "center"},
                    ),
                    html.Div(
                        [
                            html.Span("Strategy: ", className="muted"),
                            html.Span(data.BOOKS[key].label, className="strong"),
                            html.Span("  |  Benchmark: ", className="muted"),
                            html.Span(data.BOOKS[benchmark].label, className="strong"),
                        ],
                        className="mono-sm",
                    ),
                ],
                className="table-head",
            ),
            t.table(rows, columns, highlight=lambda r: r["period"] == "Holdout"),
            html.Div(
                [
                    html.Span(
                        "Returns compounded within each year, not annualised; partial years "
                        "are marked *."
                    ),
                    html.Span(
                        "Development rows: CPCV path mean (ML, rules) or walk-forward "
                        "(allocations)."
                    ),
                ],
                className="table-foot mono-sm",
            ),
        ],
        className="panel flush",
    )


def years_frame(key: str, benchmark: str) -> pd.DataFrame:
    """Both periods' calendar years, labelled, as the table and the export show them."""
    frames = []
    for period in data.PERIODS:
        try:
            frame = data.calendar_years(key, benchmark, period).copy()
        except data.DataError:
            continue
        frame["period"] = views.PERIOD_LABELS[period]
        frames.append(frame)
    if not frames:
        raise data.DataError("No returns", f"{data.BOOKS[key].label} has no saved returns")
    out = pd.concat(frames)
    out["label"] = [
        f"{y}{'*' if p else ''}" for y, p in zip(out.index, out["partial"], strict=True)
    ]
    return out


def diagnostics() -> list[Any]:
    """Trials, validation design and the holdout attempt, one line each."""
    frozen = data.freeze()
    metrics = data.research_metrics()
    started = data.holdout_started()
    complete = data.holdout_complete()

    def diag(icon_name: str, label: str, main: str, side: str, tone: str) -> html.Div:
        return html.Div(
            html.Div(
                [
                    html.Div(
                        [
                            t.icon(icon_name),
                            html.Div(
                                [
                                    html.Span(label, className="caps muted"),
                                    html.Span(main, className="mono-data strong"),
                                ],
                                style={"display": "flex", "flexDirection": "column"},
                            ),
                        ],
                        className="diag-main",
                    ),
                    html.Span(side, className=f"mono-sm {tone}"),
                ],
                className="diag",
            ),
            className="c-4",
        )

    headline = frozen["headline"]
    return [
        diag(
            "tune",
            "Trials (frozen before the holdout)",
            f"N = {headline['n_trials']} · luck threshold {headline['luck_threshold_annual']:.2f}",
            f"dev DSR {headline['dsr']:.3f}",
            "neg",
        ),
        diag(
            "account_tree",
            "Validation",
            f"CPCV {int(metrics['n_splits'])} splits · {int(metrics['n_paths'])} paths · "
            f"{int(metrics['embargo_days'])}-day embargo",
            "purged, embargoed",
            "pos",
        ),
        diag(
            "lock",
            "Holdout",
            f"Attempt {started['attempt']['number']} · runner {started['runner_commit'][:7]}",
            complete["conclusion"],
            "soft",
        ),
    ]


# --- Callbacks --------------------------------------------------------------------


@callback(
    Output("ov-strip", "children"),
    Output("ov-verdict", "children"),
    Output("ov-quality", "children"),
    Output("ov-diag", "children"),
    RELOAD,
)
def static_blocks(_token: int) -> tuple:
    """Blocks that change only when the saved results change."""
    return (
        t.guarded(strip, "the research strip"),
        t.guarded(verdict, "the verdict"),
        t.guarded(quality, "the data quality panel"),
        t.guarded(diagnostics, "the diagnostics row"),
    )


@callback(Output("ov-kpis", "children"), *INPUTS, RELOAD)
def kpis(*values: Any) -> list:
    """The five KPI cards for the selected model, strategy, benchmark and period."""
    c = selection(*values[:-1])
    cards: list[Any] = []
    for build, where in (
        (lambda: model_cards(c["model"]), "the model cards"),
        (lambda: strategy_cards(c["strategy"], c["benchmark"], c["period"]), "the strategy cards"),
        (lambda: [dsr_card()], "the DSR card"),
    ):
        built = t.guarded(build, where)
        cards.extend(built if isinstance(built, list) else [built])
    return cards


@callback(
    Output("ov-quintile", "children"),
    Output("ov-growth", "children"),
    Output("ov-years", "children"),
    *INPUTS,
    RELOAD,
)
def charts_and_table(*values: Any) -> tuple:
    """The two charts and the year table, redrawn when their inputs change."""
    c = selection(*values[:-1])
    return (
        t.guarded(lambda: quintile_panel(c["model"]), "the quintile chart"),
        t.guarded(lambda: growth_panel(c["strategy"], c["benchmark"]), "the growth chart"),
        t.guarded(lambda: years_table(c["strategy"], c["benchmark"]), "the year table"),
    )


@callback(Output("ov-download", "data"), Input("ov-report", "n_clicks"), prevent_initial_call=True)
def download_report(_clicks: int) -> Any:
    """The holdout attempt's report as written by the run."""
    text = data.text_file(data.HOLDOUT_ATTEMPT / "report.md", "Holdout report")
    return {"content": text, "filename": "holdout_attempt-1_report.md"}


def export(controls: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    """The calendar-year table for the selected strategy and benchmark."""
    frame = years_frame(controls["strategy"], controls["benchmark"])
    return f"overview_years_{controls['strategy']}", frame.drop(columns=["label"])
