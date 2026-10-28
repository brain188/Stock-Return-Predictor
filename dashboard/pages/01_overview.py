"""Overview: the research question, the honest verdict, the headline numbers and the evidence.

Reading order: the question and the holdout verdict, the decision rule that
produced it, the headline figures for the selected model and strategy, the
two charts behind them, then year-by-year detail and the method. The verdict
is built from the holdout's own record (evaluation.json and the declaration),
so it states the predeclared conclusion exactly, whatever it is.
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
SUBTITLE = "Question, verdict, evidence"
ICON = "space_dashboard"
USES = {"model", "strategy", "benchmark", "period"}
QUESTION = "Do tree-based forecasts of next-day returns beat passive holdings after costs?"


def layout() -> html.Div:
    """The page skeleton; callbacks fill each block."""
    return html.Div(
        [
            dcc.Download(id="ov-download"),
            html.Div(id="ov-strip"),
            html.Div(
                [
                    html.Div(id="ov-verdict", className="c-8 stack-md"),
                    html.Div(id="ov-quality", className="c-4 stack-md"),
                ],
                className="grid",
            ),
            html.Div(id="ov-kpis", className="kpis", style={"--n": 5}),
            html.Div(
                [
                    html.Div(id="ov-quintile", className="c-6"),
                    html.Div(id="ov-growth", className="c-6"),
                ],
                className="grid",
            ),
            html.Div(id="ov-years"),
            html.Div(id="ov-diag"),
        ],
        className="page",
    )


# --- Blocks ------------------------------------------------------------------


def strip() -> html.Header:
    """The research question, its scope and the holdout verdict."""
    evaluation = data.holdout_evaluation()
    primary = evaluation["primary"]
    edge = bool(primary["evidence_of_edge"])
    window = evaluation["window"]
    return t.page_header(
        QUESTION,
        eyebrow="Research question",
        meta=[
            ("Target", "next-day return, 1-day horizon"),
            ("Universe", f"{len(data.tickers())} US tickers incl. SPY"),
            ("Trials searched", f"N = {primary['n_trials']} (frozen)"),
            ("Holdout", f"{window['first_row_and_trade']} → {window['last_date']}"),
        ],
        status=t.badge(
            f"Holdout verdict: {primary['conclusion']}",
            "pos" if edge else "neg",
            "check_circle" if edge else "gavel",
        ),
        actions=[
            html.Button(
                [t.icon("description", "sm"), "Holdout report"],
                id="ov-report",
                className="btn small",
                title="Download the holdout attempt's report (Markdown)",
            ),
            dcc.Link(
                [t.icon("rule", "sm"), "Trials audit"], href="/experiments", className="btn small"
            ),
        ],
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
        "Hit rate · development out-of-fold",
        rows=[
            ("Always-up rate", fmt.pct(row["up_rate"], 2), "soft"),
            (
                "Edge over always-up",
                fmt.pp(row["edge_over_always_up"]),
                fmt.tone(row["edge_over_always_up"]),
            ),
            ("Rows scored", fmt.integer(row["n"]), "soft"),
        ],
        title=f"{label}: share of next-day directions called correctly",
    )
    ic = row["ic"]
    information = t.kpi(
        "Information coefficient",
        fmt.number(ic, 4, signed=True),
        "Pooled rank IC · development out-of-fold",
        rows=[
            ("Cross-sectional IC", fmt.number(row.get("cs_spearman_mean"), 4, signed=True), "soft"),
            ("t-stat of daily IC", fmt.number(row.get("cs_spearman_t_stat"), 2), "soft"),
            ("Positive days", fmt.pct(row.get("cs_spearman_positive_share"), 1), "soft"),
        ],
        title="Rank correlation of forecast and realised next-day return",
    )
    return [accuracy, information]


def strategy_cards(key: str, benchmark: str, period: str) -> list[Any]:
    """Net Sharpe (with the trial count beside it) and drawdown of the selected strategy."""
    s = data.book_stats(key, period)
    b = data.book_stats(benchmark, period)
    label = data.BOOKS[key].label
    period_label = views.PERIOD_LABELS[period]
    sharpe = t.kpi(
        "Net Sharpe ratio",
        fmt.sharpe(s["net_sharpe"]),
        f"{label} · {period_label.lower()}, annualised",
        rows=[
            ("Gross Sharpe", fmt.sharpe(s["gross_sharpe"]), "soft"),
            (
                data.BOOKS[benchmark].label,
                fmt.sharpe(b["net_sharpe"]),
                views.tone_vs(b["net_sharpe"], s["net_sharpe"]),
            ),
            ("Trials searched", views.trials_note(), "soft"),
        ],
        title=views.period_caption(key, period),
    )
    drawdown = t.kpi(
        "Maximum drawdown",
        fmt.pct(s["net_max_drawdown"], 1),
        f"{label} · net, {period_label.lower()}",
        rows=[
            ("Gross drawdown", fmt.pct(s["gross_max_drawdown"], 1), "soft"),
            (data.BOOKS[benchmark].label, fmt.pct(b["net_max_drawdown"], 1), "soft"),
            ("Window", fmt.date_range(s["start"], s["end"]), "soft"),
        ],
        title=views.period_caption(key, period),
    )
    return [sharpe, drawdown]


def dsr_card() -> Any:
    """The primary's deflated Sharpe ratio, holdout and development, against the bar."""
    evaluation = data.holdout_evaluation()["primary"]
    frozen = data.freeze()["headline"]
    level = data.declaration()["statistics"]["decision_rule"]["dsr_above"]
    passed = bool(evaluation["dsr_above_level"])
    primary = data.BOOKS[data.declaration()["candidates"]["primary"]].label
    return t.kpi(
        "Deflated Sharpe ratio",
        fmt.number(evaluation["dsr"], 3),
        f"{primary} · holdout, primary candidate",
        badge_text=("Pass", "pos") if passed else ("Fail", "neg"),
        rows=[
            ("Required", f"> {level:.2f}", "soft"),
            ("Development DSR", fmt.number(frozen["dsr"], 3), "soft"),
            ("Deflated by", f"N = {evaluation['n_trials']} trials", "soft"),
        ],
        title="Holdout DSR with the frozen N and V[SR]; not a probability of future profit",
    )


def _num(text: str, tone: str = "") -> html.Span:
    return html.Span(text, className=f"num {tone}".strip())


def verdict() -> html.Section:
    """The predeclared decision: the rule, what was observed, and what it means."""
    evaluation = data.holdout_evaluation()
    declaration = data.declaration()
    primary_key = declaration["candidates"]["primary"]
    reference = declaration["references"]["decision"]
    rule = declaration["statistics"]["decision_rule"]
    p = evaluation["primary"]
    stats = data.book_stats(primary_key, data.HOLDOUT)
    bench = data.book_stats(reference, data.HOLDOUT)
    dev = data.book_stats(primary_key, data.DEVELOPMENT)
    edge = bool(p["evidence_of_edge"])
    met = int(bool(p["dsr_above_level"])) + int(bool(p["interval_entirely_above_zero"]))
    outcome = {0: "Neither condition", 1: "Only one of the two conditions", 2: "Both conditions"}[
        met
    ]
    sentence = [
        "In the one-time holdout the predeclared primary, ",
        html.Strong(data.BOOKS[primary_key].label),
        ", earned a net Sharpe of ",
        _num(fmt.sharpe(stats["net_sharpe"])),
        f" against {fmt.sharpe(bench['net_sharpe'])} for {data.BOOKS[reference].label.lower()}; "
        "in development it had scored ",
        _num(fmt.sharpe(dev["net_sharpe"])),
        f". {outcome} of the rule fixed before opening "
        f"{'were' if met == 2 else 'was'} met, so the conclusion is ",
        html.Strong(p["conclusion"]),
        ".",
    ]
    rows = [
        {
            "condition": f"Deflated Sharpe ratio, N = {p['n_trials']} trials",
            "required": f"> {rule['dsr_above']:.2f}",
            "observed": fmt.number(p["dsr"], 3),
            "met": bool(p["dsr_above_level"]),
        },
        {
            "condition": "95% paired interval, net Sharpe minus "
            f"{data.BOOKS[reference].label.lower()}",
            "required": "entirely > 0",
            "observed": views.interval_text(p["interval"]),
            "met": bool(p["interval_entirely_above_zero"]),
        },
    ]
    columns = [
        t.Col("Condition", "condition", str, numeric=False),
        t.Col("Required", "required", str),
        t.Col("Observed", "observed", str),
        t.Col(
            "Met", "met", lambda v: "Yes" if v else "No", tone=lambda v, r: "pos" if v else "neg"
        ),
    ]
    started = data.holdout_started()
    window = evaluation["window"]
    return t.panel(
        [
            html.P(sentence, className="prose"),
            html.Div(t.table(rows, columns, compact=True), style={"marginTop": "12px"}),
        ],
        title="Research verdict",
        subtitle=f"Decision rule approved before the holdout was opened · "
        f"{window['first_row_and_trade']} → {window['last_date']}, attempt 1",
        cls="callout pos" if edge else "callout",
        footer=[
            html.Span(["Runner ", _num(started["runner_commit"][:7])]),
            html.Span(["Frozen code ", _num(started["frozen_code_version"])]),
            html.Span("The DSR is not a probability of future profit."),
        ],
    )


def quality() -> html.Section:
    """Data coverage and the biases the design cannot remove, stated plainly."""
    panel_data = data.panel()
    cells = len(panel_data)
    priced = int(panel_data["adj_close"].notna().sum())
    cfg = data.config()
    p = data.provenance()
    rows = [
        t.kv("Bar coverage", f"{fmt.pct(priced / cells, 2)} of {fmt.integer(cells)}", "strong"),
        t.kv(
            "Forward-filled · gaps flagged",
            f"{fmt.integer(panel_data['filled'].sum())} · "
            f"{fmt.integer(panel_data['gap_flag'].sum())}",
            "strong",
        ),
        t.kv("Last price bar", fmt.date(p["last_bar"]), "strong"),
        t.kv("Execution", f"{cfg.backtest.execution_lag}-bar lag after each signal", "strong"),
        t.kv(
            "Costs",
            f"{cfg.backtest.commission_bps:g} + {cfg.backtest.slippage_bps:g} bps per side, "
            f"{cfg.backtest.borrow_rate_annual:.1%} borrow",
            "strong",
        ),
        t.kv("Survivorship", "Not controlled: hindsight universe", "warn"),
        t.kv("Earlier exploration", "Notebooks saw 2008–2024 data", "warn"),
    ]
    return t.panel(
        html.Div(rows, className="list-rows"),
        title="Data quality and known biases",
        subtitle="Amber rows limit what any result here can claim",
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
        body = t.graph(charts.quintiles(q), 260, label="Mean realised return by forecast quintile")
        spread = q["realised"].iloc[-1] - q["realised"].iloc[0]
        foot = [
            html.Span(["Q5 − Q1 realised ", _num(fmt.bps(spread, 2, signed=True))]),
            html.Span(["Rows per quintile ", _num(fmt.integer(q["n"].iloc[0]))]),
            html.Span("Holdout forecasts were not recorded by the holdout run."),
        ]
    return t.panel(
        body,
        title="Forecast quintiles and realised next-day return",
        subtitle="Development 2005-07 → 2019-12 · mean of 5 out-of-fold forecasts per row · 95% CI",
        legend=t.legend([("Realised", t.PRIMARY, "line"), ("Mean forecast", t.OUTLINE, "dash")]),
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
                    f"{views.PERIOD_LABELS[period]}: net ",
                    _num(fmt.pct(net, 1, signed=True), views.tone_vs(net, bench)),
                    " vs benchmark ",
                    _num(fmt.pct(bench, 1, signed=True)),
                ]
            )
        )
    return t.panel(
        t.graph(figure, 260, label="Growth of 100, development then holdout"),
        title="Growth of 100: gross, net and benchmark",
        subtitle=f"{data.BOOKS[key].label} · log scale · development, then the holdout (shaded)",
        legend=t.legend(
            [
                ("Net", t.NET, "line"),
                ("Gross", t.GROSS, "dash"),
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
        t.Col("Gross", "gross", lambda v: fmt.pct(v, 1, True), tone=lambda v, r: fmt.tone(v)),
        t.Col("Net", "net", lambda v: fmt.pct(v, 1, True), tone=lambda v, r: fmt.tone(v)),
        t.Col("Cost drag", "cost_drag", lambda v: fmt.pp(-v)),
        t.Col(
            "Benchmark",
            "benchmark",
            lambda v: fmt.pct(v, 1, True),
            title=data.BOOKS[benchmark].label,
        ),
        t.Col("Net excess", "excess", lambda v: fmt.pct(v, 1, True), tone=lambda v, r: fmt.tone(v)),
        t.Col("Net Sharpe", "net_sharpe", fmt.sharpe),
        t.Col("Max drawdown", "net_max_drawdown", lambda v: fmt.pct(v, 1)),
    ]
    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2(
                                "Calendar-year performance and cost erosion",
                                className="section-title",
                            ),
                            html.Span(
                                f"{data.BOOKS[key].label} against "
                                f"{data.BOOKS[benchmark].label.lower()} · "
                                "holdout rows highlighted",
                                className="caption",
                            ),
                        ],
                        className="titles",
                    ),
                ],
                className="table-head",
            ),
            html.Div(
                t.table(rows, columns, highlight=lambda r: r["period"] == "Holdout"),
                className="scroll-y",
            ),
            html.Div(
                [
                    html.Span(
                        "Returns are compounded within each calendar year, not annualised; "
                        "* marks a partial year."
                    ),
                    html.Span(
                        "Development rows: mean of 5 CPCV paths (ML, rules) or one walk-forward "
                        "path (allocations)."
                    ),
                ],
                className="table-foot",
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


def diagnostics() -> html.Section:
    """How the evidence was produced: trial count, validation design, holdout run."""
    headline = data.freeze()["headline"]
    metrics = data.research_metrics()
    started = data.holdout_started()
    complete = data.holdout_complete()
    return t.panel(
        t.stat_grid(
            [
                t.stat(
                    "Trials searched (frozen)",
                    f"N = {headline['n_trials']}",
                    f"Luck threshold: annual Sharpe {headline['luck_threshold_annual']:.2f}",
                ),
                t.stat(
                    "Validation",
                    f"CPCV · {int(metrics['n_splits'])} splits",
                    f"{int(metrics['n_paths'])} paths · "
                    f"{int(metrics['embargo_days'])}-day embargo · purged",
                ),
                t.stat(
                    "Holdout evaluation",
                    f"Attempt {started['attempt']['number']} · {started['runner_commit'][:7]}",
                    f"Completed {complete['completed_at'][:10]} · {complete['conclusion']}",
                ),
            ]
        ),
        title="Method",
        subtitle="What every number on this page rests on",
    )


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
        t.guarded(strip, "the page header"),
        t.guarded(verdict, "the verdict"),
        t.guarded(quality, "the data quality panel"),
        t.guarded(diagnostics, "the method panel"),
    )


@callback(Output("ov-kpis", "children"), *INPUTS, RELOAD)
def kpis(*values: Any) -> list:
    """The five headline figures for the selected model, strategy, benchmark and period."""
    c = selection(*values[:-1])
    cards: list[Any] = []
    for build, where in (
        (lambda: model_cards(c["model"]), "the model figures"),
        (
            lambda: strategy_cards(c["strategy"], c["benchmark"], c["period"]),
            "the strategy figures",
        ),
        (lambda: [dsr_card()], "the DSR figure"),
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
