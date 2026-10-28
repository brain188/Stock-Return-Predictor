"""Experiments registry: every logged trial, the frozen trial count, the luck threshold, files.

The trial count, V[SR] and luck threshold shown are the frozen values from
holdout_freeze.json. Filtering the log below changes only which rows are
listed; it never changes the official DSR inputs, which were fixed before the
holdout was opened.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from dash import Input, Output, State, callback, ctx, dash_table, dcc, html, no_update

from dashboard import charts, data, views
from dashboard import formatting as fmt
from dashboard import theme as t
from dashboard.controls import RELOAD

PATH = "/experiments"
TITLE = "Experiments Registry"
SUBTITLE = "Trials audit & DSR"
ICON = "science"
USES: set[str] = set()
SEARCH_HINT = "Filter name, fingerprint, notes"
KEY_METRICS = ("metric_sharpe", "metric_holdout_sharpe_net", "metric_ic_mean", "metric_ic")
DOWNLOADS = {
    "ex-dl-log": ("trial_log.csv", lambda: data.config().evaluation.trial_log),
    "ex-dl-report": ("research_report.md", lambda: data.RESEARCH_REPORT),
    "ex-dl-holdout": ("holdout_attempt-1_report.md", lambda: data.HOLDOUT_ATTEMPT / "report.md"),
    "ex-dl-freeze": ("holdout_freeze.json", lambda: data.FREEZE),
    "ex-dl-declaration": ("holdout_declaration.json", lambda: data.DECLARATION),
}
TABLE_STYLE = {
    "style_table": {"overflowX": "auto", "maxHeight": "460px", "overflowY": "auto"},
    "style_header": {
        "backgroundColor": t.SURFACE_LOW,
        "color": t.OUTLINE,
        "fontFamily": t.FONT_MONO,
        "fontSize": "10px",
        "fontWeight": 600,
        "letterSpacing": "0.06em",
        "textTransform": "uppercase",
        "border": "none",
        "padding": "6px 8px",
    },
    "style_cell": {
        "backgroundColor": t.SURFACE,
        "color": t.ON_SURFACE,
        "fontFamily": t.FONT_MONO,
        "fontSize": "12px",
        "border": "none",
        "borderTop": f"1px solid {t.SURFACE_HIGHEST}",
        "padding": "5px 8px",
        "textAlign": "left",
        "maxWidth": "280px",
        "overflow": "hidden",
        "textOverflow": "ellipsis",
    },
    "style_cell_conditional": [
        {"if": {"column_id": c}, "textAlign": "right"} for c in ("trial_id", "metric")
    ],
    "style_data_conditional": [
        {
            "if": {"state": "active"},
            "backgroundColor": t.SURFACE_HIGH,
            "border": f"1px solid {t.PRIMARY_CONTAINER}",
        },
        {"if": {"filter_query": '{kind} = "holdout"'}, "color": t.PRIMARY},
        {"if": {"filter_query": '{status} = "failed"'}, "color": t.NEG},
    ],
}


def layout() -> html.Div:
    """The page skeleton with the log's own filters."""
    return html.Div(
        [
            dcc.Download(id="ex-download"),
            html.Div(id="ex-head"),
            html.Div(id="ex-banner"),
            html.Div(id="ex-kpis", className="kpis", style={"--n": 4}),
            html.Div(
                [
                    html.Div(
                        html.Section(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H2(
                                                    "Trial execution log", className="h-sm strong"
                                                ),
                                                html.Span(id="ex-count", className="tag"),
                                            ],
                                            style={
                                                "display": "flex",
                                                "gap": "8px",
                                                "alignItems": "center",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    [
                                                        html.Span(
                                                            "Filter trials", className="sr-only"
                                                        ),
                                                        dcc.Input(
                                                            id="ex-search",
                                                            type="search",
                                                            placeholder=SEARCH_HINT,
                                                            debounce=True,
                                                            className="search",
                                                        ),
                                                    ]
                                                ),
                                                dcc.Dropdown(
                                                    id="ex-kind",
                                                    placeholder="All kinds",
                                                    clearable=True,
                                                    searchable=False,
                                                    style={"minWidth": "140px"},
                                                ),
                                            ],
                                            className="chip",
                                            style={
                                                "flexDirection": "row",
                                                "alignItems": "center",
                                                "gap": "8px",
                                                "minHeight": "32px",
                                            },
                                        ),
                                    ],
                                    className="table-head",
                                ),
                                dash_table.DataTable(
                                    id="ex-table",
                                    columns=[
                                        {"name": "ID", "id": "trial_id"},
                                        {"name": "Date", "id": "date"},
                                        {"name": "Kind", "id": "kind"},
                                        {"name": "Name", "id": "name"},
                                        {"name": "Status", "id": "status"},
                                        {"name": "Key metric", "id": "metric"},
                                        {"name": "Code", "id": "code_version"},
                                    ],
                                    page_size=14,
                                    page_action="native",
                                    sort_action="native",
                                    cell_selectable=True,
                                    **TABLE_STYLE,
                                ),
                                html.Div(id="ex-table-foot", className="table-foot mono-sm"),
                            ],
                            className="panel flush",
                        ),
                        className="c-8 stack-md",
                    ),
                    html.Div(id="ex-inspector", className="c-4 stack-md"),
                ],
                className="grid",
            ),
            html.Div(
                [
                    html.Div(id="ex-counts", className="c-6"),
                    html.Div(id="ex-variance", className="c-6"),
                ],
                className="grid",
            ),
            html.Div(id="ex-bundle"),
        ],
        className="page",
    )


def head() -> html.Div:
    """Breadcrumb, title and the identifiers that pin the audit."""
    freeze = data.freeze()
    declaration = data.declaration()
    started = data.holdout_started()
    return html.Div(
        [
            html.Div(
                [
                    html.Span("Governance", className="caps muted"),
                    html.Span(" / ", className="muted"),
                    html.Span(
                        f"Protocol approved {declaration['approval']['approved_on']}",
                        className="caps primary",
                    ),
                    html.Span(" / ", className="muted"),
                    html.Span(f"Trials frozen at {freeze['code_version']}", className="caps pos"),
                ]
            ),
            html.H1(
                "Trial registry & multiple-testing governance",
                className="h-lg strong",
                style={"margin": "4px 0"},
            ),
            html.Div(
                [
                    t.tag(
                        f"freeze {started['preflight']['freeze_sha256'][:12]}…",
                        "",
                        "SHA-256 of the committed freeze file",
                    ),
                    t.tag(
                        f"declaration {started['declaration_sha256'][:12]}…",
                        "",
                        "SHA-256 of the approved declaration",
                    ),
                    t.tag(f"config {freeze['config_hash']}", ""),
                ],
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
            ),
        ]
    )


def banner() -> html.Div:
    """The multiple-testing defence, stated with the frozen numbers."""
    freeze = data.freeze()
    h = freeze["headline"]
    p = data.holdout_evaluation()["primary"]
    counts = freeze["count_scenarios"]
    scenario = counts["all_fingerprints"]
    return html.Div(
        [
            html.Div(t.icon("shield", "lg"), className="brand-mark", style={"flexShrink": 0}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Multiple-testing defence", className="caps neg"),
                            html.Span(" | ", className="muted"),
                            html.Span(
                                f"Trials counted: N = {h['n_trials']} "
                                f"({scenario['pipeline']} pipeline + "
                                f"{scenario['notebook']} notebook)",
                                className="mono-sm strong",
                            ),
                        ]
                    ),
                    html.P(
                        [
                            html.Span(
                                "Deflated Sharpe ratio (Bailey & López de Prado). ",
                                className="strong",
                            ),
                            "The best of ",
                            html.Span(str(h["n_trials"]), className="num"),
                            " unskilled trials with V[SR] = ",
                            html.Span(f"{h['variance_used_annual']:.3f}", className="num"),
                            " would be expected to show an annual Sharpe of ",
                            html.Span(f"{h['luck_threshold_annual']:.2f}", className="num"),
                            (
                                ". The best development candidate "
                                f"({data.BOOKS[h['selected']].label}, "
                            ),
                            html.Span(f"{h['sharpe_annual']:.3f}", className="num"),
                            ") had DSR ",
                            html.Span(f"{h['dsr']:.3f}", className="num neg"),
                            "; in the holdout it scored ",
                            html.Span(f"{p['sharpe_annual']:.3f}", className="num"),
                            ", DSR ",
                            html.Span(f"{p['dsr']:.3f}", className="num neg"),
                            ". Both are far below the 0.95 required.",
                        ],
                        className="prose body-sm",
                    ),
                ],
                style={"flex": 1, "minWidth": 0},
            ),
            html.Div(
                [
                    html.Span("Count scenarios", className="caps muted"),
                    html.Span(
                        " / ".join(str(v["n_trials"]) for v in counts.values()),
                        className="mono-display strong",
                    ),
                    html.Span("headline uses the largest", className="mono-sm muted"),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "flex-end",
                    "flexShrink": 0,
                },
            ),
        ],
        className="panel high",
        style={
            "flexDirection": "row",
            "gap": "12px",
            "alignItems": "center",
            "borderLeft": f"2px solid {t.NEG}",
        },
    )


def kpis() -> list[Any]:
    """Luck threshold, the DSR bar, the development-to-holdout decay and the trials' spread."""
    freeze = data.freeze()
    h = freeze["headline"]
    p = data.holdout_evaluation()["primary"]
    decay = p["sharpe_annual"] / h["sharpe_annual"] - 1
    sharpes = data.backtest_trial_sharpes()
    figure = charts.trial_sharpes(
        sharpes, h["luck_threshold_annual"], [("best dev", h["sharpe_annual"], t.PRIMARY)]
    )
    figure.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    figure.update_xaxes(title_text=None)
    figure.update_yaxes(title_text=None, showticklabels=False)
    return [
        t.kpi(
            "Expected max Sharpe (null)",
            fmt.number(h["luck_threshold_annual"], 3),
            "annual",
            "insights",
            rows=[
                ("Trials N", str(h["n_trials"]), "soft"),
                ("V[SR] used", f"{h['variance_used_annual']:.4f} ({h['variance_used']})", "soft"),
            ],
        ),
        t.kpi(
            "Deflated Sharpe bar",
            "> 0.95",
            "required to claim an edge",
            "rule",
            tone="primary",
            rows=[
                ("Development DSR", fmt.number(h["dsr"], 3), "neg"),
                ("Holdout DSR", fmt.number(p["dsr"], 3), "neg"),
            ],
        ),
        t.kpi(
            "Development → holdout",
            fmt.pct(decay, 1, True),
            "primary's net Sharpe",
            "trending_down",
            tone=fmt.tone(decay),
            rows=[
                ("Development", fmt.sharpe(h["sharpe_annual"]), "soft"),
                ("Holdout", fmt.sharpe(p["sharpe_annual"]), "soft"),
            ],
        ),
        html.Div(
            [
                html.Div(
                    [html.Span("Trial Sharpe spread", className="caps"), t.icon("analytics", "sm")],
                    className="kpi-top",
                ),
                dcc.Graph(
                    figure=figure,
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": "70px"},
                ),
                t.kv(
                    f"{len(sharpes)} backtest trials",
                    f"threshold {h['luck_threshold_annual']:.2f}",
                    "neg",
                ),
            ],
            className="kpi",
        ),
    ]


def log_rows() -> pd.DataFrame:
    """The trial log, one row per execution, with each row's key metric."""
    table = data.trial_table()
    metric = pd.Series(pd.NA, index=table.index, dtype="object")
    for column in KEY_METRICS:
        if column in table:
            label = column.removeprefix("metric_")
            values = table[column].map(
                lambda v, label=label: f"{label} {v:+.3f}" if pd.notna(v) else pd.NA
            )
            metric = metric.fillna(values)
    out = pd.DataFrame(
        {
            "trial_id": table["trial_id"],
            "date": pd.to_datetime(table["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "kind": table["kind"],
            "name": table["name"],
            "status": table["status"],
            "metric": metric.fillna(""),
            "code_version": table["code_version"],
            "fingerprint": table["fingerprint"],
            "notes": table["notes"],
        }
    )
    return out


def filtered(search: str | None, kind: str | None) -> pd.DataFrame:
    """The log rows matching the filters (display only)."""
    rows = log_rows()
    if kind:
        rows = rows[rows["kind"] == kind]
    if search:
        text = search.strip().lower()
        hay = (rows["name"] + " " + rows["fingerprint"] + " " + rows["notes"]).str.lower()
        rows = rows[hay.str.contains(text, regex=False)]
    return rows


def inspector(trial_id: int | None) -> html.Section:
    """Everything the log recorded for one execution."""
    table = data.trial_table()
    if trial_id is None:
        trial_id = int(table["trial_id"].iloc[-1])
    row = table[table["trial_id"] == trial_id]
    if row.empty:
        raise data.DataError("Trial not found", f"no trial {trial_id} in the log")
    r = row.iloc[0]
    metrics = {
        k.removeprefix("metric_"): v
        for k, v in r.items()
        if k.startswith("metric_") and pd.notna(v)
    }
    details = {"model_params": r["model_params"], "metrics": metrics, "features": r["features"]}
    tone = "pos" if r["status"] == "ok" else "neg"
    return t.panel(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Kind", className="caps muted"),
                            html.Span(r["kind"], className="mono-data strong"),
                        ],
                        className="kpi",
                        style={"background": "var(--surface-low)"},
                    ),
                    html.Div(
                        [
                            html.Span("Status", className="caps muted"),
                            html.Span(r["status"], className=f"mono-data {tone}"),
                        ],
                        className="kpi",
                        style={"background": "var(--surface-low)"},
                    ),
                ],
                className="kpis",
                style={"--n": 2},
            ),
            html.Div(
                [
                    t.kv("Name", r["name"], "strong"),
                    t.kv("Logged", str(r["timestamp"]), "strong"),
                    t.kv("Fingerprint", r["fingerprint"] or fmt.MISSING, "strong"),
                    t.kv(
                        "Code version",
                        r["code_version"],
                        "warn" if "dirty" in str(r["code_version"]) else "strong",
                    ),
                    t.kv("Config hash", r["config_hash"] or fmt.MISSING, "strong"),
                ],
                className="list-rows",
                style={"margin": "8px 0"},
            ),
            html.P(
                r["notes"] or "No notes.",
                className="body-sm soft",
                style={"fontStyle": "italic", "marginBottom": "8px"},
            ),
            html.Div(
                [
                    html.Span("Recorded settings & metrics", className="caps muted"),
                    html.Span("trial_log.csv", className="mono-sm primary"),
                ],
                className="kv",
            ),
            html.Pre(
                json.dumps(details, indent=2, default=str),
                className="mono-sm",
                style={
                    "background": "var(--surface-lowest)",
                    "padding": "8px",
                    "borderRadius": "4px",
                    "maxHeight": "260px",
                    "overflow": "auto",
                    "margin": "4px 0 0",
                    "color": "var(--on-surface-variant)",
                    "whiteSpace": "pre-wrap",
                },
            ),
        ],
        title_icon="dataset",
        title=f"Trial inspector: #{trial_id}",
        actions=t.tag(r["kind"], "info"),
    )


def counts_panel() -> html.Section:
    """The frozen trial-count scenarios (the headline uses the first)."""
    counts = data.freeze()["count_scenarios"]
    rows = [{"scenario": k, **v} for k, v in counts.items()]
    columns = [
        t.Col("Scenario", "scenario", str, numeric=False),
        t.Col("N", "n_trials", fmt.integer),
        t.Col("Pipeline", "pipeline", fmt.integer),
        t.Col("Notebook", "notebook", fmt.integer),
        t.Col("Rule", "rule", str, numeric=False),
    ]
    return t.panel(
        t.table(
            rows, columns, compact=True, highlight=lambda r: r["scenario"] == "all_fingerprints"
        ),
        title="Trial count (frozen)",
        subtitle=(
            "From holdout_freeze.json · the headline uses every distinct fingerprint, "
            "the upper bound"
        ),
    )


def variance_panel() -> html.Section:
    """The frozen V[SR] scenarios (the headline uses the largest)."""
    freeze = data.freeze()
    variances = freeze["variance_scenarios"]
    used = freeze["headline"]["variance_used"]
    rows = []
    for k, v in variances.items():
        best = v.get("best")
        rows.append(
            {
                "scenario": k,
                **v,
                "best_label": data.BOOKS[best].label if best in data.BOOKS else best,
            }
        )
    columns = [
        t.Col("Scenario", "scenario", str, numeric=False),
        t.Col("Candidates", "candidates", fmt.integer),
        t.Col("V[SR] annual", "variance_annual", lambda v: fmt.number(v, 4)),
        t.Col("Best", "best_label", str, numeric=False),
        t.Col("Best Sharpe", "best_sharpe_annual", fmt.sharpe),
    ]
    return t.panel(
        t.table(rows, columns, compact=True, highlight=lambda r: r["scenario"] == used),
        title="Sharpe variance across trials (frozen)",
        subtitle=f"The headline used {used}, the largest pre-registered option",
    )


def bundle() -> html.Section:
    """Hashes, the attempt record, downloads and the MLflow runs."""
    complete = data.holdout_complete()
    started = data.holdout_started()
    manifest = data.manifest()
    try:
        runs = data.mlflow_runs(8)
        mlflow = t.table(
            [
                {
                    "name": r["name"],
                    "status": r["status"],
                    "start": r["start_time"],
                    "id": r["run_uuid"][:8],
                }
                for _, r in runs.iterrows()
            ],
            [
                t.Col("MLflow run", "name", str, numeric=False),
                t.Col("Status", "status", str, numeric=False),
                t.Col("Started", "start", lambda v: str(v)[:16]),
                t.Col("Run ID", "id", str),
            ],
            compact=True,
        )
    except data.DataError as exc:
        mlflow = t.empty(
            exc.title, exc.detail + " MLflow is optional; the trial log is the record.", "missing"
        )
    buttons = [
        html.Button(
            [t.icon("download", "sm"), label],
            id=key,
            className="btn" if key != "ex-dl-log" else "btn primary-btn",
        )
        for key, (label, _) in DOWNLOADS.items()
    ]
    facts = [
        (
            "Holdout attempt",
            f"#{started['attempt']['number']} complete · {complete['completed_at'][:10]}",
            "pos",
        ),
        ("Runner commit", started["runner_commit"][:12], "strong"),
        (
            "Artifacts",
            f"model + pairs, prepared {manifest['prepared_at'][:10]} on {manifest['code_version']}",
            "strong",
        ),
        ("Outputs hashed", f"{len(complete['outputs_sha256'])} files in COMPLETE.json", "strong"),
        ("Trial log prefix", complete["trial_log_prefix_after_run"], "pos"),
    ]
    return t.panel(
        [
            html.Div(
                [
                    html.Div([t.kv(k, v, tone) for k, v, tone in facts], className="list-rows c-6"),
                    html.Div(
                        [
                            html.Div(
                                buttons,
                                style={
                                    "display": "flex",
                                    "gap": "8px",
                                    "flexWrap": "wrap",
                                    "marginBottom": "8px",
                                },
                            ),
                            mlflow,
                        ],
                        className="c-6",
                    ),
                ],
                className="grid",
            ),
        ],
        title_icon="lock",
        title="Replication & reproducibility bundle",
        subtitle=(
            "Every file the audit rests on, as committed; reports are Markdown (no PDF "
            "export exists)"
        ),
    )


@callback(
    Output("ex-head", "children"),
    Output("ex-banner", "children"),
    Output("ex-kpis", "children"),
    Output("ex-counts", "children"),
    Output("ex-variance", "children"),
    Output("ex-bundle", "children"),
    Output("ex-kind", "options"),
    RELOAD,
)
def fixed_blocks(_token: int) -> tuple:
    """Blocks fixed by the freeze and the trial log."""
    try:
        kinds = sorted(data.trial_table()["kind"].unique())
    except data.DataError:
        kinds = []
    return (
        t.guarded(head, "the header"),
        t.guarded(banner, "the DSR banner"),
        t.guarded(kpis, "the luck threshold cards"),
        t.guarded(counts_panel, "the count scenarios"),
        t.guarded(variance_panel, "the variance scenarios"),
        t.guarded(bundle, "the replication bundle"),
        [{"label": k, "value": k} for k in kinds],
    )


@callback(
    Output("ex-table", "data"),
    Output("ex-count", "children"),
    Output("ex-table-foot", "children"),
    Input("ex-search", "value"),
    Input("ex-kind", "value"),
    RELOAD,
)
def table_rows(search: str | None, kind: str | None, _token: int) -> tuple:
    """Filter the listed rows; the frozen N above never changes with them."""
    try:
        rows = filtered(search, kind)
        total = len(data.trial_table())
    except data.DataError as exc:
        return [], "0 rows", exc.detail
    foot = [
        html.Span(
            f"Showing {len(rows)} of {total} executions · filters change the list only, "
            "never the frozen N"
        ),
        html.Span(
            f"Frozen N = {data.freeze()['headline']['n_trials']} distinct selection trials · "
            f"{views.trials_note()}"
        ),
    ]
    return rows.drop(columns=["fingerprint", "notes"]).to_dict("records"), f"{total} logged", foot


@callback(
    Output("ex-inspector", "children"),
    Input("ex-table", "active_cell"),
    State("ex-table", "derived_viewport_data"),
    RELOAD,
)
def inspect(active: dict | None, viewport: list | None, _token: int) -> Any:
    """Show the clicked trial, or the latest one."""
    trial_id = None
    if active and viewport and active["row"] < len(viewport):
        trial_id = int(viewport[active["row"]]["trial_id"])
    return t.guarded(lambda: inspector(trial_id), "the trial inspector")


@callback(
    Output("ex-download", "data"),
    [Input(key, "n_clicks") for key in DOWNLOADS],
    prevent_initial_call=True,
)
def download(*_clicks: Any) -> Any:
    """Send the committed file the clicked button names."""
    key = ctx.triggered_id
    if key not in DOWNLOADS or not any(_clicks):
        return no_update
    filename, path = DOWNLOADS[key]
    return dcc.send_file(str(path()), filename=filename)


def export(controls: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    """The whole trial log as displayed (every execution, key metric included)."""
    return "trial_log", log_rows()
