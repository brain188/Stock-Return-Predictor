"""Model performance: the one fitted model against its baselines, out-of-fold and in the holdout.

The project fitted one model (a random forest regression) and four rule
baselines; those five are the only choices here. Every metric is out-of-fold
(CPCV, development period). In-sample fit metrics were never saved, and the
holdout run recorded strategy returns but not forecasts, so the holdout panel
shows only the ML strategies' results and says what is missing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from dash import Output, callback, html

from dashboard import charts, data, views
from dashboard import formatting as fmt
from dashboard import theme as t
from dashboard.controls import INPUTS, MODEL_KEY, MODEL_LABELS, RELOAD, selection

PATH = "/model"
TITLE = "Model Performance"
SUBTITLE = "Q3: CPCV out-of-fold & holdout"
ICON = "model_training"
USES = {"model"}
DESCRIPTIONS = {
    MODEL_KEY: "Random forest regression of the next-day return on 19 features",
    "always_up": "Predicts up every day (the passive benchmark's bet)",
    "random": "A seeded coin flip: what luck looks like",
    "last_sign": "Tomorrow repeats today's direction (naive momentum)",
    "reverse_last_sign": "Tomorrow reverses today's direction (naive reversal)",
}
ML_BOOKS = ("ml_long_short", "ml_long_only", "ml_thresholded")


def layout() -> html.Div:
    """The page skeleton."""
    return html.Div(
        [
            html.Div(id="mp-tabs"),
            html.Div(id="mp-params"),
            html.Div(
                [
                    html.Div(id="mp-dev", className="c-6"),
                    html.Div(id="mp-holdout", className="c-6"),
                ],
                className="grid",
            ),
            html.Div(
                [
                    html.Div(id="mp-years", className="c-7 stack-md"),
                    html.Div(id="mp-quintiles", className="c-5 stack-md"),
                ],
                className="grid",
            ),
            html.Div(id="mp-board"),
            html.Div(id="mp-paths"),
        ],
        className="page",
    )


def tabs(model: str) -> html.Div:
    """One tab per evaluated forecast; the active one is the global model."""
    validation = data.research_validation()
    metrics = data.research_metrics()
    buttons = [
        html.Button(
            [label, t.tag("model", "") if key == MODEL_KEY else None],
            id={"type": "pick-model", "key": key},
            className="tab active" if key == model else "tab",
            title=DESCRIPTIONS[key],
            **{"aria-pressed": "true" if key == model else "false"},
        )
        for key, label in MODEL_LABELS.items()
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Span("Forecast ::", className="mono-sm muted"),
                    html.Div(buttons, className="tabs"),
                ],
                style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    t.kv("Run fingerprint", validation["fingerprint"], "strong"),
                    t.kv("Shuffled", "no (purged, embargoed)", "pos"),
                    t.kv(
                        "Paths",
                        f"{int(metrics['n_paths'])} from {int(metrics['n_splits'])} splits",
                        "strong",
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
        ],
        className="strip",
    )


def params(model: str) -> html.Div:
    """The model's configuration and the validation engine, as the run used them."""
    cfg = data.config()
    m = data.research_metrics()
    if model == MODEL_KEY:
        rf = cfg.model.random_forest
        settings = (
            f"Estimators: {rf['n_estimators']} | Max depth: {rf['max_depth']} "
            f"| Min leaf: {rf['min_samples_leaf']} | Max features: {rf['max_features']} "
            f"| Task: {cfg.model.task} | Labels: {cfg.labelling.method} "
            f"({cfg.labelling.fixed_horizon.horizon_days} day) | Features: 19 "
            f"| Fracdiff d: {m['fracdiff_d']:.2f}"
        )
    else:
        settings = f"Rule baseline · {DESCRIPTIONS[model]} · no parameters fitted"
    validation = (
        f"CPCV ({cfg.validation.cpcv.n_groups} groups, "
        f"{cfg.validation.cpcv.n_test_groups} test) · {int(m['embargo_days'])}-day embargo · "
        f"{fmt.integer(m['train_rows_mean'])} train rows per split"
    )
    return html.Div(
        [
            html.Div(
                [t.tag("Configuration"), html.Span(settings, className="mono-sm soft")],
                style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    t.tag("Validation engine", "info"),
                    html.Span(validation, className="mono-sm soft"),
                ],
                style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
            ),
        ],
        className="strip",
        style={"flexDirection": "column", "alignItems": "flex-start"},
    )


def _metric(label: str, main: str, sub: str, tone: str = "strong") -> html.Div:
    return html.Div(
        [
            html.Span(label, className="caps muted"),
            html.Span(main, className=f"mono-display {tone}"),
            html.Span(sub, className="mono-sm muted"),
        ],
        className="kpi",
        style={"background": "var(--surface-low)"},
    )


def development_panel(model: str) -> html.Section:
    """Out-of-fold metrics of the selected forecast, with their spread over paths."""
    table = data.forecast_comparison()
    row = table.loc[model]
    paths = data.research_paths() if model == MODEL_KEY else None

    def spread(column: str, digits: int) -> str:
        if paths is None or column not in paths:
            return "single rule, no paths"
        return f"path s.d. {paths[column].std():.1e}"

    cells = [
        _metric("MAE", fmt.number(row.get("mae"), 4), spread("mae", 5)),
        _metric("RMSE", fmt.number(row.get("rmse"), 4), spread("rmse", 5)),
        _metric(
            "Hit rate",
            fmt.pct(row["hit_rate"], 2),
            f"up rate {fmt.pct(row['up_rate'], 2)}",
            fmt.tone(row["edge_over_always_up"]),
        ),
        _metric(
            "Rank IC", fmt.number(row["ic"], 4, signed=True), spread("ic", 4), fmt.tone(row["ic"])
        ),
        _metric("R²", fmt.number(row.get("r2"), 4), spread("r2", 4)),
    ]
    return t.panel(
        [
            html.Div(cells, className="kpis", style={"--n": 5}),
            html.Div(
                [
                    t.icon("info", "sm"),
                    html.Span(
                        " MAE, RMSE and R² exist only for the regression model; direction "
                        "rules have none. In-sample fit metrics were not saved: every number "
                        "here is out-of-fold.",
                        className="body-sm soft",
                    ),
                ],
                style={"marginTop": "8px", "display": "flex", "gap": "4px"},
            ),
        ],
        kicker="Development · out-of-fold",
        title=f"{MODEL_LABELS[model]}: 2005-07-22 → 2019-12-30",
        subtitle=f"n = {fmt.integer(row['n'])} rows · each predicted by models that never saw it",
        actions=t.tag("CPCV", "info"),
        cls="low",
    )


def holdout_panel(model: str) -> html.Section:
    """What the holdout says about the model: only the ML strategies' results exist."""
    if model != MODEL_KEY:
        body = t.empty(
            "Not part of the holdout", f"{MODEL_LABELS[model]} was a development baseline only."
        )
    else:
        summary = data.holdout_summary()
        rows = [
            {
                "book": data.BOOKS[k].label,
                "sharpe_net": summary.loc[k, "net_sharpe"],
                "sharpe_gross": summary.loc[k, "gross_sharpe"],
                "dev": data.book_stats(k, data.DEVELOPMENT)["net_sharpe"],
                "turnover": summary.loc[k, "turnover_annual"],
            }
            for k in ML_BOOKS
            if k in summary.index
        ]
        columns = [
            t.Col("ML strategy", "book", str, numeric=False),
            t.Col(
                "Holdout net",
                "sharpe_net",
                fmt.sharpe,
                tone=lambda v, r: views.tone_vs(v, r["dev"]),
            ),
            t.Col("Holdout gross", "sharpe_gross", fmt.sharpe),
            t.Col("Dev net", "dev", fmt.sharpe),
            t.Col("Turnover / yr", "turnover", lambda v: fmt.number(v, 1) + "×"),
        ]
        body = [
            t.empty(
                "Holdout forecasts were not recorded",
                "The one-time run saved each book's returns, not the model's forecasts, so "
                "holdout IC, hit rate and errors are unavailable. Computing them now would "
                "be a new analysis of opened data.",
                "missing",
            ),
            html.Div(t.table(rows, columns, compact=True), style={"marginTop": "8px"}),
        ]
    window = views.holdout_window()
    return t.panel(
        body,
        kicker="Holdout · one-time evaluation",
        title=f"Holdout: {fmt.date_range(*window)}" if window else "Holdout",
        subtitle="Final model fitted once on all development rows · strategies' net Sharpe, "
        + views.trials_note(),
        actions=t.tag("Attempt 1", "pos"),
    )


def years_panel(model: str) -> html.Section:
    """Pooled IC per calendar year, mean and spread over the 5 paths."""
    if model != MODEL_KEY:
        raise data.DataError(
            "Saved for the model only",
            "IC by year was computed for the random forest, not the baselines.",
        )
    years = data.ic_by_year()
    figure = charts.bars_with_error(
        years["period"].astype(str), years["ic_mean"], years["ic_std"], "pooled IC", "+.3f"
    )
    positive = int((years["ic_mean"] > 0).sum())
    return t.panel(
        t.graph(figure, 250, label="IC by calendar year"),
        title="IC by calendar year (out-of-fold)",
        subtitle="Pooled Spearman IC per year · whisker: s.d. over 5 CPCV paths · development",
        actions=t.legend([("IC > 0", t.POS, "box"), ("IC < 0", t.NEG, "box")]),
        footer=[
            html.Span(
                [
                    "Years with IC > 0: ",
                    html.Span(f"{positive} of {len(years)}", className="strong"),
                ]
            ),
            html.Span(
                [
                    "t-stat across years: ",
                    html.Span(
                        fmt.number(data.research_metrics()["ic_year_t_stat"], 2), className="strong"
                    ),
                ]
            ),
            html.Span("Exploratory: not adjusted for the trials searched", className="muted"),
        ],
    )


def quintiles_panel(model: str) -> html.Section:
    """Realised return by forecast quintile: does the ranking sort outcomes?"""
    if model != MODEL_KEY:
        raise data.DataError(
            "No continuous forecast", "Direction rules cannot be sorted into quintiles."
        )
    q = data.forecast_quintiles()
    monotone = bool(q["realised"].is_monotonic_increasing)
    return t.panel(
        t.graph(charts.quintiles(q), 250, label="Realised return by forecast quintile"),
        title="Monotonicity sort",
        subtitle="Mean realised next-day return by forecast quintile · 95% CI · development",
        actions=t.tag("Q1 to Q5 spread"),
        footer=[
            html.Span(
                [
                    "Q5 − Q1: ",
                    html.Span(
                        fmt.bps(q["realised"].iloc[-1] - q["realised"].iloc[0], 2, True),
                        className="strong",
                    ),
                ]
            ),
            html.Span(
                [
                    "Strictly sorted: ",
                    html.Span("yes" if monotone else "no", className="pos" if monotone else "neg"),
                ]
            ),
        ],
    )


def board(model: str) -> html.Section:
    """Every forecast's out-of-fold metrics side by side."""
    table = data.forecast_comparison()
    rows = []
    for key, row in table.iterrows():
        rows.append(
            {
                "key": key,
                "name": MODEL_LABELS.get(key, key),
                **row.to_dict(),
                "status": "Model" if key == MODEL_KEY else "Baseline",
            }
        )
    columns = [
        t.Col("Forecast", "name", str, numeric=False),
        t.Col("Rank IC", "ic", lambda v: fmt.number(v, 4, True), tone=lambda v, r: fmt.tone(v)),
        t.Col("Hit rate", "hit_rate", lambda v: fmt.pct(v, 2)),
        t.Col("Edge vs up", "edge_over_always_up", fmt.pp, tone=lambda v, r: fmt.tone(v)),
        t.Col("MAE", "mae", lambda v: fmt.number(v, 4)),
        t.Col("RMSE", "rmse", lambda v: fmt.number(v, 4)),
        t.Col("R²", "r2", lambda v: fmt.number(v, 4)),
        t.Col("CS IC", "cs_spearman_mean", lambda v: fmt.number(v, 4, True)),
        t.Col("CS t", "cs_spearman_t_stat", lambda v: fmt.number(v, 2)),
        t.Col("Status", "status", str, tone=lambda v, r: "primary" if v == "Model" else "muted"),
    ]
    strategy = data.book_stats("ml_long_short", data.DEVELOPMENT)
    return html.Section(
        [
            html.Div(
                [
                    html.H2(
                        [t.icon("table_chart"), " Forecast comparison: model against baselines"],
                        className="h-sm strong",
                    ),
                    html.Span(
                        f"{len(rows)} forecasts evaluated · development, out-of-fold",
                        className="mono-sm muted",
                    ),
                ],
                className="table-head",
            ),
            t.table(rows, columns, highlight=lambda r: r["key"] == model),
            html.Div(
                [
                    html.Span(
                        f"Strategy built on the model (ML long/short): development net Sharpe "
                        f"{fmt.sharpe(strategy['net_sharpe'])}, deflated against "
                        f"{views.trials_note()}."
                    ),
                    html.Span(
                        "No LightGBM or LSTM was run in the pipeline: configured, but no "
                        "results exist."
                    ),
                ],
                className="table-foot mono-sm",
            ),
        ],
        className="panel flush",
    )


def paths_panel(model: str) -> Any:
    """The model's metrics on each CPCV path."""
    if model != MODEL_KEY:
        return None
    paths = data.research_paths().reset_index()
    columns = [
        t.Col("Path", "path", str, numeric=False),
        t.Col("Rank IC", "ic", lambda v: fmt.number(v, 4, True), tone=lambda v, r: fmt.tone(v)),
        t.Col("Hit rate", "hit_rate", lambda v: fmt.pct(v, 2)),
        t.Col("Edge vs up", "edge_over_always_up", fmt.pp, tone=lambda v, r: fmt.tone(v)),
        t.Col("MAE", "mae", lambda v: fmt.number(v, 5)),
        t.Col("RMSE", "rmse", lambda v: fmt.number(v, 5)),
        t.Col("CS IC", "cs_spearman_mean", lambda v: fmt.number(v, 4, True)),
        t.Col("CS t", "cs_spearman_t_stat", lambda v: fmt.number(v, 2)),
    ]
    return t.panel(
        t.table(paths.to_dict("records"), columns, compact=True),
        title="Performance across CPCV test paths",
        subtitle="Each path covers every development date once, predicted out-of-fold",
    )


@callback(
    Output("mp-tabs", "children"),
    Output("mp-params", "children"),
    Output("mp-dev", "children"),
    Output("mp-holdout", "children"),
    Output("mp-years", "children"),
    Output("mp-quintiles", "children"),
    Output("mp-board", "children"),
    Output("mp-paths", "children"),
    *INPUTS,
    RELOAD,
)
def update(*values: Any) -> tuple:
    """Rebuild the page for the selected forecast."""
    model = selection(*values[:-1])["model"]
    return (
        t.guarded(lambda: tabs(model), "the model tabs"),
        t.guarded(lambda: params(model), "the configuration"),
        t.guarded(lambda: development_panel(model), "the development metrics"),
        t.guarded(lambda: holdout_panel(model), "the holdout panel"),
        t.guarded(lambda: years_panel(model), "the IC by year"),
        t.guarded(lambda: quintiles_panel(model), "the quintile sort"),
        t.guarded(lambda: board(model), "the comparison table"),
        t.guarded(lambda: paths_panel(model), "the path table"),
    )


def export(controls: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    """The forecast comparison table."""
    return "model_comparison", data.forecast_comparison()
