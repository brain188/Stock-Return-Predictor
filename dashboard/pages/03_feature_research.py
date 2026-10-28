"""Feature and signal research: importance, redundancy, IC over time and regimes, and the tests run.

Everything comes from the saved research run (CPCV out-of-fold, development
period only). The hypothesis table lists only tests the project actually ran,
with the statistic it saved; none is computed here, so the dashboard adds no
untracked trials. Exploratory, development and confirmatory evidence are
labelled apart.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from dash import Output, callback, html

from dashboard import charts, data, views
from dashboard import formatting as fmt
from dashboard import theme as t
from dashboard.controls import INPUTS, RELOAD, selection

PATH = "/features"
TITLE = "Feature & Signal"
SUBTITLE = "Q2: IC, importance & regimes"
ICON = "hub"
USES = {"asset", "dates"}
REDUNDANCY = 0.7  # The research pipeline's clustering threshold (REDUNDANCY_THRESHOLD).
T_CRITICAL = 1.96
SERIES = (
    ("mom_63d", "63-day momentum", ".0%"),
    ("ret_5d", "5-day lagged return", ".1%"),
    ("vol_21d", "21-day volatility", ".1%"),
)


def layout() -> html.Div:
    """The page skeleton."""
    return html.Div(
        [
            html.Div(id="fs-strip"),
            html.Div(id="fs-kpis", className="kpis", style={"--n": 5}),
            html.Div(
                [
                    html.Div(id="fs-importance", className="c-7 stack-md"),
                    html.Div(id="fs-redundancy", className="c-5 stack-md"),
                ],
                className="grid",
            ),
            html.Div(
                [
                    html.Div(id="fs-ic", className="c-7 stack-md"),
                    html.Div(id="fs-regimes", className="c-5 stack-md"),
                ],
                className="grid",
            ),
            html.Div(id="fs-tests"),
            html.Div(id="fs-series"),
        ],
        className="page",
    )


def strip() -> html.Div:
    """The research question and the validation design behind every number on the page."""
    m = data.research_metrics()
    v = data.research_validation()
    cfg = data.config().validation
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            t.tag("Research question Q2", "info"),
                            html.Span(
                                f"CPCV (N = {cfg.cpcv.n_groups}, k = "
                                f"{cfg.cpcv.n_test_groups}) · "
                                f"{int(m['embargo_days'])}-day embargo · fingerprint "
                                f"{v['fingerprint']}",
                                className="mono-sm pos",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "8px",
                            "alignItems": "center",
                            "flexWrap": "wrap",
                        },
                    ),
                    html.H1(
                        "Do the 19 features carry information about next-day returns?",
                        className="h-sm strong",
                    ),
                    html.Div(
                        [
                            html.Span(
                                [
                                    "Target: ",
                                    html.Span(
                                        "next-day return (fixed horizon, 1 day)",
                                        className="primary",
                                    ),
                                ]
                            ),
                            html.Span(
                                [
                                    "Sample: ",
                                    html.Span(
                                        f"{fmt.integer(m['n_rows'])} rows · "
                                        f"{fmt.integer(m['cs_dates'])} dates · "
                                        f"{int(m['cs_mean_tickers'])} tickers",
                                        className="strong",
                                    ),
                                ]
                            ),
                            html.Span(
                                [
                                    "Period: ",
                                    html.Span(
                                        "development 2005-07 → 2019-12, out-of-fold",
                                        className="strong",
                                    ),
                                ]
                            ),
                        ],
                        className="meta mono-sm",
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "gap": "2px", "minWidth": 0},
            ),
            html.Div(
                [t.tag("Exploratory + development evidence"), t.tag(views.trials_note())],
                className="panel-actions",
            ),
        ],
        className="strip",
    )


def kpis() -> list[Any]:
    """Headline signal statistics of the research run."""
    m = data.research_metrics()
    red = data.redundancy()
    matrix = red.drop(columns=["cluster"])
    pairs = int((np.triu(matrix.abs().to_numpy(), k=1) > REDUNDANCY).sum())
    cs_t = float(m["cs_spearman_t_stat"])
    return [
        t.kpi(
            "Mean cross-sectional IC",
            fmt.number(m["cs_spearman_mean"], 4, signed=True),
            "Spearman",
            "insights",
            tone=fmt.tone(m["cs_spearman_mean"]),
            rows=[
                ("Positive days", fmt.pct(m["cs_spearman_positive_share"], 1), "soft"),
                ("Dates", fmt.integer(m["cs_dates"]), "soft"),
            ],
        ),
        t.kpi(
            "Pooled IC (5 paths)",
            fmt.number(m["ic_mean"], 4, signed=True),
            f"± {fmt.number(m['ic_std'], 4)} s.d.",
            "analytics",
            tone=fmt.tone(m["ic_mean"]),
            rows=[
                ("Positive paths", f"{int(m['positive_ic_paths'])} of {int(m['n_paths'])}", "soft"),
                ("Range", f"{fmt.number(m['ic_min'], 4)} to {fmt.number(m['ic_max'], 4)}", "soft"),
            ],
        ),
        t.kpi(
            "t-stat of daily IC",
            fmt.number(cs_t, 2),
            "not significant" if abs(cs_t) < T_CRITICAL else "significant at 5%",
            "rule",
            tone="neg" if abs(cs_t) < T_CRITICAL else "pos",
            rows=[
                ("Two-sided 5% bar", f"|t| > {T_CRITICAL}", "soft"),
                ("Yearly-IC t-stat", fmt.number(m["ic_year_t_stat"], 2), "soft"),
            ],
        ),
        t.kpi(
            "Feature pool",
            str(len(red)),
            "features",
            "hub",
            rows=[
                ("Redundancy clusters", str(red["cluster"].nunique()), "soft"),
                ("Fracdiff order d", fmt.number(m["fracdiff_d"], 2), "soft"),
            ],
        ),
        t.kpi(
            "Collinearity",
            str(pairs),
            f"pairs |ρ| > {REDUNDANCY}",
            "warning",
            tone="warn" if pairs else "pos",
            rows=[
                ("Largest cluster", f"{red['cluster'].value_counts().iloc[0]} features", "soft"),
                ("Threshold", "research pipeline", "soft"),
            ],
        ),
    ]


def importance_panel() -> html.Section:
    """Permutation importance with its spread across the 15 CPCV folds."""
    imp = data.importance()
    stable = imp[imp["positive_share"] >= 0.8]
    return t.panel(
        t.graph(charts.importance(imp), 470, label="Permutation importance by feature"),
        kicker="Stability analysis",
        title="Permutation importance across CPCV folds",
        subtitle="Mean IC lost when a feature is shuffled · whisker: s.d. over 15 folds",
        actions=t.legend([("Mean", t.PRIMARY_CONTAINER, "box"), ("Fold s.d.", t.POS, "line")]),
        footer=[
            html.Span(
                [
                    "Positive in ≥ 80% of folds: ",
                    html.Span(f"{len(stable)} of {len(imp)}", className="strong"),
                ]
            ),
            html.Span(
                (
                    "Out-of-fold permutation; small values: importance here is IC on the order "
                    "of 0.005"
                ),
                className="muted",
            ),
        ],
    )


def redundancy_panel() -> html.Section:
    """The features' correlation matrix ordered by redundancy cluster."""
    red = data.redundancy().sort_values("cluster")
    order = list(red.index)
    matrix = red.drop(columns=["cluster"]).loc[order, order]
    return t.panel(
        t.graph(
            charts.heatmap(matrix, "ρ", show_text=False), 470, label="Feature correlation matrix"
        ),
        kicker="Multi-collinearity",
        title="Feature correlation by cluster",
        subtitle=f"Ordered by the research run's clusters (|ρ| > {REDUNDANCY} joins)",
        actions=t.tag(f"{red['cluster'].nunique()} clusters"),
        footer=[
            html.Span("Hover a cell for the exact correlation; clusters run along the diagonal.")
        ],
    )


def ic_panel() -> html.Section:
    """252-day trailing mean of the daily cross-sectional IC, with a band."""
    roll = data.rolling_ic()
    daily = data.daily_cross_sectional_ic()["ic"]
    last = roll.iloc[-1]
    below = float((roll["high"] < 0).mean())
    return t.panel(
        t.graph(charts.rolling_ic(roll), 260, label="Rolling cross-sectional IC"),
        kicker="Dynamic stability",
        title="12-month rolling IC with 95% band",
        subtitle=(
            "Daily cross-sectional Spearman IC, mean of 5 paths · band ±1.96 SE "
            "(understates: days overlap)"
        ),
        actions=t.legend(
            [("Rolling IC", t.PRIMARY, "line"), ("±1.96 SE", "rgba(142,213,255,0.4)", "box")]
        ),
        footer=[
            html.Span(
                [
                    "Latest (2019-12): ",
                    html.Span(
                        fmt.number(last["ic"], 4, signed=True), className=fmt.tone(last["ic"])
                    ),
                ]
            ),
            html.Span(
                [
                    "Lag-1 autocorrelation of daily IC: ",
                    html.Span(fmt.number(daily.autocorr(1), 3), className="strong"),
                ]
            ),
            html.Span(
                [
                    "Share of windows wholly below 0: ",
                    html.Span(fmt.pct(below, 1), className="strong"),
                ]
            ),
        ],
    )


def regimes_panel() -> html.Section:
    """IC and the ML long/short Sharpe by volatility regime."""
    ic = data.ic_by_regime().set_index("regime")
    strat = data.strategy_regimes()
    strat = strat[strat["strategy"] == "ml_long_short"].set_index("regime")
    rows = []
    for regime in ("low", "normal", "high"):
        if regime not in ic.index:
            continue
        r = {"regime": f"{regime.capitalize()} vol", **ic.loc[regime].to_dict()}
        if regime in strat.index:
            r |= {
                "sharpe": strat.loc[regime, "sharpe"],
                "vs": strat.loc[regime, "vs_reference"],
                "lo": strat.loc[regime, "vs_reference_low"],
                "hi": strat.loc[regime, "vs_reference_high"],
            }
        rows.append(r)
    best = max(rows, key=lambda r: r["ic_mean"])
    worst = min(rows, key=lambda r: r["ic_mean"])
    columns = [
        t.Col("Regime", "regime", str, numeric=False),
        t.Col(
            "Mean IC", "ic_mean", lambda v: fmt.number(v, 4, True), tone=lambda v, r: fmt.tone(v)
        ),
        t.Col("Paths > 0", "positive_paths", lambda v: f"{int(v)}/5"),
        t.Col("ML L/S Sharpe", "sharpe", fmt.sharpe),
        t.Col(
            "vs EW hold [95%]",
            "vs",
            lambda v: fmt.sharpe(v, True),
            tone=lambda v, r: views.interval_verdict({"low": r.get("lo"), "high": r.get("hi")})[1],
        ),
    ]
    return t.panel(
        [
            t.table(rows, columns, compact=True),
            html.Div(
                [
                    t.icon("info", "sm"),
                    html.Span(
                        f" IC is highest in {best['regime'].lower()} "
                        f"({fmt.number(best['ic_mean'], 4, True)}) and lowest in "
                        f"{worst['regime'].lower()} "
                        f"({fmt.number(worst['ic_mean'], 4, True)}). No regime's "
                        "strategy interval lies wholly above equal-weight hold; regimes are "
                        "labelled with past volatility only.",
                        className="body-sm soft",
                    ),
                ],
                style={"marginTop": "8px", "display": "flex", "gap": "4px"},
            ),
        ],
        kicker="Conditional performance",
        title="Volatility regime conditioning",
        subtitle="Development, out-of-fold · Sharpe vs equal-weight hold, paired bootstrap",
    )


def tests_rows() -> list[dict[str, Any]]:
    """Every hypothesis test the project ran and saved, with its evidence type."""
    m = data.research_metrics()
    emb = data.research_dir().parents[1] / "diagnostics" / "embargo_sensitivity" / "summary.csv"
    rows: list[dict[str, Any]] = []
    cs_t = float(m["cs_spearman_t_stat"])
    rows.append(
        {
            "hypothesis": "Forecasts rank next-day returns across tickers",
            "family": "Signal · daily cross-sectional IC",
            "null": "mean IC = 0",
            "evidence": "Development",
            "n": m["cs_dates"],
            "stat": f"t = {cs_t:+.2f}",
            "result": f"mean {m['cs_spearman_mean']:+.4f}",
            "reject": abs(cs_t) > T_CRITICAL,
        }
    )
    rows.append(
        {
            "hypothesis": "Pooled IC is positive on every CPCV path",
            "family": "Signal · pooled Spearman IC",
            "null": "IC ≤ 0",
            "evidence": "Development",
            "n": m["n_rows"],
            "stat": f"{int(m['positive_ic_paths'])}/{int(m['n_paths'])} paths > 0",
            "result": f"{m['ic_mean']:+.4f} ± {m['ic_std']:.4f}",
            "reject": int(m["positive_ic_paths"]) == int(m["n_paths"]),
        }
    )
    yt = float(m["ic_year_t_stat"])
    years = data.ic_by_year()
    rows.append(
        {
            "hypothesis": "IC is positive year after year",
            "family": "Signal · IC by calendar year",
            "null": "mean yearly IC = 0",
            "evidence": "Exploratory",
            "n": len(years),
            "stat": f"t = {yt:+.2f}",
            "result": f"{int((years['ic_mean'] > 0).sum())}/{len(years)} years > 0",
            "reject": abs(yt) > T_CRITICAL,
        }
    )
    edge, sd = float(m["edge_over_always_up_mean"]), float(m["edge_over_always_up_std"])
    rows.append(
        {
            "hypothesis": "Hit rate beats always predicting up",
            "family": "Direction · hit rate, across paths",
            "null": "edge ≤ 0",
            "evidence": "Development",
            "n": m["n_rows"],
            "stat": f"path s.d. {sd * 100:.2f} pp",
            "result": f"{edge * 100:+.2f} pp",
            "reject": edge - T_CRITICAL * sd > 0,
        }
    )
    if emb.is_file():
        e = pd.read_csv(emb).set_index("variant")
        if "embargo_0" in e.index:
            r = e.loc["embargo_0"]
            rows.append(
                {
                    "hypothesis": "Removing the embargo changes the IC (leakage)",
                    "family": "Validation · paired by split",
                    "null": "no difference",
                    "evidence": "Diagnostic",
                    "n": 15,
                    "stat": f"t = {r['paired_t']:+.2f}",
                    "result": f"{r['paired_diff_mean']:+.4f} IC",
                    "reject": abs(r["paired_t"]) > T_CRITICAL,
                }
            )
    comp = data.strategy_comparison().loc["ml_long_short"]
    rows.append(
        {
            "hypothesis": "ML long/short beats equal-weight hold (net)",
            "family": "Strategy · paired block bootstrap",
            "null": "Sharpe difference ≤ 0",
            "evidence": "Development",
            "n": 5,
            "stat": f"[{comp['vs_reference_low']:+.2f}, {comp['vs_reference_high']:+.2f}]",
            "result": f"{comp['sharpe_net_vs_equal_weight_hold']:+.2f} Sharpe",
            "reject": comp["vs_reference_low"] > 0,
        }
    )
    p = data.holdout_evaluation()["primary"]
    primary = data.declaration()["candidates"]["primary"]
    rows.append(
        {
            "hypothesis": f"{data.BOOKS[primary].label} has an edge (predeclared rule)",
            "family": "Strategy · DSR and paired interval",
            "null": "no edge over equal-weight hold",
            "evidence": "Confirmatory (holdout)",
            "n": p["moments"]["n"],
            "stat": (
                f"DSR {p['dsr']:.3f}; [{p['interval']['low']:+.2f}, {p['interval']['high']:+.2f}]"
            ),
            "result": p["conclusion"],
            "reject": bool(p["evidence_of_edge"]),
        }
    )
    return rows


def tests_panel() -> html.Section:
    """The hypothesis table, with what each verdict means."""
    rows = tests_rows()
    for r in rows:
        r["verdict"] = "Reject H0" if r["reject"] else "Fail to reject H0"
    evidence_tone = {
        "Confirmatory (holdout)": "pos",
        "Development": "",
        "Exploratory": "warn",
        "Diagnostic": "",
    }
    columns = [
        t.Col("Hypothesis", "hypothesis", str, numeric=False),
        t.Col("Test", "family", str, numeric=False),
        t.Col("H0", "null", str, numeric=False),
        t.Col(
            "Evidence", "evidence", str, numeric=False, tone=lambda v, r: evidence_tone.get(v, "")
        ),
        t.Col("n", "n", fmt.integer, title="Dates, rows, years, splits or paths, as the test used"),
        t.Col("Statistic", "stat", str),
        t.Col("Estimate", "result", str),
        t.Col("Verdict", "verdict", str, tone=lambda v, r: "pos" if r["reject"] else "neg"),
    ]
    rejected = sum(r["reject"] for r in rows)
    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Saved tests only", className="caps pos"),
                            html.H2("Hypothesis tests and significance", className="h-sm strong"),
                        ],
                        style={"display": "flex", "flexDirection": "column"},
                    ),
                    t.tag(f"{rejected} of {len(rows)} reject H0"),
                ],
                className="table-head",
            ),
            t.table(rows, columns, highlight=lambda r: r["evidence"].startswith("Confirmatory")),
            html.Div(
                [
                    html.Span(
                        "Two-sided 5% bar |t| > 1.96. Only the holdout row is confirmatory; "
                        "it alone is deflated for the N = 98 trials searched."
                    ),
                    html.Span(
                        "Rows 1–3 test one forecast three ways and are not independent; no "
                        "further correction is applied to them."
                    ),
                ],
                className="table-foot mono-sm",
            ),
        ],
        className="panel flush",
    )


def series_panel(c: dict[str, Any]) -> html.Section:
    """Three of the model's inputs for the selected asset over the date window."""
    feats = data.features()
    if c["asset"] not in feats.index.get_level_values("ticker"):
        raise data.DataError("No features", f"{c['asset']} has no saved features")
    frame = data.window(feats.xs(c["asset"], level="ticker"), c["start"], c["end"])
    if len(frame) < 30:
        raise data.DataError(
            "Window too short", "Choose a date window of at least 30 trading days."
        )
    return t.panel(
        t.graph(
            charts.feature_series(frame, SERIES), 330, label=f"Feature series for {c['asset']}"
        ),
        kicker="Inputs",
        title=f"{c['asset']}: momentum, lagged return and rolling volatility",
        subtitle=(
            "Model features as built (no look-ahead) · "
            f"{fmt.date_range(frame.index.min(), frame.index.max())}"
        ),
        actions=t.legend(
            [
                (label, colour, "line")
                for (_, label, _), colour in zip(
                    SERIES, [t.PRIMARY, t.SECONDARY, t.WARN], strict=True
                )
            ]
        ),
    )


@callback(
    Output("fs-strip", "children"),
    Output("fs-kpis", "children"),
    Output("fs-importance", "children"),
    Output("fs-redundancy", "children"),
    Output("fs-ic", "children"),
    Output("fs-regimes", "children"),
    Output("fs-tests", "children"),
    RELOAD,
)
def research_blocks(_token: int) -> tuple:
    """Blocks fixed by the saved research run."""
    return (
        t.guarded(strip, "the research strip"),
        t.guarded(kpis, "the signal cards"),
        t.guarded(importance_panel, "the importance chart"),
        t.guarded(redundancy_panel, "the redundancy matrix"),
        t.guarded(ic_panel, "the rolling IC"),
        t.guarded(regimes_panel, "the regime table"),
        t.guarded(tests_panel, "the hypothesis table"),
    )


@callback(Output("fs-series", "children"), *INPUTS, RELOAD)
def asset_series(*values: Any) -> Any:
    """The feature series for the selected asset and window."""
    c = selection(*values[:-1])
    return t.guarded(lambda: series_panel(c), "the feature series")


def export(controls: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    """Feature importance with its fold stability."""
    return "feature_importance", data.importance()
