"""Portfolio and risk: weights against risk contributions, the frontier, forward risk and stress.

Allocation-level results come from the portfolio pipeline's saved tables
(development period); the holdout column comes from the one-time evaluation.
Weight and contribution to risk are always shown as separate columns: a 10%
weight can carry 20% of the risk.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

from dashboard import charts, data, views
from dashboard import formatting as fmt
from dashboard import theme as t
from dashboard.controls import RELOAD

PATH = "/portfolio"
TITLE = "Portfolio & Risk"
SUBTITLE = "Frontier, VaR & risk budget"
ICON = "pie_chart"
USES: set[str] = set()
RULES = {
    "minimum_variance": "Minimum variance",
    "equal_weight": "Equal weight",
    "inverse_volatility": "Inverse volatility",
    "risk_parity": "Risk parity",
    "maximum_sharpe": "Maximum Sharpe",
}
BOOK_OF = {
    "minimum_variance": "allocation_minimum_variance",
    "equal_weight": "equal_weight",
    "inverse_volatility": "allocation_inverse_volatility",
    "risk_parity": "allocation_risk_parity",
    "maximum_sharpe": "allocation_maximum_sharpe",
}
METHODS = {"bootstrap": "Block bootstrap", "student_t": "Student t", "normal": "Normal"}


def layout() -> html.Div:
    """The page skeleton with the page's own allocation and method selectors."""
    return html.Div(
        [
            html.Div(id="pr-engine"),
            t.page_controls(
                [
                    t.field(
                        "Allocation",
                        dcc.RadioItems(
                            id="pr-rule",
                            options=[{"label": v, "value": k} for k, v in RULES.items()],
                            value="minimum_variance",
                            className="seg",
                            inline=True,
                        ),
                    ),
                    t.field(
                        "Forward-risk model",
                        dcc.RadioItems(
                            id="pr-method",
                            options=[{"label": v, "value": k} for k, v in METHODS.items()],
                            value="bootstrap",
                            className="seg",
                            inline=True,
                        ),
                    ),
                ]
            ),
            html.Div(id="pr-kpis", className="kpis", style={"--n": 4}),
            html.Div(
                [
                    html.Div(id="pr-frontier", className="c-7 stack-md"),
                    html.Div(id="pr-weights", className="c-5 stack-md"),
                ],
                className="grid",
            ),
            html.Div(
                [
                    html.Div(id="pr-mc", className="c-7 stack-md"),
                    html.Div(id="pr-stress", className="c-5 stack-md"),
                ],
                className="grid",
            ),
            html.Div(id="pr-compare"),
        ],
        className="page",
    )


def engine() -> html.Header:
    """The page header: what it answers, and how every allocation is estimated."""
    p = data.config().portfolio
    return t.page_header(
        "Portfolio and risk",
        "How the allocation rules spread weight and risk, what the frontier promised against "
        "what it delivered, and the saved forward-risk and stress results.",
        eyebrow="Research question Q5",
        meta=[
            (
                "Covariance",
                f"{p.covariance_method.replace('_', '-').title()}, {p.covariance_window}-day",
            ),
            ("Constraints", "long only, fully invested" if p.long_only else "long/short"),
            ("Expected returns", f"{p.expected_return_method}, rf = {p.risk_free_rate:.0%}"),
            ("Assets", str(len(data.tickers()))),
        ],
        status=t.tag("Development, with holdout column"),
    )


def kpis(rule: str, method: str) -> list[Any]:
    """Volatility in both periods, one-year VaR and CVaR, and diversification."""
    key = BOOK_OF[rule]
    dev = data.book_stats(key, data.DEVELOPMENT)
    hold = data.book_stats(key, data.HOLDOUT)
    risk = data.forward_risk()
    row = risk[(risk["allocation"] == rule) & (risk["method"] == method)]
    if row.empty:
        raise data.DataError("Forward risk not saved", f"no {method} simulation for {RULES[rule]}")
    r = row.iloc[0]
    weights = data.latest_weights(rule)
    effective = 1 / float((weights**2).sum())
    summary = data.risk_summary().loc[rule]
    meta = data.portfolio_metadata()["forward_risk"]
    return [
        t.kpi(
            "Annualised volatility",
            fmt.pct(dev["net_annual_volatility"], 1),
            "Development, net of costs",
            rows=[
                (
                    "Holdout, net",
                    fmt.pct(hold["net_annual_volatility"], 1),
                    views.tone_vs(dev["net_annual_volatility"], hold["net_annual_volatility"]),
                ),
                ("At latest weights", fmt.pct(summary["volatility"], 1), "soft"),
            ],
        ),
        t.kpi(
            "95% one-year VaR",
            fmt.pct(-r["var_95"], 1),
            f"Simulated loss; CVaR {fmt.pct(-r['cvar_95'], 1)}",
            rows=[
                ("Model", METHODS[method], "soft"),
                (
                    "Paths · horizon",
                    f"{fmt.integer(meta['runs'])} · {meta['horizon_days']} days",
                    "soft",
                ),
            ],
        ),
        t.kpi(
            "99% one-year VaR",
            fmt.pct(-r["var_99"], 1),
            f"Simulated loss; CVaR {fmt.pct(-r['cvar_99'], 1)}",
            rows=[
                ("CVaR ÷ VaR (99%)", fmt.number(r["cvar_99"] / r["var_99"], 2), "soft"),
                ("P(ruin, −50%)", fmt.pct(r["ruin_probability"], 2), "soft"),
            ],
        ),
        t.kpi(
            "Effective N",
            fmt.number(effective, 1),
            f"Of {len(weights)} assets, from the latest weights",
            rows=[
                ("Diversification ratio", fmt.number(summary["diversification_ratio"], 2), "soft"),
                ("Largest weight", f"{fmt.pct(weights.max(), 1)} {weights.idxmax()}", "soft"),
            ],
        ),
    ]


def frontier_panel(rule: str) -> html.Section:
    """The in-sample frontier against what its portfolios realised out of sample."""
    meta = data.portfolio_metadata()["frontier"]
    train, test = "–".join(d[:4] for d in meta["train"]), "–".join(d[:4] for d in meta["test"])
    curve = data.frontier()
    portfolios = data.frontier_portfolios()
    figure = charts.frontier(curve, data.frontier_assets(), portfolios, train, test)
    sharpe_in = curve["sharpe"].max()
    realised = curve["realised_return"] / curve["realised_volatility"]
    return t.panel(
        t.graph(figure, 360, label="Efficient frontier in-sample against realised"),
        title="Markowitz frontier: in-sample vs out-of-sample",
        subtitle=(
            f"Estimated on {fmt.date_range(*meta['train'])} · realised "
            f"{fmt.date_range(*meta['test'])}"
        ),
        footer=[
            html.Span(
                ["Best in-sample Sharpe: ", html.Span(fmt.sharpe(sharpe_in), className="primary")]
            ),
            html.Span(
                [
                    "Same portfolios, realised: ",
                    html.Span(
                        f"{fmt.sharpe(realised.min())} to {fmt.sharpe(realised.max())}",
                        className="warn",
                    ),
                ]
            ),
            html.Span(
                "Development data only; the frontier was not re-estimated on the holdout.",
                className="muted",
            ),
        ],
    )


def weights_panel(rule: str) -> html.Section:
    """Each asset's weight next to its share of portfolio variance."""
    weights = data.latest_weights(rule)
    contrib = data.risk_contributions().loc[rule]
    rows = []
    for ticker in weights.index:
        w, rc = float(weights[ticker]), float(contrib.get(ticker, np.nan))
        rows.append(
            {"ticker": ticker, "weight": w, "risk": rc, "ratio": rc / w if w > 1e-6 else np.nan}
        )
    rows.sort(key=lambda r: -r["risk"] if not np.isnan(r["risk"]) else 0)
    top = rows[0]
    held = [r["ratio"] for r in rows if r["weight"] > 1e-6]
    if held and max(abs(x - 1) for x in held) < 0.02:
        # A property of the optimum, not a display bug: say so.
        why = (
            " At the minimum-variance optimum every held asset adds the same marginal risk, "
            "so each one's share of the variance equals its weight."
        )
    else:
        why = (
            f" {top['ticker']}: {fmt.pct(top['weight'], 1)} of weight carries "
            f"{fmt.pct(top['risk'], 1)} of the variance."
        )
    columns = [
        t.Col("Asset", "ticker", str, numeric=False),
        t.Col("Weight", "weight", lambda v: fmt.pct(v, 1)),
        t.Col(
            "Risk share",
            "risk",
            lambda v: fmt.pct(v, 1),
            tone=lambda v, r: "neg" if r["ratio"] > 1.25 else ("pos" if r["ratio"] < 0.8 else ""),
        ),
        t.Col("Risk ÷ weight", "ratio", lambda v: fmt.number(v, 2) + "×"),
    ]
    return t.panel(
        [
            t.table(rows, columns, compact=True),
            t.note(why.strip()),
        ],
        title="Weight vs contribution to risk",
        subtitle=(
            f"{RULES[rule]} at its last development rebalance, {fmt.date(weights.name)} "
            "· the weights the holdout entered"
        ),
    )


def mc_panel(rule: str, method: str) -> html.Section:
    """The simulated one-year drawdown: how likely each depth is."""
    risk = data.forward_risk()
    r = risk[(risk["allocation"] == rule) & (risk["method"] == method)].iloc[0]
    probabilities = {f"> {k}%": r[f"drawdown_p_drawdown_{k}"] for k in (10, 20, 30, 50)}
    meta = data.portfolio_metadata()["forward_risk"]
    return t.panel(
        t.graph(
            charts.exceedance(probabilities),
            260,
            label="Probability of each simulated drawdown depth",
        ),
        title=f"Monte Carlo drawdown: {fmt.integer(meta['runs'])} paths, one-year horizon",
        subtitle=(
            f"{METHODS[method]} model fitted to development returns · seed "
            f"{meta['seed']} · at the latest weights · saved percentiles only"
        ),
        footer=[
            html.Span(
                [
                    "Median max drawdown: ",
                    html.Span(fmt.pct(-r["drawdown_median"], 1), className="primary"),
                ]
            ),
            html.Span(
                ["95th percentile: ", html.Span(fmt.pct(-r["drawdown_p95"], 1), className="neg")]
            ),
            html.Span(
                [
                    "Median days under water: ",
                    html.Span(fmt.integer(r["underwater_median_days"]), className="strong"),
                ]
            ),
        ],
    )


def stress_panel(rule: str) -> html.Section:
    """The latest weights replayed through the pre-2020 crises."""
    s = data.stress()
    s = s[s["allocation"] == rule]
    if s.empty:
        raise data.DataError("No stress results", f"no saved scenarios for {RULES[rule]}")
    worst = s["max_drawdown"].min()
    items = []
    for _, r in s.iterrows():
        share = r["max_drawdown"] / worst if worst else 0
        items.append(
            html.Div(
                [
                    t.kv(
                        r["scenario"],
                        fmt.pct(r["total_return"], 1, True),
                        fmt.tone(r["total_return"]),
                    ),
                    html.Div(
                        [
                            html.Span(f"{r['start']} → {r['end']} · {int(r['days'])} days"),
                            html.Span(
                                f"Max drawdown {fmt.pct(r['max_drawdown'], 1)} · worst day "
                                f"{fmt.pct(r['worst_day'], 1)}"
                            ),
                        ],
                        className="stress-meta caption",
                    ),
                    html.Div(
                        html.Div(
                            className="bar-fill neg",
                            style={"width": f"{max(3, share * 100):.0f}%"},
                        ),
                        className="bar-track",
                        title=f"Max drawdown relative to the worst scenario ({fmt.pct(worst, 1)})",
                    ),
                ],
                className="stress-row",
            )
        )
    return t.panel(
        items,
        title="Stress tests: historical crises",
        subtitle="Latest development weights replayed through each crisis window (before 2020)",
        footer=[
            html.Span(["Worst drawdown: ", html.Span(fmt.pct(worst, 1), className="neg")]),
            html.Span(
                "No holdout crisis is replayed here; see the backtest page for 2020–2024.",
                className="muted",
            ),
        ],
    )


def comparison() -> html.Section:
    """Every allocation in both periods, against monthly equal weight."""
    rows = []
    for rule, label in RULES.items():
        key = BOOK_OF[rule]
        dev, hold = data.book_stats(key, data.DEVELOPMENT), data.book_stats(key, data.HOLDOUT)
        interval = data.interval_against(key, "equal_weight", data.DEVELOPMENT)
        rows.append(
            {
                "label": label,
                "dev_sharpe": dev["net_sharpe"],
                "dev_vol": dev["net_annual_volatility"],
                "dev_mdd": dev["net_max_drawdown"],
                "vs": views.interval_text(interval) if interval else "reference",
                "hold_sharpe": hold["net_sharpe"],
                "hold_vol": hold["net_annual_volatility"],
                "hold_mdd": hold["net_max_drawdown"],
                "turnover": dev["turnover_annual"],
            }
        )
    columns = [
        t.Col("Allocation", "label", str, numeric=False),
        t.Col("Dev net Sharpe", "dev_sharpe", fmt.sharpe),
        t.Col("Dev vol", "dev_vol", lambda v: fmt.pct(v, 1)),
        t.Col("Dev max DD", "dev_mdd", lambda v: fmt.pct(v, 1)),
        t.Col("Dev vs 1/N [95%]", "vs", str),
        t.Col(
            "Holdout net Sharpe",
            "hold_sharpe",
            fmt.sharpe,
            tone=lambda v, r: views.tone_vs(v, r["dev_sharpe"]),
        ),
        t.Col("Holdout vol", "hold_vol", lambda v: fmt.pct(v, 1)),
        t.Col("Holdout max DD", "hold_mdd", lambda v: fmt.pct(v, 1)),
        t.Col("Turnover / yr", "turnover", lambda v: fmt.number(v, 2) + "×"),
    ]
    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2(
                                "Allocation rules: development and holdout",
                                className="section-title",
                            ),
                            html.Span(views.trials_note(), className="caption"),
                        ],
                        className="titles",
                    ),
                ],
                className="table-head",
            ),
            t.table(rows, columns, highlight=lambda r: r["label"] == "Minimum variance"),
            html.Div(
                [
                    html.Span(
                        "Holdout tone compares each rule with its own development Sharpe. "
                        "Minimum variance is the predeclared primary."
                    ),
                    html.Span(
                        "Development: walk-forward 2006-01-31 → 2019-12-31 · holdout: "
                        "attempt 1, 2020-01-03 → 2024-12-30"
                    ),
                ],
                className="table-foot",
            ),
        ],
        className="panel flush",
    )


@callback(Output("pr-engine", "children"), Output("pr-compare", "children"), RELOAD)
def fixed_blocks(_token: int) -> tuple:
    """Blocks that depend only on the saved results."""
    return t.guarded(engine, "the optimiser settings"), t.guarded(
        comparison, "the allocation comparison"
    )


@callback(
    Output("pr-kpis", "children"),
    Output("pr-frontier", "children"),
    Output("pr-weights", "children"),
    Output("pr-mc", "children"),
    Output("pr-stress", "children"),
    Input("pr-rule", "value"),
    Input("pr-method", "value"),
    RELOAD,
)
def update(rule: str, method: str, _token: int) -> tuple:
    """Redraw the allocation-specific panels."""
    rule, method = rule or "minimum_variance", method or "bootstrap"
    return (
        t.guarded(lambda: kpis(rule, method), "the risk cards"),
        t.guarded(lambda: frontier_panel(rule), "the frontier"),
        t.guarded(lambda: weights_panel(rule), "the weight table"),
        t.guarded(lambda: mc_panel(rule, method), "the drawdown distribution"),
        t.guarded(lambda: stress_panel(rule), "the stress tests"),
    )


def export(controls: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    """Weights and risk contributions of every allocation at its latest rebalance."""
    contrib = data.risk_contributions()
    frames = {
        rule: pd.DataFrame({"weight": data.latest_weights(rule), "risk_share": contrib.loc[rule]})
        for rule in RULES
    }
    return "portfolio_weights_and_risk", pd.concat(frames, names=["allocation", "ticker"])
