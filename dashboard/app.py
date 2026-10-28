"""The dashboard shell: sidebar, global controls, routing and the shared actions.

    python -m dashboard.app            # http://127.0.0.1:8050
    python -m dashboard.app --port 8060 --debug

The header's controls are the single source of truth for the shared
selections (asset, date window, period, sampling, model, strategy,
benchmark). Pages read them as callback inputs, so a selection survives
navigation. Each page declares which controls it uses; the others are dimmed
there, so a control never appears to do something it does not.

Every page is a fast structure (built once per navigation) with slow parts
filled by callbacks, so changing a control redraws only the figures that
depend on it and the page never blanks. Nothing here writes a result, logs a
trial or opens the holdout: Reload re-reads the saved files, Export writes a
CSV of what the page shows.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import dash
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import controls, data
from dashboard import formatting as fmt
from dashboard import theme as t
from src.logger import get_logger

logger = get_logger(__name__)

ASSETS = Path(__file__).resolve().parent / "assets"
PAGE_MODULES = (
    "01_overview",
    "02_market_data",
    "03_feature_research",
    "04_model_performance",
    "05_backtesting",
    "06_portfolio_risk",
    "07_experiments",
)


def load_pages() -> dict[str, ModuleType]:
    """Import the seven page modules, keyed by their URL path."""
    pages = {}
    for name in PAGE_MODULES:
        module = importlib.import_module(f"dashboard.pages.{name}")
        pages[module.PATH] = module
    return pages


PAGES = load_pages()


CONTROL_STATES = controls.STATES
CONTROLS = controls.CONTROLS


# --- Shell ---------------------------------------------------------------------------


def sidebar() -> html.Aside:
    """Brand, the seven pages with the active one marked, and the run's status."""
    try:
        version = data.provenance()["code_version"]
    except data.DataError:
        version = "unknown"
    links = [
        dcc.Link(
            [
                t.icon(page.ICON),
                html.Span(
                    [
                        html.Span(page.TITLE, className="nav-title"),
                        html.Span(page.SUBTITLE, className="nav-sub"),
                    ]
                ),
            ],
            href=path,
            id={"type": "nav", "path": path},
            className="nav-item",
        )
        for path, page in PAGES.items()
    ]
    try:
        conclusion = data.holdout_complete().get("conclusion", fmt.MISSING)
        holdout = ("Attempt 1 complete", "strong")
    except data.DataError:
        holdout, conclusion = ("Not run", "warn"), fmt.MISSING
    return html.Aside(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(t.icon("terminal"), className="brand-mark"),
                                    html.Div(
                                        [
                                            html.Div("QuantLab", className="brand-name"),
                                            html.Div("Research terminal", className="brand-sub"),
                                        ]
                                    ),
                                ],
                                className="brand-row",
                            ),
                            html.Div(
                                [
                                    html.Span("stock-return-predictor", className="caption"),
                                    t.tag(
                                        version,
                                        "" if "dirty" not in version else "warn",
                                        "Code version (git commit)",
                                    ),
                                ],
                                className="brand-meta",
                            ),
                        ],
                        className="brand",
                    ),
                    html.Div(
                        html.Span("Research questions", className="eyebrow"), className="nav-label"
                    ),
                    html.Nav(links, className="nav", **{"aria-label": "Pages"}),
                ]
            ),
            html.Div(
                [
                    t.kv("Holdout", holdout[0], holdout[1]),
                    t.kv(
                        "Verdict", conclusion, "neg" if "no convincing" in conclusion else "strong"
                    ),
                    t.kv("Trials (frozen)", _frozen_n(), "strong"),
                ],
                className="sidebar-foot",
            ),
        ],
        className="sidebar",
        id="sidebar",
    )


def _frozen_n() -> str:
    try:
        return f"N = {data.freeze()['headline']['n_trials']}"
    except (data.DataError, KeyError):
        return fmt.MISSING


def _dropdown(control_id: str, options: list[dict], value: str) -> dcc.Dropdown:
    return dcc.Dropdown(
        id=control_id, options=options, value=value, clearable=False, searchable=False
    )


def toolbar() -> html.Header:
    """Global filters in two groups, the data status and the shared actions.

    Every control is always mounted (so its value survives navigation); each
    page shows only the ones it uses, and a page with none says so.
    """
    d = controls.defaults()
    seg = {"className": "seg", "inline": True}
    strategies = [{"label": data.BOOKS[k].label, "value": k} for k in data.STRATEGIES]
    benchmarks = [{"label": data.BOOKS[k].label, "value": k} for k in data.BENCHMARKS]
    models = [{"label": v, "value": k} for k, v in controls.MODEL_LABELS.items()]
    market = html.Div(
        [
            t.field(
                "Asset",
                _dropdown(
                    "ctl-asset", [{"label": x, "value": x} for x in data.tickers()], d["asset"]
                ),
                "field-asset",
                "w-sm",
            ),
            t.field(
                "Window",
                dcc.DatePickerRange(
                    id="ctl-dates",
                    start_date=d["start"],
                    end_date=d["end"],
                    min_date_allowed=d["start"],
                    max_date_allowed=d["end"],
                    display_format="YYYY-MM-DD",
                    minimum_nights=20,
                    first_day_of_week=1,
                ),
                "field-dates",
            ),
            t.field(
                "Sampling",
                dcc.RadioItems(
                    id="ctl-freq",
                    options=[
                        {"label": "Daily", "value": "1D"},
                        {"label": "Weekly", "value": "1W"},
                        {"label": "Monthly", "value": "1M"},
                    ],
                    value=d["freq"],
                    **seg,
                ),
                "field-freq",
            ),
        ],
        className="filter-group",
        id="group-market",
        role="group",
        **{"aria-label": "Market data filters"},
    )
    evaluation = html.Div(
        [
            t.field(
                "Period",
                dcc.RadioItems(
                    id="ctl-period",
                    options=[
                        {
                            "label": html.Span(
                                "Development", title="2005–2019, out-of-sample CPCV"
                            ),
                            "value": data.DEVELOPMENT,
                        },
                        {
                            "label": html.Span("Holdout", title="2020–2024, one-time evaluation"),
                            "value": data.HOLDOUT,
                        },
                    ],
                    value=d["period"],
                    **seg,
                ),
                "field-period",
            ),
            t.field("Model", _dropdown("ctl-model", models, d["model"]), "field-model", "w-lg"),
            t.field(
                "Strategy", _dropdown("ctl-strategy", strategies, d["strategy"]), "field-strategy"
            ),
            t.field(
                "Benchmark",
                _dropdown("ctl-benchmark", benchmarks, d["benchmark"]),
                "field-benchmark",
            ),
        ],
        className="filter-group",
        id="group-evaluation",
        role="group",
        **{"aria-label": "Evaluation filters"},
    )
    note = html.Span(
        "No global filters on this page; its own controls are under the page title.",
        id="toolbar-note",
        className="toolbar-note hidden",
    )
    actions = html.Div(
        [
            html.Div(id="data-status", className="status"),
            html.Button(
                t.icon("refresh", "sm"),
                id="btn-reload",
                className="btn icon-only",
                title="Reload: re-read every saved result from disk. "
                "Nothing is recomputed or rerun.",
                **{"aria-label": "Reload saved results"},
            ),
            html.Button(
                [t.icon("download", "sm"), "Export CSV"],
                id="btn-export",
                className="btn",
                title="Download the main table of this page, as shown, as CSV",
            ),
            html.Button(
                t.icon("restart_alt", "sm"),
                id="btn-reset",
                className="btn icon-only",
                title="Reset every filter to its default",
                **{"aria-label": "Reset filters"},
            ),
        ],
        className="actions",
    )
    menu = html.Button(
        t.icon("menu"),
        id="btn-menu",
        className="btn icon-only menu-toggle",
        title="Show pages",
        **{"aria-label": "Show pages"},
    )
    filters = html.Div(
        [market, evaluation, note],
        className="filters",
        role="toolbar",
        **{"aria-label": "Global filters"},
    )
    return html.Header([menu, filters, actions], className="toolbar")


def serve_layout() -> html.Div:
    """The whole app: shell around a page container the router fills."""
    return html.Div(
        [
            dcc.Location(id="url"),
            dcc.Store(id="reload-token", data=0),
            dcc.Download(id="download"),
            sidebar(),
            html.Div(
                [
                    toolbar(),
                    html.Main(
                        dcc.Loading(
                            html.Div(id="page", className="page"),
                            type="dot",
                            color=t.PRIMARY,
                            delay_show=400,
                        ),
                        className="main",
                        id="main",
                    ),
                ],
                className="content",
            ),
            html.Div(id="export-status", role="status", className="sr-only"),
        ]
    )


def create_app() -> Dash:
    """Build the Dash app with the assets folder and the global callbacks."""
    app = Dash(
        __name__,
        assets_folder=str(ASSETS),
        title="QuantLab · Research Terminal",
        update_title=None,
        suppress_callback_exceptions=True,
    )
    app.layout = serve_layout
    return app


# --- Global callbacks --------------------------------------------------------------


@dash.callback(Output("page", "children"), Input("url", "pathname"))
def route(pathname: str) -> Any:
    """Build the structure of the page at this path."""
    page = PAGES.get(pathname or "/")
    if page is None:
        return t.empty("Page not found", f"{pathname} is not one of the seven pages.", "missing")
    return t.guarded(page.layout, f"the {page.TITLE} page")


GROUPS = {
    "market": ("asset", "dates", "freq"),
    "evaluation": ("period", "model", "strategy", "benchmark"),
}


@dash.callback(
    Output({"type": "nav", "path": ALL}, "className"),
    # dcc.Link has no aria-current; its title names the current page instead.
    Output({"type": "nav", "path": ALL}, "title"),
    [Output(f"field-{c}", "className") for c in CONTROLS],
    [Output(f"group-{g}", "className") for g in GROUPS],
    Output("toolbar-note", "className"),
    Input("url", "pathname"),
)
def mark_page(pathname: str) -> tuple:
    """Highlight the current page and show only the global filters it uses."""
    paths = list(PAGES)
    current = pathname if pathname in PAGES else "/"
    classes = ["nav-item active" if p == current else "nav-item" for p in paths]
    titles = ["Current page" if p == current else PAGES[p].TITLE for p in paths]
    uses = PAGES[current].USES
    sizes = {"asset": " w-sm", "model": " w-lg"}
    fields = [f"field{sizes.get(c, '')}" + ("" if c in uses else " hidden") for c in CONTROLS]
    groups = [
        "filter-group" if any(c in uses for c in members) else "filter-group hidden"
        for members in GROUPS.values()
    ]
    note = "toolbar-note" if not uses else "toolbar-note hidden"
    return classes, titles, *fields, *groups, note


@dash.callback(
    Output("sidebar", "className"),
    Input("btn-menu", "n_clicks"),
    Input("url", "pathname"),
    State("sidebar", "className"),
)
def toggle_menu(clicks: int | None, pathname: str, current: str) -> str:
    """Open and close the sidebar on narrow screens; navigating closes it."""
    if ctx.triggered_id == "btn-menu" and "open" not in (current or ""):
        return "sidebar open"
    return "sidebar"


@dash.callback(
    Output("reload-token", "data"),
    Input("btn-reload", "n_clicks"),
    State("reload-token", "data"),
    prevent_initial_call=True,
)
def reload_results(clicks: int | None, token: int) -> int:
    """Forget the cached files so every page re-reads the saved results."""
    data.clear_cache()
    return (token or 0) + 1


@dash.callback(Output("data-status", "children"), Input("reload-token", "data"))
def show_status(token: int) -> Any:
    """Last price date, when results were written and when they were loaded here."""
    try:
        p = data.provenance()
    except data.DataError as exc:
        return html.Span(exc.title, className="caption neg")
    loaded = datetime.now(UTC).strftime("%H:%M UTC")
    return [
        html.Span(
            [html.Span(className="dot"), f"Prices to {fmt.date(p['last_bar'])}"],
            className="caption",
        ),
        html.Span(f"Results loaded {loaded}", className="caption"),
    ]


@dash.callback(
    [Output(s.component_id, s.component_property) for s in CONTROL_STATES],
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_controls(clicks: int | None) -> tuple:
    """Every control back to its default."""
    d = controls.defaults()
    return (
        d["asset"],
        d["start"],
        d["end"],
        d["period"],
        d["freq"],
        d["model"],
        d["strategy"],
        d["benchmark"],
    )


@dash.callback(
    Output("ctl-strategy", "value", allow_duplicate=True),
    Input({"type": "pick-strategy", "key": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def pick_strategy(clicks: list) -> Any:
    """A strategy tab on a page sets the global strategy (one source of truth)."""
    if not ctx.triggered_id or not any(clicks or []):
        return no_update
    return ctx.triggered_id["key"]


@dash.callback(
    Output("ctl-model", "value", allow_duplicate=True),
    Input({"type": "pick-model", "key": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def pick_model(clicks: list) -> Any:
    """A model tab on the model page sets the global model."""
    if not ctx.triggered_id or not any(clicks or []):
        return no_update
    return ctx.triggered_id["key"]


@dash.callback(
    Output("ctl-period", "value", allow_duplicate=True),
    Input({"type": "pick-period", "key": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def pick_period(clicks: list) -> Any:
    """A period toggle on a page sets the global evaluation period."""
    if not ctx.triggered_id or not any(clicks or []):
        return no_update
    return ctx.triggered_id["key"]


@dash.callback(
    Output("download", "data"),
    Output("export-status", "children"),
    Input("btn-export", "n_clicks"),
    State("url", "pathname"),
    *CONTROL_STATES,
    prevent_initial_call=True,
)
def export(clicks: int | None, pathname: str, *values: Any) -> tuple:
    """Write the current page's main table, as displayed, to a CSV download."""
    page = PAGES.get(pathname or "/", PAGES["/"])
    try:
        name, frame = page.export(controls.selection(*values))
    except data.DataError as exc:
        return no_update, f"Nothing to export: {exc.detail}"
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    return dcc.send_data_frame(frame.to_csv, f"{name}_{stamp}.csv"), f"Exported {name}"


app = create_app()
server = app.server


def main(argv: list[str] | None = None) -> None:
    """Serve the dashboard locally."""
    parser = argparse.ArgumentParser(description="Serve the research dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true", help="hot reload and the Dash dev tools")
    args = parser.parse_args(argv)
    logger.info("Dashboard on http://%s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
