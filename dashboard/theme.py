"""The design system: colour tokens, the chart template and the layout components.

The tokens match assets/style.css, which takes them from the Stitch design.
Pages build their structure from these components only, so a panel, a KPI
card or a table looks the same everywhere.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc, html

from dashboard import formatting as fmt
from dashboard.data import DataError
from src.logger import get_logger

logger = get_logger(__name__)

# --- Tokens ------------------------------------------------------------------

BG = "#051424"
SURFACE_LOWEST = "#010f1f"
SURFACE_LOW = "#0d1c2d"
SURFACE = "#122131"
SURFACE_HIGH = "#1c2b3c"
SURFACE_HIGHEST = "#273647"
ON_SURFACE = "#d4e4fa"
ON_SURFACE_VARIANT = "#bdc8d1"
OUTLINE = "#87929a"
OUTLINE_VARIANT = "#3e484f"
PRIMARY = "#8ed5ff"
PRIMARY_DIM = "#7bd0ff"
PRIMARY_CONTAINER = "#38bdf8"
SECONDARY = "#c0c1ff"
POS = "#4edea3"
NEG = "#ffb4ab"
WARN = "#f0c36b"

FONT_UI = "Inter, Segoe UI, system-ui, sans-serif"
FONT_MONO = "JetBrains Mono, Cascadia Mono, Consolas, monospace"

# Series roles: the strategy is always the brightest line, benchmarks recede.
NET = PRIMARY
GROSS = PRIMARY_DIM
BENCHMARK = OUTLINE
BENCHMARK_2 = SECONDARY
HOLDOUT_BAND = "rgba(39, 54, 71, 0.55)"

# The icons bundled in assets/fonts (Material Symbols subset).
ICONS = [
    "account_tree",
    "analytics",
    "candlestick_chart",
    "check_circle",
    "dataset",
    "description",
    "download",
    "error",
    "gavel",
    "hub",
    "info",
    "insights",
    "model_training",
    "pie_chart",
    "query_stats",
    "radar",
    "refresh",
    "restart_alt",
    "rule",
    "science",
    "space_dashboard",
    "storage",
    "terminal",
    "trending_down",
    "trending_up",
    "tune",
    "verified",
    "warning",
    "waterfall_chart",
    "table_chart",
    "shield",
    "history",
    "lock",
    "folder_open",
    "schema",
    "menu",
]

CHART_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def _template() -> go.layout.Template:
    """Restrained dark template: thin lines, faint dashed grid, mono tick labels."""
    axis = {
        "showgrid": True,
        "gridcolor": SURFACE_HIGHEST,
        "griddash": "dot",
        "gridwidth": 1,
        "zeroline": True,
        "zerolinecolor": OUTLINE_VARIANT,
        "zerolinewidth": 1,
        "linecolor": SURFACE_HIGHEST,
        "tickfont": {"family": FONT_MONO, "size": 10, "color": OUTLINE},
        "title": {"font": {"family": FONT_MONO, "size": 10, "color": OUTLINE}, "standoff": 6},
        "ticks": "",
        "automargin": True,
    }
    layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_UI, "size": 11, "color": ON_SURFACE_VARIANT},
        colorway=[NET, BENCHMARK, POS, SECONDARY, WARN, NEG, PRIMARY_CONTAINER],
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        xaxis=axis,
        yaxis=axis,
        hovermode="x unified",
        hoverlabel={
            "bgcolor": SURFACE_HIGH,
            "bordercolor": OUTLINE_VARIANT,
            "font": {"family": FONT_MONO, "size": 11, "color": ON_SURFACE},
        },
        legend={"visible": False},
        showlegend=False,
        bargap=0.25,
    )
    return go.layout.Template(layout=layout)


pio.templates["quantlab"] = _template()
pio.templates.default = "quantlab"


# --- Components ----------------------------------------------------------------


def icon(name: str, size: str = "") -> html.Span:
    """A Material Symbols glyph, hidden from screen readers."""
    return html.Span(name, className=f"icon {size}".strip(), **{"aria-hidden": "true"})


def tag(text: str, tone: str = "", title: str | None = None) -> html.Span:
    """A small uppercase label, optionally toned pos, neg, warn or info."""
    return html.Span(text, className=f"tag {tone}".strip(), title=title)


def badge(text: str, tone: str = "neutral", icon_name: str | None = None) -> html.Span:
    """A status badge: a verdict stated in words, toned but never colour-only."""
    children = ([icon(icon_name, "sm")] if icon_name else []) + [text]
    return html.Span(children, className=f"verdict-badge {tone}", role="status")


def value(text: str, tone: str = "strong", cls: str = "mono-data") -> html.Span:
    """A formatted number in its semantic colour."""
    return html.Span(text, className=f"{cls} {tone}")


def kv(label: Any, val: Any, tone: str = "strong", cls: str = "mono-sm") -> html.Div:
    """One label and value on a line."""
    return html.Div(
        [html.Span(label, className="muted"), html.Span(val, className=tone)], className=f"kv {cls}"
    )


def legend(items: Sequence[tuple[str, str, str]]) -> html.Div:
    """A chart legend: (label, colour, style) where style is line, dash or box."""
    out = []
    for label, colour, style in items:
        swatch_cls = {"line": "swatch", "dash": "swatch dash", "box": "swatch box"}[style]
        out.append(
            html.Span(
                [
                    html.Span(className=swatch_cls, style={"background": colour, "color": colour}),
                    label,
                ],
                className="legend-item mono-sm",
                style={"color": colour},
            )
        )
    return html.Div(out, className="legend")


def panel(
    children: Any,
    title: str | None = None,
    subtitle: Any = None,
    kicker: str | None = None,
    actions: Any = None,
    footer: Any = None,
    cls: str = "",
    span: str = "",
    panel_id: str | None = None,
    title_icon: str | None = None,
) -> html.Section:
    """A design panel: optional kicker, title, subtitle and actions above the content."""
    head = []
    if title or kicker or actions:
        titles = []
        if kicker:
            titles.append(html.Span(kicker, className="caps kicker"))
        if title:
            heading = [icon(title_icon, "lg"), " "] if title_icon else []
            titles.append(html.H2([*heading, title], className="h-sm strong"))
        if subtitle:
            titles.append(html.Span(subtitle, className="mono-sm muted"))
        head.append(
            html.Div(
                [
                    html.Div(titles, className="titles"),
                    html.Div(actions, className="panel-actions"),
                ],
                className="panel-head",
            )
        )
    body = [*head, html.Div(children)]
    if footer:
        body.append(html.Div(footer, className="panel-foot mono-sm"))
    kwargs = {"id": panel_id} if panel_id else {}
    return html.Section(body, className=f"panel {cls} {span}".strip(), **kwargs)


def kpi(
    label: str,
    main: Any,
    unit: str = "",
    icon_name: str | None = None,
    rows: Iterable[tuple[str, Any, str]] = (),
    tone: str = "strong",
    title: str | None = None,
) -> html.Div:
    """A KPI card: caps label, one large monospaced value, then supporting rows."""
    top = [html.Span(label, className="caps")]
    if icon_name:
        top.append(icon(icon_name, "sm"))
    body = [
        html.Div(top, className="kpi-top"),
        html.Div(
            [
                html.Span(main, className=f"mono-display {tone}"),
                html.Span(unit, className="mono-sm muted"),
            ],
            className="kpi-value",
        ),
        html.Div([kv(k, v, t) for k, v, t in rows], className="kpi-rows"),
    ]
    return html.Div(body, className="kpi", title=title)


def kpi_pair(
    label: str,
    left: tuple[str, str, str],
    right: tuple[str, str, str],
    rows: Iterable[tuple[str, Any, str]] = (),
    note: str | None = None,
    note_tone: str = "muted",
) -> html.Div:
    """A KPI card comparing two values side by side, for gross against net."""
    top = [html.Span(label, className="caps")]
    if note:
        top.append(html.Span(note, className=f"tag {note_tone}"))

    def side(item: tuple[str, str, str]) -> html.Div:
        caption, text, tone = item
        return html.Div(
            [
                html.Span(caption, className="caps muted"),
                html.Span(text, className=f"mono-display {tone}"),
            ]
        )

    return html.Div(
        [
            html.Div(top, className="kpi-top"),
            html.Div([side(left), side(right)], className="kpi-pair"),
            html.Div([kv(k, v, t) for k, v, t in rows], className="kpi-rows"),
        ],
        className="kpi",
    )


@dataclass(frozen=True)
class Col:
    """A table column: header, the row field, a formatter and an optional tone rule."""

    header: str
    field: str
    format: Callable[[Any], str] = str
    numeric: bool = True
    tone: Callable[[Any, dict], str] | None = None
    title: str | None = None


def table(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[Col],
    *,
    compact: bool = False,
    highlight: Callable[[dict], bool] | None = None,
    caption: str | None = None,
) -> html.Div:
    """A dense research table: mono numbers right-aligned, labels left, tones on values."""
    head = html.Thead(
        html.Tr(
            [
                html.Th(c.header, className="num" if c.numeric else "", title=c.title, scope="col")
                for c in columns
            ]
        )
    )
    body = []
    for row in rows:
        cells = []
        for c in columns:
            raw = row.get(c.field)
            text = fmt.MISSING if fmt.is_missing(raw) and c.format is not str else c.format(raw)
            tone = c.tone(raw, row) if c.tone else ""
            cls = ("num " if c.numeric else "lbl ") + tone
            cells.append(html.Td(text, className=cls.strip()))
        body.append(html.Tr(cells, className="highlight" if highlight and highlight(row) else ""))
    parts = (
        [html.Caption(caption, className="sr-only", style={"display": "none"})] if caption else []
    )
    return html.Div(
        html.Table(
            [*parts, head, html.Tbody(body)], className=f"data {'compact' if compact else ''}"
        ),
        className="table-wrap",
    )


def empty(title: str, detail: str = "", kind: str = "info") -> html.Div:
    """An honest empty state: what is missing and why, never a chart of zeros."""
    glyph = {"info": "info", "missing": "folder_open", "error": "error", "invalid": "warning"}[kind]
    return html.Div(
        [
            icon(glyph),
            html.Div(
                [html.Div(title, className="empty-title"), html.Div(detail, className="body-sm")]
            ),
        ],
        className=f"empty {kind}",
        role="note",
    )


def graph(figure: go.Figure, height: int, graph_id: str | None = None, label: str = "") -> html.Div:
    """A chart in its dark well, sized by the panel and resizing with it."""
    kwargs = {"id": graph_id} if graph_id else {}
    return html.Div(
        dcc.Graph(
            figure=figure,
            config=CHART_CONFIG,
            style={"height": f"{height}px"},
            responsive=True,
            **kwargs,
        ),
        className="chart-well",
        role="img",
        **{"aria-label": label} if label else {},
    )


def guarded(build: Callable[[], Any], where: str) -> Any:
    """Build a view, turning a data problem into an empty state and logging anything else.

    The reader sees one plain sentence; the traceback goes to the log.
    """
    try:
        return build()
    except DataError as exc:
        return empty(exc.title, exc.detail, "missing")
    # One broken panel must not take the page down with it.
    except Exception as exc:
        logger.exception("Dashboard panel failed: %s", where)
        return empty(
            "This panel could not be built",
            f"{type(exc).__name__} while building {where}. Details are in logs/pipeline.log.",
            "error",
        )
