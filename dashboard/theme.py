"""The design system: colour tokens, the chart template and the layout components.

The tokens match assets/style.css. Pages build their structure from these
components only, so a page header, a panel, a KPI, a stat strip or a table
looks the same everywhere. The rules the components enforce:

- Inter for what a person reads, JetBrains Mono only for numbers and IDs.
- Colour carries meaning (sign, status, selection), never decoration.
- A panel's title stays on one line; legends and subtitles sit beneath it.
- Figures inside a panel are a stat strip with dividers, not nested cards.
"""

from __future__ import annotations

import re
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

# --- Tokens (mirrors :root in assets/style.css) ----------------------------------

BG = "#07111e"
SURFACE_LOWEST = BG
SURFACE_LOW = "#0a1626"
SURFACE = "#0e1b2b"
SURFACE_HIGH = "#132337"
SURFACE_HIGHEST = "#1a2d44"
ON_SURFACE = "#dde7f2"
ON_SURFACE_VARIANT = "#b0bccb"
OUTLINE = "#7f8d9d"
OUTLINE_VARIANT = "#2a3e57"
GRID = "rgba(221, 231, 242, 0.06)"
PRIMARY = "#4db4e6"
PRIMARY_DIM = "#8fd0f2"
PRIMARY_CONTAINER = "#4db4e6"
SECONDARY = "#a99cf0"
POS = "#3ec28b"
NEG = "#e8776d"
WARN = "#d9a64a"

FONT_UI = "Inter, Segoe UI, system-ui, sans-serif"
FONT_MONO = "JetBrains Mono, Cascadia Mono, Consolas, monospace"

# Series roles, the same on every page: the strategy's net line is the accent,
# its gross line a lighter dash, benchmarks recede into grey.
NET = PRIMARY
GROSS = PRIMARY_DIM
BENCHMARK = OUTLINE
BENCHMARK_2 = SECONDARY
HOLDOUT_BAND = "rgba(77, 180, 230, 0.07)"

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
    """Quiet axes: faint solid grid, mono tick labels, sans axis titles."""
    axis = {
        "showgrid": True,
        "gridcolor": GRID,
        "gridwidth": 1,
        "zeroline": True,
        "zerolinecolor": OUTLINE_VARIANT,
        "zerolinewidth": 1,
        "showline": False,
        "tickfont": {"family": FONT_MONO, "size": 10, "color": OUTLINE},
        "title": {"font": {"family": FONT_UI, "size": 11, "color": OUTLINE}, "standoff": 8},
        "ticks": "",
        "automargin": True,
    }
    layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_UI, "size": 11, "color": ON_SURFACE_VARIANT},
        colorway=[NET, BENCHMARK, POS, SECONDARY, WARN, NEG],
        margin={"l": 4, "r": 4, "t": 6, "b": 4},
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

# A value is set in the UI face, not mono, when it reads as words.
_WORDS = re.compile(r"[A-Za-z]{4,}")


# --- Primitives ---------------------------------------------------------------------


def icon(name: str, size: str = "") -> html.Span:
    """A Material Symbols glyph, hidden from screen readers."""
    return html.Span(name, className=f"icon {size}".strip(), **{"aria-hidden": "true"})


def tag(text: str, tone: str = "", title: str | None = None) -> html.Span:
    """A small outlined label for status or provenance (pos, neg, warn or info)."""
    return html.Span(text, className=f"tag {tone}".strip(), title=title)


def badge(text: str, tone: str = "neutral", icon_name: str | None = None) -> html.Span:
    """A status statement: the verdict in words, toned, never colour alone."""
    children = ([icon(icon_name, "sm")] if icon_name else []) + [text]
    return html.Span(children, className=f"verdict-badge {tone}", role="status")


def value(text: str, tone: str = "strong", cls: str = "num") -> html.Span:
    """A formatted number in its semantic colour."""
    return html.Span(text, className=f"{cls} {tone}")


def kv(label: Any, val: Any, tone: str = "strong", cls: str = "") -> html.Div:
    """One label and value on a line; words in the UI face, numbers in mono."""
    words = isinstance(val, str) and bool(_WORDS.search(val))
    return html.Div(
        [
            html.Span(label, title=label if isinstance(label, str) else None),
            html.Span(val, className=tone, title=val if isinstance(val, str) else None),
        ],
        className=f"kv {'text' if words else ''} {cls}".strip(),
    )


def note(text: Any, icon_name: str = "info") -> html.Div:
    """An explanatory line under a table or chart."""
    return html.Div([icon(icon_name, "sm"), html.Span(text)], className="note")


def field(label: str, control: Any, field_id: str | None = None, cls: str = "") -> html.Div:
    """A labelled control: label left of the control, one height for all."""
    kwargs = {"id": field_id} if field_id else {}
    return html.Div(
        [html.Span(label, className="label"), control], className=f"field {cls}".strip(), **kwargs
    )


def legend(items: Sequence[tuple[str, str, str]]) -> html.Div:
    """A chart legend: (label, colour, style) where style is line, dash or box."""
    out = []
    for label, colour, style in items:
        swatch_cls = {"line": "swatch", "dash": "swatch dash", "box": "swatch box"}[style]
        out.append(
            html.Span(
                [
                    html.Span(
                        className=swatch_cls, style={"background": colour, "borderColor": colour}
                    ),
                    label,
                ],
                className="legend-item",
            )
        )
    return html.Div(out, className="legend")


# --- Page structure ---------------------------------------------------------------------


def page_header(
    title: str,
    description: Any = None,
    eyebrow: str | None = None,
    meta: Iterable[tuple[str, Any]] = (),
    status: Any = None,
    actions: Any = None,
) -> html.Header:
    """The page's title, what it answers, its scope, and its status or actions."""
    main = []
    if eyebrow:
        main.append(html.Span(eyebrow, className="eyebrow"))
    main.append(html.H1(title, className="page-title"))
    if description:
        main.append(html.P(description, className="description"))
    items = [html.Span([f"{label} ", html.B(val)], className="meta") for label, val in meta]
    if items:
        main.append(html.Div(items, className="meta-row"))
    side = []
    if status is not None:
        side.append(status)
    if actions:
        side.append(html.Div(actions, className="page-actions"))
    return html.Header(
        [html.Div(main, className="page-head-main"), html.Div(side, className="page-head-side")],
        className="page-head",
    )


def page_controls(children: Any) -> html.Div:
    """Controls that belong to this page only, under its header."""
    return html.Div(
        children, className="page-controls", role="group", **{"aria-label": "Page controls"}
    )


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
    legend: Any = None,
) -> html.Section:
    """A section of the page: one-line title, then subtitle and legend, then content.

    The title row holds only the title and small actions, so it never wraps;
    the legend sits on the subtitle row beneath it.
    """
    head = []
    if kicker:
        head.append(html.Span(kicker, className="kicker"))
    if title or actions:
        titles = [icon(title_icon)] if title_icon else []
        if title:
            titles.append(html.H2(title, className="section-title", title=title))
        head.append(
            html.Div(
                [
                    html.Div(titles, className="titles"),
                    html.Div(actions, className="panel-actions"),
                ],
                className="panel-head-row",
            )
        )
    if subtitle or legend:
        head.append(
            html.Div(
                [html.Span(subtitle, className="caption") if subtitle else html.Span(), legend],
                className="panel-sub",
            )
        )
    body = []
    if head:
        body.append(html.Div(head, className="panel-head"))
    body.append(html.Div(children, className="panel-body"))
    if footer:
        body.append(html.Div(footer, className="panel-foot"))
    kwargs = {"id": panel_id} if panel_id else {}
    return html.Section(body, className=f"panel {cls} {span}".strip(), **kwargs)


def stat(
    label: str, val: Any, caption: Any = None, tone: str = "", share: float | None = None
) -> html.Div:
    """One figure in a stat strip: label, value, optional caption and proportion bar."""
    words = isinstance(val, str) and bool(_WORDS.search(val))
    parts = [
        html.Span(label, className="label"),
        html.Span(
            val, className=f"stat-value {'text' if words else ''} {tone}".strip(), title=str(val)
        ),
    ]
    if caption:
        parts.append(html.Span(caption, className="caption"))
    if share is not None:
        width = f"{max(2.0, min(100.0, share * 100)):.0f}%"
        parts.append(
            html.Div(html.Div(className="bar-fill", style={"width": width}), className="bar-track")
        )
    return html.Div(parts, className="stat")


def stat_grid(items: Sequence[html.Div]) -> html.Div:
    """Figures side by side, separated by rules: the in-panel alternative to cards."""
    return html.Div(list(items), className="stat-grid", style={"--n": len(items)})


def kpi(
    label: str,
    main: Any,
    caption: Any = "",
    rows: Iterable[tuple[str, Any, str]] = (),
    tone: str = "strong",
    title: str | None = None,
    badge_text: tuple[str, str] | None = None,
) -> html.Div:
    """A headline figure: label, one large value, a caption, then supporting rows."""
    top = [html.Span(label, className="label")]
    if badge_text:
        top.append(tag(*badge_text))
    body = [
        html.Div(top, className="kpi-top"),
        html.Div(main, className=f"kpi-value {tone}".strip()),
        html.Div(
            caption, className="kpi-caption", title=caption if isinstance(caption, str) else None
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
    note_tone: str = "",
) -> html.Div:
    """Two figures side by side under one label, for gross against net."""
    top = [html.Span(label, className="label")]
    if note:
        top.append(tag(note, note_tone))

    def side(item: tuple[str, str, str]) -> html.Div:
        caption, text, tone = item
        return html.Div(
            [
                html.Span(caption, className="kpi-caption"),
                html.Span(text, className=f"kpi-value {tone}"),
            ],
            className="side",
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
    """A dense research table: numbers right-aligned in mono, labels left in the UI face."""
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
    parts = [html.Caption(caption, className="sr-only")] if caption else []
    return html.Div(
        html.Table(
            [*parts, head, html.Tbody(body)],
            className=f"data {'compact' if compact else ''}".strip(),
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
    """A chart sized by its panel and resizing with it."""
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
