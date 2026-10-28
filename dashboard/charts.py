"""Every chart the dashboard draws, built from data the pages pass in.

Charts compute nothing a reader would need to check: they plot the series
they are given, with units in the axis titles, exact values on hover, and the
holdout shaded wherever a time axis crosses 2020-01-01.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from dashboard import theme as t

# Diverging scale centred on zero: red for negative, neutral surface at zero,
# blue for positive; equal lightness steps either side.
DIVERGING = [
    [0.0, "#b4483f"],
    [0.25, "#5e2f34"],
    [0.5, t.SURFACE_HIGH],
    [0.75, "#1d4f6e"],
    [1.0, "#3a9fd0"],
]


@dataclass(frozen=True)
class Line:
    """One series on a time-series chart."""

    name: str
    series: pd.Series
    colour: str
    dash: str = "solid"
    width: float = 1.5
    fill: str | None = None


def _hover_pct(name: str, digits: int = 2) -> str:
    return f"{name}: %{{y:.{digits}%}}<extra></extra>"


def shade_holdout(figure: go.Figure, start: pd.Timestamp | None, end: pd.Timestamp | None) -> None:
    """Shade the holdout window and label it, when the axis reaches it."""
    if start is None or end is None or end < start:
        return
    figure.add_vrect(x0=start, x1=end, fillcolor=t.HOLDOUT_BAND, line_width=0, layer="below")
    figure.add_vline(x=start, line={"color": t.OUTLINE, "dash": "dot", "width": 1})
    figure.add_annotation(
        x=start,
        y=0,
        yref="paper",
        xanchor="left",
        yanchor="bottom",
        xshift=4,
        yshift=2,
        showarrow=False,
        text="Holdout",
        font={"family": t.FONT_UI, "size": 10, "color": t.OUTLINE},
    )


def growth(lines: Sequence[Line], holdout: tuple | None = None, log_y: bool = False) -> go.Figure:
    """Value of 100 invested, one line per series, holdout shaded."""
    figure = go.Figure()
    for line in lines:
        figure.add_trace(
            go.Scatter(
                x=line.series.index,
                y=line.series.to_numpy(),
                name=line.name,
                mode="lines",
                line={"color": line.colour, "width": line.width, "dash": line.dash},
                hovertemplate=f"{line.name}: %{{y:.1f}}<extra></extra>",
            )
        )
    if holdout:
        shade_holdout(figure, *holdout)
    figure.update_yaxes(title_text="value of 100 invested", type="log" if log_y else "linear")
    if log_y:
        figure.update_yaxes(tickvals=_doubling_ticks(lines), tickformat=",.0f")
    figure.update_xaxes(hoverformat="%Y-%m-%d")
    return figure


def _doubling_ticks(lines: Sequence[Line]) -> list[float]:
    """Tick values at 100 times powers of two spanning the data, for a readable log axis."""
    values = pd.concat([line.series for line in lines]).dropna()
    if values.empty or values.min() <= 0:
        return []
    low, high = np.floor(np.log2(values.min() / 100)), np.ceil(np.log2(values.max() / 100))
    return [100 * 2.0**k for k in np.arange(low, high + 1)]


def drawdowns(lines: Sequence[Line], holdout: tuple | None = None) -> go.Figure:
    """Fall from the running peak, the strategy filled, benchmarks as lines."""
    figure = go.Figure()
    for line in lines:
        figure.add_trace(
            go.Scatter(
                x=line.series.index,
                y=line.series.to_numpy(),
                name=line.name,
                mode="lines",
                line={"color": line.colour, "width": line.width, "dash": line.dash},
                fill=line.fill,
                fillcolor="rgba(255, 180, 171, 0.12)" if line.fill else None,
                hovertemplate=_hover_pct(line.name, 1),
            )
        )
    if holdout:
        shade_holdout(figure, *holdout)
    figure.update_yaxes(title_text="drawdown from peak", tickformat=".0%")
    figure.update_xaxes(hoverformat="%Y-%m-%d")
    return figure


def quintiles(frame: pd.DataFrame) -> go.Figure:
    """Mean realised next-day return per forecast quintile, with 95% intervals, in bps."""
    x = list(frame.index.astype(str))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=frame["forecast"] * 1e4,
            name="Mean forecast",
            mode="lines+markers",
            line={"color": t.OUTLINE, "width": 1, "dash": "dot"},
            marker={"size": 5},
            hovertemplate="Mean forecast: %{y:.2f} bps<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=frame["realised"] * 1e4,
            name="Realised",
            mode="lines+markers",
            line={"color": t.PRIMARY, "width": 2},
            marker={"size": 7},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": (frame["high"] - frame["realised"]) * 1e4,
                "arrayminus": (frame["realised"] - frame["low"]) * 1e4,
                "color": t.PRIMARY_DIM,
                "thickness": 1,
                "width": 4,
            },
            customdata=np.c_[frame["low"] * 1e4, frame["high"] * 1e4, frame["n"]],
            hovertemplate=(
                "Realised: %{y:.2f} bps<br>95% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
                "<br>n = %{customdata[2]:,.0f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(hovermode="x")
    figure.update_yaxes(title_text="next-day return, bps")
    figure.update_xaxes(title_text="forecast quintile (Q1 lowest)", showgrid=False)
    return figure


def price_panel(prices: pd.Series, volume: pd.Series, returns: pd.Series, ticker: str) -> go.Figure:
    """Adjusted close, volume and daily returns on one shared date axis."""
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.18, 0.22], vertical_spacing=0.03
    )
    figure.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices,
            name="Adjusted close",
            line={"color": t.PRIMARY, "width": 1.4},
            hovertemplate="Adj. close: %{y:$,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=volume.index,
            y=volume,
            name="Volume",
            marker={"color": t.SURFACE_HIGHEST, "line": {"width": 0}},
            hovertemplate="Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    colours = np.where(returns.to_numpy() >= 0, t.POS, t.NEG)
    figure.add_trace(
        go.Bar(
            x=returns.index,
            y=returns,
            name="Return",
            marker={"color": colours, "line": {"width": 0}},
            hovertemplate="Return: %{y:.2%}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    figure.update_yaxes(title_text=f"{ticker} adj. close, $", row=1, col=1, tickprefix="$")
    figure.update_yaxes(title_text="volume", row=2, col=1, showgrid=False)
    figure.update_yaxes(title_text="return", row=3, col=1, tickformat=".0%")
    figure.update_xaxes(hoverformat="%Y-%m-%d")
    figure.update_layout(bargap=0)
    return figure


def distribution(returns: pd.Series, label: str) -> go.Figure:
    """Histogram of returns as a density, against the normal with the same mean and spread."""
    data = returns.dropna()
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=data,
            histnorm="probability density",
            nbinsx=min(80, max(10, len(data) // 25)),
            marker={"color": t.PRIMARY_CONTAINER, "line": {"width": 0}},
            opacity=0.55,
            name="Empirical",
            hovertemplate=f"{label} return %{{x:.2%}}<br>density %{{y:.1f}}<extra></extra>",
        )
    )
    if len(data) > 2 and data.std() > 0:
        grid = np.linspace(data.min(), data.max(), 200)
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=stats.norm.pdf(grid, data.mean(), data.std()),
                name="Normal",
                line={"color": t.OUTLINE, "width": 1.2, "dash": "dot"},
                hovertemplate="normal density %{y:.1f}<extra></extra>",
            )
        )
    figure.update_layout(hovermode="closest", bargap=0.02)
    figure.update_xaxes(title_text=f"{label} return", tickformat=".1%")
    figure.update_yaxes(title_text="density")
    return figure


def volatility(frame: pd.DataFrame, median: float, high: float) -> go.Figure:
    """Rolling annualised volatility with the median and 95th percentile of its history."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["volatility"],
            name="21-day volatility",
            line={"color": t.PRIMARY, "width": 1.3},
            hovertemplate="21-day vol: %{y:.1%}<extra></extra>",
        )
    )
    for level, name, colour in ((median, "Median", t.OUTLINE), (high, "95th pct", t.NEG)):
        figure.add_hline(y=level, line={"color": colour, "dash": "dash", "width": 1})
        figure.add_annotation(
            x=1,
            xref="paper",
            y=level,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            text=f"{name} {level:.1%}",
            font={"family": t.FONT_UI, "size": 10, "color": colour},
        )
    figure.update_yaxes(title_text="annualised volatility", tickformat=".0%")
    figure.update_xaxes(hoverformat="%Y-%m-%d")
    return figure


def autocorrelation(frame: pd.DataFrame) -> go.Figure:
    """ACF of returns (bars) and of absolute returns (markers) with the 95% white-noise band."""
    band = float(frame["band"].iloc[0]) if len(frame) else 0.0
    figure = go.Figure()
    figure.add_hrect(y0=-band, y1=band, fillcolor="rgba(135,146,154,0.10)", line_width=0)
    figure.add_trace(
        go.Bar(
            x=frame["lag"],
            y=frame["returns"],
            name="Returns",
            marker={"color": t.PRIMARY, "line": {"width": 0}},
            width=0.35,
            hovertemplate="lag %{x}: returns %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["lag"],
            y=frame["absolute"],
            name="|Returns|",
            mode="markers",
            marker={"color": t.WARN, "size": 6, "symbol": "diamond"},
            hovertemplate="lag %{x}: |returns| %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(hovermode="x")
    figure.update_xaxes(title_text="lag, trading days", dtick=2, showgrid=False)
    figure.update_yaxes(title_text="autocorrelation")
    return figure


def heatmap(matrix: pd.DataFrame, label: str = "ρ", show_text: bool = True) -> go.Figure:
    """A correlation matrix on a diverging scale centred on zero, exact values on hover."""
    z = matrix.to_numpy(dtype=float)
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=list(matrix.columns),
            y=list(matrix.index),
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=DIVERGING,
            text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v:+.2f}")(z)
            if show_text
            else None,
            texttemplate="%{text}" if show_text else None,
            textfont={"family": t.FONT_MONO, "size": 9, "color": t.ON_SURFACE},
            hovertemplate=f"%{{y}} × %{{x}}<br>{label} = %{{z:.3f}}<extra></extra>",
            xgap=1,
            ygap=1,
            colorbar={
                "thickness": 8,
                "len": 0.9,
                "tickfont": {"family": t.FONT_MONO, "size": 9, "color": t.OUTLINE},
                "outlinewidth": 0,
            },
        )
    )
    figure.update_layout(hovermode="closest")
    crowded = len(matrix.columns) > 12
    figure.update_xaxes(showgrid=False, side="top", tickangle=-60 if crowded else 0, zeroline=False)
    figure.update_yaxes(showgrid=False, autorange="reversed", zeroline=False)
    return figure


def importance(frame: pd.DataFrame) -> go.Figure:
    """Permutation importance per feature with its spread across folds, largest at the top."""
    data = frame.sort_values("perm_mean")
    colours = np.where(data["perm_mean"] > 0, t.PRIMARY_CONTAINER, t.OUTLINE_VARIANT)
    figure = go.Figure(
        go.Bar(
            x=data["perm_mean"],
            y=data["feature"],
            orientation="h",
            marker={"color": colours, "line": {"width": 0}},
            error_x={
                "type": "data",
                "array": data["perm_std"],
                "color": t.POS,
                "thickness": 1.2,
                "width": 3,
            },
            customdata=np.c_[data["perm_std"], data["positive_share"]],
            hovertemplate=(
                "%{y}<br>mean %{x:.2e}<br>fold s.d. %{customdata[0]:.2e}"
                "<br>positive in %{customdata[1]:.0%} of folds<extra></extra>"
            ),
        )
    )
    figure.update_layout(hovermode="closest", bargap=0.35)
    figure.update_xaxes(
        title_text="permutation importance (IC lost when shuffled)", tickformat=".1e"
    )
    figure.update_yaxes(showgrid=False, tickfont={"size": 10})
    return figure


def rolling_ic(frame: pd.DataFrame) -> go.Figure:
    """Trailing mean of the daily cross-sectional IC with its +/- 1.96 SE band."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame.index, y=frame["high"], line={"width": 0}, hoverinfo="skip", showlegend=False
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["low"],
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(142, 213, 255, 0.12)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["ic"],
            name="Rolling IC",
            line={"color": t.PRIMARY, "width": 1.6},
            customdata=np.c_[frame["low"], frame["high"]],
            hovertemplate=(
                "IC %{y:+.4f}<br>band [%{customdata[0]:+.4f}, %{customdata[1]:+.4f}]<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0, line={"color": t.OUTLINE_VARIANT, "width": 1})
    figure.update_yaxes(title_text="252-day mean Spearman IC", tickformat="+.3f")
    figure.update_xaxes(hoverformat="%Y-%m-%d")
    return figure


def bars_with_error(
    labels: Sequence[str],
    values: Sequence[float],
    errors: Sequence[float],
    axis_title: str,
    fmt: str,
) -> go.Figure:
    """Bars with symmetric error whiskers, positive and negative toned."""
    values = np.asarray(values, dtype=float)
    colours = np.where(values >= 0, t.POS, t.NEG)
    figure = go.Figure(
        go.Bar(
            x=list(labels),
            y=values,
            marker={"color": colours, "line": {"width": 0}},
            error_y={
                "type": "data",
                "array": list(errors),
                "color": t.OUTLINE,
                "thickness": 1,
                "width": 3,
            },
            customdata=np.asarray(errors, dtype=float),
            hovertemplate=f"%{{x}}: %{{y:{fmt}}} (± %{{customdata:{fmt}}})<extra></extra>",
        )
    )
    figure.update_layout(hovermode="closest")
    figure.update_yaxes(title_text=axis_title, tickformat=fmt)
    figure.update_xaxes(showgrid=False)
    return figure


def feature_series(frame: pd.DataFrame, columns: Sequence[tuple[str, str, str]]) -> go.Figure:
    """Up to three feature series on stacked panels sharing the date axis."""
    figure = make_subplots(rows=len(columns), cols=1, shared_xaxes=True, vertical_spacing=0.05)
    for row, (column, title, tickformat) in enumerate(columns, start=1):
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                name=title,
                line={"color": [t.PRIMARY, t.SECONDARY, t.WARN][row - 1], "width": 1.1},
                hovertemplate=f"{title}: %{{y:{tickformat}}}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(title_text=title, tickformat=tickformat, row=row, col=1)
    figure.update_xaxes(hoverformat="%Y-%m-%d")
    return figure


def frontier(
    curve: pd.DataFrame, assets: pd.DataFrame, portfolios: pd.DataFrame, train: str, test: str
) -> go.Figure:
    """The in-sample frontier as estimated, and what its portfolios realised out of sample."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=curve["volatility"],
            y=curve["target_return"],
            name=f"In-sample frontier ({train})",
            mode="lines",
            line={"color": t.PRIMARY, "width": 2},
            hovertemplate="expected: vol %{x:.1%}, return %{y:.1%}<extra>in-sample</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=curve["realised_volatility"],
            y=curve["realised_return"],
            name=f"Same portfolios, realised ({test})",
            mode="lines+markers",
            line={"color": t.WARN, "width": 1.5, "dash": "dash"},
            marker={"size": 4},
            hovertemplate="realised: vol %{x:.1%}, return %{y:.1%}<extra>out of sample</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=assets["expected_volatility"],
            y=assets["expected_return"],
            text=list(assets.index),
            name="Assets (in-sample)",
            mode="markers+text",
            textposition="top center",
            textfont={"family": t.FONT_UI, "size": 10, "color": t.OUTLINE},
            marker={"color": t.OUTLINE, "size": 5},
            hovertemplate="%{text}: vol %{x:.1%}, return %{y:.1%}<extra>in-sample</extra>",
        )
    )
    names = {"equal_weight": "1/N", "minimum_variance": "Min var", "maximum_sharpe": "Max Sharpe"}
    colours = {
        "equal_weight": t.OUTLINE,
        "minimum_variance": t.POS,
        "maximum_sharpe": t.PRIMARY_CONTAINER,
    }
    for key, row in portfolios.iterrows():
        label = names.get(str(key), str(key))
        colour = colours.get(str(key), t.ON_SURFACE)
        figure.add_trace(
            go.Scatter(
                x=[row["expected_volatility"], row["realised_volatility"]],
                y=[row["expected_return"], row["realised_return"]],
                mode="lines+markers+text",
                text=[f"{label} (exp.)", f"{label} (real.)"],
                textposition=["middle right", "bottom right"],
                textfont={"family": t.FONT_UI, "size": 10, "color": colour},
                line={"color": colour, "width": 1, "dash": "dot"},
                marker={"color": colour, "size": [9, 9], "symbol": ["square", "circle"]},
                name=label,
                hovertemplate=f"{label}: vol %{{x:.1%}}, return %{{y:.1%}}<extra></extra>",
            )
        )
    figure.update_layout(hovermode="closest")
    figure.update_xaxes(title_text="annualised volatility", tickformat=".0%")
    figure.update_yaxes(title_text="annualised return", tickformat=".0%")
    return figure


def exceedance(probabilities: dict[str, float]) -> go.Figure:
    """Probability that the one-year simulated drawdown exceeds each threshold."""
    labels = list(probabilities)
    values = np.asarray(list(probabilities.values()), dtype=float)
    shades = [t.PRIMARY_CONTAINER, t.WARN, "#e0866f", t.NEG][: len(values)]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker={"color": shades, "line": {"width": 0}},
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            textfont={"family": t.FONT_MONO, "size": 10, "color": t.ON_SURFACE_VARIANT},
            hovertemplate="P(max drawdown %{x}) = %{y:.2%}<extra></extra>",
        )
    )
    figure.update_layout(hovermode="closest")
    figure.update_yaxes(
        title_text="probability over one year", tickformat=".0%", rangemode="tozero"
    )
    figure.update_xaxes(title_text="simulated maximum drawdown", showgrid=False)
    return figure


def trial_sharpes(
    sharpes: pd.Series, threshold: float, marks: Sequence[tuple[str, float, str]]
) -> go.Figure:
    """Distribution of the trials' Sharpe ratios against the luck threshold."""
    figure = go.Figure(
        go.Histogram(
            x=sharpes,
            nbinsx=20,
            marker={"color": t.SURFACE_HIGHEST, "line": {"color": t.OUTLINE_VARIANT, "width": 1}},
            hovertemplate="Sharpe %{x}<br>%{y} trials<extra></extra>",
        )
    )
    figure.add_vline(x=threshold, line={"color": t.NEG, "dash": "dash", "width": 1.2})
    figure.add_annotation(
        x=threshold,
        y=1,
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        text=f" luck threshold {threshold:.2f}",
        font={"family": t.FONT_UI, "size": 10, "color": t.NEG},
    )
    for label, x, colour in marks:
        figure.add_vline(x=x, line={"color": colour, "width": 1.2})
        figure.add_annotation(
            x=x,
            y=0.82,
            yref="paper",
            xanchor="right",
            yanchor="top",
            showarrow=False,
            text=f"{label} {x:.2f} ",
            font={"family": t.FONT_UI, "size": 10, "color": colour},
        )
    figure.update_layout(hovermode="closest", bargap=0.05)
    figure.update_xaxes(title_text="annualised net Sharpe of each backtest trial")
    figure.update_yaxes(title_text="trials")
    return figure
