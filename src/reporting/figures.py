"""Every exploratory figure, rebuilt from the cleaned panel.

A figure that cannot be regenerated cannot be defended. Each function here
takes the cleaned panel, draws one figure, saves it under reports/figures, and
returns the Figure so tests and notebooks can inspect it. Re-running
overwrites the file.

Figures are drawn on matplotlib.figure.Figure directly rather than through
pyplot, so nothing is held in pyplot's global state and no GUI backend is
needed.

Run it with:

    python -m src.reporting.figures
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from src.config import PROJECT_ROOT, Config, load_config
from src.logger import get_logger, log_step

logger = get_logger(__name__)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Categorical slots in fixed order, validated for colour vision deficiency.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# Neutral ink for benchmarks, reference curves and bands.
GREY, DARK_GREY, SHADE = "#8a8984", "#52514e", "#e4e3df"

TRADING_DAYS = 252

STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.6,
    "axes.edgecolor": GREY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "font.size": 10,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
}


@contextmanager
def _style() -> Generator[None, None]:
    """Apply the project style without changing matplotlib's global settings."""
    with mpl.rc_context(STYLE):
        yield


def _save(figure: Figure, name: str, out_dir: Path | None) -> Path:
    """Write a figure to disk as PNG, replacing any earlier version."""
    out_dir = out_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    figure.savefig(path, bbox_inches="tight")
    logger.info("Saved figure %s", path.name)

    return path


def _wide(cleaned: pd.DataFrame, column: str) -> pd.DataFrame:
    """One column of the cleaned panel as a wide frame, a column per ticker."""
    return cleaned[column].unstack("ticker")


def _returns(cleaned: pd.DataFrame, ticker: str) -> pd.Series:
    """One ticker's daily returns, without the leading NaN."""
    return _wide(cleaned, "return")[ticker].dropna()


def _runs(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Start and end dates of each consecutive run of True values."""
    mask = mask.fillna(False).astype(bool)
    starts = mask & ~mask.shift(1, fill_value=False)
    ends = mask & ~mask.shift(-1, fill_value=False)

    return list(zip(mask.index[starts], mask.index[ends], strict=True))


def volatility_regimes(
    returns: pd.Series,
    window: int = 21,
    min_history: int = TRADING_DAYS,
) -> pd.Series:
    """Label each day low, normal or high volatility, using only the past.

    Realised volatility on day t is compared against the terciles of every
    realised volatility before t. Full-sample terciles would be simpler, but
    would classify 2008 using the knowledge that 2020 was coming.

    Args:
        returns: Daily returns.
        window: Days in the realised volatility estimate.
        min_history: Days of volatility history required before labelling.

    Returns:
        "low", "normal" or "high" per day, NaN during the warm-up.
    """
    vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    history = vol.expanding(min_periods=min_history)
    lower = history.quantile(1 / 3).shift(1)
    upper = history.quantile(2 / 3).shift(1)

    labels = np.select([vol <= lower, vol >= upper], ["low", "high"], default="normal")
    regimes = pd.Series(labels, index=returns.index, name="regime", dtype=object)
    regimes[lower.isna() | vol.isna()] = np.nan

    return regimes


def price_and_cumulative_return(
    cleaned: pd.DataFrame,
    config: Config,
    out_dir: Path | None = None,
) -> Figure:
    """Growth of 1 for every ticker, each in its own panel against the benchmark.

    Growth of 1 is the price rebased to its first value, so it is the price
    and the cumulative return in one line. Small multiples rather than eleven
    overlapping lines, so each ticker can be read.
    """
    prices = _wide(cleaned, config.data.price_column)
    growth = prices / prices.bfill().iloc[0]
    benchmark = config.universe.benchmark
    tickers = [t for t in prices.columns if t != benchmark]

    columns = 4
    rows = int(np.ceil(len(tickers) / columns))

    with _style():
        figure = Figure(figsize=(14, 2.6 * rows + 0.6))
        axes = figure.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)

        for axis, ticker in zip(axes.flat, tickers, strict=False):
            axis.plot(growth.index, growth[benchmark], color=GREY, lw=1.0, label=benchmark)
            axis.plot(growth.index, growth[ticker], color=BLUE, lw=1.2, label="this ticker")
            axis.set_yscale("log")
            final = growth[ticker].dropna().iloc[-1]
            axis.set_title(f"{ticker}  {final:.1f}x", fontsize=10)

        for axis in axes.flat[len(tickers) :]:
            axis.set_visible(False)

        for axis in axes[:, 0]:
            axis.set_ylabel("growth of 1 (log)")

        start, end = growth.index[0].date(), growth.index[-1].date()
        figure.suptitle(
            f"Cumulative return per ticker, {start} to {end}, against {benchmark} in grey",
            x=0.01,
            ha="left",
            fontweight="bold",
        )
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper right", ncol=2)
        figure.tight_layout()

    _save(figure, "eda_cumulative_returns", out_dir)

    return figure


def return_distribution(
    cleaned: pd.DataFrame,
    config: Config,
    ticker: str | None = None,
    out_dir: Path | None = None,
) -> Figure:
    """Histogram of daily returns against a fitted normal, tails annotated."""
    ticker = ticker or config.universe.benchmark
    returns = _returns(cleaned, ticker)
    mean, std = returns.mean(), returns.std()

    beyond = int((((returns - mean) / std).abs() > 3).sum())
    expected = len(returns) * 2 * stats.norm.sf(3)

    with _style():
        figure = Figure(figsize=(11, 4.5))
        axis = figure.subplots()

        axis.hist(returns, bins=150, density=True, color=BLUE, alpha=0.85, label="daily returns")
        grid = np.linspace(returns.min(), returns.max(), 400)
        axis.plot(
            grid,
            stats.norm.pdf(grid, mean, std),
            color=DARK_GREY,
            ls="--",
            lw=1.4,
            label="normal, same mean and std",
        )

        note = (
            f"skew {returns.skew():.2f}\n"
            f"excess kurtosis {returns.kurtosis():.1f}\n"
            f"days beyond 3 std: {beyond} (normal: {expected:.0f})\n"
            f"n = {len(returns):,}"
        )
        axis.text(0.02, 0.95, note, transform=axis.transAxes, va="top", family="monospace")

        axis.set_xlabel("daily return")
        axis.set_ylabel("density")
        axis.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        axis.set_title(f"{ticker} daily returns: fatter tails than a normal distribution")
        axis.legend(loc="upper right")
        figure.tight_layout()

    _save(figure, f"eda_return_distribution_{ticker}", out_dir)

    return figure


def rolling_volatility(
    cleaned: pd.DataFrame,
    config: Config,
    ticker: str | None = None,
    windows: tuple[int, int] = (21, 63),
    out_dir: Path | None = None,
) -> Figure:
    """Rolling realised volatility at two windows, annualised, showing clustering."""
    ticker = ticker or config.universe.benchmark
    returns = _returns(cleaned, ticker)

    with _style():
        figure = Figure(figsize=(12, 4.5))
        axis = figure.subplots()

        for window, colour in zip(windows, [BLUE, ORANGE], strict=True):
            vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
            axis.plot(vol.index, vol, color=colour, lw=1.1, label=f"{window} day")

        axis.set_ylabel("annualised volatility")
        axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        axis.set_title(f"{ticker} rolling volatility: calm follows calm, panic follows panic")
        axis.legend(loc="upper right")
        figure.tight_layout()

    _save(figure, f"eda_rolling_volatility_{ticker}", out_dir)

    return figure


def return_autocorrelation(
    cleaned: pd.DataFrame,
    config: Config,
    ticker: str | None = None,
    max_lag: int = 21,
    out_dir: Path | None = None,
) -> Figure:
    """Autocorrelation of returns and absolute returns, lags 1 to max_lag.

    The dashed band is +/- 1.96 / sqrt(n), the 95% range for white noise.
    Returns sit inside it; absolute returns sit well above it, which is
    volatility clustering seen from another angle.
    """
    ticker = ticker or config.universe.benchmark
    returns = _returns(cleaned, ticker)
    lags = np.arange(1, max_lag + 1)
    band = 1.96 / np.sqrt(len(returns))

    raw = [returns.autocorr(lag) for lag in lags]
    absolute = [returns.abs().autocorr(lag) for lag in lags]

    with _style():
        figure = Figure(figsize=(12, 4.5))
        axis = figure.subplots()
        width = 0.4

        axis.bar(lags - width / 2, raw, width=width, color=BLUE, label="returns")
        axis.bar(lags + width / 2, absolute, width=width, color=ORANGE, label="absolute returns")
        axis.axhline(0, color=GREY, lw=0.8)
        for level in (band, -band):
            axis.axhline(level, color=DARK_GREY, ls="--", lw=1.0)
        axis.text(max_lag + 0.6, band, "95% white noise band", va="bottom", ha="right", fontsize=8)

        axis.set_xlabel("lag (trading days)")
        axis.set_ylabel("autocorrelation")
        axis.set_xticks(lags)
        axis.set_title(f"{ticker}: returns carry almost no memory, their size does")
        axis.legend(loc="upper right")
        figure.tight_layout()

    _save(figure, f"eda_autocorrelation_{ticker}", out_dir)

    return figure


def correlation_heatmap(
    cleaned: pd.DataFrame,
    config: Config,
    out_dir: Path | None = None,
) -> Figure:
    """Full-sample correlation of daily returns across the universe.

    A full-sample matrix hides the fact that correlations rise in a crisis;
    the rolling view belongs to the risk module. This one is for structure.
    """
    correlation = _wide(cleaned, "return").corr()
    n = len(correlation)

    with _style():
        figure = Figure(figsize=(8, 6.5))
        axis = figure.subplots()
        image = axis.imshow(correlation, cmap="Blues", vmin=0, vmax=1)

        axis.set_xticks(range(n), correlation.columns, rotation=45, ha="right")
        axis.set_yticks(range(n), correlation.index)
        axis.grid(False)

        for i in range(n):
            for j in range(n):
                value = correlation.iloc[i, j]
                ink = "white" if value > 0.6 else "black"
                axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=ink)

        figure.colorbar(image, ax=axis, shrink=0.8, label="correlation")
        axis.set_title("Daily return correlation, full sample")
        figure.tight_layout()

    _save(figure, "eda_correlation", out_dir)

    return figure


def volatility_regime_chart(
    cleaned: pd.DataFrame,
    config: Config,
    out_dir: Path | None = None,
) -> Figure:
    """The benchmark's growth of 1 with high volatility periods shaded."""
    benchmark = config.universe.benchmark
    price = _wide(cleaned, config.data.price_column)[benchmark].dropna()
    growth = price / price.iloc[0]
    regimes = volatility_regimes(_returns(cleaned, benchmark))
    high = regimes.reindex(growth.index).eq("high")
    share = high.mean()

    with _style():
        figure = Figure(figsize=(12, 4.5))
        axis = figure.subplots()

        for start, end in _runs(high):
            axis.axvspan(start, end, color=SHADE, lw=0)
        axis.plot(growth.index, growth, color=BLUE, lw=1.2, label=f"{benchmark} growth of 1")
        axis.fill_between([], [], color=SHADE, label=f"high volatility ({share:.0%} of days)")

        axis.set_yscale("log")
        plain = mpl.ticker.FuncFormatter(lambda value, _: f"{value:g}")
        axis.yaxis.set_major_formatter(plain)
        axis.yaxis.set_minor_formatter(plain)
        axis.set_ylabel("growth of 1 (log)")
        axis.set_title(
            f"{benchmark} volatility regimes: 21 day vol above the top tercile of its own past"
        )
        axis.legend(loc="upper left")
        figure.tight_layout()

    _save(figure, "eda_volatility_regimes", out_dir)

    return figure


def build_all_figures(
    cleaned: pd.DataFrame,
    config: Config,
    out_dir: Path | None = None,
) -> dict[str, Figure]:
    """Rebuild every exploratory figure.

    Args:
        cleaned: Output of clean_panel.
        config: Loaded project configuration.
        out_dir: Directory to write to. Defaults to reports/figures.

    Returns:
        Each figure by name.
    """
    return {
        "cumulative_returns": price_and_cumulative_return(cleaned, config, out_dir),
        "return_distribution": return_distribution(cleaned, config, out_dir=out_dir),
        "rolling_volatility": rolling_volatility(cleaned, config, out_dir=out_dir),
        "autocorrelation": return_autocorrelation(cleaned, config, out_dir=out_dir),
        "correlation": correlation_heatmap(cleaned, config, out_dir),
        "volatility_regimes": volatility_regime_chart(cleaned, config, out_dir),
    }


def main() -> None:
    """Rebuild every figure from the cleaned panel."""
    from src.data.cleaning import load_clean_panel

    config = load_config()

    with log_step(logger, "Exploratory figures"):
        figures = build_all_figures(load_clean_panel(config), config)

    print(f"{len(figures)} figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
