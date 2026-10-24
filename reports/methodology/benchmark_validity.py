"""How much of the equal-weight benchmark comes from picking survivors?

Run from the project root:

    python reports/methodology/benchmark_validity.py

Development period only (the dates the CPCV paths cover, before 2020). For
each ticker: its own buy-and-hold result; then the equal-weight book with
that ticker left out, which shows how much the benchmark leans on any one
name; then the book without the two biggest winners.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.backtest.engine import run_backtest  # noqa: E402
from src.backtest.performance import performance_stats, sharpe_standard_error  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.cleaning import load_clean_panel  # noqa: E402
from src.pipelines.evaluate import build_context  # noqa: E402


def hold(tickers: list[str], returns: pd.DataFrame, config) -> dict:
    """Equal weight in the given tickers on day one, then drift."""
    weights = pd.DataFrame(1 / len(tickers), index=returns.index, columns=tickers)
    result = run_backtest(weights.shift(1).fillna(0.0), returns[tickers], config)
    stats = performance_stats(result.net)

    return {
        "sharpe": stats["sharpe"],
        "sharpe_se": sharpe_standard_error(result.net),
        "annual_return": stats["annual_return"],
        "max_drawdown": stats["max_drawdown"],
        "growth_of_1": float(result.equity_net.iloc[-1] / config.backtest.initial_capital),
    }


def main() -> None:
    pd.set_option("display.width", 200)
    config = load_config(create_dirs=False)
    panel = load_clean_panel(config)
    context = build_context(config, panel, fit_models=False)
    returns = context.returns.reindex(context.dates)
    tickers = list(returns.columns)
    print(f"Development period: {context.dates[0].date()} to {context.dates[-1].date()}\n")

    single = pd.DataFrame({t: hold([t], returns, config) for t in tickers}).T
    print("Each ticker held alone:")
    print(single.sort_values("growth_of_1", ascending=False).round(3).to_string())

    full = hold(tickers, returns, config)
    rows = {"all 11": full}
    for t in tickers:
        rows[f"without {t}"] = hold([x for x in tickers if x != t], returns, config)
    winners = single["growth_of_1"].sort_values(ascending=False).index[:2].tolist()
    rows[f"without {' and '.join(winners)}"] = hold(
        [x for x in tickers if x not in winners], returns, config
    )
    leave_out = pd.DataFrame(rows).T
    leave_out["sharpe_change"] = leave_out["sharpe"] - full["sharpe"]
    print("\nEqual weight, leaving names out:")
    print(leave_out.sort_values("sharpe_change").round(3).to_string())


if __name__ == "__main__":
    main()
