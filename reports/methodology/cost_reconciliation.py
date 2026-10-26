"""Reconcile notebook 4's cost drag with Phase 6's, one difference at a time.

Run from the project root:

    python reports/methodology/cost_reconciliation.py

Step 0 re-runs notebook 4's own code (its features, model, walk-forward
splits and backtest function), extracted from the .ipynb. Each later step
changes exactly one thing and pushes the same inputs through the Phase 6
engine, so the gap between "0.31 Sharpe lost to costs" and "0.06" can be
attributed rather than guessed.

The notebook's raw bars are cut at 2019-12-31 before its code runs, so no
2020-2024 data is read: the holdout stays untouched. The published notebook
figures (0.5901 gross, 0.2787 net, cost_drag 0.9035) used data to 2024 and
come from the notebook's saved outputs; this script's step 0 is the same
code on the development years only, so its numbers differ from them.

The step table and the share-of-days figures are also saved to
reports/results/reconciliation/, so the results notebook reads them instead
of retyping them.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.backtest.engine import run_backtest  # noqa: E402
from src.backtest.performance import sharpe_ratio  # noqa: E402
from src.config import load_config  # noqa: E402

SETUP_CELL = 1
MODEL_CELLS = (3, 5, 6, 8, 10)

# Last date the notebook may see: the day before the holdout starts.
LAST_DEVELOPMENT_DATE = "2019-12-31"

RESEARCH = ROOT / "reports/results/research/research_random_forest_regression_fixed_horizon"
OUTPUT = ROOT / "reports/results/reconciliation"


def run_notebook() -> dict:
    """Execute notebook 4's modelling cells on pre-2020 data only; return its namespace."""
    notebook = json.loads((ROOT / "notebooks" / "04_backtest.ipynb").read_text(encoding="utf-8"))

    def source(i: int) -> str:
        return "".join(notebook["cells"][i]["source"])

    namespace: dict = {}
    here = Path.cwd()
    os.chdir(ROOT / "notebooks")
    try:
        exec(compile(source(SETUP_CELL), "04_backtest.ipynb", "exec"), namespace)
        # Cut every ticker's bars before the model code sees them.
        namespace["bars"] = {
            ticker: frame.loc[:LAST_DEVELOPMENT_DATE] for ticker, frame in namespace["bars"].items()
        }
        code = "\n\n".join(source(i) for i in MODEL_CELLS)
        exec(compile(code, "04_backtest.ipynb", "exec"), namespace)
    finally:
        os.chdir(here)

    latest = max(frame.index.max() for frame in namespace["bars"].values())
    assert latest <= pd.Timestamp(LAST_DEVELOPMENT_DATE), "holdout data reached the notebook"

    return namespace


def stats(name: str, gross: pd.Series, net: pd.Series, traded: pd.Series, cost: pd.Series) -> dict:
    """One row of the reconciliation table."""
    years = len(net) / 252

    return {
        "step": name,
        "sharpe_gross": sharpe_ratio(gross),
        "sharpe_net": sharpe_ratio(net),
        "sharpe_lost": sharpe_ratio(gross) - sharpe_ratio(net),
        "turnover_per_year": float(traded.sum() / years),
        "cost_per_year": float(cost.sum() / years),
        "volatility": float(net.std() * np.sqrt(252)),
        "cumulative_cost": float(cost.sum()),
        "days": len(net),
    }


def main() -> None:
    pd.set_option("display.width", 220)
    ns = run_notebook()
    config = load_config(create_dirs=False)
    no_borrow = dataclasses.replace(
        config, backtest=dataclasses.replace(config.backtest, borrow_rate_annual=0.0)
    )
    rows = []

    # Step 0: the notebook's own backtest function, development years only.
    book = ns["books"]["binary long/short"]
    rows.append(
        stats("0 notebook code", book["gross"], book["net"], book["turnover"], book["cost"])
    )

    # Step 1: the same positions through the Phase 6 engine, same timing.
    # Notebook position p_t multiplies the return dated t (close t-1 to t);
    # Phase 6 weight w_t earns the return dated t+1, so w = p shifted back a day.
    returns = ns["period_returns"].rename("SPY").to_frame()
    position = ns["strategies"]["binary long/short"].reindex(returns.index).fillna(0.0)
    weights = position.shift(-1).fillna(0.0).to_frame("SPY")
    same = run_backtest(weights, returns, no_borrow, name="notebook positions")
    rows.append(
        stats(
            "1 same positions, Phase 6 engine",
            same.gross,
            same.net,
            same.turnover,
            same.costs["total"],
        )
    )

    # Where do the two engines' gross returns differ? The notebook holds a
    # constant weight every day (an unpriced daily rebalance); Phase 6 lets an
    # unchanged position drift. For a long of weight 1 drift changes nothing;
    # for a short of -1 it does (a rising price makes the short a bigger share
    # of the book). So after the first day, differences should sit only on
    # days that began with a short.
    notebook_gross = (position * returns["SPY"]).reindex(same.gross.index)
    gap = (same.gross - notebook_gross).abs()
    began_short = same.weights["SPY"].shift(1) < 0
    drift_check = {
        "max gross gap, days held long": float(gap.iloc[1:][~began_short.iloc[1:]].max()),
        "max gross gap, days held short": float(gap.iloc[1:][began_short.iloc[1:]].max()),
        "days held short": int(began_short.sum()),
        "first-day gap (nothing held yet)": float(gap.iloc[0]),
    }

    # Step 2: add the borrow fee on shorts.
    borrowed = run_backtest(weights, returns, config, name="with borrow")
    rows.append(
        stats(
            "2 + borrow 0.5%/yr",
            borrowed.gross,
            borrowed.net,
            borrowed.turnover,
            borrowed.costs["total"],
        )
    )

    # Step 3: the one-bar delay applied to the notebook's model.
    delayed = run_backtest(weights.shift(1).fillna(0.0), returns, config, name="delayed")
    rows.append(
        stats(
            "3 + one-bar delay",
            delayed.gross,
            delayed.net,
            delayed.turnover,
            delayed.costs["total"],
        )
    )

    # Step 4: Phase 6's pooled forecasts, SPY column only, same timing and costs.
    # Returns come from the cleaned panel: the notebook's series starts in 2008.
    forecasts = pd.read_parquet(RESEARCH / "predictions.parquet")["path_0"].unstack("ticker")
    panel_returns = pd.read_parquet(ROOT / "data/02-bars/bars.parquet")["return"].unstack("ticker")
    panel_returns = panel_returns.reindex(forecasts.index).fillna(0.0)
    spy_only = np.sign(forecasts[["SPY"]]).shift(1).fillna(0.0)
    phase6_spy = run_backtest(spy_only, panel_returns[["SPY"]], config, name="phase 6 SPY")
    rows.append(
        stats(
            "4 Phase 6 forecasts, SPY only",
            phase6_spy.gross,
            phase6_spy.net,
            phase6_spy.turnover,
            phase6_spy.costs["total"],
        )
    )

    # Step 5: the full Phase 6 book: 11 tickers at 1/11 each.
    book11 = (np.sign(forecasts) / forecasts.shape[1]).shift(1).fillna(0.0)
    phase6 = run_backtest(book11, panel_returns, config, name="phase 6 book")
    rows.append(
        stats(
            "5 Phase 6 book, 11 tickers, path 0",
            phase6.gross,
            phase6.net,
            phase6.turnover,
            phase6.costs["total"],
        )
    )

    print(pd.DataFrame(rows).set_index("step").round(4).to_string())

    breakdown = phase6.costs[["commission", "slippage", "borrow"]].sum() / (len(phase6.net) / 252)
    print("\nPhase 6 book, cost per year by component:")
    print(breakdown.round(5).to_string())

    # How often the forecast's sign changes, the driver of turnover.
    nb_sign = np.sign(ns["predictions"])
    sign = np.sign(forecasts)
    flips = {
        "notebook SPY forecast changes sign": float((nb_sign != nb_sign.shift()).iloc[1:].mean()),
        "Phase 6 SPY forecast changes sign": float((sign["SPY"].diff().abs() > 0).iloc[1:].mean()),
        "Phase 6 forecasts, mean over tickers": float(
            (sign.diff().abs() > 0).iloc[1:].mean().mean()
        ),
        "notebook forecasts > 0": float((ns["predictions"] > 0).mean()),
        "Phase 6 forecasts > 0": float((forecasts > 0).to_numpy().mean()),
    }
    print("\nShare of days:")
    for key, value in flips.items():
        print(f"  {key:40s} {value:.3f}")

    # Saved for the results notebook; every figure printed here, none retyped.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).set_index("step").to_csv(OUTPUT / "notebook4_costs.csv")
    pd.Series({**flips, **drift_check}, name="value").to_csv(OUTPUT / "notebook4_signals.csv")
    # Every notebook rule and its forecast quality on the development years, the
    # notebook's own definitions (Spearman IC, sign agreement).
    rules = pd.DataFrame(
        [stats(n, b["gross"], b["net"], b["turnover"], b["cost"]) for n, b in ns["books"].items()]
    ).set_index("step")
    rules.index.name = "rule"
    rules.to_csv(OUTPUT / "notebook4_rules.csv")
    predictions, realised = ns["predictions"], ns["realised"]
    pd.Series(
        {
            "ic": spearmanr(predictions, realised).statistic,
            "hit_rate": float((np.sign(predictions) == np.sign(realised)).mean()),
            "first_prediction": str(predictions.index[0].date()),
            "last_prediction": str(predictions.index[-1].date()),
        },
        name="value",
    ).to_csv(OUTPUT / "notebook4_forecast.csv")

    print("\nWhere the two engines' gross returns differ (step 0 vs step 1):")
    for key, value in drift_check.items():
        print(f"  {key:40s} {value}")

    first, last = ns["predictions"].index[[0, -1]]
    print(f"\nNotebook predictions: {first.date()} to {last.date()}")
    print(
        f"Cost per side: {config.backtest.cost_per_side * 1e4:.1f} bps here, "
        f"{ns['COST_BPS']} in the notebook"
    )


if __name__ == "__main__":
    main()
