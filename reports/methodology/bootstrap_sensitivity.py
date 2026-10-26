"""How much do the Monte Carlo conclusions depend on the bootstrap block length?

Run from the project root:

    python reports/methodology/bootstrap_sensitivity.py [block ...]

Reads only saved development-period results (the Phase 6 path returns and
the Phase 7 allocation returns): nothing is refitted, nothing is logged as a
trial, and no date on or after 2020-01-01 is read. Each block length uses the
same seeds, so the comparison isolates the block length.

"Probability of beating" here means the share of paired resamples (strategy
and benchmark drawn on the same days) in which the strategy's Sharpe ratio
is higher. It describes how stable the historical comparison is under
reshuffling; it is not a probability that the strategy will win in future.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.evaluation.monte_carlo import path_statistics, simulate_paths  # noqa: E402
from src.pipelines.evaluate import load_path_returns, monte_carlo_table  # noqa: E402

BLOCKS = (10, 21, 42)
STRATEGIES = (
    "ml_long_short",
    "ml_long_only",
    "ml_thresholded",
    "momentum",
    "pairs",
    "buy_and_hold",
    "equal_weight_hold",
)
EVALUATION = ROOT / "reports/results/evaluation"
PORTFOLIO = ROOT / "reports/results/portfolio"


def strategy_sensitivity(config, blocks) -> pd.DataFrame:
    """Phase 6 strategies against equal weight and SPY, per block length."""
    saved = load_path_returns(EVALUATION)
    returns = {name: saved[name] for name in STRATEGIES}
    holdout = pd.Timestamp(config.validation.test_start)
    assert all(s.index.max() < holdout for paths in returns.values() for s in paths)

    rows = []
    for block in blocks:
        sized = dataclasses.replace(
            config, evaluation=dataclasses.replace(config.evaluation, bootstrap_block_days=block)
        )
        table = monte_carlo_table(returns, sized, config.evaluation.monte_carlo_runs)
        table["block_days"] = block
        rows.append(table)

    return pd.concat(rows).set_index("block_days", append=True)


def allocation_sensitivity(config, blocks) -> pd.DataFrame:
    """Phase 7 allocations against monthly equal weight, per block length."""
    frame = pd.read_parquet(PORTFOLIO / "allocation_returns.parquet")
    assert frame.index.max() < pd.Timestamp(config.validation.test_start)
    net = {c.split("|")[0]: frame[c] for c in frame.columns if c.endswith("|net")}
    base = net["equal_weight"]
    rows = {}

    for block in blocks:
        for name, series in net.items():
            pair = pd.concat([series, base], axis=1, join="inner")
            sims = simulate_paths(pair, "block", config, block_size=block)
            ours = path_statistics(sims[:, :, 0])
            theirs = path_statistics(sims[:, :, 1])["sharpe"].to_numpy()
            rows[(name, block)] = {
                "sharpe_median": float(ours["sharpe"].median()),
                "sharpe_low": float(ours["sharpe"].quantile(0.025)),
                "sharpe_high": float(ours["sharpe"].quantile(0.975)),
                "max_drawdown_p05": float(ours["max_drawdown"].quantile(0.05)),
                "p_beat_equal_weight": float(np.mean(ours["sharpe"].to_numpy() > theirs)),
                "runs": float(len(ours)),
                "seed": float(config.project.random_seed),
            }

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.names = ["allocation", "block_days"]

    return table


def main() -> None:
    blocks = tuple(int(b) for b in sys.argv[1:]) or BLOCKS
    config = load_config(create_dirs=False)
    pd.set_option("display.width", 220)

    started = time.perf_counter()
    strategies = strategy_sensitivity(config, blocks)
    allocations = allocation_sensitivity(config, blocks)
    elapsed = time.perf_counter() - started

    strategies.to_csv(EVALUATION / "monte_carlo_block_sensitivity.csv")
    allocations.to_csv(PORTFOLIO / "monte_carlo_block_sensitivity.csv")

    columns = [
        "sharpe_median",
        "sharpe_low",
        "sharpe_high",
        "max_drawdown_p05",
        "p_beat_equal_weight_hold",
        "p_beat_buy_and_hold",
    ]
    print(strategies[columns].round(3).to_string())
    print()
    print(allocations.round(3).to_string())
    print(f"\n{len(blocks)} block lengths in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
