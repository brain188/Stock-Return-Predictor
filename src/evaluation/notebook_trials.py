"""The 42 experiments run in notebooks 2 to 4, recovered for the trial log.

Notebook 4 counted them before the pipeline existed: 21 fractional
differencing orders searched in notebook 2, 17 features tested one at a time
in notebook 3, and 3 position rules and 1 model configuration in notebook 4.
They shaped the pipeline's choices, so they count towards the number of trials
a deflated Sharpe ratio corrects for.

Everything here is parsed from the notebooks' saved outputs, never typed in.
What the outputs do not show is not invented: most of these experiments have
no Sharpe ratio at all, and their exact code is not recoverable (the
notebooks were edited after they ran). Each record says so in its notes.

All of these experiments used data to 2024, before the 2020-2024 holdout was
set aside. Their metric names therefore carry their period (for example
sharpe_net_2008_2024), so they can never be mistaken for development
statistics, and they are left out of the Sharpe variance used by the DSR.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from src.config import PROJECT_ROOT
from src.evaluation.trial_log import TrialLog, trial_fingerprint
from src.logger import get_logger

logger = get_logger(__name__)

NOTEBOOKS = PROJECT_ROOT / "notebooks"
VERSION = "notebook, not recorded"
EXPECTED = {"fracdiff_order": 21, "feature": 17, "position_rule": 3, "model": 1}

# Notebook experiments the pipeline later re-ran as its own trial, so a count
# that adds both may count one choice twice. "Scaled by conviction" and the
# 38 differencing and feature tests have no pipeline counterpart.
OVERLAPS = {
    "notebook04_rule_binary_long_short": "ml_long_short",
    "notebook04_rule_long_only": "ml_long_only",
    "notebook04_model_random_forest_spy": "research_random_forest_regression_fixed_horizon",
}


class NotebookTrialError(ValueError):
    """Raised when the notebooks' saved outputs do not hold what is expected."""


def _outputs(notebook: str) -> list[str]:
    """The text output of every code cell, in order."""
    cells = json.loads((NOTEBOOKS / f"{notebook}.ipynb").read_text(encoding="utf-8"))["cells"]
    out = []

    for cell in cells:
        text = []
        for output in cell.get("outputs", []):
            value = output.get("text") or output.get("data", {}).get("text/plain") or ""
            text.append("".join(value))
        out.append("\n".join(text))

    return out


def _table_after(text: str, header: str) -> list[list[str]]:
    """Whitespace-split rows of the printed table that starts with `header`."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if header in line), None)
    if start is None:
        raise NotebookTrialError(f"no table with header '{header}'")

    rows = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        rows.append(line.split())

    return rows


def historical_notebook_trials() -> pd.DataFrame:
    """Every notebook experiment that counts as a trial, with what its outputs show.

    Returns:
        One row per experiment: notebook, experiment (the kind), name, the
        metrics recoverable from the saved output, and notes.

    Raises:
        NotebookTrialError: If the outputs no longer hold the expected tables,
            or the counts differ from notebook 4's own tally of 42.
    """
    rows = []

    # Notebook 2: the differencing search on SPY log prices.
    search = next(t for t in _outputs("02_stationarity") if "correlation  rows_kept" in t)
    for parts in _table_after(search, "correlation  rows_kept"):
        d, adf, correlation = float(parts[1]), float(parts[2]), float(parts[3])
        rows.append(
            {
                "notebook": "02",
                "experiment": "fracdiff_order",
                "name": f"notebook02_fracdiff_d_{d:.2f}",
                "metrics": {"adf_spy_2005_2024": adf, "correlation_spy_2005_2024": correlation},
            }
        )

    # Notebook 3: each feature's information coefficient on SPY.
    information = next(t for t in _outputs("03_feature_research") if "p_value  abs_ic" in t)
    for parts in _table_after(information, "feature")[0:]:
        if parts[0] == "feature" or len(parts) < 4:
            continue
        rows.append(
            {
                "notebook": "03",
                "experiment": "feature",
                "name": f"notebook03_feature_{parts[0]}",
                "metrics": {
                    "ic_spy_2005_2024": float(parts[1]),
                    "p_value_spy_2005_2024": float(parts[2]),
                },
            }
        )

    # Notebook 4: the position rules' Sharpe ratios and the model's IC.
    outputs = _outputs("04_backtest")
    summary = next(t for t in outputs if "sharpe_gross  sharpe_net" in t)
    for line in summary.splitlines()[1:]:
        match = re.match(r"^(.+?)\s{2,}(-?\d\.\d+)\s+(-?\d\.\d+)", line)
        if match and match.group(1).strip() != "buy and hold":
            rule = match.group(1).strip()
            rows.append(
                {
                    "notebook": "04",
                    "experiment": "position_rule",
                    "name": f"notebook04_rule_{rule.replace(' ', '_').replace('/', '_')}",
                    "metrics": {
                        "sharpe_gross_2008_2024": float(match.group(2)),
                        "sharpe_net_2008_2024": float(match.group(3)),
                    },
                }
            )

    model = next(t for t in outputs if "Information coefficient" in t)
    ic = float(re.search(r"Information coefficient\s*:\s*([+-]?\d\.\d+)", model).group(1))
    hit = float(re.search(r"Directional hit rate\s*:\s*(\d\.\d+)", model).group(1))
    rows.append(
        {
            "notebook": "04",
            "experiment": "model",
            "name": "notebook04_model_random_forest_spy",
            "metrics": {"ic_spy_2008_2024": ic, "hit_rate_spy_2008_2024": hit},
        }
    )

    table = pd.DataFrame(rows)
    counts = table["experiment"].value_counts().to_dict()
    if counts != EXPECTED:
        raise NotebookTrialError(f"expected {EXPECTED} experiments, found {counts}")

    table["notes"] = (
        "Historical notebook experiment, recorded from its saved output. Run on data to 2024, "
        "before the holdout was set aside. Exact code and settings not recoverable; metrics "
        "are only what the output printed."
    )

    return table


def record_notebook_trials(log: TrialLog) -> int:
    """Append the 42 notebook experiments to the log, once.

    Returns:
        The number of rows appended.

    Raises:
        NotebookTrialError: If notebook records are already in the log.
    """
    if log.trial_count({"kind": "notebook"}) if log.path.is_file() else 0:
        raise NotebookTrialError(f"{log.path} already holds notebook records")

    table = historical_notebook_trials()
    for row in table.itertuples():
        log.log_trial(
            row.name,
            kind="notebook",
            fingerprint=trial_fingerprint({"pipeline": "notebook", "name": row.name}),
            status="ok",
            label=row.experiment,
            metrics=row.metrics,
            notes=row.notes,
            version=VERSION,
        )

    logger.info("Recorded %s notebook experiments in %s", len(table), log.path)

    return len(table)
