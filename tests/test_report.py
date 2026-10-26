"""End-to-end check on the report: real pipeline results in, every number traceable out.

Research and evaluation run on the small synthetic panel used by the
evaluation tests, the report is built from their result objects, and each
number checked here is read back from those objects rather than typed in.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.evaluation.notebook_trials import record_notebook_trials
from src.evaluation.trial_log import TrialLog
from src.pipelines.evaluate import REFERENCE, run_evaluation
from src.pipelines.research import run_research
from src.reporting.report import (
    ReportData,
    ReportError,
    build_report,
    findings_section,
    summary_table,
    trial_summary,
)
from tests.test_evaluate import panel, shrink  # noqa: F401  (panel is a fixture)


@pytest.fixture(scope="module")
def run(panel, tmp_path_factory, config):  # noqa: F811
    """Research, evaluation, the notebook records and the report, once."""
    folder = tmp_path_factory.mktemp("report")
    small = shrink(config, folder / "trials.csv")
    log = TrialLog.from_config(small)
    # Constant volume makes the volume features constant; research needs them to vary.
    panel = panel.assign(volume=np.random.default_rng(0).uniform(5e5, 2e6, len(panel)))

    research = run_research(small, panel, trial_log=log, permutation=False)
    evaluation = run_evaluation(small, panel, trial_log=log)
    record_notebook_trials(log)
    path = build_report(research, evaluation, small, folder / "report.md", trial_log=log)

    return research, evaluation, small, log, path.read_text(encoding="utf-8")


def test_the_report_states_numbers_from_the_result_objects(run):
    research, evaluation, config, _, text = run

    assert f"{research.metrics['ic_mean']:.4f}" in text
    best = summary_table(evaluation).index[0]
    assert f"| {best} |" in text
    assert f"{evaluation.comparison.loc[best, 'sharpe_net']:.3f}" in text
    assert f"{evaluation.comparison.loc[REFERENCE, 'sharpe_net']:.2f}" in text
    assert f"{len(config.universe.tickers)} tickers" in text


def test_the_report_counts_the_trials_in_the_log(run):
    _, _, _, log, text = run
    scenarios = trial_summary(log)["scenarios"]

    for scenario, row in scenarios.iterrows():
        assert f"| {scenario} | {row['n_trials']} |" in text
    assert "notebook, not recorded" in text


def test_the_report_has_every_section_and_says_the_holdout_is_untouched(run):
    text = run[-1]

    for heading in (
        "## Findings",
        "## Forecast quality",
        "## Strategies and benchmarks",
        "## Trials",
        "## Deflated Sharpe ratio",
        "### Headline",
        "### Correlated trials",
        "## Limitations",
    ):
        assert heading in text
    assert "holdout has not been evaluated" in text
    assert "not a probability of future profit" in text
    assert "Trial dependence" in text
    assert "## Allocations" not in text


def test_building_the_report_logs_no_trial(run, tmp_path):
    research, evaluation, config, log, _ = run
    before = log.trial_count()

    build_report(research, evaluation, config, tmp_path / "again.md", trial_log=log)

    assert log.trial_count() == before


def test_a_result_that_reaches_the_holdout_is_refused(run, tmp_path):
    research, evaluation, config, log, _ = run
    early = dataclasses.replace(
        config,
        validation=dataclasses.replace(
            config.validation, test_start=pd.Timestamp("2007-01-01").date()
        ),
    )

    with pytest.raises(ReportError, match="holdout"):
        build_report(research, evaluation, early, tmp_path / "bad.md", trial_log=log)


def test_results_that_disagree_with_the_log_are_refused(run, tmp_path):
    research, evaluation, config, log, _ = run
    altered = dataclasses.replace(evaluation, comparison=evaluation.comparison.copy())
    name = altered.comparison.index[altered.comparison["kind"] == "backtest"][0]
    altered.comparison.loc[name, "sharpe_net"] = 9.0

    with pytest.raises(ValueError, match=name):
        build_report(research, altered, config, tmp_path / "bad.md", trial_log=log)


def _findings(run, low, high):
    """Findings for the real run with the best strategy's interval replaced."""
    research, evaluation, config, log, _ = run
    comparison = evaluation.comparison.copy()
    comparison["vs_reference_low"] = -1.0
    comparison["vs_reference_high"] = 1.0
    best = pd.to_numeric(comparison.loc[comparison["kind"] == "backtest", "sharpe_net"]).idxmax()
    comparison.loc[best, ["vs_reference_low", "vs_reference_high"]] = [low, high]
    data = ReportData(
        research=research,
        evaluation=dataclasses.replace(evaluation, comparison=comparison),
        config=config,
        trial_log=log,
    )

    return best, findings_section(data)


def test_findings_say_plainly_when_nothing_beats_equal_weight(run):
    _, text = _findings(run, -0.2, 0.3)

    assert "Nothing distinguishably beats equal weight" in text


def test_findings_name_a_strategy_whose_whole_interval_is_above_zero(run):
    best, text = _findings(run, 0.1, 0.4)

    assert f"Beats equal weight distinguishably:** {best}" in text
    assert "Nothing distinguishably" not in text


def test_summary_table_is_sorted_by_net_sharpe(run):
    table = summary_table(run[1])

    assert np.all(np.diff(table["Sharpe net"].to_numpy()) <= 0)
