"""The research report, written from result objects and saved development artifacts.

No number in the report is typed in. Each one is read from a ResearchResult, an
EvaluationResult, the portfolio comparison, the config or the trial log, and
formatted here. A rerun of the pipeline therefore changes the report, and a
number in the report can always be traced back to the table it came from.

The report covers the development period only, before validation.test_start.
Nothing here reads, evaluates or summarises the 2020-2024 holdout.

Run it with:

    python -m src.reporting.report
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, Config, load_config
from src.evaluation.trial_log import SELECTION_KINDS, TrialLog, code_version, config_hash
from src.evaluation.trial_scenarios import (
    HEADLINE_COUNT,
    PERIODS,
    candidate_daily_returns,
    candidate_sharpes,
    deflate,
    deflation_table,
    effective_trials,
    freeze_record,
    headline,
    path_moments,
    trial_count_scenarios,
    trials_where_dsr_falls_below,
    variance_scenarios,
    write_freeze,
)
from src.logger import get_logger, log_frame, setup_logging
from src.pipelines.evaluate import DEFAULT_OUTPUT as EVALUATION_OUTPUT
from src.pipelines.evaluate import REFERENCE, EvaluationResult, load_path_returns
from src.pipelines.portfolio import REFERENCE as ALLOCATION_REFERENCE
from src.pipelines.research import DEFAULT_OUTPUT as RESEARCH_OUTPUT
from src.pipelines.research import ResearchResult, run_name

logger = get_logger(__name__)

PORTFOLIO_OUTPUT = PROJECT_ROOT / "reports" / "results" / "portfolio"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "research_report.md"
FREEZE_FILE = PROJECT_ROOT / "reports" / "results" / "holdout_freeze.json"

# The columns of the strategy table, in reading order.
SUMMARY_COLUMNS = {
    "kind": "kind",
    "sharpe_gross": "Sharpe gross",
    "sharpe_net": "Sharpe net",
    "sharpe_net_std": "Sharpe net, sd over paths",
    f"sharpe_net_vs_{REFERENCE}": "vs equal weight",
    "vs_reference_low": "95% low",
    "vs_reference_high": "95% high",
    "exposure_net_mean": "net exposure",
    "turnover_annual": "turnover / yr",
    "cost_annual": "cost / yr",
    "max_drawdown_net": "max drawdown",
}

ALLOCATION_COLUMNS = {
    "sharpe_net": "Sharpe net",
    "vs_equal_weight": "vs equal weight",
    "vs_equal_weight_low": "95% low",
    "vs_equal_weight_high": "95% high",
    "p_beat_equal_weight": "share of resamples beating",
    "max_drawdown_net": "max drawdown",
    "turnover_annual": "turnover / yr",
}

RESEARCH_METRICS = {
    "ic_mean": "pooled IC, mean over paths",
    "ic_std": "pooled IC, sd over paths",
    "positive_ic_paths": "paths with positive IC",
    "n_paths": "paths",
    "ic_year_t_stat": "t-stat of yearly IC",
    "cs_spearman_mean": "within-date rank IC, mean",
    "cs_spearman_t_stat": "within-date rank IC, t-stat",
    "hit_rate_mean": "hit rate",
    "edge_over_always_up_mean": "hit-rate edge over always-up",
    "n_rows": "labelled rows",
}

DSR_LEVELS = (0.95, 0.5)


class ReportError(ValueError):
    """Raised when the results needed for the report are missing or inconsistent."""


@dataclass
class ReportData:
    """Everything the report reads, so every section draws on the same objects."""

    research: ResearchResult
    evaluation: EvaluationResult
    config: Config
    trial_log: TrialLog
    portfolio: pd.DataFrame | None = None
    allocation_returns: pd.DataFrame | None = None


def _number(value: object, digits: int = 3) -> str:
    """A number for the report: fixed digits, n/a for missing values."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if isinstance(value, bool):
        return "yes" if value else "no"
    # Counts stay whole numbers whatever the table's precision.
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer() and value):
        return f"{int(value)}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(table: pd.DataFrame, digits: int = 3) -> str:
    """A DataFrame as a GitHub markdown table, numbers rounded for reading."""
    header = [table.index.name or ""] + [str(c) for c in table.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for index, row in table.iterrows():
        cells = [str(index)] + [_number(v, digits) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def load_research_result(
    config: Config, trial_log: TrialLog, folder: Path = RESEARCH_OUTPUT
) -> ResearchResult:
    """The saved research run for the configured model, as a ResearchResult.

    Raises:
        ReportError: If the run was never saved or never logged.
    """
    name = run_name(config)
    root = Path(folder) / name
    if not (root / "metrics.csv").is_file():
        raise ReportError(f"no saved research run in {root}; run make research first")

    def read(table: str) -> pd.DataFrame:
        path = root / f"{table}.csv"
        return pd.read_csv(path, index_col=0) if path.is_file() else pd.DataFrame()

    logged = trial_log.trial_table()
    logged = logged[(logged["name"] == name) & (logged["kind"] == "research")]
    if logged.empty:
        raise ReportError(f"research run '{name}' is saved but not in the trial log")

    return ResearchResult(
        name=name,
        trial_id=int(logged["trial_id"].iloc[-1]),
        metrics=read("metrics")["value"].astype(float).to_dict(),
        path_metrics=read("paths"),
        comparison=read("comparison"),
        predictions=pd.read_parquet(root / "predictions.parquet"),
        ic_by_year=read("ic_by_year"),
        ic_by_regime=read("ic_by_regime"),
        cs_ic_daily=read("cs_ic_daily"),
        redundancy=read("redundancy"),
        splits=read("splits"),
        importance=read("importance") if (root / "importance.csv").is_file() else None,
        files=sorted(root.iterdir()),
    )


def load_evaluation_result(folder: Path = EVALUATION_OUTPUT) -> EvaluationResult:
    """The saved strategy comparison, as an EvaluationResult without refitting.

    The per-path backtests are not rebuilt (evaluations is empty); their
    daily net returns are read from path_returns.parquet when needed.
    """
    folder = Path(folder)
    if not (folder / "comparison.csv").is_file():
        raise ReportError(f"no saved evaluation in {folder}; run make evaluate first")

    def read(name: str) -> pd.DataFrame:
        path = folder / name
        return pd.read_csv(path, index_col=0) if path.is_file() else pd.DataFrame()

    return EvaluationResult(
        evaluations={},
        comparison=read("comparison.csv"),
        paths=read("paths.csv"),
        regimes=read("regimes.csv"),
        pairs_selection=read("pairs_selection.csv"),
        path_check=read("path_check.csv"),
        monte_carlo=read("monte_carlo.csv") if (folder / "monte_carlo.csv").is_file() else None,
        files=sorted(folder.iterdir()),
    )


def load_report_data(config: Config) -> ReportData:
    """Every saved development result the report needs."""
    trial_log = TrialLog.from_config(config)
    portfolio = allocation_returns = None

    if (PORTFOLIO_OUTPUT / "comparison.csv").is_file():
        portfolio = pd.read_csv(PORTFOLIO_OUTPUT / "comparison.csv", index_col=0)
        allocation_returns = pd.read_parquet(PORTFOLIO_OUTPUT / "allocation_returns.parquet")

    return ReportData(
        research=load_research_result(config, trial_log),
        evaluation=load_evaluation_result(),
        config=config,
        trial_log=trial_log,
        portfolio=portfolio,
        allocation_returns=allocation_returns,
    )


def _check_development_only(data: ReportData) -> None:
    """Refuse any result whose returns reach validation.test_start."""
    holdout = pd.Timestamp(data.config.validation.test_start)
    frames = {"research predictions": data.research.predictions.index.get_level_values(0)}
    if data.allocation_returns is not None:
        frames["allocation returns"] = data.allocation_returns.index
    for name, series in _path_returns(data).items():
        frames[f"{name} path returns"] = series[0].index

    for name, index in frames.items():
        if len(index) and pd.Timestamp(index.max()) >= holdout:
            raise ReportError(f"{name} reach {index.max()}, inside the holdout")


def _path_returns(data: ReportData) -> dict[str, list[pd.Series]]:
    """Daily net returns per path: from the result object if it has them, else saved."""
    if data.evaluation.evaluations:
        return {
            name: [result.net for result in evaluation.results]
            for name, evaluation in data.evaluation.evaluations.items()
        }

    return load_path_returns(EVALUATION_OUTPUT)


def candidate_returns(data: ReportData):
    """Maps a candidate name to its saved net returns, one series per path."""
    paths = _path_returns(data)

    def returns_of(name: str) -> list[pd.Series]:
        if name in paths:
            return paths[name]
        allocation = name.removeprefix("allocation_")
        if data.allocation_returns is not None and f"{allocation}|net" in data.allocation_returns:
            return [data.allocation_returns[f"{allocation}|net"].dropna()]
        raise ReportError(f"no saved returns for '{name}'")

    return returns_of


def summary_table(results: EvaluationResult | pd.DataFrame) -> pd.DataFrame:
    """Strategies and benchmarks, net Sharpe first, with the comparison to equal weight.

    Args:
        results: An EvaluationResult or its comparison table.

    Returns:
        One row per strategy, sorted by net Sharpe, with readable column names.
    """
    comparison = results.comparison if isinstance(results, EvaluationResult) else results
    if comparison.empty:
        raise ReportError("the comparison table is empty")

    columns = [c for c in SUMMARY_COLUMNS if c in comparison.columns]
    table = comparison[columns].copy()
    for column in columns[1:]:
        table[column] = pd.to_numeric(table[column], errors="coerce")

    table = table.sort_values("sharpe_net", ascending=False).rename(columns=SUMMARY_COLUMNS)
    table.index.name = "strategy"

    return table


def trial_summary(trial_log: TrialLog) -> dict[str, pd.DataFrame]:
    """Executions and distinct trials by kind, and the trial-count scenarios.

    Returns:
        by_kind: executions and unique trials for every kind in the log.
        provenance: where the selection trials came from and how documented.
        scenarios: from trial_count_scenarios().
    """
    table = trial_log.trial_table()
    unique = trial_log.unique_trials().reset_index()

    by_kind = pd.DataFrame(
        {
            "executions": table.groupby("kind").size(),
            "unique_trials": unique.groupby("kind").size(),
            "names": unique.groupby("kind")["name"].nunique(),
        }
    ).fillna(0)
    by_kind["counts_towards_N"] = [kind in SELECTION_KINDS for kind in by_kind.index]
    by_kind.index.name = "kind"

    # code_version says how reproducible each record is: a commit, a commit with
    # uncommitted changes (-dirty), unrecorded legacy rows, or notebook outputs.
    provenance = (
        table[table["kind"].isin(SELECTION_KINDS)]
        .groupby(["kind", "code_version"])
        .size()
        .rename("executions")
        .to_frame()
    )

    return {
        "by_kind": by_kind.astype({"executions": int, "unique_trials": int, "names": int}),
        "provenance": provenance,
        "scenarios": trial_count_scenarios(trial_log),
    }


def deflation(data: ReportData) -> dict[str, pd.DataFrame]:
    """V[SR] scenarios, the DSR grid, and where the DSR crosses 0.95 and 0.5."""
    variances = variance_scenarios(data.evaluation.comparison, data.portfolio, data.trial_log)
    counts = trial_count_scenarios(data.trial_log)
    grid = deflation_table(counts, variances, candidate_returns(data))

    crossings = {}
    for v_name, rows in grid.groupby("variance_scenario", sort=False):
        moments = rows.iloc[0][["sharpe_period", "n", "skew", "kurtosis"]].to_dict()
        variance = float(variances.loc[v_name, "variance_period"])
        crossings[v_name] = {
            "selected": rows.iloc[0]["selected"],
            **{
                f"N where DSR < {level}": trials_where_dsr_falls_below(moments, variance, level)
                for level in DSR_LEVELS
            },
        }

    crossing = pd.DataFrame(crossings).T
    crossing.index.name = "variance_scenario"

    # The pre-registered headline, then what correlation between candidates
    # does to it: the candidates' clusters stand in for their raw count, the
    # other trials (research, notebooks, earlier pairs) are kept as counted.
    returns_of = candidate_returns(data)
    top = headline(counts, variances, returns_of)
    candidates = candidate_sharpes(data.evaluation.comparison, data.portfolio).index
    estimate = effective_trials(candidate_daily_returns(candidates, returns_of))
    others = int(counts.loc["distinct_less_overlap", "n_trials"]) - len(candidates)
    moments = path_moments(returns_of(top["selected"]))
    effective = {}
    for method in ("clusters", "participation"):
        n_trials = round(estimate[method]) + others
        effective[method] = {
            "candidates": estimate["candidates"],
            "independent_candidates": estimate[method],
            "other_trials": others,
            "n_trials": n_trials,
            **deflate(moments, n_trials, top["variance_used_annual"] / PERIODS),
        }
    effective = pd.DataFrame(effective).T
    effective.index.name = "estimate"

    return {
        "variances": variances,
        "counts": counts,
        "grid": grid,
        "crossings": crossing,
        "headline": top,
        "effective": effective,
        "cluster_correlation": estimate["cluster_correlation"],
    }


def findings_section(results: ReportData) -> str:
    """The findings, each sentence built from the numbers it states.

    Says plainly when nothing distinguishably beats equal weight: a strategy
    counts as better only when the whole 95% interval of its Sharpe difference
    lies above zero.
    """
    comparison = results.evaluation.comparison
    metrics = results.research.metrics
    lines = []

    # The forecast itself.
    lines.append(
        f"- **Forecast.** The pooled information coefficient averaged "
        f"{_number(metrics.get('ic_mean'), 4)} over {_number(metrics.get('n_paths'), 0)} CPCV "
        f"paths (positive on {_number(metrics.get('positive_ic_paths'), 0)}), and the "
        f"within-date rank IC {_number(metrics.get('cs_spearman_mean'), 4)} "
        f"(t = {_number(metrics.get('cs_spearman_t_stat'), 2)}). The hit rate beat always "
        f"predicting up by {_number(metrics.get('edge_over_always_up_mean'), 4)}."
    )

    # Strategies against equal weight.
    strategies = comparison[comparison["kind"] == "backtest"].copy()
    for column in ("sharpe_net", "vs_reference_low", "vs_reference_high", "exposure_net_mean"):
        strategies[column] = pd.to_numeric(strategies[column], errors="coerce")
    reference = float(comparison.loc[REFERENCE, "sharpe_net"])
    best = strategies["sharpe_net"].idxmax()
    better = strategies.index[strategies["vs_reference_low"] > 0].tolist()
    worse = strategies.index[strategies["vs_reference_high"] < 0].tolist()

    lines.append(
        f"- **Best strategy.** {best}, net Sharpe {_number(strategies.loc[best, 'sharpe_net'], 2)}"
        f", against {_number(reference, 2)} for {REFERENCE}; 95% interval of the difference "
        f"[{_number(strategies.loc[best, 'vs_reference_low'], 2)}, "
        f"{_number(strategies.loc[best, 'vs_reference_high'], 2)}]."
    )
    if better:
        lines.append(f"- **Beats equal weight distinguishably:** {', '.join(better)}.")
    else:
        lines.append(
            f"- **Nothing distinguishably beats equal weight.** No strategy's 95% interval for "
            f"the Sharpe difference against {REFERENCE} lies above zero."
        )
    if worse:
        lines.append(
            f"- **Distinguishably worse than equal weight:** {', '.join(worse)} "
            f"(whole interval below zero)."
        )

    exposure = strategies.loc[best, "exposure_net_mean"]
    if exposure > 0.5:
        lines.append(
            f"- **Exposure.** {best} was net long {_number(exposure, 2)} of capital on average, "
            f"so much of its return is market exposure rather than forecasting skill."
        )

    # Costs.
    if "sharpe_lost_to_costs" in strategies.columns:
        lost = pd.to_numeric(strategies["sharpe_lost_to_costs"], errors="coerce")
        lines.append(
            f"- **Costs.** Costs took {_number(lost.min(), 2)} to {_number(lost.max(), 2)} of "
            f"annual Sharpe, the most from {lost.idxmax()}."
        )

    # Allocations.
    if results.portfolio is not None:
        allocations = results.portfolio.drop(index=ALLOCATION_REFERENCE)
        ahead = allocations.index[allocations["vs_equal_weight_low"] > 0].tolist()
        top = allocations["sharpe_net"].idxmax()
        lines.append(
            f"- **Allocations.** The best, {top}, had net Sharpe "
            f"{_number(allocations.loc[top, 'sharpe_net'], 2)} against "
            f"{_number(results.portfolio.loc[ALLOCATION_REFERENCE, 'sharpe_net'], 2)} for "
            f"monthly equal weight; "
            + (
                f"{', '.join(ahead)} beat it distinguishably."
                if ahead
                else "no allocation beats equal weight distinguishably."
            )
        )

    return "\n".join(lines)


def _dsr_findings(tables: dict, counts: pd.DataFrame) -> str:
    """The headline DSR, then the range over every scenario."""
    top = tables["headline"]
    grid = tables["grid"]
    low, high = grid["dsr"].min(), grid["dsr"].max()
    rows = grid.loc[[grid["dsr"].idxmin(), grid["dsr"].idxmax()]]

    return (
        f"Pre-registered headline: {top['selected']} (annual Sharpe "
        f"{_number(top['sharpe_annual'], 3)}) against a luck threshold of "
        f"{_number(top['luck_threshold_annual'], 3)} for N = {top['n_trials']} trials and "
        f"V[SR] = {_number(top['variance_used_annual'], 4)} ({top['variance_used']}): "
        f"DSR {_number(top['dsr'], 3)}. "
        f"Across {counts['n_trials'].min()} to {counts['n_trials'].max()} trials and every "
        f"variance scenario, the DSR of the selected strategy ranges from {_number(low, 3)} "
        f"({rows.iloc[0]['selected']}, {rows.iloc[0]['count_scenario']}, "
        f"{rows.iloc[0]['variance_scenario']}) to {_number(high, 3)} "
        f"({rows.iloc[1]['selected']}, {rows.iloc[1]['count_scenario']}, "
        f"{rows.iloc[1]['variance_scenario']}). "
        + (
            "None reaches 0.95, so no selected Sharpe ratio is distinguishable from the "
            "best of that many unskilled trials."
            if high < 0.95
            else "Some scenarios reach 0.95; read them with the dependence caveat below."
        )
    )


def limitations_section(config: Config) -> str:
    """What the results cannot show, from the config where a number is involved."""
    tickers = config.universe.tickers

    return "\n".join(
        [
            f"- **Survivorship.** The universe is {len(tickers)} tickers "
            f"({', '.join(tickers)}) known to have survived to {config.universe.end_date}; "
            f"bias status in the config: {config.universe.survivorship_bias}. Every backtest "
            f"is flattered by it, the long books most.",
            "- **One model, one universe.** The strategies share one random forest's forecasts, "
            "so their results are not independent evidence.",
            "- **Trial dependence.** The DSR assumes independent trials. The trials here are "
            "strongly correlated (shared forecasts, shared pair legs, the same eleven stocks), "
            "so the effective N is smaller than any count above and those DSRs are "
            "conservative. The raw count is not evidence that the trials are independent.",
            "- **Notebook trials.** The 42 notebook experiments are recorded from their saved "
            "outputs only. Their code and settings are not recoverable, they ran on data to "
            "2024, and none has a development Sharpe ratio, so they enter N but not V[SR].",
            "- **DSR meaning.** The DSR is the probability, under its assumptions, that the "
            "selected strategy's true Sharpe ratio over the development sample exceeded the "
            "luck threshold. It is not a probability of future profit.",
            f"- **Holdout.** Everything here uses data before {config.validation.test_start}. "
            "The holdout has not been evaluated.",
        ]
    )


def build_report(
    research_result: ResearchResult,
    evaluation_result: EvaluationResult,
    config: Config,
    path: str | Path = DEFAULT_REPORT,
    *,
    trial_log: TrialLog | None = None,
    portfolio: pd.DataFrame | None = None,
    allocation_returns: pd.DataFrame | None = None,
) -> Path:
    """Write the markdown report.

    Args:
        research_result: The research run, fresh or from load_research_result().
        evaluation_result: The strategy comparison, fresh or from
            load_evaluation_result().
        config: The project config.
        path: Where to write the report.
        trial_log: The trial log; defaults to the configured one.
        portfolio: The allocation comparison table, if the portfolio step ran.
        allocation_returns: Its saved daily returns (name|gross, name|net).

    Returns:
        The path written.

    Raises:
        ReportError: If any input reaches the holdout, or the saved results
            disagree with the trial log.
    """
    data = ReportData(
        research=research_result,
        evaluation=evaluation_result,
        config=config,
        trial_log=trial_log or TrialLog.from_config(config),
        portfolio=portfolio,
        allocation_returns=allocation_returns,
    )
    _check_development_only(data)

    trials = trial_summary(data.trial_log)
    tables = deflation(data)
    research = pd.DataFrame(
        {
            "value": [research_result.metrics.get(key, math.nan) for key in RESEARCH_METRICS],
        },
        index=list(RESEARCH_METRICS.values()),
    )
    research.index.name = "metric"
    baselines = research_result.comparison[
        [
            c
            for c in ("ic", "hit_rate", "edge_over_always_up", "cs_spearman_mean")
            if c in research_result.comparison.columns
        ]
    ]

    grid = tables["grid"].set_index(["variance_scenario", "count_scenario"])[
        ["selected", "n_trials", "sharpe_annual", "luck_threshold_annual", "psr_vs_zero", "dsr"]
    ]
    grid.index = [f"{v} / {n}" for v, n in grid.index]
    grid.index.name = "variance / count scenario"

    sections = [
        "# Stock return predictor: development results",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from saved development results. "
        f"Code version {code_version()}, config {config_hash(config)}. Development period "
        f"{config.universe.start_date} to the day before {config.validation.test_start}. "
        f"**The holdout has not been evaluated:** {config.validation.test_start} to "
        f"{config.universe.end_date} is untouched.",
        "",
        "## Findings",
        "",
        findings_section(data),
        f"- **Deflated Sharpe.** {_dsr_findings(tables, trials['scenarios'])}",
        "",
        f"## Forecast quality ({research_result.name}, trial {research_result.trial_id})",
        "",
        _markdown(research, 4),
        "",
        "Against naive forecasts:",
        "",
        _markdown(baselines, 4),
        "",
        "## Strategies and benchmarks (mean over CPCV paths, net of costs)",
        "",
        "Intervals are 95% paired block-bootstrap intervals for the Sharpe difference "
        f"against {REFERENCE}, averaged over paths.",
        "",
        _markdown(summary_table(evaluation_result), 3),
    ]
    if portfolio is not None:
        allocations = portfolio[[c for c in ALLOCATION_COLUMNS if c in portfolio.columns]]
        sections += [
            "",
            "## Allocations (monthly rebalanced, one development path)",
            "",
            _markdown(allocations.rename(columns=ALLOCATION_COLUMNS), 3),
        ]
    sections += [
        "",
        "## Trials",
        "",
        _markdown(trials["by_kind"], 0),
        "",
        "Where the selection trials came from:",
        "",
        _markdown(trials["provenance"].reset_index().set_index("kind"), 0),
        "",
        "Defensible values of N:",
        "",
        _markdown(trials["scenarios"], 0),
        "",
        "## Deflated Sharpe ratio",
        "",
        "### Headline",
        "",
        "Fixed in advance by reports/methodology/trial_accounting.md (rules 2 and 3): N "
        f"counts every distinct fingerprint ({HEADLINE_COUNT}); V[SR] is the largest of the "
        "pre-registered options (the trial log's variance, the current candidates' with and "
        "without the allocations, and the selected strategy's own sampling variance), the "
        "one giving the highest luck threshold. A minimum "
        "track record of inf means the observed Sharpe is below the luck threshold, so no "
        "length of record at that Sharpe would clear it.",
        "",
        _markdown(pd.Series(tables["headline"], name="value").to_frame().rename_axis("item"), 4),
        "",
        "### Correlated trials",
        "",
        "The DSR assumes independent trials. Clustering the current candidates' daily net "
        f"returns (average linkage, cut at correlation {tables['cluster_correlation']}) and "
        "the participation ratio of their correlation matrix estimate how many independent "
        "candidates they amount to. The other trials are kept at their counted value. This "
        "sits beside the headline and does not replace it.",
        "",
        _markdown(tables["effective"], 3),
        "",
        "### Every scenario",
        "",
        "Variance of annual net Sharpe ratios across candidates (benchmarks and notebook "
        "records excluded; notebook records have no verifiable development Sharpe):",
        "",
        _markdown(tables["variances"][["candidates", "variance_annual", "best", "source"]], 4),
        "",
        "DSR of each variance scenario's best candidate, under each trial count. The luck "
        "threshold is the expected maximum annual Sharpe of N unskilled trials.",
        "",
        _markdown(grid, 3),
        "",
        "Number of trials at which the DSR falls below each level:",
        "",
        _markdown(tables["crossings"], 0),
        "",
        "## Limitations",
        "",
        limitations_section(config),
        "",
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Report written to %s", path)

    return path


def freeze(data: ReportData, path: Path = FREEZE_FILE) -> Path:
    """Write N, V[SR], the headline threshold and the trial list, once, before the holdout.

    Raises:
        ScenarioError: If a freeze exists already or the code is uncommitted.
    """
    tables = deflation(data)
    record = freeze_record(
        data.trial_log,
        tables["counts"],
        tables["variances"],
        tables["headline"],
        str(data.config.validation.test_start),
        config_hash(data.config),
    )

    return write_freeze(record, path)


def main(argv: list[str] | None = None) -> None:
    """Build the report from the saved development results, or freeze the DSR inputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--freeze",
        action="store_true",
        help=f"write {FREEZE_FILE.relative_to(PROJECT_ROOT)} once, from committed code",
    )
    args = parser.parse_args(argv)

    setup_logging()
    config = load_config()
    data = load_report_data(config)
    log_frame(logger, data.evaluation.comparison, "Saved comparison")

    if args.freeze:
        freeze(data)
        return

    build_report(
        data.research,
        data.evaluation,
        config,
        trial_log=data.trial_log,
        portfolio=data.portfolio,
        allocation_returns=data.allocation_returns,
    )


if __name__ == "__main__":
    main()
