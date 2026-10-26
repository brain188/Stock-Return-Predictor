"""How many trials, how dispersed, and what the best one is worth after deflation.

The deflated Sharpe ratio needs two inputs that no log can state exactly: the
number of independent trials N and the variance of their Sharpe ratios. This
module does not pick one answer. It reports defensible scenarios side by side,
each with the rule that produced it:

- Trial counts come from the trial log. The upper bound counts every distinct
  fingerprint, including versions superseded by bug fixes; lower counts merge
  versions of one named choice, remove notebook experiments the pipeline
  re-ran, or leave the notebooks out altogether.
- Sharpe ratios come from the saved development comparison tables (the
  result objects, not retyped), checked against the trial log. Benchmarks are
  not candidates and notebook records are excluded: none has a verifiable
  development Sharpe ratio.

One combination is the headline, fixed in advance by
reports/methodology/trial_accounting.md (rules 2 and 3) before any DSR was
computed: N counts every distinct fingerprint, and V[SR] is the larger of the
log-wide variance and the selected strategy's own sampling variance.

Trials here are far from independent: the strategies share one model's
forecasts, the pairs share legs, and the allocations hold the same eleven
stocks. Correlated trials behave like fewer independent ones, so every raw
count overstates the effective N and the DSR computed from it is
conservative; effective_trials() estimates how much. The DSR is still a
probability about the past sample under the model's assumptions, not a
probability of future profit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src.evaluation.deflated_sharpe import (
    _sharpe_dispersion,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    return_moments,
    to_annual_sharpe,
)
from src.evaluation.notebook_trials import OVERLAPS
from src.evaluation.trial_log import METRIC_PREFIX, SELECTION_KINDS, TrialLog, code_version
from src.logger import get_logger

logger = get_logger(__name__)

PERIODS = 252

# Agreement required between a saved Sharpe ratio and the trial log's copy.
LOG_TOLERANCE = 1e-9

# Largest N searched when finding where the DSR crosses a level.
MAX_TRIALS = 1_000_000

# Candidates whose daily returns correlate above this are one cluster when
# estimating the effective number of independent trials.
CLUSTER_CORRELATION = 0.5

HEADLINE_COUNT = "all_fingerprints"
HEADLINE_VARIANCE = "every_backtest_fingerprint"


class ScenarioError(ValueError):
    """Raised when saved results and the trial log disagree, or a scenario is empty."""


def trial_count_scenarios(log: TrialLog) -> pd.DataFrame:
    """The defensible values of N, from the most to the least generous.

    Only selection trials count (research, backtest, notebook): benchmarks and
    diagnostics were not candidates for selection.

    Returns:
        Indexed by scenario, with n_trials, pipeline and notebook (its two
        parts) and the rule that produced it.
    """
    unique = log.unique_trials().reset_index()
    unique = unique[unique["kind"].isin(SELECTION_KINDS)]
    names = unique[["kind", "name"]].drop_duplicates()

    def split(frame: pd.DataFrame) -> tuple[int, int]:
        notebook = int((frame["kind"] == "notebook").sum())
        return len(frame) - notebook, notebook

    pipeline_names = set(names.loc[names["kind"] != "notebook", "name"])
    overlaps = sorted(
        notebook
        for notebook, pipeline in OVERLAPS.items()
        if notebook in set(names["name"]) and pipeline in pipeline_names
    )

    rows = {}
    pipeline, notebook = split(unique)
    rows["all_fingerprints"] = (
        pipeline,
        notebook,
        "every distinct (kind, fingerprint), including versions superseded by bug fixes; "
        "upper bound",
    )
    pipeline, notebook = split(names)
    rows["distinct_choices"] = (
        pipeline,
        notebook,
        "versions of one named trial merged: a corrected rerun is the same choice",
    )
    rows["distinct_less_overlap"] = (
        pipeline,
        notebook - len(overlaps),
        f"as distinct_choices, less {len(overlaps)} notebook experiments the pipeline "
        f"re-ran ({', '.join(overlaps) or 'none'})",
    )
    rows["pipeline_only"] = (
        pipeline,
        0,
        "pipeline trials only; ignores the notebook search, so understates N",
    )

    table = pd.DataFrame(rows, index=["pipeline", "notebook", "rule"]).T
    table.insert(0, "n_trials", table["pipeline"] + table["notebook"])
    table.index.name = "scenario"

    return table.astype({"n_trials": int, "pipeline": int, "notebook": int})


def candidate_sharpes(
    comparison: pd.DataFrame,
    allocations: pd.DataFrame | None = None,
    reference: str = "equal_weight",
) -> pd.Series:
    """Annualised net Sharpe of every selection candidate in the saved results.

    Args:
        comparison: The evaluation comparison table (kind, sharpe_net columns).
        allocations: The portfolio comparison table, or None to leave the
            allocations out. Its reference allocation is a benchmark.
        reference: Name of the reference allocation.

    Returns:
        Named like the trial log: strategies by name, allocations as
        allocation_<name>.
    """
    sharpes = comparison.loc[comparison["kind"] == "backtest", "sharpe_net"].astype(float)

    if allocations is not None:
        rest = allocations.drop(index=reference)["sharpe_net"].astype(float)
        sharpes = pd.concat([sharpes, rest.rename(lambda name: f"allocation_{name}")])

    return sharpes.rename("sharpe_annual")


def check_against_log(sharpes: pd.Series, log: TrialLog) -> None:
    """Refuse saved Sharpe ratios that differ from their latest trial log entry.

    Raises:
        ScenarioError: If a candidate is missing from the log, or its value
            differs, which would mean the saved results are not the run logged.
    """
    table = log.trial_table()
    column = METRIC_PREFIX + "sharpe"

    for name, value in sharpes.items():
        rows = table[(table["name"] == name) & (table["status"] == "ok")]
        if rows.empty:
            raise ScenarioError(f"'{name}' is in the saved results but not in the trial log")
        logged = float(rows.iloc[-1][column])
        if not math.isclose(logged, value, rel_tol=0, abs_tol=LOG_TOLERANCE):
            raise ScenarioError(f"'{name}': saved Sharpe {value} but the log's latest is {logged}")


def variance_scenarios(
    comparison: pd.DataFrame,
    allocations: pd.DataFrame | None,
    log: TrialLog,
    periods: int = PERIODS,
) -> pd.DataFrame:
    """V[SR] with and without the allocations, and from every logged fingerprint.

    Returns:
        Indexed by scenario: candidates, variance_annual, variance_period
        (annual / periods, the unit the DSR uses), best and best_sharpe_annual.
    """
    sets = {"strategies": candidate_sharpes(comparison)}
    if allocations is not None:
        sets["strategies_and_allocations"] = candidate_sharpes(comparison, allocations)

    rows = {}
    for scenario, sharpes in sets.items():
        check_against_log(sharpes, log)
        rows[scenario] = {
            "candidates": len(sharpes),
            "variance_annual": float(sharpes.var(ddof=1)),
            "best": sharpes.idxmax(),
            "best_sharpe_annual": float(sharpes.max()),
            "source": "saved development results",
        }

    # Every backtest fingerprint ever logged, superseded versions included (the
    # pre-registered V[SR]). A superseded version cannot be selected, so the
    # selected strategy is the best current candidate.
    logged = log.unique_trials({"kind": "backtest"})[METRIC_PREFIX + "sharpe"].dropna()
    current = sets["strategies_and_allocations" if allocations is not None else "strategies"]
    rows[HEADLINE_VARIANCE] = {
        "candidates": len(logged),
        "variance_annual": float(logged.var(ddof=1)),
        "best": current.idxmax(),
        "best_sharpe_annual": float(current.max()),
        "source": "trial log, superseded versions included",
    }

    table = pd.DataFrame(rows).T
    table["variance_period"] = table["variance_annual"].astype(float) / periods
    table.index.name = "scenario"

    return table


def path_moments(paths: Sequence[pd.Series]) -> dict[str, float]:
    """Per-period Sharpe, n, skewness and raw kurtosis, averaged over CPCV paths.

    Paths cover the same dates, so they are averaged, not pooled: pooling would
    count each day several times and overstate n. A path on which the strategy
    never traded has no Sharpe ratio and is skipped, as in the comparison table.

    Raises:
        ScenarioError: If no path has returns that vary.
    """
    traded = [path.dropna() for path in paths if path.dropna().std() > 0]
    if not traded:
        raise ScenarioError("no path has varying returns: the Sharpe ratio is undefined")
    moments = np.array([return_moments(path.to_numpy()) for path in traded])

    return {
        "sharpe_period": float(moments[:, 0].mean()),
        "n": round(moments[:, 1].mean()),
        "skew": float(moments[:, 2].mean()),
        "kurtosis": float(moments[:, 3].mean()),
        "paths": len(traded),
    }


def deflate(
    moments: Mapping[str, float],
    n_trials: int,
    variance_period: float,
    periods: int = PERIODS,
) -> dict[str, float]:
    """The selected strategy's PSR, luck threshold, DSR and minimum track record.

    Args:
        moments: From path_moments().
        n_trials: N, the number of trials searched.
        variance_period: V[SR] across trials, per-period units.
        periods: Return periods a year, to annualise for reading.

    Returns:
        sharpe_annual, luck_threshold_annual, psr_vs_zero, dsr, and
        min_track_record_years (the years of returns needed for the DSR's
        confidence to reach 95% at this Sharpe; inf when it cannot).
    """
    sharpe, n = moments["sharpe_period"], int(moments["n"])
    skew, kurtosis = moments["skew"], moments["kurtosis"]
    threshold = expected_max_sharpe(n_trials, variance_period)

    return {
        "sharpe_annual": to_annual_sharpe(sharpe, periods),
        "luck_threshold_annual": to_annual_sharpe(threshold, periods),
        "psr_vs_zero": probabilistic_sharpe_ratio(sharpe, n, skew, kurtosis, 0.0),
        "dsr": deflated_sharpe_ratio(sharpe, n_trials, n, skew, kurtosis, variance_period),
        "min_track_record_years": minimum_track_record_length(
            sharpe, skew, kurtosis, target=threshold
        )
        / periods,
    }


def trials_where_dsr_falls_below(
    moments: Mapping[str, float],
    variance_period: float,
    level: float,
) -> int | None:
    """The smallest N at which the DSR drops below `level`.

    The DSR falls as N grows, so this is a bisection. Returns 1 when even a
    single trial is below the level, and None when N up to MAX_TRIALS stays
    above it.
    """

    def dsr(n_trials: int) -> float:
        return deflate(moments, n_trials, variance_period)["dsr"]

    if dsr(1) < level:
        return 1
    if dsr(MAX_TRIALS) >= level:
        return None

    low, high = 1, MAX_TRIALS
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (middle, high) if dsr(middle) >= level else (low, middle)

    return high


def deflation_table(
    counts: pd.DataFrame,
    variances: pd.DataFrame,
    returns_of: Callable[[str], Sequence[pd.Series]],
) -> pd.DataFrame:
    """The DSR of each variance scenario's best candidate under each trial count.

    Args:
        counts: From trial_count_scenarios().
        variances: From variance_scenarios(); rows without a best (the
            log-wide scenario) are skipped, having no saved returns.
        returns_of: Maps a candidate name to its saved daily net returns,
            one series per path.

    Returns:
        One row per (variance scenario, count scenario).
    """
    rows = []

    for v_name, v_row in variances.dropna(subset=["best"]).iterrows():
        moments = path_moments(returns_of(v_row["best"]))
        for n_name, n_row in counts.iterrows():
            rows.append(
                {
                    "variance_scenario": v_name,
                    "count_scenario": n_name,
                    "selected": v_row["best"],
                    "n_trials": int(n_row["n_trials"]),
                    "variance_annual": float(v_row["variance_annual"]),
                    **moments,
                    **deflate(moments, int(n_row["n_trials"]), float(v_row["variance_period"])),
                }
            )

    if not rows:
        raise ScenarioError("no variance scenario names a candidate with saved returns")

    return pd.DataFrame(rows)


def sharpe_sampling_variance(moments: Mapping[str, float]) -> float:
    """Variance of the selected strategy's own per-period Sharpe estimate.

    The PSR's denominator: (1 - skew SR + (kurtosis - 1)/4 SR^2) / (n - 1),
    the analytic counterpart of the spread of Sharpe ratios across Phase 7's
    resampled paths (trial_accounting.md, rule 3's cross-check).
    """
    dispersion = _sharpe_dispersion(moments["sharpe_period"], moments["skew"], moments["kurtosis"])

    return dispersion / (int(moments["n"]) - 1)


def headline(
    counts: pd.DataFrame,
    variances: pd.DataFrame,
    returns_of: Callable[[str], Sequence[pd.Series]],
    periods: int = PERIODS,
) -> dict[str, Any]:
    """The pre-registered DSR: trial_accounting.md, rules 2 and 3.

    N counts every distinct fingerprint. V[SR] is the one variance, of all the
    pre-registered options, that gives the highest luck threshold (the
    largest): the log-wide variance (rule 3), the current candidates' variance
    with and without the allocations ("uses the larger luck threshold"), and
    the selected strategy's own sampling variance (rule 3's cross-check).

    Returns:
        The selected strategy, N, every variance option (annual), the one
        used, the deflate() results, and the N at which the DSR falls below 0.95.
    """
    selected = variances.loc[HEADLINE_VARIANCE, "best"]
    moments = path_moments(returns_of(selected))
    options = variances["variance_period"].astype(float).to_dict()
    options["own_sampling_variance"] = sharpe_sampling_variance(moments)
    source = max(options, key=options.get)
    used = options[source]
    n_trials = int(counts.loc[HEADLINE_COUNT, "n_trials"])

    return {
        "selected": selected,
        "n_trials": n_trials,
        **{f"variance_{name}_annual": value * periods for name, value in options.items()},
        "variance_used": source,
        "variance_used_annual": used * periods,
        **moments,
        **deflate(moments, n_trials, used, periods),
        "n_where_dsr_below_0.95": trials_where_dsr_falls_below(moments, used, 0.95),
    }


def candidate_daily_returns(
    names: Sequence[str], returns_of: Callable[[str], Sequence[pd.Series]]
) -> pd.DataFrame:
    """One daily net return series per candidate: the mean over its CPCV paths.

    Paths cover the same dates, so their mean is the candidate's typical day.
    Candidates are aligned on the dates they share; a candidate that never
    varies (a pair that never traded) is dropped, having no correlation.
    """
    frame = pd.concat(
        {name: pd.concat(list(returns_of(name)), axis=1).mean(axis=1) for name in names}, axis=1
    ).dropna()

    return frame.loc[:, frame.std() > 0]


def effective_trials(
    returns: pd.DataFrame, correlation: float = CLUSTER_CORRELATION
) -> dict[str, float]:
    """How many independent trials the candidates amount to (trial_accounting.md, rule 4).

    Two estimates, neither of which replaces the headline count:

    - clusters: average-linkage clusters of the distance sqrt((1 - rho) / 2),
      cut where the correlation falls to `correlation`;
    - participation: the participation ratio of the correlation matrix's
      eigenvalues, (sum lambda)^2 / sum lambda^2, which is N for independent
      series and 1 for identical ones.

    Returns:
        candidates, clusters, participation and the threshold used.
    """
    corr = returns.corr().to_numpy()
    distance = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average")
    clusters = fcluster(tree, t=math.sqrt((1 - correlation) / 2), criterion="distance")
    eigen = np.linalg.eigvalsh(corr)

    return {
        "candidates": returns.shape[1],
        "clusters": len(set(clusters)),
        "participation": float(eigen.sum() ** 2 / (eigen**2).sum()),
        "cluster_correlation": correlation,
    }


def freeze_record(
    log: TrialLog,
    counts: pd.DataFrame,
    variances: pd.DataFrame,
    result: Mapping[str, Any],
    holdout_start: str,
    config: str = "",
) -> dict[str, Any]:
    """Everything the holdout will be judged against (trial_accounting.md, rule 6).

    Records N, V[SR], the headline threshold, the config hash and the list of
    trials, with a hash of the trial log, so trials added later are visible as
    additions. The primary candidate for the holdout is not chosen here: that
    is a decision for the holdout protocol, made by a person.
    """
    trials = log.unique_trials().reset_index()
    trials = trials[trials["kind"].isin(SELECTION_KINDS)]

    return {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": code_version(),
        "holdout_start": holdout_start,
        "config_hash": config,
        "trial_log_rows": log.trial_count(),
        "trial_log_sha256": hashlib.sha256(log.path.read_bytes()).hexdigest(),
        "headline": dict(result),
        "count_scenarios": counts.to_dict(orient="index"),
        "variance_scenarios": variances.to_dict(orient="index"),
        "trials": trials[["kind", "fingerprint", "name", "executions"]].to_dict(orient="records"),
    }


def write_freeze(record: Mapping[str, Any], path: str | Path, *, allow_dirty: bool = False) -> Path:
    """Write the freeze file once, from committed code.

    Raises:
        ScenarioError: If the file exists already (a freeze is never
            rewritten), or the code has uncommitted changes and allow_dirty
            is False.
    """
    path = Path(path)
    if path.exists():
        raise ScenarioError(f"{path} exists: a freeze is written once and never rewritten")
    if str(record["code_version"]).endswith("-dirty") and not allow_dirty:
        raise ScenarioError("commit the code first: the freeze must record a clean commit")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    logger.info("Froze N = %s and V[SR] in %s", record["headline"]["n_trials"], path)

    return path
