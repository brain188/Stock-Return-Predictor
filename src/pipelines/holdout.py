"""The one-time 2020-2024 holdout evaluation: declaration, preflight, preparation and run.

The holdout can be opened once. Everything that decides how it is judged is
fixed before any holdout price is read, and checked again by preflight()
immediately before the run:

- The freeze (reports/results/holdout_freeze.json) is the copy committed in
  FREEZE_COMMIT. It holds N and V[SR], written by code at FROZEN_CODE_VERSION.
  The freeze can only exist in a later commit than the code that wrote it,
  and the runner is later still, so HEAD never equals FROZEN_CODE_VERSION. What
  is checked instead: FROZEN_CODE_VERSION is an ancestor of HEAD, and the diff
  between them touches only ALLOWED_PATHS (plus the one KINDS edit).
- The trial log still starts with the frozen 175 rows. Their hash is checked
  as written (CRLF, as the csv module writes it on the freezing machine) or
  LF-normalised (as git stores it and checks it out on Linux); both forms are
  derived from the committed blob. Rows after them must be holdout rows.
- The declaration (reports/results/holdout_declaration.json) names the
  candidates, benchmarks, window, costs and statistics, and pins the protocol
  by hash. It is written once; a change before opening is a numbered
  replacement that cites the file it replaces, never an overwrite.
- The development-only artifacts (the final model and the pairs selection)
  match the hashes in their manifest, written once by prepare_artifacts().
- No earlier attempt is overwritten.

Only then is an attempt folder made, STARTED.json written with exclusive
create, and the price loader called. Every output is created exclusively too.
A failed or interrupted attempt is kept as it is; the next one goes in its own
folder, needs a stated reason (--new-attempt), and is refused until the earlier
folder has been committed.

Timing (holdout_protocol.md, section 3), in the engine's convention that a
weight held after the close of t earns the return of t + 1:

- D0, the first trading date on or after validation.test_start, is the
  decision date. Each book's target at D0 is what its own development code
  asks for at the close of D0, using only prices up to D0.
- Every book is flat until D1 = D0 + execution_lag bars, when it trades from
  flat to its D0 target and pays the entry cost. D1 is the first row of the
  evaluation window; its gross return is zero for every book.
- The bar after D1 is the first return a book earns. The window ends at the
  last date on or before universe.end_date on which every ticker has a price.

Nothing here is run by `make all` or CI.

    python entrypoint/holdout.py check    # preflight only: reads no price
"""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import io
import json
import math
import subprocess
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult, run_backtest
from src.backtest.momentum import DAYS_PER_MONTH, momentum_signal
from src.backtest.pairs import find_pairs, hedge_ratio, spread_zscore
from src.backtest.performance import (
    exposure,
    performance_stats,
    sharpe_difference_interval,
    sharpe_standard_error,
)
from src.backtest.strategy import STRATEGIES, MeanReversionPairs, delay, enforce_limits
from src.config import PROJECT_ROOT, Config
from src.data.cleaning import load_clean_panel
from src.evaluation.deflated_sharpe import (
    SharpeError,
    probabilistic_sharpe_ratio,
    return_moments,
)
from src.evaluation.trial_log import TrialLog, code_version, config_hash, trial_fingerprint
from src.evaluation.trial_scenarios import deflate
from src.features.build import build_features, warmup_rows
from src.logger import get_logger, log_frame
from src.models.baseline import Forecaster
from src.models.estimator import build_estimator, fit_with_weights
from src.models.persistence import load_model, save_model
from src.pipelines.portfolio import allocation_weights
from src.pipelines.research import (
    development_panel,
    forecast,
    prepare_research_data,
    target_for,
)
from src.portfolio.construction import rebalance
from src.reporting.report import _markdown

logger = get_logger(__name__)

# --- What was frozen -------------------------------------------------------

FROZEN_CODE_VERSION = "20cdb7a"
FREEZE_COMMIT = "65263c7"
# The V[SR] the headline used: the largest pre-registered option.
VARIANCE_SCENARIO = "strategies_and_allocations"
PERIODS = 252

RESULTS_ROOT = PROJECT_ROOT / "reports" / "results"
FREEZE_FILE = RESULTS_ROOT / "holdout_freeze.json"
DECLARATION_FILE = RESULTS_ROOT / "holdout_declaration.json"
ARTIFACT_DIR = RESULTS_ROOT / "holdout_artifacts"
HOLDOUT_DIR = RESULTS_ROOT / "holdout"
PROTOCOL_FILE = PROJECT_ROOT / "reports" / "methodology" / "holdout_protocol.md"

MANIFEST = "manifest.json"
MODEL_DIR = "model"
PAIRS_FILE = "pairs_selection.csv"
STARTED = "STARTED.json"
FAILED = "FAILED.json"
COMPLETE = "COMPLETE.json"
ATTEMPT_PREFIX = "attempt-"

# Hashed as stored; every other file is hashed LF-normalised, because git may
# check text out with CRLF on one machine and LF on another.
BINARY_SUFFIXES = {".joblib", ".parquet", ".pkl"}

# --- What is declared ------------------------------------------------------

PRIMARY = "allocation_minimum_variance"
SECONDARIES = (
    "allocation_inverse_volatility",
    "allocation_risk_parity",
    "allocation_maximum_sharpe",
    "ml_long_short",
    "ml_long_only",
    "ml_thresholded",
    "momentum",
    "pairs",
)
BENCHMARKS = ("buy_and_hold", "equal_weight_hold", "equal_weight")
DECISION_REFERENCE = "equal_weight_hold"
DEVELOPMENT_REFERENCE = "equal_weight"
DSR_LEVEL = 0.95
INTERVAL_LEVEL = 0.95
BOOTSTRAP_BLOCK = 21
BOOTSTRAP_RESAMPLES = 1000
# The seed every development comparison used (compare_strategies, seed=0).
BOOTSTRAP_SEED = 0

EDGE = "evidence of an edge"
NO_EDGE = "no convincing evidence of an edge"

# Allocation books run portfolio.allocation_weights unchanged; the others run
# their strategy class from backtest.strategy unchanged.
ALLOCATION_BOOKS = {
    "allocation_minimum_variance": "minimum_variance",
    "allocation_inverse_volatility": "inverse_volatility",
    "allocation_risk_parity": "risk_parity",
    "allocation_maximum_sharpe": "maximum_sharpe",
    "equal_weight": "equal_weight",
}
ML_BOOKS = ("ml_long_short", "ml_long_only", "ml_thresholded")
RULE_BOOKS = ("momentum", "pairs", "buy_and_hold", "equal_weight_hold")
SUPPORTED_BOOKS = frozenset(ALLOCATION_BOOKS) | frozenset(ML_BOOKS) | frozenset(RULE_BOOKS)

# --- What may change after the freeze ---------------------------------------

# git diff FROZEN_CODE_VERSION..HEAD may touch only these (fnmatch patterns).
# Data, config and every module that computes a result are absent on purpose.
ALLOWED_PATHS = (
    "reports/results/holdout_freeze.json",
    "reports/results/holdout_declaration.json",
    "reports/results/holdout_declaration.r*.json",
    "reports/results/holdout_artifacts/*",
    "reports/results/holdout/*",
    # Its frozen prefix and holdout-only tail are checked row by row.
    "reports/trial_log.csv",
    "reports/methodology/holdout_protocol.md",
    "reports/methodology/trial_accounting.md",
    "src/pipelines/holdout.py",
    "entrypoint/holdout.py",
    "tests/test_holdout.py",
    "Makefile",
)
# A changed file allowed only if HEAD equals the frozen text with this edit.
TRIAL_LOG_MODULE = "src/evaluation/trial_log.py"
KINDS_BEFORE = 'KINDS = {"research", "backtest", "benchmark", "diagnostic", "notebook", "other"}\n'
KINDS_AFTER = (
    "# Holdout rows record the one-time 2020-2024 evaluation. They are never\n"
    "# selection candidates, so they cannot change N.\n"
    'KINDS = {"research", "backtest", "benchmark", "diagnostic", "notebook", "holdout", "other"}\n'
)
ALLOWED_EDITS = {TRIAL_LOG_MODULE: (KINDS_BEFORE, KINDS_AFTER)}

# --- The rules, as recorded in the declaration and the report ---------------

PRIMARY_RATIONALE = (
    "Chosen on development evidence only. It had the highest development net Sharpe "
    "(0.785, one walk-forward path, 2006-01-31 to 2019-12-31) among the 22 current "
    "candidates of the frozen strategies_and_allocations scenario. It is long only, "
    "rebalanced monthly and has low turnover, and it needs no trained model or pair "
    "selection, so its holdout run depends on no newly prepared artifact. The evidence "
    "is inconclusive: its frozen DSR is 0.207; its paired interval against monthly equal "
    "weight (0.724) included zero; and its lead over ml_long_short (0.759, an average "
    "over CPCV paths) compares a single path with a path average and is within noise."
)

WINDOW_RULES = {
    "calendar": "the trading dates of the cleaned panel (data/02-bars); no weekend or holiday rows",
    "decision_date": "D0 = the first trading date on or after validation.test_start "
    "(2020-01-01). Expected 2020-01-02; the run records the actual date.",
    "first_row": "D1 = D0 + execution_lag bars (expected 2020-01-03): every book trades "
    "from flat to its D0 target at the close of D1 and pays the entry cost. The gross "
    "return of D1 is zero for every book; its net return is minus the entry cost.",
    "first_return_earned": "the bar after D1 (expected 2020-01-06)",
    "last_date": "the last trading date on or before universe.end_date (2024-12-31) on "
    "which every ticker has a price; recorded by the run",
    "common_window": "every candidate and benchmark is evaluated on the same rows, D1 to "
    "the last date; no book starts later or ends earlier",
    "fair_start": "the window starts at D1 only if every book can be evaluated there: the "
    "development history before D0 must cover every warm-up (required_history) and every "
    "book's D0 target must be defined; otherwise the run fails rather than moving the "
    "window",
    "missing_prices": "as in development: the cleaned panel forward-fills at most three "
    "days; a return still missing is treated as zero, and the count of missing cells in "
    "the window is recorded",
}

EXECUTION_RULES = {
    "execution_lag": "one bar: a target set at the close of t is held from the close of "
    "t + 1 and earns the return of t + 2 (backtest.engine)",
    "starting_state": "every book is flat (all cash) until the close of D1",
    "december_2019_weights": "not carried over. The development books' 2019 positions "
    "belong to the development period. A monthly book's D0 target is its weights from "
    "the last scheduled rebalance on or before D0 (2019-12-31, estimated on the trailing "
    "window ending that day, development data only), entered at D1 from flat",
    "rebalance": "allocations and momentum: the last trading date of each month, traded "
    "one bar later, with allocations forced back to target on each rebalance (as in "
    "development); the other books trade whenever their target changes",
    "warm_up": "signals may use any price up to the decision date, so development prices "
    "serve as warm-up history; no holdout return is used to fit anything",
    "ml_signal": "forecasts from the prepared model exist only for dates from D0: the "
    "model was trained on every earlier date, so earlier forecasts would be in-sample. "
    "The thresholded rule's held state therefore starts flat at D0",
    "pairs_state": "the spread rule runs on the unbroken price history, as in development, "
    "so a trade open at the close of D0 (decided from past prices) is entered at D1",
}

PAIRS_PROCEDURE = (
    "The pairs book is one predeclared strategy. Its pairs are selected once, by "
    "backtest.pairs.find_pairs on every development date (before 2020-01-01, one "
    "contiguous segment), with the frozen screen: daily log-return correlation of at "
    "least 0.5, Engle-Granger cointegration at 5%, and a half-life above 0 and at most "
    "126 days. The selection table is a prepared artifact, hashed in the manifest. The "
    "book trades every selected pair with an equal share of the gross budget, a rolling "
    "252-day hedge ratio, a 63-day z score, entry at 2, exit at 0.5 and stop at 4. The "
    "individual historical pair trials are not candidates. If no pair is selected, the "
    "book is flat, reported as such, and its Sharpe is undefined."
)

ML_PROCEDURE = (
    "One model with the frozen settings (random forest regression, fixed-horizon labels, "
    "the frozen features and fracdiff order) is fitted once on every labelled development "
    "row, with the development sample weights, and saved as a prepared artifact. The three "
    "ML books share its forecasts, made on features built from prices up to each date."
)

DSR_NOTE = (
    "The full frozen trial count N = 98 deflates the single predeclared primary. This is "
    "a deliberately conservative choice: the selection happened on development data, and "
    "the rule charges the holdout result for it again."
)

PROBABILITY_NOTE = (
    "Neither the PSR nor the DSR is a probability of future profit. Each is the "
    "probability, under the method's assumptions, that the true Sharpe ratio over the "
    "holdout period exceeded a benchmark (zero, or the luck threshold for N trials)."
)

SURVIVORSHIP_NOTE = (
    "The universe is 11 tickers chosen with hindsight about which companies survived "
    "(survivorship bias, acknowledged in config). The result applies to this universe "
    "only and is not evidence about an investable universe."
)


class HoldoutError(Exception):
    """Raised when the holdout cannot be declared, prepared or run safely."""


class PreflightError(HoldoutError):
    """Raised when one or more preflight checks fail; nothing has been opened."""

    def __init__(self, failures: Sequence[str], record: Mapping[str, Any]) -> None:
        super().__init__("preflight failed:\n- " + "\n- ".join(failures))
        self.failures = list(failures)
        self.record = dict(record)


# --- Small helpers ----------------------------------------------------------


def sha256(data: bytes) -> str:
    """Hex SHA-256 of some bytes."""
    return hashlib.sha256(data).hexdigest()


def lf(data: bytes) -> bytes:
    """Text bytes with CRLF line endings turned into LF."""
    return data.replace(b"\r\n", b"\n")


def file_digest(path: Path) -> str:
    """SHA-256 of a file: raw for binary files, LF-normalised for text."""
    data = Path(path).read_bytes()

    return sha256(data if Path(path).suffix in BINARY_SUFFIXES else lf(data))


def _plain(value: Any) -> Any:
    """JSON-safe values: no NaN or infinity (written as text), no numpy scalars."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date()) if value == value.normalize() else str(value)

    return value


def json_text(record: Mapping[str, Any]) -> str:
    """Strict JSON (no Infinity, unlike the freeze), indented, newline-terminated."""
    return json.dumps(_plain(record), indent=2, allow_nan=False, default=str) + "\n"


def write_new(path: Path, text: str) -> str:
    """Create a file that must not exist yet, with LF endings, and return its hash.

    pandas writes os.linesep (CRLF on Windows), so the text is normalised
    first: the bytes written are the bytes hashed, on any machine.

    Raises:
        FileExistsError: If the file exists; nothing is overwritten.
    """
    text = text.replace("\r\n", "\n")
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)

    return sha256(text.encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    """A JSON file as a dict. Python's reader accepts the freeze's Infinity."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Git:
    """The few git queries the provenance checks need. Tests pass a fake."""

    def __init__(self, root: Path = PROJECT_ROOT) -> None:
        self.root = Path(root)

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(["git", *args], cwd=self.root, capture_output=True, check=check)

    def head(self) -> str:
        """The full commit hash of HEAD."""
        return self._run("rev-parse", "HEAD").stdout.decode().strip()

    def status(self) -> list[str]:
        """Changed and untracked files (ignored ones excluded), one line each."""
        return self._run("status", "--porcelain").stdout.decode().splitlines()

    def is_ancestor(self, commit: str) -> bool:
        """Whether commit is HEAD or one of its ancestors."""
        return self._run("merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode == 0

    def show(self, commit: str, path: str) -> bytes | None:
        """A file as committed, or None if it is not in that commit."""
        result = self._run("show", f"{commit}:{path}", check=False)

        return result.stdout if result.returncode == 0 else None

    def changed_files(self, base: str) -> list[str]:
        """Paths that differ between base and HEAD."""
        return self._run("diff", "--name-only", base, "HEAD").stdout.decode().splitlines()


@dataclass(frozen=True)
class HoldoutPaths:
    """Where everything lives. Tests point these at a temporary folder."""

    root: Path = PROJECT_ROOT
    freeze: Path = FREEZE_FILE
    declaration: Path = DECLARATION_FILE
    protocol: Path = PROTOCOL_FILE
    artifacts: Path = ARTIFACT_DIR
    results: Path = HOLDOUT_DIR
    trial_log: Path | None = None

    def log_path(self, config: Config) -> Path:
        """The trial log: the one given, or the config's."""
        return Path(self.trial_log or config.evaluation.trial_log)

    def relative(self, path: Path) -> str:
        """A path as git names it, relative to the root."""
        return Path(path).resolve().relative_to(Path(self.root).resolve()).as_posix()


# --- The freeze and the trial log -------------------------------------------


def frozen_dsr_inputs(freeze: Mapping[str, Any]) -> dict[str, Any]:
    """N and V[SR] for the holdout DSR, from the explicitly named frozen fields.

    Raises:
        HoldoutError: If the headline used a different variance scenario, or
            the frozen fields disagree with each other.
    """
    headline = freeze["headline"]
    scenario = freeze["variance_scenarios"][VARIANCE_SCENARIO]
    annual, period = float(scenario["variance_annual"]), float(scenario["variance_period"])

    if headline["variance_used"] != VARIANCE_SCENARIO:
        raise HoldoutError(f"the freeze's headline used {headline['variance_used']}")
    if not math.isclose(period, annual / PERIODS, rel_tol=1e-12):
        raise HoldoutError("the frozen per-period variance is not the annual one / 252")
    if not math.isclose(float(headline["variance_used_annual"]), annual, rel_tol=1e-12):
        raise HoldoutError("the headline variance differs from its named scenario")

    return {
        "n_trials": int(headline["n_trials"]),
        "n_trials_field": "headline.n_trials",
        "variance_scenario": VARIANCE_SCENARIO,
        "variance_field": f"variance_scenarios.{VARIANCE_SCENARIO}.variance_period",
        "variance_annual": annual,
        "variance_period": period,
        "periods_per_year": PERIODS,
    }


@dataclass(frozen=True)
class LogPrefix:
    """The frozen first rows of the trial log, hashed both ways, and what follows."""

    rows: int
    raw_sha256: str
    lf_sha256: str
    appended_kinds: tuple[str, ...]


def _prefix_bytes(data: bytes, rows: int) -> bytes:
    """The header and the first `rows` rows, checked by parsing them as CSV."""
    lines = data.splitlines(keepends=True)
    if len(lines) < rows + 1:
        raise HoldoutError(f"the trial log has {len(lines) - 1} rows, fewer than the {rows} frozen")

    prefix = b"".join(lines[: rows + 1])
    parsed = list(csv.reader(io.StringIO(prefix.decode("utf-8"), newline="")))
    ids = [row[0] for row in parsed[1:]]
    if len(parsed) != rows + 1 or ids != [str(i) for i in range(1, rows + 1)]:
        raise HoldoutError(f"the trial log's first {rows} lines are not trials 1 to {rows}")

    return prefix


def read_log_prefix(path: Path, rows: int) -> LogPrefix:
    """Hash the frozen prefix of the trial log, and list the kinds logged after it."""
    data = Path(path).read_bytes()
    prefix = _prefix_bytes(data, rows)
    frame = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding="utf-8")

    return LogPrefix(
        rows=rows,
        raw_sha256=sha256(prefix),
        lf_sha256=sha256(lf(prefix)),
        appended_kinds=tuple(frame["kind"].iloc[rows:]),
    )


def committed_prefix_hashes(blob: bytes, rows: int) -> dict[str, str]:
    """Both representations of the frozen prefix, from the committed (LF) blob."""
    prefix = lf(_prefix_bytes(blob, rows))

    return {"lf": sha256(prefix), "crlf": sha256(prefix.replace(b"\n", b"\r\n"))}


def check_log_prefix(prefix: LogPrefix, raw: str, lf_hash: str) -> str:
    """Which declared representation the prefix matches.

    Raises:
        HoldoutError: If it matches neither; no other hash is accepted.
    """
    if prefix.raw_sha256 == raw:
        return "raw bytes match the frozen SHA-256 (CRLF, as written when frozen)"
    if prefix.lf_sha256 == lf_hash:
        return "LF-normalised bytes match the committed copy's SHA-256"

    raise HoldoutError(
        f"the first {prefix.rows} trial-log rows changed: SHA-256 {prefix.raw_sha256} "
        f"(LF {prefix.lf_sha256}) matches neither {raw} nor {lf_hash}"
    )


# --- The declaration --------------------------------------------------------


def declaration_record(
    config: Config,
    *,
    paths: HoldoutPaths | None = None,
    git: Git | None = None,
    approved_on: str,
    approved_by: str,
) -> dict[str, Any]:
    """Everything the holdout is declared to be, checked against the frozen history.

    Raises:
        HoldoutError: If the freeze, the trial log or the config differ from
            what was committed in FREEZE_COMMIT.
    """
    paths = paths or HoldoutPaths()
    git = git or Git(paths.root)
    blob = git.show(FREEZE_COMMIT, paths.relative(paths.freeze))
    if blob is None:
        raise HoldoutError(f"{paths.freeze} is not in commit {FREEZE_COMMIT}")
    if not paths.freeze.is_file() or lf(paths.freeze.read_bytes()) != lf(blob):
        raise HoldoutError(f"{paths.freeze} differs from its copy in {FREEZE_COMMIT}")

    freeze = json.loads(blob)
    rows = int(freeze["trial_log_rows"])
    log_path = paths.log_path(config)
    log_blob = git.show(FREEZE_COMMIT, paths.relative(log_path))
    if log_blob is None:
        raise HoldoutError(f"{log_path} is not in commit {FREEZE_COMMIT}")
    committed = committed_prefix_hashes(log_blob, rows)
    if committed["crlf"] != freeze["trial_log_sha256"]:
        raise HoldoutError("the frozen trial-log SHA-256 does not reproduce from the commit")
    matched = check_log_prefix(
        read_log_prefix(log_path, rows), freeze["trial_log_sha256"], committed["lf"]
    )
    if config_hash(config) != freeze["config_hash"]:
        raise HoldoutError(f"config hash {config_hash(config)} is not {freeze['config_hash']}")
    if freeze["code_version"] != FROZEN_CODE_VERSION:
        raise HoldoutError(f"the freeze records {freeze['code_version']}")

    backtest = config.backtest

    return {
        "declaration_version": 1,
        "created_at": _now(),
        "created_with_code_version": code_version(),
        "approval": {
            "status": "approved",
            "approved_on": approved_on,
            "approved_by": approved_by,
            "scope": "the candidates, benchmarks, timing, costs and statistical rules below, "
            "and the protocol text pinned by its hash",
        },
        "protocol": {
            "path": paths.relative(paths.protocol),
            "sha256_lf": file_digest(paths.protocol),
        },
        "freeze": {
            "path": paths.relative(paths.freeze),
            "commit": FREEZE_COMMIT,
            "code_version": freeze["code_version"],
            "sha256_committed": sha256(blob),
            "frozen_at": freeze["frozen_at"],
            "holdout_start": freeze["holdout_start"],
        },
        "trial_log": {
            "path": paths.relative(log_path),
            "frozen_rows": rows,
            "prefix_sha256_raw": freeze["trial_log_sha256"],
            "prefix_sha256_lf": committed["lf"],
            "representations": "raw = CRLF, as the csv module wrote it on the freezing "
            "machine (this is the freeze's own trial_log_sha256); lf = as git stores it. "
            "Both were derived from the trial log committed in FREEZE_COMMIT",
            "verified_at_declaration": matched,
            "rows_after_prefix": "kind holdout only",
        },
        "config_hash": freeze["config_hash"],
        "candidates": {
            "primary": PRIMARY,
            "secondary": list(SECONDARIES),
            "benchmarks": list(BENCHMARKS),
            "excluded": {
                "ml_long_short_same_close": "a diagnostic of notebook 4's timing, not a candidate",
                "pair_<A>_<B>": "individual development pair trials; the pairs book is "
                "one strategy",
            },
        },
        "primary_rationale": PRIMARY_RATIONALE,
        "references": {
            "decision": DECISION_REFERENCE,
            "development": DEVELOPMENT_REFERENCE,
            "note": "The primary's development comparison was against monthly equal weight "
            "(equal_weight); the holdout decision compares it against equal-weight buy "
            "and hold (equal_weight_hold), as the protocol fixed before this declaration. "
            "The equal_weight comparison is reported, as descriptive only",
        },
        "window": WINDOW_RULES,
        "execution": {
            **EXECUTION_RULES,
            "costs": {
                "commission_bps": backtest.commission_bps,
                "slippage_bps": backtest.slippage_bps,
                "borrow_rate_annual": backtest.borrow_rate_annual,
            },
            "execution_lag_bars": backtest.execution_lag,
            "max_position_size": backtest.max_position_size,
            "max_gross_leverage": backtest.max_gross_leverage,
            "allow_short": backtest.allow_short,
            "allocations_long_only": config.portfolio.long_only,
            "cash_return": 0.0,
        },
        "pairs_procedure": PAIRS_PROCEDURE,
        "ml_procedure": ML_PROCEDURE,
        "statistics": {
            "primary_metric": "annualised net Sharpe ratio of the primary over the window",
            "dsr": {
                **frozen_dsr_inputs(freeze),
                "moments": "per-period Sharpe, n, skewness and raw kurtosis of the primary's "
                "daily net returns over the window",
                "note": DSR_NOTE,
            },
            "psr": "The PSR against zero of every book is reported, separately from the DSR. "
            + PROBABILITY_NOTE,
            "bootstrap": {
                "method": "paired block bootstrap of the net Sharpe difference "
                "(backtest.performance.sharpe_difference_interval), the same blocks of "
                "dates for both series",
                "block_days": BOOTSTRAP_BLOCK,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "level": INTERVAL_LEVEL,
            },
            "decision_rule": {
                "dsr_above": DSR_LEVEL,
                "interval_against": DECISION_REFERENCE,
                "interval_low_above": 0.0,
                "both_required": True,
                "pass": EDGE,
                "otherwise": NO_EDGE,
                "strict": "a DSR of exactly 0.95 or an interval bound of exactly 0 fails",
            },
            "secondary_and_benchmarks": "reported in full, descriptive only; they never "
            "replace the primary or change the rule",
        },
        "development_artifacts": {
            "status": "not prepared",
            "manifest": paths.relative(paths.artifacts / MANIFEST),
            "required": [f"{MODEL_DIR}/model.joblib", f"{MODEL_DIR}/metadata.json", PAIRS_FILE],
            "note": "Built by `python entrypoint/holdout.py prepare` only after separate "
            "authorisation. Their hashes are recorded in the manifest, which cites this "
            "declaration's SHA-256; this file is not rewritten when they are built",
        },
        "runner": {
            "code_version": None,
            "note": "recorded in each attempt's STARTED.json and COMPLETE.json, with all "
            "attempt metadata",
        },
        "replacement_policy": "This file is never overwritten. Before the holdout is opened, "
        "a change is made by `python entrypoint/holdout.py declare --replace REASON`, "
        "which writes holdout_declaration.r<k>.json citing the SHA-256 of the file it "
        "replaces. Once any attempt exists, no replacement is accepted.",
    }


def replacement_path(path: Path, number: int) -> Path:
    """The k-th replacement of a declaration: holdout_declaration.r<k>.json."""
    return path.with_name(f"{path.stem}.r{number}{path.suffix}")


def declaration_chain(path: Path) -> list[tuple[Path, dict[str, Any], str]]:
    """The original declaration and every replacement, in order, with their LF hashes.

    Raises:
        HoldoutError: If a replacement does not cite the file before it.
    """
    if not Path(path).is_file():
        return []

    chain = [(Path(path), read_json(path), file_digest(path))]
    number = 2
    while (nxt := replacement_path(Path(path), number)).is_file():
        record = read_json(nxt)
        cited = record.get("replaces", {}).get("sha256")
        if cited != chain[-1][2]:
            raise HoldoutError(f"{nxt.name} does not cite the SHA-256 of {chain[-1][0].name}")
        chain.append((nxt, record, file_digest(nxt)))
        number += 1

    return chain


def write_declaration(
    config: Config,
    *,
    paths: HoldoutPaths | None = None,
    git: Git | None = None,
    approved_on: str,
    approved_by: str,
    replace: str | None = None,
) -> Path:
    """Write the declaration once, or a documented replacement before opening.

    Raises:
        HoldoutError: For an existing declaration without --replace, a
            replacement after any attempt exists, or one with no reason.
        FileExistsError: If the target file appeared meanwhile.
    """
    paths = paths or HoldoutPaths()
    chain = declaration_chain(paths.declaration)
    record = declaration_record(
        config, paths=paths, git=git, approved_on=approved_on, approved_by=approved_by
    )

    if replace is None:
        if chain:
            raise HoldoutError(
                f"{paths.declaration} exists and is never rewritten; a change before opening "
                "needs --replace REASON"
            )
        target = paths.declaration
    else:
        if not replace.strip():
            raise HoldoutError("a replacement needs a stated reason")
        if not chain:
            raise HoldoutError("there is no declaration to replace")
        if attempt_states(paths.results):
            raise HoldoutError("the holdout has been opened: the declaration can no longer change")
        previous, _, digest = chain[-1]
        record["replaces"] = {"path": paths.relative(previous), "sha256": digest}
        record["replacement_reason"] = replace.strip()
        target = replacement_path(paths.declaration, len(chain) + 1)

    write_new(target, json_text(record))
    logger.info("Declaration written to %s", target)

    return target


def check_declaration(
    declaration: Mapping[str, Any],
    freeze: Mapping[str, Any],
    freeze_sha: str,
    protocol_sha: str,
) -> None:
    """Refuse a declaration that is unapproved, inconsistent, or cites another freeze.

    Raises:
        HoldoutError: Naming the first problem found.
    """
    if declaration["approval"]["status"] != "approved":
        raise HoldoutError("the declaration is not approved")
    cited = declaration["freeze"]
    if cited["commit"] != FREEZE_COMMIT or cited["code_version"] != FROZEN_CODE_VERSION:
        raise HoldoutError(
            f"the declaration cites freeze {cited['commit']}/{cited['code_version']}"
        )
    if cited["sha256_committed"] != freeze_sha:
        raise HoldoutError("the declaration cites a different freeze file")
    if declaration["protocol"]["sha256_lf"] != protocol_sha:
        raise HoldoutError("the protocol changed after it was approved in the declaration")
    if declaration["config_hash"] != freeze["config_hash"]:
        raise HoldoutError("the declaration's config hash is not the frozen one")

    candidates = declaration["candidates"]
    names = [candidates["primary"], *candidates["secondary"], *candidates["benchmarks"]]
    if len(set(names)) != len(names):
        raise HoldoutError(f"a book is declared twice: {names}")
    if declaration["references"]["decision"] not in candidates["benchmarks"]:
        raise HoldoutError("the decision reference is not a declared benchmark")

    dsr = declaration["statistics"]["dsr"]
    frozen = frozen_dsr_inputs(freeze)
    for key in ("n_trials", "variance_scenario", "variance_annual", "variance_period"):
        if dsr[key] != frozen[key]:
            raise HoldoutError(f"the declared DSR {key} {dsr[key]} is not the frozen {frozen[key]}")


def declared_books(declaration: Mapping[str, Any]) -> dict[str, str]:
    """Every declared book and its role, primary first."""
    candidates = declaration["candidates"]

    return {
        candidates["primary"]: "primary",
        **dict.fromkeys(candidates["secondary"], "secondary"),
        **dict.fromkeys(candidates["benchmarks"], "benchmark"),
    }


# --- Attempts ---------------------------------------------------------------


def attempt_states(results: Path) -> dict[int, str]:
    """Each attempt folder's state: complete, failed or stale (started, never finished).

    Raises:
        HoldoutError: If the results folder holds anything but attempt folders.
    """
    results = Path(results)
    if not results.exists():
        return {}

    states = {}
    for child in sorted(results.iterdir()):
        number = child.name.removeprefix(ATTEMPT_PREFIX)
        if not (child.is_dir() and child.name.startswith(ATTEMPT_PREFIX) and number.isdigit()):
            raise HoldoutError(f"unexpected {child.name} in {results}: only attempt folders belong")
        if (child / COMPLETE).is_file():
            states[int(number)] = "complete"
        elif (child / FAILED).is_file():
            states[int(number)] = "failed"
        else:
            states[int(number)] = "stale"

    return states


def plan_attempt(results: Path, new_attempt: str | None) -> dict[str, Any]:
    """The attempt to run, under the recovery policy.

    The first attempt needs no reason. Any later one, after a failed, stale
    (interrupted) or complete attempt, needs an explicit reason, and every
    earlier folder is kept and reported.

    Raises:
        HoldoutError: For a later attempt without a reason, or a reason with
            no earlier attempt.
    """
    states = attempt_states(results)

    if not states:
        if new_attempt is not None:
            raise HoldoutError("no earlier attempt exists: run without --new-attempt")
        return {"number": 1, "reason": "", "previous": {}}

    if not (new_attempt or "").strip():
        listed = ", ".join(f"{k} ({v})" for k, v in states.items())
        raise HoldoutError(
            f"attempt(s) {listed} exist: the holdout has been opened. A further attempt "
            "needs --new-attempt REASON; earlier attempts are kept and reported"
        )

    return {
        "number": max(states) + 1,
        "reason": new_attempt.strip(),
        "previous": {str(k): v for k, v in states.items()},
    }


# --- Preflight --------------------------------------------------------------


@dataclass
class Preflight:
    """What preflight checked and loaded, for the run to record."""

    record: dict[str, Any]
    freeze: dict[str, Any]
    declaration: dict[str, Any]
    declaration_sha256: str
    attempt: dict[str, Any] | None


def _allowed(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in ALLOWED_PATHS)


def preflight(
    config: Config,
    *,
    paths: HoldoutPaths | None = None,
    git: Git | None = None,
    stage: str = "run",
    new_attempt: str | None = None,
) -> Preflight:
    """Every check that must pass before the holdout loader may be called.

    Reads git, the freeze, the declaration, the protocol, the trial log, the
    config and the prepared artifacts. Never reads a price.

    Args:
        config: Loaded project configuration.
        paths: Where the files are.
        git: Git queries; a fake in tests.
        stage: "run" (everything) or "prepare" (no artifact or attempt checks,
            and the artifacts must not exist yet).
        new_attempt: Reason for a further attempt, as for plan_attempt().

    Returns:
        The checks and what they loaded.

    Raises:
        PreflightError: Listing every failed check.
    """
    if stage not in ("run", "prepare"):
        raise HoldoutError(f"unknown preflight stage {stage}")
    paths = paths or HoldoutPaths()
    git = git or Git(paths.root)
    checks: dict[str, str] = {}
    failures: list[str] = []
    state: dict[str, Any] = {}

    def check(name: str, test: Callable[[], str]) -> None:
        try:
            checks[name] = test()
        except (HoldoutError, OSError, KeyError, ValueError, TypeError) as exc:
            checks[name] = f"FAILED: {exc}"
            failures.append(f"{name}: {exc}")

    def clean_tree() -> str:
        if changes := git.status():
            raise HoldoutError(f"uncommitted or untracked files: {changes[:5]}")
        return "clean, untracked files included"

    def ancestry() -> str:
        state["head"] = git.head()
        if not git.is_ancestor(FROZEN_CODE_VERSION):
            raise HoldoutError(f"{FROZEN_CODE_VERSION} is not an ancestor of HEAD")
        return f"{FROZEN_CODE_VERSION} is an ancestor of HEAD {state['head'][:7]}"

    def freeze_file() -> str:
        blob = git.show(FREEZE_COMMIT, paths.relative(paths.freeze))
        if blob is None:
            raise HoldoutError(f"no freeze in commit {FREEZE_COMMIT}")
        if not paths.freeze.is_file():
            raise HoldoutError(f"{paths.freeze} is missing")
        if lf(paths.freeze.read_bytes()) != lf(blob):
            raise HoldoutError(f"{paths.freeze} differs from its copy in {FREEZE_COMMIT}")
        freeze = json.loads(blob)
        if freeze["code_version"] != FROZEN_CODE_VERSION:
            raise HoldoutError(f"the freeze records code {freeze['code_version']}")
        if freeze["holdout_start"] != str(config.validation.test_start):
            raise HoldoutError(f"the freeze's holdout starts {freeze['holdout_start']}")
        frozen_dsr_inputs(freeze)
        state["freeze"], state["freeze_sha"] = freeze, sha256(blob)
        return f"identical to {FREEZE_COMMIT} (LF-normalised), SHA-256 {sha256(blob)}"

    def config_file() -> str:
        if config_hash(config) != state["freeze"]["config_hash"]:
            raise HoldoutError(f"config hash {config_hash(config)} is not the frozen one")
        return f"config hash {config_hash(config)} as frozen"

    def declaration() -> str:
        chain = declaration_chain(paths.declaration)
        if not chain:
            raise HoldoutError(f"{paths.declaration} is missing")
        path, record, digest = chain[-1]
        check_declaration(record, state["freeze"], state["freeze_sha"], file_digest(paths.protocol))
        state["declaration"], state["declaration_sha"] = record, digest
        state["chain"] = [d for _, _, d in chain]
        return f"{path.name} (SHA-256 {digest}), {len(chain) - 1} replacement(s)"

    def trial_log() -> str:
        record = state["declaration"]["trial_log"]
        rows = int(state["freeze"]["trial_log_rows"])
        if (
            record["frozen_rows"] != rows
            or record["prefix_sha256_raw"] != (state["freeze"]["trial_log_sha256"])
        ):
            raise HoldoutError("the declaration's trial-log record is not the frozen one")
        prefix = read_log_prefix(paths.log_path(config), rows)
        matched = check_log_prefix(prefix, record["prefix_sha256_raw"], record["prefix_sha256_lf"])
        if other := sorted(set(prefix.appended_kinds) - {"holdout"}):
            raise HoldoutError(f"rows of kind {other} were logged after the freeze")
        state["appended"] = len(prefix.appended_kinds)
        return f"{rows} frozen rows: {matched}; {len(prefix.appended_kinds)} holdout rows after"

    def books() -> str:
        names = declared_books(state["declaration"])
        if unknown := sorted(set(names) - SUPPORTED_BOOKS):
            raise HoldoutError(f"the runner does not support {unknown}")
        return f"{len(names)} books, all supported"

    def frozen_paths() -> str:
        changed = git.changed_files(FROZEN_CODE_VERSION)
        refused = []
        for path in changed:
            if _allowed(path):
                continue
            if path in ALLOWED_EDITS:
                before, after = ALLOWED_EDITS[path]
                base = git.show(FROZEN_CODE_VERSION, path)
                now = (paths.root / path).read_bytes() if (paths.root / path).is_file() else None
                if base is not None and now is not None:
                    text = lf(base).decode("utf-8")
                    if before in text and lf(now).decode("utf-8") == text.replace(before, after):
                        continue
            refused.append(path)
        if refused:
            raise HoldoutError(f"changed since {FROZEN_CODE_VERSION} and not allowed: {refused}")
        return f"{len(changed)} changed path(s), all allowlisted"

    def artifacts() -> str:
        manifest_path = paths.artifacts / MANIFEST
        if not manifest_path.is_file():
            raise HoldoutError(
                f"{manifest_path} is missing: the development-only artifacts have not been "
                "prepared (a separately authorised step)"
            )
        manifest = read_json(manifest_path)
        if manifest["declaration_sha256"] not in state["chain"]:
            raise HoldoutError("the artifacts were prepared under a different declaration")
        for name, digest in manifest["files"].items():
            path = paths.artifacts / name
            if not path.is_file():
                raise HoldoutError(f"artifact {name} is missing")
            if file_digest(path) != digest:
                raise HoldoutError(f"artifact {name} does not match its manifest hash")
        if pd.Timestamp(manifest["training_end"]) >= pd.Timestamp(config.validation.test_start):
            raise HoldoutError("the model was trained on a date in the holdout")
        state["manifest_sha"] = file_digest(manifest_path)
        return f"{len(manifest['files'])} files match the manifest ({state['manifest_sha']})"

    def no_artifacts_yet() -> str:
        if paths.artifacts.exists():
            raise HoldoutError(f"{paths.artifacts} exists: artifacts are prepared once")
        if attempt_states(paths.results):
            raise HoldoutError("the holdout has been opened: too late to prepare")
        return "no artifacts yet"

    def attempts() -> str:
        plan = plan_attempt(paths.results, new_attempt)
        if plan["number"] == 1 and state.get("appended"):
            raise HoldoutError("holdout rows are in the trial log but no attempt folder exists")
        state["attempt"] = plan
        return f"attempt {plan['number']}" + (f" ({plan['reason']})" if plan["reason"] else "")

    check("1_clean_tree", clean_tree)
    check("2_ancestry", ancestry)
    check("3_freeze", freeze_file)
    if "freeze" in state:
        check("4_config", config_file)
        check("7_declaration", declaration)
    if "declaration" in state:
        check("5_6_trial_log_prefix", trial_log)
        check("8_books", books)
    if stage == "run":
        if "chain" in state:
            check("9_artifacts", artifacts)
        check("10_attempt", attempts)
    else:
        check("9_no_artifacts_yet", no_artifacts_yet)
    check("11_frozen_paths", frozen_paths)

    record = {
        "stage": stage,
        "checked_at": _now(),
        "head": state.get("head"),
        "frozen_code_version": FROZEN_CODE_VERSION,
        "freeze_commit": FREEZE_COMMIT,
        "freeze_sha256": state.get("freeze_sha"),
        "declaration_sha256": state.get("declaration_sha"),
        "manifest_sha256": state.get("manifest_sha"),
        "protocol_sha256_lf": file_digest(paths.protocol) if paths.protocol.is_file() else None,
        "config_hash": config_hash(config),
        "checks": checks,
    }
    if failures:
        raise PreflightError(failures, record)

    return Preflight(
        record=record,
        freeze=state["freeze"],
        declaration=state["declaration"],
        declaration_sha256=state["declaration_sha"],
        attempt=state.get("attempt"),
    )


# --- Development-only artifacts ---------------------------------------------


@dataclass
class DevelopmentArtifacts:
    """The final model and the pairs selection, both from development data only."""

    model: Forecaster
    training_index: pd.Index
    fracdiff_d: float
    pairs: pd.DataFrame


def build_development_artifacts(panel: pd.DataFrame, config: Config) -> DevelopmentArtifacts:
    """Fit the final model and select the pairs, on development rows only.

    The panel is cut at validation.test_start first (development_panel), so
    nothing after the boundary can change either artifact.
    """
    panel = development_panel(panel, config)
    data = prepare_research_data(panel, config)
    model = build_estimator(config)
    fit_with_weights(model, data.X, target_for(data, config.model.task), data.weights)

    prices = panel[config.data.price_column].unstack("ticker").sort_index()
    pairs = find_pairs(prices, config)
    log_frame(logger, data.X, "Final model training rows")

    return DevelopmentArtifacts(
        model=model, training_index=data.X.index, fracdiff_d=data.fracdiff_d, pairs=pairs
    )


def prepare_artifacts(
    config: Config,
    *,
    authorised_by: str,
    paths: HoldoutPaths | None = None,
    git: Git | None = None,
    loader: Callable[[Config], pd.DataFrame] | None = None,
) -> Path:
    """Build and save the development-only artifacts once, with a manifest of hashes.

    Needs separate authorisation: it fits the final model. It never reads a
    holdout price into anything: the panel is cut before features, labels,
    the fit or the pairs screen see it.

    Raises:
        HoldoutError: Without an authoriser, or if preflight fails.
    """
    if not authorised_by.strip():
        raise HoldoutError("preparing the artifacts needs the name of whoever authorised it")
    paths = paths or HoldoutPaths()
    check = preflight(config, paths=paths, git=git, stage="prepare")

    built = build_development_artifacts((loader or load_clean_panel)(config), config)
    dates = built.training_index.get_level_values("date")
    paths.artifacts.mkdir(parents=True, exist_ok=False)
    save_model(built.model, config, {}, paths.artifacts / MODEL_DIR, built.training_index)
    write_new(paths.artifacts / PAIRS_FILE, built.pairs.to_csv(index=False))

    files = [f"{MODEL_DIR}/model.joblib", f"{MODEL_DIR}/metadata.json", PAIRS_FILE]
    manifest = {
        "prepared_at": _now(),
        "authorised_by": authorised_by.strip(),
        "code_version": code_version(),
        "declaration_sha256": check.declaration_sha256,
        "config_hash": config_hash(config),
        "training_start": dates.min(),
        "training_end": dates.max(),
        "training_rows": len(built.training_index),
        "fracdiff_d": built.fracdiff_d,
        "pairs_selected": [
            f"{a}_{b}" for a, b in built.pairs.loc[built.pairs["selected"], ["a", "b"]].to_numpy()
        ],
        "boundary": f"panel cut before {config.validation.test_start} before anything was built",
        "files": {name: file_digest(paths.artifacts / name) for name in files},
    }
    write_new(paths.artifacts / MANIFEST, json_text(manifest))
    logger.info("Development-only artifacts prepared in %s", paths.artifacts)

    return paths.artifacts


@dataclass
class LoadedArtifacts:
    """The prepared artifacts, as the run uses them."""

    model: Forecaster
    fracdiff_d: float
    pairs: list[tuple[str, str]]
    selection: pd.DataFrame
    manifest: dict[str, Any]


def load_artifacts(paths: HoldoutPaths, config: Config) -> LoadedArtifacts:
    """The model bundle and pairs selection named by the manifest."""
    manifest = read_json(paths.artifacts / MANIFEST)
    bundle = load_model(paths.artifacts / MODEL_DIR, config)
    selection = pd.read_csv(paths.artifacts / PAIRS_FILE)
    chosen = selection.loc[selection["selected"].astype(str).str.lower() == "true"]

    return LoadedArtifacts(
        model=bundle.model,
        fracdiff_d=float(manifest["fracdiff_d"]),
        pairs=list(zip(chosen["a"], chosen["b"], strict=True)),
        selection=selection,
        manifest=manifest,
    )


# --- Evaluation -------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """The evaluation window every book shares."""

    decision: pd.Timestamp
    first_trade: pd.Timestamp
    first_return: pd.Timestamp
    last: pd.Timestamp
    lag: int

    def as_dict(self) -> dict[str, str]:
        return {
            "decision_date": str(self.decision.date()),
            "first_row_and_trade": str(self.first_trade.date()),
            "first_return_earned": str(self.first_return.date()),
            "last_date": str(self.last.date()),
            "execution_lag": str(self.lag),
        }


def evaluation_window(prices: pd.DataFrame, config: Config) -> Window:
    """D0, D1, the first earned return and the last date, from the calendar.

    Raises:
        HoldoutError: If the lag is below one bar or the data ends too soon.
    """
    lag = config.backtest.execution_lag
    if lag < 1:
        raise HoldoutError("the holdout needs an execution lag of at least one bar")

    calendar = pd.DatetimeIndex(prices.index)
    after = calendar[calendar >= pd.Timestamp(config.validation.test_start)]
    complete = prices.notna().all(axis=1).to_numpy() & (
        calendar <= pd.Timestamp(config.universe.end_date)
    )
    if after.empty or not complete.any():
        raise HoldoutError("no holdout dates in the data")

    position = calendar.get_loc(after[0])
    last = calendar[complete][-1]
    if position + lag + 1 >= len(calendar) or calendar[position + lag + 1] > last:
        raise HoldoutError("the data ends before the first holdout return")

    return Window(
        decision=after[0],
        first_trade=calendar[position + lag],
        first_return=calendar[position + lag + 1],
        last=last,
        lag=lag,
    )


def required_history(config: Config, d: float | None) -> dict[str, int]:
    """Bars of history before D0 each kind of book needs for its D0 target."""
    portfolio, momentum, pairs = (
        config.portfolio,
        config.strategies.momentum,
        config.strategies.pairs,
    )

    return {
        # Allocation returns drop the first row, as development_returns does.
        "allocations": max(portfolio.covariance_window, portfolio.volatility_window) + 1,
        "momentum": momentum.lookback_months * DAYS_PER_MONTH + 1,
        "pairs": pairs.hedge_window + pairs.zscore_window,
        "ml_features": warmup_rows(config, d) + 1,
    }


@dataclass
class Book:
    """One declared book's pre-delay targets on the calendar, and how it trades."""

    name: str
    role: str
    targets: pd.DataFrame
    rebalance: pd.Series | None = None
    limits: bool = True
    ready: str = ""


@dataclass
class BookResult:
    """One book over the evaluation window."""

    name: str
    role: str
    net: pd.Series
    gross: pd.Series
    turnover: pd.Series
    costs: pd.DataFrame
    weights: pd.DataFrame


def run_book(book: Book, returns: pd.DataFrame, window: Window, config: Config) -> BookResult:
    """Flat until D1, then the book's own targets, one bar late, with costs.

    The backtest starts at D0 with the position forced flat (the delay leaves
    nothing before it), so the trade into the D0 target happens at D1 and is
    charged there. The result is cut to the window, D1 to the last date.
    """
    span = returns.loc[window.decision : window.last]
    targets = book.targets.reindex(index=span.index, columns=span.columns).fillna(0.0)
    positions = delay(targets, window.lag)
    if book.limits:
        positions = enforce_limits(positions, config)
    trade_on = (
        None
        if book.rebalance is None
        else book.rebalance.reindex(span.index, fill_value=False).shift(
            window.lag, fill_value=False
        )
    )
    result: BacktestResult = run_backtest(
        positions, span, config, name=f"holdout/{book.name}", rebalance=trade_on
    )
    cut = slice(window.first_trade, window.last)

    return BookResult(
        name=book.name,
        role=book.role,
        net=result.net.loc[cut],
        gross=result.gross.loc[cut],
        turnover=result.turnover.loc[cut],
        costs=result.costs.loc[cut],
        weights=result.weights.loc[cut],
    )


def build_books(
    roles: Mapping[str, str],
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    window: Window,
    config: Config,
    artifacts: LoadedArtifacts | None,
) -> dict[str, Book]:
    """Each declared book's targets, from its unchanged development code.

    Raises:
        HoldoutError: If a book's D0 target is undefined (too little history,
            or a missing forecast), rather than starting it later.
    """
    d0 = window.decision
    books: dict[str, Book] = {}
    names = list(roles)

    if any(name in ALLOCATION_BOOKS for name in names):
        weights = allocation_weights(returns.iloc[1:], config)
        for name in (n for n in names if n in ALLOCATION_BOOKS):
            frame = weights[ALLOCATION_BOOKS[name]]
            set_on = frame.notna().any(axis=1)
            before = set_on.loc[:d0]
            if not before.any():
                raise HoldoutError(f"{name} has no rebalance on or before {d0.date()}")
            books[name] = Book(
                name,
                roles[name],
                rebalance(frame, set_on).reindex(prices.index).fillna(0.0),
                rebalance=set_on.reindex(prices.index, fill_value=False),
                limits=False,
                ready=f"weights from the {before[before].index[-1].date()} rebalance",
            )

    if any(name in ML_BOOKS for name in names):
        if artifacts is None:
            raise HoldoutError("the ML books need the prepared model")
        features = build_features(panel, config, artifacts.fracdiff_d)
        dates = features.index.get_level_values("date")
        X = features[dates >= d0]
        signal = (
            pd.Series(forecast(artifacts.model, X, config.model.task), index=X.index)
            .unstack("ticker")
            .reindex(index=prices.loc[d0:].index, columns=prices.columns)
        )
        if signal.loc[d0].isna().any():
            raise HoldoutError(f"no forecast for every ticker on {d0.date()}")
        for name in (n for n in names if n in ML_BOOKS):
            targets = STRATEGIES[name]().targets(signal, prices.loc[signal.index], config)
            books[name] = Book(name, roles[name], targets, ready="forecasts from D0")

    for name in (n for n in names if n in RULE_BOOKS):
        strategy = STRATEGIES[name]()
        ready = "constant target"
        if isinstance(strategy, MeanReversionPairs):
            strategy.selection_ = artifacts.selection if artifacts else None
            strategy.pairs_ = artifacts.pairs if artifacts else []
            if strategy.selection_ is None:
                raise HoldoutError("the pairs book needs the prepared selection")
            settings = config.strategies.pairs
            for a, b in strategy.pairs_:
                hedge = hedge_ratio(prices[a], prices[b], settings.hedge_window)
                z = spread_zscore(
                    prices[a],
                    prices[b],
                    hedge,
                    settings.zscore_window,
                    hedge_window=settings.hedge_window,
                )
                if np.isnan(z.loc[d0]):
                    raise HoldoutError(f"pair {a}_{b} has no z score on {d0.date()}")
            ready = f"{len(strategy.pairs_)} pair(s)" if strategy.pairs_ else "no pair: flat"
        elif name == "momentum":
            settings = config.strategies.momentum
            signal = momentum_signal(prices, settings.lookback_months, settings.skip_months)
            if signal.loc[d0].isna().any():
                raise HoldoutError(f"momentum has no signal for every ticker on {d0.date()}")
            ready = "12-1 signal defined at D0"
        books[name] = Book(name, roles[name], strategy.targets(None, prices, config), ready=ready)

    return {name: books[name] for name in names}


def book_statistics(book: BookResult) -> dict[str, Any]:
    """Return, risk, cost and uncertainty of one book over the window."""
    exposures = exposure(book.weights)
    try:
        sharpe, n, skew, kurtosis = return_moments(book.net.to_numpy())
        psr = probabilistic_sharpe_ratio(sharpe, n, skew, kurtosis, 0.0)
    except SharpeError:
        psr = float("nan")

    return {
        "role": book.role,
        "days": len(book.net),
        **{f"net_{k}": v for k, v in performance_stats(book.net).items()},
        **{f"gross_{k}": v for k, v in performance_stats(book.gross).items()},
        "sharpe_standard_error": sharpe_standard_error(book.net),
        "psr_vs_zero": psr,
        "turnover_annual": float(book.turnover.mean() * PERIODS),
        "cost_annual": float(book.costs["total"].mean() * PERIODS),
        "exposure_net_mean": float(exposures["net"].mean()),
        "exposure_gross_mean": float(exposures["gross"].mean()),
    }


def compare_books(
    results: Mapping[str, BookResult],
    benchmarks: Sequence[str],
    primary: str,
    bootstrap: Mapping[str, Any],
) -> pd.DataFrame:
    """Paired bootstrap of each book's net Sharpe minus each benchmark's.

    Only primary versus DECISION_REFERENCE enters the decision; every other
    row is descriptive.
    """
    rows = []

    for name, book in results.items():
        for benchmark in benchmarks:
            if benchmark == name:
                continue
            interval = sharpe_difference_interval(
                book.net,
                results[benchmark].net,
                n_boot=int(bootstrap["resamples"]),
                block=int(bootstrap["block_days"]),
                level=float(bootstrap["level"]),
                seed=int(bootstrap["seed"]),
            )
            if name == primary and benchmark == DECISION_REFERENCE:
                role = "decision"
            elif name == primary and benchmark == DEVELOPMENT_REFERENCE:
                role = "development reference, descriptive"
            else:
                role = "descriptive"
            rows.append({"book": name, "benchmark": benchmark, "role": role, **interval})

    return pd.DataFrame(rows)


def decide(dsr: float, interval_low: float, level: float = DSR_LEVEL) -> dict[str, Any]:
    """The predeclared rule: both the DSR and the paired interval must pass, strictly."""
    dsr_passes = bool(np.isfinite(dsr) and dsr > level)
    interval_passes = bool(np.isfinite(interval_low) and interval_low > 0)
    edge = dsr_passes and interval_passes

    return {
        "dsr_above_level": dsr_passes,
        "interval_entirely_above_zero": interval_passes,
        "evidence_of_edge": edge,
        "conclusion": EDGE if edge else NO_EDGE,
    }


def primary_result(
    net: pd.Series,
    frozen: Mapping[str, Any],
    decision_interval: Mapping[str, float],
    level: float = DSR_LEVEL,
) -> dict[str, Any]:
    """The primary's PSR, DSR from the frozen N and V[SR], and the decision."""
    try:
        sharpe, n, skew, kurtosis = return_moments(net.to_numpy())
        moments = {"sharpe_period": sharpe, "n": n, "skew": skew, "kurtosis": kurtosis}
        deflated = deflate(moments, int(frozen["n_trials"]), float(frozen["variance_period"]))
    except SharpeError as exc:
        moments, deflated = {"error": str(exc)}, {"dsr": float("nan"), "psr_vs_zero": float("nan")}

    return {
        "moments": moments,
        **deflated,
        "n_trials": int(frozen["n_trials"]),
        "variance_field": frozen["variance_field"],
        "variance_period": float(frozen["variance_period"]),
        "variance_annual": float(frozen["variance_annual"]),
        "interval": dict(decision_interval),
        **decide(deflated["dsr"], decision_interval["low"], level),
    }


@dataclass
class HoldoutOutcome:
    """Everything one run computed."""

    window: Window
    books: dict[str, BookResult]
    summary: pd.DataFrame
    comparisons: pd.DataFrame
    primary: dict[str, Any]
    data: dict[str, Any] = field(default_factory=dict)


def evaluate_holdout(
    panel: pd.DataFrame,
    config: Config,
    declaration: Mapping[str, Any],
    frozen: Mapping[str, Any],
    artifacts: LoadedArtifacts | None,
) -> HoldoutOutcome:
    """Run every declared book over the common window and apply the decision rule."""
    panel = panel[panel.index.get_level_values("date") <= pd.Timestamp(config.universe.end_date)]
    prices = panel[config.data.price_column].unstack("ticker").sort_index()
    window = evaluation_window(prices, config)
    prices = prices.loc[: window.last]
    panel = panel[panel.index.get_level_values("date") <= window.last]
    raw = panel["return"].unstack("ticker").reindex(index=prices.index, columns=prices.columns)
    log_frame(logger, prices, "Holdout run prices, development warm-up included")

    history = prices.index.get_loc(window.decision)
    needed = required_history(config, artifacts.fracdiff_d if artifacts else None)
    if short := {k: v for k, v in needed.items() if history < v}:
        raise HoldoutError(f"{history} bars before D0 is too little history for {short}")

    roles = declared_books(declaration)
    returns = raw.fillna(0.0)
    books = build_books(roles, panel, prices, returns, window, config, artifacts)
    results = {name: run_book(book, returns, window, config) for name, book in books.items()}

    summary = pd.DataFrame({name: book_statistics(r) for name, r in results.items()}).T
    summary.index.name = "book"
    primary = declaration["candidates"]["primary"]
    comparisons = compare_books(
        results,
        declaration["candidates"]["benchmarks"],
        primary,
        declaration["statistics"]["bootstrap"],
    )
    decision_row = comparisons[comparisons["role"] == "decision"].iloc[0]
    primary_outcome = primary_result(
        results[primary].net,
        frozen,
        decision_row[["difference", "low", "high", "share_positive"]].to_dict(),
        float(declaration["statistics"]["decision_rule"]["dsr_above"]),
    )

    return HoldoutOutcome(
        window=window,
        books=results,
        summary=summary,
        comparisons=comparisons,
        primary=primary_outcome,
        data={
            "panel_rows": len(panel),
            "panel_sha256": sha256(
                pd.util.hash_pandas_object(panel, index=True).to_numpy().tobytes()
            ),
            "history_bars_before_decision": history,
            "required_history": needed,
            "missing_return_cells_in_window": int(raw.loc[window.first_trade :].isna().sum().sum()),
            "readiness": {name: book.ready for name, book in books.items()},
        },
    )


# --- Report -----------------------------------------------------------------


def holdout_report(
    outcome: HoldoutOutcome,
    started: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> str:
    """The attempt's report, built only from the outcome and the declaration."""
    primary = declaration["candidates"]["primary"]
    p = outcome.primary
    window = outcome.window.as_dict()
    attempt = started["attempt"]
    headline = pd.DataFrame(
        {
            "value": {
                "net Sharpe (annual)": p.get("sharpe_annual", float("nan")),
                "PSR against 0": p.get("psr_vs_zero", float("nan")),
                "luck threshold (annual)": p.get("luck_threshold_annual", float("nan")),
                "DSR": p["dsr"],
                f"Sharpe difference vs {DECISION_REFERENCE}": p["interval"]["difference"],
                "95% interval low": p["interval"]["low"],
                "95% interval high": p["interval"]["high"],
            }
        }
    )
    headline.index.name = "primary"
    shown = [
        "role",
        "days",
        "net_sharpe",
        "gross_sharpe",
        "net_annual_return",
        "net_annual_volatility",
        "net_max_drawdown",
        "net_sortino",
        "sharpe_standard_error",
        "psr_vs_zero",
        "turnover_annual",
        "cost_annual",
        "exposure_net_mean",
    ]
    comparisons = outcome.comparisons.set_index("book")
    earlier = attempt["previous"] or {}

    sections = [
        f"# Holdout evaluation, attempt {attempt['number']}",
        f"Runner commit {started['runner_commit']}, frozen code {FROZEN_CODE_VERSION}, freeze "
        f"commit {FREEZE_COMMIT}, declaration SHA-256 {started['declaration_sha256']}, config "
        f"{started['preflight']['config_hash']}.",
        f"Window: decision date {window['decision_date']}; every book flat until "
        f"{window['first_row_and_trade']}, when it trades to its target; first return earned "
        f"{window['first_return_earned']}; last date {window['last_date']}. "
        f"Missing return cells set to zero: {outcome.data['missing_return_cells_in_window']}.",
        "## Primary decision (predeclared)",
        f"Primary: `{primary}`. Rule: DSR above {DSR_LEVEL} using the frozen N = "
        f"{p['n_trials']} and V[SR] = {p['variance_annual']:.4f} a year, "
        f"{p['variance_period']:.6f} a day (`{p['variance_field']}`), and the paired 95% "
        "interval of its net Sharpe minus "
        f"`{DECISION_REFERENCE}`'s entirely above zero.",
        _markdown(headline),
        f"**Conclusion: {p['conclusion']}.**",
        DSR_NOTE,
        PROBABILITY_NOTE,
        "## Every declared book (descriptive)",
        "Secondary candidates and benchmarks are reported in full. They do not replace the "
        "primary or change the rule.",
        _markdown(outcome.summary[shown]),
        "## Paired comparisons with each benchmark",
        f"Only `{primary}` against `{DECISION_REFERENCE}` enters the decision. Its comparison "
        f"with `{DEVELOPMENT_REFERENCE}` (the development reference) is descriptive.",
        _markdown(comparisons),
        "## Limitations",
        f"- **Survivorship.** {SURVIVORSHIP_NOTE}",
        "- **One period, one path.** Five years, one realisation; regimes in it may not recur.",
        "- **Costs are a model.** 2 + 3 bps per side and 0.5% a year to borrow; real fills "
        "and borrow availability can differ.",
        "- **Readiness.** " + "; ".join(f"{k}: {v}" for k, v in outcome.data["readiness"].items()),
        "- **Earlier attempts.** "
        + (
            ", ".join(f"attempt {k} {v}" for k, v in earlier.items())
            + f". This attempt's reason: {attempt['reason']}. Every attempt is reported; "
            "the first complete attempt remains the primary record."
            if earlier
            else "None."
        ),
    ]

    return "\n\n".join(sections) + "\n"


# --- The run ----------------------------------------------------------------


def load_holdout_panel(config: Config) -> pd.DataFrame:
    """The full cleaned panel, holdout included. Called only after STARTED.json exists."""
    return load_clean_panel(config)


def _holdout_metrics(stats: Mapping[str, Any], primary: Mapping[str, Any] | None) -> dict:
    """Trial-log metrics, prefixed holdout_ so they never mix with development ones."""
    metrics = {
        "holdout_sharpe_net": stats["net_sharpe"],
        "holdout_sharpe_gross": stats["gross_sharpe"],
        "holdout_annual_return_net": stats["net_annual_return"],
        "holdout_max_drawdown_net": stats["net_max_drawdown"],
        "holdout_psr_vs_zero": stats["psr_vs_zero"],
    }
    if primary is not None:
        metrics |= {
            "holdout_dsr": primary["dsr"],
            "holdout_evidence_of_edge": float(primary["evidence_of_edge"]),
        }

    return {k: (float(v) if np.isfinite(float(v)) else None) for k, v in metrics.items()}


def run_holdout(
    config: Config,
    *,
    paths: HoldoutPaths | None = None,
    git: Git | None = None,
    loader: Callable[[Config], pd.DataFrame] | None = None,
    new_attempt: str | None = None,
) -> Path:
    """Open the holdout once: preflight, STARTED.json, load, evaluate, record.

    Args:
        config: Loaded project configuration.
        paths: Where the files are.
        git: Git queries; a fake in tests.
        loader: Returns the cleaned panel, holdout included. Called only after
            preflight passed and STARTED.json was created.
        new_attempt: Reason for an attempt after an earlier one.

    Returns:
        The attempt folder.

    Raises:
        PreflightError: Before anything is opened.
        Exception: Whatever failed after opening, once FAILED.json records it.
    """
    paths = paths or HoldoutPaths()
    git = git or Git(paths.root)
    check = preflight(config, paths=paths, git=git, stage="run", new_attempt=new_attempt)
    artifacts = load_artifacts(paths, config)
    frozen = frozen_dsr_inputs(check.freeze)
    attempt = check.attempt
    folder = paths.results / f"{ATTEMPT_PREFIX}{attempt['number']}"
    folder.mkdir(parents=True, exist_ok=False)

    started = {
        "attempt": attempt,
        "started_at": _now(),
        "runner_commit": check.record["head"],
        "runner_code_version": code_version(),
        "frozen_code_version": FROZEN_CODE_VERSION,
        "freeze_commit": FREEZE_COMMIT,
        "declaration_sha256": check.declaration_sha256,
        "frozen_dsr_inputs": frozen,
        "bootstrap": check.declaration["statistics"]["bootstrap"],
        "candidates": check.declaration["candidates"],
        "costs": check.declaration["execution"]["costs"],
        "preflight": check.record,
    }
    write_new(folder / STARTED, json_text(started))
    logger.info("Holdout opened: attempt %s in %s", attempt["number"], folder)
    log = TrialLog(paths.log_path(config))

    try:
        panel = (loader or load_holdout_panel)(config)
        outcome = evaluate_holdout(panel, config, check.declaration, frozen, artifacts)

        daily = pd.concat(
            {(n, k): getattr(r, k) for n, r in outcome.books.items() for k in ("gross", "net")},
            axis=1,
        )
        daily.columns = ["|".join(c) for c in daily.columns]
        outputs = {
            "daily_returns.csv": daily.to_csv(),
            "summary.csv": outcome.summary.to_csv(),
            "comparisons.csv": outcome.comparisons.to_csv(index=False),
            "evaluation.json": json_text(
                {"window": outcome.window.as_dict(), "primary": outcome.primary, **outcome.data}
            ),
            "report.md": holdout_report(outcome, started, check.declaration),
        }
        hashes = {name: write_new(folder / name, text) for name, text in outputs.items()}

        primary = check.declaration["candidates"]["primary"]
        trial_ids = {}
        for name, stats in outcome.summary.iterrows():
            trial_ids[name] = log.log_trial(
                f"holdout_{name}",
                kind="holdout",
                fingerprint=trial_fingerprint(
                    {"holdout": name, "declaration": check.declaration_sha256, **attempt}
                ),
                status="ok",
                config=config,
                model_params={"role": stats["role"], "attempt": attempt["number"]},
                metrics=_holdout_metrics(stats, outcome.primary if name == primary else None),
                notes=f"Holdout attempt {attempt['number']}, {stats['role']}; window "
                f"{outcome.window.first_trade.date()} to {outcome.window.last.date()}",
            )

        rows = int(check.freeze["trial_log_rows"])
        record = check.declaration["trial_log"]
        after = read_log_prefix(paths.log_path(config), rows)
        matched = check_log_prefix(after, record["prefix_sha256_raw"], record["prefix_sha256_lf"])
        write_new(
            folder / COMPLETE,
            json_text(
                {
                    "completed_at": _now(),
                    "outputs_sha256": hashes,
                    "trial_ids": trial_ids,
                    "trial_log_prefix_after_run": matched,
                    "conclusion": outcome.primary["conclusion"],
                }
            ),
        )
    except Exception as exc:
        failure = {
            "failed_at": _now(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        try:
            failure["trial_id"] = log.log_trial(
                "holdout_attempt",
                kind="holdout",
                fingerprint=trial_fingerprint({"holdout_attempt": attempt["number"]}),
                status="failed",
                config=config,
                notes=f"Holdout attempt {attempt['number']} failed: {failure['error']}",
            )
        except Exception as log_error:
            failure["trial_log_error"] = str(log_error)
        write_new(folder / FAILED, json_text(failure))
        logger.error("Holdout attempt %s failed; %s keeps the record", attempt["number"], folder)
        raise

    logger.info("Holdout attempt %s complete: %s", attempt["number"], folder)

    return folder
