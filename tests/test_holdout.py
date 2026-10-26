"""Checks on the holdout runner's safeguards, timing and statistics.

Everything here is synthetic: a temporary "repository" with its own freeze,
trial log, protocol and declaration, a fake git, and a synthetic price panel
whose "holdout" starts on 2008-06-01. No test reads the real 2020-2024 prices,
and none calls the real loader.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest.engine import BacktestError
from src.config import PROJECT_ROOT
from src.evaluation.deflated_sharpe import deflated_sharpe_ratio, return_moments
from src.evaluation.trial_log import SELECTION_KINDS, TrialLog, config_hash
from src.evaluation.trial_scenarios import trial_count_scenarios
from src.pipelines import holdout as h
from src.pipelines.research import development_panel, prepare_research_data
from tests.test_evaluate import TICKERS, shrink

N = 1000
ROWS = 5
VARIANCE = 0.158


def make_panel(perturb_from: str | None = None) -> pd.DataFrame:
    """Six synthetic tickers from 2005, AAA and BBB cointegrated; optionally wreck later prices."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2005-01-03", periods=N)
    walks = np.cumsum(rng.normal(0.0003, 0.012, (N, len(TICKERS))), axis=0)
    spread = np.zeros(N)
    for t in range(1, N):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.01)
    walks[:, 1] = 0.2 + walks[:, 2] + spread
    volume = rng.integers(1_000_000, 9_000_000, (N, len(TICKERS))).astype(float)

    if perturb_from is not None:
        later = dates >= pd.Timestamp(perturb_from)
        walks[later] += np.random.default_rng(99).normal(0, 0.05, (later.sum(), len(TICKERS)))

    frames = {}
    for i, ticker in enumerate(TICKERS):
        price = 50 * np.exp(walks[:, i])
        frames[ticker] = pd.DataFrame(
            {
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "adj_close": price,
                "volume": volume[:, i],
                "return": pd.Series(price).pct_change().to_numpy(),
            },
            index=pd.DatetimeIndex(dates, name="date"),
        )

    return pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()


class FakeGit:
    """Answers the provenance questions from fields a test can change."""

    def __init__(self, blobs: dict[tuple[str, str], bytes]) -> None:
        self.blobs = blobs
        self.dirty: list[str] = []
        self.ancestor = True
        self.changed: list[str] = []

    def head(self) -> str:
        return "f" * 40

    def status(self) -> list[str]:
        return list(self.dirty)

    def is_ancestor(self, commit: str) -> bool:
        return self.ancestor

    def show(self, commit: str, path: str) -> bytes | None:
        return self.blobs.get((commit, path))

    def changed_files(self, base: str) -> list[str]:
        return list(self.changed)


def build_repo(root: Path, config) -> SimpleNamespace:
    """A temporary repository holding a freeze, a 5-row trial log, a protocol and a declaration."""
    cfg = shrink(config, root / "reports" / "trial_log.csv")
    paths = h.HoldoutPaths(
        root=root,
        freeze=root / "reports/results/holdout_freeze.json",
        declaration=root / "reports/results/holdout_declaration.json",
        protocol=root / "reports/methodology/holdout_protocol.md",
        artifacts=root / "reports/results/holdout_artifacts",
        results=root / "reports/results/holdout",
        trial_log=root / "reports/trial_log.csv",
    )
    log = TrialLog(paths.trial_log)
    for i in range(ROWS):
        log.log_trial(f"s{i}", kind="backtest", fingerprint=f"f{i}", metrics={"sharpe": 0.1 * i})
    raw = paths.trial_log.read_bytes()

    freeze = {
        "frozen_at": "2026-10-26T17:01:49+00:00",
        "code_version": h.FROZEN_CODE_VERSION,
        "holdout_start": str(cfg.validation.test_start),
        "config_hash": config_hash(cfg),
        "trial_log_rows": ROWS,
        "trial_log_sha256": h.sha256(raw),
        "headline": {
            "n_trials": 98,
            "variance_used": h.VARIANCE_SCENARIO,
            "variance_used_annual": VARIANCE,
            "min_track_record_years": float("inf"),
        },
        "variance_scenarios": {
            h.VARIANCE_SCENARIO: {"variance_annual": VARIANCE, "variance_period": VARIANCE / 252}
        },
    }
    paths.freeze.parent.mkdir(parents=True, exist_ok=True)
    paths.freeze.write_bytes(json.dumps(freeze, indent=2).encode())
    paths.protocol.parent.mkdir(parents=True, exist_ok=True)
    paths.protocol.write_bytes(b"# Holdout protocol\n\nApproved.\n")

    git = FakeGit(
        {
            (h.FREEZE_COMMIT, "reports/results/holdout_freeze.json"): paths.freeze.read_bytes(),
            (h.FREEZE_COMMIT, "reports/trial_log.csv"): h.lf(raw),
        }
    )
    h.write_declaration(cfg, paths=paths, git=git, approved_on="2026-10-26", approved_by="tester")

    return SimpleNamespace(config=cfg, paths=paths, git=git, freeze=freeze, log=log)


def fake_artifacts(repo: SimpleNamespace) -> None:
    """Placeholder artifacts with a valid manifest, for checks that never load them."""
    folder = repo.paths.artifacts
    (folder / h.MODEL_DIR).mkdir(parents=True)
    (folder / h.MODEL_DIR / "model.joblib").write_bytes(b"\x00model")
    (folder / h.MODEL_DIR / "metadata.json").write_bytes(b"{}\n")
    (folder / h.PAIRS_FILE).write_bytes(b"a,b,selected\n")
    names = [f"{h.MODEL_DIR}/model.joblib", f"{h.MODEL_DIR}/metadata.json", h.PAIRS_FILE]
    manifest = {
        "declaration_sha256": h.file_digest(repo.paths.declaration),
        "training_end": "2008-05-30",
        "fracdiff_d": 0.5,
        "files": {name: h.file_digest(folder / name) for name in names},
    }
    (folder / h.MANIFEST).write_text(h.json_text(manifest), encoding="utf-8")


@pytest.fixture
def repo(tmp_path, config):
    repo = build_repo(tmp_path, config)
    fake_artifacts(repo)
    return repo


def failed_checks(repo, **kwargs) -> list[str]:
    with pytest.raises(h.PreflightError) as caught:
        h.preflight(repo.config, paths=repo.paths, git=repo.git, **kwargs)
    return [failure.split(":")[0] for failure in caught.value.failures]


# --- Preflight ---------------------------------------------------------------


def test_a_complete_setup_passes_every_check(repo):
    result = h.preflight(repo.config, paths=repo.paths, git=repo.git)

    assert len(result.record["checks"]) == 10
    assert result.attempt["number"] == 1
    assert result.declaration["candidates"]["primary"] == h.PRIMARY


def test_a_dirty_or_untracked_tree_is_refused(repo):
    repo.git.dirty = ["?? stray.py"]

    assert failed_checks(repo) == ["1_clean_tree"]


def test_the_frozen_code_must_be_an_ancestor(repo):
    repo.git.ancestor = False

    assert failed_checks(repo) == ["2_ancestry"]


def test_a_missing_or_edited_freeze_is_refused(repo):
    original = repo.paths.freeze.read_bytes()
    repo.paths.freeze.write_bytes(original.replace(b"98", b"12"))
    assert "3_freeze" in failed_checks(repo)

    repo.paths.freeze.unlink()
    assert "3_freeze" in failed_checks(repo)


def test_a_freeze_unlike_its_committed_copy_is_refused(repo):
    key = (h.FREEZE_COMMIT, "reports/results/holdout_freeze.json")
    repo.git.blobs[key] = repo.git.blobs[key].replace(b"98", b"99")

    assert "3_freeze" in failed_checks(repo)


def test_a_changed_config_is_refused(repo, tmp_path):
    edited = tmp_path / "config.yaml"
    edited.write_text(repo.config.source_file.read_text(encoding="utf-8") + "\n# edit\n")
    changed = dataclasses.replace(repo.config, source_file=edited)

    with pytest.raises(h.PreflightError) as caught:
        h.preflight(changed, paths=repo.paths, git=repo.git)
    assert any(f.startswith("4_config") for f in caught.value.failures)


def test_an_edited_historical_row_is_refused(repo):
    data = repo.paths.trial_log.read_bytes()
    repo.paths.trial_log.write_bytes(data.replace(b"s2", b"s9", 1))

    assert failed_checks(repo) == ["5_6_trial_log_prefix"]


@pytest.mark.parametrize("ending", ["crlf", "lf"])
def test_both_line_endings_of_the_frozen_prefix_are_accepted(repo, ending):
    if ending == "lf":
        repo.paths.trial_log.write_bytes(h.lf(repo.paths.trial_log.read_bytes()))
    assert (b"\r\n" in repo.paths.trial_log.read_bytes()) == (ending == "crlf")

    checks = h.preflight(repo.config, paths=repo.paths, git=repo.git).record["checks"]

    assert ("raw bytes" if ending == "crlf" else "LF-normalised") in checks["5_6_trial_log_prefix"]


def test_only_holdout_rows_may_follow_the_frozen_prefix(repo):
    repo.log.log_trial("late", kind="research", fingerprint="late", metrics={"ic": 0.0})

    assert failed_checks(repo) == ["5_6_trial_log_prefix"]


def test_holdout_rows_without_an_attempt_folder_are_refused(repo):
    repo.log.log_trial("holdout_x", kind="holdout", fingerprint="x", metrics={"a": 1.0})

    assert failed_checks(repo) == ["10_attempt"]


def test_an_unknown_candidate_is_refused(repo):
    record = json.loads(repo.paths.declaration.read_text(encoding="utf-8"))
    record["candidates"]["secondary"].append("magic_book")
    repo.paths.declaration.write_text(h.json_text(record), encoding="utf-8")

    assert "8_books" in failed_checks(repo)


def test_an_unapproved_or_inconsistent_declaration_is_refused(repo):
    record = json.loads(repo.paths.declaration.read_text(encoding="utf-8"))
    record["statistics"]["dsr"]["variance_annual"] = 0.05
    repo.paths.declaration.write_text(h.json_text(record), encoding="utf-8")

    assert "7_declaration" in failed_checks(repo)


def test_an_edited_protocol_is_refused(repo):
    repo.paths.protocol.write_bytes(b"# Holdout protocol\n\nChanged after approval.\n")

    assert "7_declaration" in failed_checks(repo)


def test_missing_or_modified_artifacts_are_refused(repo):
    model = repo.paths.artifacts / h.MODEL_DIR / "model.joblib"
    model.write_bytes(b"\x00other model")
    assert failed_checks(repo) == ["9_artifacts"]

    model.unlink()
    assert failed_checks(repo) == ["9_artifacts"]


def test_unprepared_artifacts_are_refused(tmp_path, config):
    repo = build_repo(tmp_path, config)

    assert failed_checks(repo) == ["9_artifacts"]


def test_changes_outside_the_allowlist_are_refused(repo):
    repo.git.changed = ["src/pipelines/holdout.py", "reports/results/holdout_declaration.json"]
    assert h.preflight(repo.config, paths=repo.paths, git=repo.git)

    repo.git.changed = ["src/backtest/costs.py"]
    assert failed_checks(repo) == ["11_frozen_paths"]
    repo.git.changed = ["config/config.yaml"]
    assert failed_checks(repo) == ["11_frozen_paths"]


def test_the_trial_log_module_may_change_only_by_the_kinds_edit(repo):
    path = h.TRIAL_LOG_MODULE
    base = "header\n" + h.KINDS_BEFORE + "footer\n"
    repo.git.blobs[(h.FROZEN_CODE_VERSION, path)] = base.encode()
    repo.git.changed = [path]
    target = repo.paths.root / path
    target.parent.mkdir(parents=True)

    target.write_bytes(base.replace(h.KINDS_BEFORE, h.KINDS_AFTER).encode().replace(b"\n", b"\r\n"))
    assert h.preflight(repo.config, paths=repo.paths, git=repo.git)

    target.write_bytes(base.replace(h.KINDS_BEFORE, h.KINDS_AFTER).replace("footer", "x").encode())
    assert failed_checks(repo) == ["11_frozen_paths"]


def test_the_real_trial_log_edit_is_exactly_the_allowed_one():
    base = h.Git().show(h.FROZEN_CODE_VERSION, h.TRIAL_LOG_MODULE)
    if base is None:
        pytest.skip("the frozen commit is not available")
    now = (PROJECT_ROOT / h.TRIAL_LOG_MODULE).read_bytes()

    assert h.lf(now).decode() == h.lf(base).decode().replace(h.KINDS_BEFORE, h.KINDS_AFTER)


# --- Declaration and attempts -------------------------------------------------


def test_the_declaration_is_written_once_and_replaced_only_with_a_reason(repo):
    with pytest.raises(h.HoldoutError, match="never rewritten"):
        h.write_declaration(
            repo.config, paths=repo.paths, git=repo.git, approved_on="x", approved_by="y"
        )
    with pytest.raises(h.HoldoutError, match="reason"):
        h.write_declaration(
            repo.config,
            paths=repo.paths,
            git=repo.git,
            approved_on="x",
            approved_by="y",
            replace=" ",
        )

    original = repo.paths.declaration.read_bytes()
    second = h.write_declaration(
        repo.config,
        paths=repo.paths,
        git=repo.git,
        approved_on="2026-10-27",
        approved_by="tester",
        replace="typo in the rationale",
    )

    assert repo.paths.declaration.read_bytes() == original
    assert second.name == "holdout_declaration.r2.json"
    assert json.loads(second.read_text())["replaces"]["sha256"] == h.file_digest(
        repo.paths.declaration
    )
    # The manifest cites the original, which is still in the chain.
    assert h.preflight(repo.config, paths=repo.paths, git=repo.git)


def test_the_declaration_records_the_frozen_inputs(repo):
    record = json.loads(repo.paths.declaration.read_text(encoding="utf-8"))

    assert record["statistics"]["dsr"]["n_trials"] == 98
    assert record["statistics"]["dsr"]["variance_annual"] == VARIANCE
    assert record["trial_log"]["prefix_sha256_raw"] == repo.freeze["trial_log_sha256"]
    assert record["development_artifacts"]["status"] == "not prepared"
    assert h.PRIMARY not in record["candidates"]["secondary"]
    bootstrap = record["statistics"]["bootstrap"]
    assert (bootstrap["block_days"], bootstrap["resamples"], bootstrap["seed"]) == (21, 1000, 0)


def test_existing_or_stale_attempts_need_a_reason_and_are_kept(repo):
    attempt = repo.paths.results / "attempt-1"
    attempt.mkdir(parents=True)
    (attempt / h.STARTED).write_text("{}")

    assert failed_checks(repo) == ["10_attempt"]
    plan = h.plan_attempt(repo.paths.results, "interrupted by a power cut")
    assert plan == {
        "number": 2,
        "reason": "interrupted by a power cut",
        "previous": {"1": "stale"},
    }
    with pytest.raises(h.HoldoutError, match="without --new-attempt"):
        h.plan_attempt(repo.paths.results.parent / "empty", "reason")

    (repo.paths.results / "notes.txt").write_text("x")
    assert failed_checks(repo, new_attempt="reason") == ["10_attempt"]


# --- Holdout kind ------------------------------------------------------------


def test_holdout_rows_never_count_as_selection_trials(tmp_path):
    log = TrialLog(tmp_path / "trials.csv")
    log.log_trial("a", kind="backtest", fingerprint="a", metrics={"sharpe": 0.3})
    before = trial_count_scenarios(log)
    log.log_trial("holdout_a", kind="holdout", fingerprint="h", metrics={"holdout_sharpe_net": 1})

    assert "holdout" not in SELECTION_KINDS
    pd.testing.assert_frame_equal(trial_count_scenarios(log), before)


# --- Timing, costs and limits -----------------------------------------------


@pytest.fixture
def calendar(config):
    """Ten dates straddling a synthetic holdout start, three tickers."""
    dates = pd.bdate_range("2008-05-26", periods=10)
    returns = pd.DataFrame(
        np.linspace(-0.01, 0.02, 30).reshape(10, 3), index=dates, columns=["A", "B", "C"]
    )
    cfg = dataclasses.replace(
        config,
        validation=dataclasses.replace(
            config.validation, test_start=pd.Timestamp("2008-06-01").date()
        ),
    )
    window = h.evaluation_window(returns, cfg)

    return SimpleNamespace(returns=returns, config=cfg, window=window, dates=dates)


def test_the_window_follows_the_engine_convention(calendar):
    w = calendar.window

    assert w.decision == pd.Timestamp("2008-06-02")
    assert w.first_trade == pd.Timestamp("2008-06-03")
    assert w.first_return == pd.Timestamp("2008-06-04")
    assert w.last == calendar.dates[-1]


def test_a_book_is_flat_until_d1_and_earns_from_the_bar_after(calendar):
    target = pd.DataFrame(0.0, index=calendar.dates, columns=calendar.returns.columns)
    # A December-style monthly target set before D0, and a different one set at D0.
    target.loc[: calendar.window.decision, "A"] = 0.5
    target.loc[calendar.window.decision :, "B"] = 0.5
    book = h.Book("x", "primary", target, limits=True)

    result = h.run_book(book, calendar.returns, calendar.window, calendar.config)
    w, r = calendar.window, calendar.returns

    assert result.net.index[0] == w.first_trade
    assert result.gross.loc[w.first_trade] == 0.0
    assert result.net.loc[w.first_trade] == pytest.approx(-result.costs.loc[w.first_trade, "total"])
    assert result.costs.loc[w.first_trade, "total"] > 0
    # The D0 target (A and B at 0.5) is held from D1 and earns D2's return, not D1's.
    assert result.weights.loc[w.first_trade, ["A", "B"]].tolist() == [0.5, 0.5]
    assert result.gross.loc[w.first_return] == pytest.approx(
        0.5 * r.loc[w.first_return, "A"] + 0.5 * r.loc[w.first_return, "B"]
    )


def test_limits_are_enforced_and_costs_charged(calendar):
    target = pd.DataFrame(0.9, index=calendar.dates, columns=calendar.returns.columns)
    limited = h.run_book(
        h.Book("x", "secondary", target), calendar.returns, calendar.window, calendar.config
    )

    gross = limited.weights.abs().sum(axis=1)
    assert (gross <= calendar.config.backtest.max_gross_leverage + 1e-9).all()
    assert (limited.net <= limited.gross + 1e-15).all()

    with pytest.raises(BacktestError, match="max_gross_leverage"):
        h.run_book(
            h.Book("x", "secondary", target, limits=False),
            calendar.returns,
            calendar.window,
            calendar.config,
        )


def test_a_zero_lag_is_refused(calendar):
    config = dataclasses.replace(
        calendar.config, backtest=dataclasses.replace(calendar.config.backtest, execution_lag=0)
    )

    with pytest.raises(h.HoldoutError, match="lag"):
        h.evaluation_window(calendar.returns, config)


# --- Statistics ----------------------------------------------------------------


def test_the_dsr_uses_the_frozen_count_and_variance():
    net = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 1250))
    frozen = h.frozen_dsr_inputs(
        {
            "headline": {
                "n_trials": 98,
                "variance_used": h.VARIANCE_SCENARIO,
                "variance_used_annual": VARIANCE,
            },
            "variance_scenarios": {
                h.VARIANCE_SCENARIO: {
                    "variance_annual": VARIANCE,
                    "variance_period": VARIANCE / 252,
                }
            },
        }
    )
    interval = {"difference": 0.1, "low": -0.2, "high": 0.4, "share_positive": 0.6}

    result = h.primary_result(net, frozen, interval)
    sharpe, n, skew, kurtosis = return_moments(net.to_numpy())

    assert result["n_trials"] == 98
    assert result["dsr"] == pytest.approx(
        deflated_sharpe_ratio(sharpe, 98, n, skew, kurtosis, VARIANCE / 252)
    )
    assert result["conclusion"] == h.NO_EDGE


def test_a_freeze_with_another_variance_scenario_is_refused():
    freeze = {
        "headline": {"n_trials": 98, "variance_used": "strategies", "variance_used_annual": 0.1},
        "variance_scenarios": {
            h.VARIANCE_SCENARIO: {"variance_annual": 0.1, "variance_period": 0.1 / 252}
        },
    }

    with pytest.raises(h.HoldoutError, match="headline used"):
        h.frozen_dsr_inputs(freeze)


@pytest.mark.parametrize(
    ("dsr", "low", "edge"),
    [
        (0.96, 0.01, True),
        (0.95, 0.01, False),
        (0.96, 0.0, False),
        (0.94, 0.5, False),
        (float("nan"), 0.5, False),
        (0.99, float("nan"), False),
    ],
)
def test_the_decision_needs_both_conditions_strictly(dsr, low, edge):
    result = h.decide(dsr, low)

    assert result["evidence_of_edge"] is edge
    assert result["conclusion"] == (h.EDGE if edge else h.NO_EDGE)


# --- Development-only artifacts -------------------------------------------------


def test_development_artifacts_ignore_every_post_boundary_price(config, tmp_path):
    cfg = shrink(config, tmp_path / "trials.csv")
    start = str(cfg.validation.test_start)
    clean, wrecked = make_panel(), make_panel(perturb_from=start)
    assert not clean.equals(wrecked)

    a = h.build_development_artifacts(clean, cfg)
    b = h.build_development_artifacts(wrecked, cfg)

    pd.testing.assert_frame_equal(a.pairs, b.pairs)
    assert a.training_index.equals(b.training_index)
    assert a.fracdiff_d == b.fracdiff_d
    assert a.training_index.get_level_values("date").max() < pd.Timestamp(start)
    X = prepare_research_data(development_panel(clean, cfg), cfg).X
    np.testing.assert_array_equal(a.model.predict(X), b.model.predict(X))


def test_preparation_needs_an_authoriser(repo):
    with pytest.raises(h.HoldoutError, match="authorised"):
        h.prepare_artifacts(repo.config, authorised_by=" ", paths=repo.paths, git=repo.git)


# --- The run, end to end on synthetic data ------------------------------------------


@pytest.fixture(scope="module")
def opened(tmp_path_factory, config):
    """Prepare the artifacts and run attempt 1 on the synthetic panel."""
    repo = build_repo(tmp_path_factory.mktemp("repo"), config)
    panel = make_panel()
    calls = []

    h.prepare_artifacts(
        repo.config, authorised_by="tester", paths=repo.paths, git=repo.git, loader=lambda c: panel
    )

    def loader(cfg):
        # The start marker must exist before any price is handed over.
        calls.append((repo.paths.results / "attempt-1" / h.STARTED).is_file())
        return panel

    folder = h.run_holdout(repo.config, paths=repo.paths, git=repo.git, loader=loader)

    return SimpleNamespace(repo=repo, folder=folder, calls=calls, panel=panel)


def test_the_start_marker_is_written_before_the_loader_is_called(opened):
    assert opened.calls == [True]
    started = json.loads((opened.folder / h.STARTED).read_text(encoding="utf-8"))
    assert started["frozen_code_version"] == h.FROZEN_CODE_VERSION
    assert started["runner_commit"] == "f" * 40
    assert started["frozen_dsr_inputs"]["n_trials"] == 98


def test_every_declared_book_shares_one_window(opened):
    daily = pd.read_csv(opened.folder / "daily_returns.csv", index_col=0, parse_dates=True)
    evaluation = json.loads((opened.folder / "evaluation.json").read_text(encoding="utf-8"))
    books = {column.split("|")[0] for column in daily.columns}

    declaration = json.loads(opened.repo.paths.declaration.read_text(encoding="utf-8"))

    assert books == set(h.declared_books(declaration))
    assert daily.notna().all().all()
    assert str(daily.index[0].date()) == evaluation["window"]["first_row_and_trade"]
    gross = daily[[c for c in daily.columns if c.endswith("|gross")]]
    assert (gross.iloc[0] == 0).all()


def test_the_report_covers_every_book_and_the_limitations(opened):
    report = (opened.folder / "report.md").read_text(encoding="utf-8")
    summary = pd.read_csv(opened.folder / "summary.csv", index_col=0)

    for name in (h.PRIMARY, *h.SECONDARIES, *h.BENCHMARKS):
        assert f"| {name} |" in report
        assert name in summary.index
    assert "Survivorship" in report
    assert h.PROBABILITY_NOTE in report
    assert h.DSR_NOTE in report
    assert re.search(
        r"\*\*Conclusion: (no convincing evidence of an edge|evidence of an edge)", report
    )


def test_the_run_is_logged_as_holdout_rows_and_leaves_the_prefix_intact(opened):
    complete = json.loads((opened.folder / h.COMPLETE).read_text(encoding="utf-8"))
    table = TrialLog(opened.repo.paths.trial_log).trial_table()

    assert set(table["kind"].iloc[ROWS:]) == {"holdout"}
    assert len(table) == ROWS + 12
    assert "raw bytes" in complete["trial_log_prefix_after_run"]
    for name, digest in complete["outputs_sha256"].items():
        assert h.file_digest(opened.folder / name) == digest


def test_a_second_run_is_refused_without_a_reason(opened):
    with pytest.raises(h.PreflightError, match="has been opened"):
        h.run_holdout(opened.repo.config, paths=opened.repo.paths, git=opened.repo.git, loader=None)


def test_the_loader_is_never_called_when_preflight_fails(repo):
    called = []
    repo.git.dirty = [" M src/backtest/costs.py"]

    with pytest.raises(h.PreflightError):
        h.run_holdout(
            repo.config, paths=repo.paths, git=repo.git, loader=lambda c: called.append(1)
        )

    assert called == []
    assert not repo.paths.results.exists()


def test_a_failed_attempt_is_kept_and_the_next_needs_a_reason(repo, monkeypatch):
    # The placeholder artifacts are never unpickled: the loader fails first.
    monkeypatch.setattr(h, "load_artifacts", lambda paths, config: None)

    def broken(cfg):
        raise RuntimeError("disk unplugged")

    with pytest.raises(RuntimeError, match="disk unplugged"):
        h.run_holdout(repo.config, paths=repo.paths, git=repo.git, loader=broken)

    first = repo.paths.results / "attempt-1"
    kept = {p.name: p.read_bytes() for p in first.iterdir()}
    assert set(kept) == {h.STARTED, h.FAILED}
    assert "disk unplugged" in json.loads(kept[h.FAILED])["error"]
    table = TrialLog(repo.paths.trial_log).trial_table()
    assert table["kind"].iloc[-1] == "holdout" and table["status"].iloc[-1] == "failed"

    assert failed_checks(repo) == ["10_attempt"]
    with pytest.raises(RuntimeError):
        h.run_holdout(
            repo.config, paths=repo.paths, git=repo.git, loader=broken, new_attempt="retry"
        )

    assert {p.name: p.read_bytes() for p in first.iterdir()} == kept
    started = json.loads((repo.paths.results / "attempt-2" / h.STARTED).read_text())
    assert started["attempt"]["previous"] == {"1": "failed"}


# --- Tooling -------------------------------------------------------------------------


def test_ci_and_make_all_never_run_the_holdout():
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/test.yml").read_text(encoding="utf-8")
    )
    commands = " ".join(step.get("run", "") for step in workflow["jobs"]["test"]["steps"])
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    all_line = next(line for line in makefile.splitlines() if line.startswith("all:"))
    ci = makefile.split("\nci:")[1].split("\n\n")[0]

    assert "holdout" not in commands
    assert "holdout" not in all_line.split("##")[0]
    assert "holdout" not in ci
    assert "entrypoint/holdout.py run" in makefile


def test_the_check_command_never_loads_prices(monkeypatch):
    from entrypoint import holdout as cli

    def forbidden(config):
        raise AssertionError("the price loader was called")

    monkeypatch.setattr(h, "load_holdout_panel", forbidden)
    monkeypatch.setattr(h, "load_clean_panel", forbidden)
    calls = []
    monkeypatch.setattr(
        h,
        "preflight",
        lambda config, **kw: calls.append(kw) or SimpleNamespace(record={"checks": {"x": "ok"}}),
    )

    assert cli.main(["check"]) == cli.EXIT_OK
    assert calls == [{"stage": "run", "new_attempt": None}]
