"""Checks on the dashboard: formatting, the data layer, honest failures, routing and integrity.

The data-layer tests read the committed results (reports/results, data/), so
they check the numbers a reader sees against the files they come from. The
integrity test runs every page and export and confirms that no frozen
artifact, trial-log row or holdout output changed, and that nothing tried
to log a trial or open the holdout.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard import data, theme
from dashboard import formatting as fmt
from dashboard.controls import defaults, selection
from src.config import PROJECT_ROOT

PAGES = (
    "01_overview",
    "02_market_data",
    "03_feature_research",
    "04_model_performance",
    "05_backtesting",
    "06_portfolio_risk",
    "07_experiments",
)


def page(name: str):
    return importlib.import_module(f"dashboard.pages.{name}")


def control_values() -> list:
    d = defaults()
    return [
        d[k] for k in ("asset", "start", "end", "period", "freq", "model", "strategy", "benchmark")
    ]


def text_of(component) -> str:
    return str(component)


# --- Formatting --------------------------------------------------------------


def test_missing_values_are_a_dash_never_zero():
    for missing in (None, float("nan"), np.nan, pd.NaT, float("inf")):
        assert fmt.pct(missing) == fmt.MISSING
        assert fmt.sharpe(missing) == fmt.MISSING
    assert fmt.pct(0.0) == "0.0%"


def test_numbers_use_a_true_minus_and_fixed_digits():
    assert fmt.pct(-0.1234, 1) == fmt.MINUS + "12.3%"
    assert fmt.pct(0.05, 1, signed=True) == "+5.0%"
    assert fmt.bps(0.0005) == "5.0 bps"
    assert fmt.pp(0.0003) == "+0.03 pp"
    assert fmt.sharpe(0.53677) == "0.54"
    assert fmt.integer(39985.0) == "39,985"


def test_tone_follows_the_sign_and_can_be_inverted():
    assert fmt.tone(0.1) == "pos"
    assert fmt.tone(-0.1) == "neg"
    assert fmt.tone(-0.1, good_when="negative") == "pos"
    assert fmt.tone(None) == "neutral"


# --- Data layer ----------------------------------------------------------------


@pytest.mark.parametrize("key", list(data.BOOKS))
def test_every_book_has_returns_in_the_right_period(key):
    boundary = pd.Timestamp(data.config().validation.test_start)

    dev = data.book_returns(key, data.DEVELOPMENT)
    hold = data.book_returns(key, data.HOLDOUT)

    assert list(dev.columns) == ["gross", "net"] == list(hold.columns)
    assert dev.index.max() < boundary <= hold.index.min()
    assert dev.index.is_unique and hold.index.is_unique
    # Net never beats gross: costs only subtract.
    assert (dev["net"] <= dev["gross"] + 1e-12).all()


def test_holdout_statistics_are_the_saved_summary():
    summary = pd.read_csv(data.HOLDOUT_ATTEMPT / "summary.csv").set_index("book")

    for key in data.BOOKS:
        stats = data.book_stats(key, data.HOLDOUT)
        assert stats["net_sharpe"] == pytest.approx(summary.loc[key, "net_sharpe"])
        assert stats["gross_max_drawdown"] == pytest.approx(summary.loc[key, "gross_max_drawdown"])


def test_development_cpcv_statistics_are_means_over_paths():
    paths = pd.read_csv(data.EVALUATION / "paths.csv")
    expected = paths[paths["strategy"] == "ml_long_short"]["net_sharpe"].mean()

    assert data.book_stats("ml_long_short", data.DEVELOPMENT)["net_sharpe"] == pytest.approx(
        expected
    )


def test_the_decision_interval_is_the_one_the_holdout_recorded():
    evaluation = data.holdout_evaluation()["primary"]["interval"]
    primary = data.declaration()["candidates"]["primary"]

    shown = data.interval_against(primary, "equal_weight_hold", data.HOLDOUT)

    assert shown["role"] == "decision"
    assert shown["low"] == pytest.approx(evaluation["low"])
    assert shown["high"] == pytest.approx(evaluation["high"])


def test_calendar_years_compound_within_each_year():
    returns = data.book_returns("equal_weight_hold", data.HOLDOUT)
    years = data.calendar_years("equal_weight_hold", "buy_and_hold", data.HOLDOUT)
    first = returns.index.year.min()

    expected = (1 + returns.loc[str(first), "net"]).prod() - 1
    assert years.loc[first, "net"] == pytest.approx(expected)
    assert years.loc[first, "cost_drag"] == pytest.approx(years.loc[first, "gross"] - expected)


def test_resampling_compounds_and_keeps_empty_periods_missing():
    dates = pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-20"])
    returns = pd.Series([0.01, 0.01, 0.02], index=dates)

    weekly = data.resample_returns(returns, "1W")

    assert weekly.loc["2020-01-10"] == pytest.approx(1.01**2 - 1)
    # The week of 13 January has no data: missing, not a 0% return.
    assert np.isnan(weekly.loc["2020-01-17"])


def test_drawdown_is_measured_from_the_running_peak():
    returns = pd.Series([0.10, -0.50, 0.20])

    dd = data.drawdown(returns)

    assert dd.iloc[0] == 0
    assert dd.iloc[1] == pytest.approx(-0.5)
    assert dd.iloc[2] == pytest.approx(1.1 * 0.5 * 1.2 / 1.1 - 1)


def test_forecast_quintiles_split_every_scored_row():
    q = data.forecast_quintiles()

    assert list(q.index.astype(str)) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert q["n"].sum() == len(data.forecasts_and_outcomes())
    assert q["forecast"].is_monotonic_increasing


def test_latest_weights_are_fully_invested_and_match_the_saved_risk_split():
    for rule in ("minimum_variance", "equal_weight", "risk_parity"):
        weights = data.latest_weights(rule)
        contrib = data.risk_contributions().loc[rule]

        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert weights.name == pd.Timestamp("2019-12-31")
        # Every asset with risk carries weight, and none without weight carries risk.
        assert ((weights > 1e-6) == (contrib.reindex(weights.index) > 1e-6)).all()


def test_frozen_inputs_are_read_not_recomputed():
    freeze = data.freeze()

    assert freeze["headline"]["n_trials"] == 98
    assert data.holdout_evaluation()["primary"]["n_trials"] == freeze["headline"]["n_trials"]


def test_a_missing_result_is_a_data_error(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "HOLDOUT_ATTEMPT", tmp_path / "attempt-9")
    data.clear_cache()

    with pytest.raises(data.DataError, match="not been written"):
        data.holdout_summary()
    data.clear_cache()


def test_the_mlflow_store_is_optional(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "MLFLOW_DB", tmp_path / "absent.db")
    data.clear_cache()

    with pytest.raises(data.DataError, match="MLflow store not found"):
        data.mlflow_runs()
    data.clear_cache()


def test_files_are_reread_after_they_change(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("a\n1\n")
    first = data._cached_file(path, data._csv(), "Table")
    path.write_text("a\n2\n")
    import os

    os.utime(path, ns=(path.stat().st_mtime_ns + 10**9,) * 2)

    assert int(first["a"].iloc[0]) == 1
    assert int(data._cached_file(path, data._csv(), "Table")["a"].iloc[0]) == 2


# --- Honest failure states ------------------------------------------------------


def test_a_data_problem_becomes_an_empty_state():
    def missing():
        raise data.DataError("Holdout summary not found", "the run has not been written")

    rendered = text_of(theme.guarded(missing, "a panel"))

    assert "Holdout summary not found" in rendered
    assert "empty missing" in rendered


def test_an_unexpected_failure_is_an_error_state_without_a_traceback(caplog):
    def broken():
        raise KeyError("column")

    rendered = text_of(theme.guarded(broken, "a panel"))

    assert "This panel could not be built" in rendered
    assert "Traceback" not in rendered
    assert "Dashboard panel failed" in caplog.text


# --- The app --------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_module():
    return importlib.import_module("dashboard.app")


def test_the_app_builds_and_routes_every_page(app_module):
    assert app_module.serve_layout() is not None
    assert len(app_module.PAGES) == 7

    for path in app_module.PAGES:
        assert "could not be built" not in text_of(app_module.route(path))
    assert "Page not found" in text_of(app_module.route("/nowhere"))


def toolbar_classes(app_module, pathname: str) -> tuple[dict, dict, str]:
    """The field and group classes and the note class mark_page gives one page."""
    out = app_module.mark_page(pathname)
    controls, groups = app_module.CONTROLS, list(app_module.GROUPS)
    fields = dict(zip(controls, out[2 : 2 + len(controls)], strict=True))
    start = 2 + len(controls)
    return fields, dict(zip(groups, out[start : start + len(groups)], strict=True)), out[-1]


def test_each_page_shows_only_the_global_controls_it_uses(app_module):
    fields, groups, note = toolbar_classes(app_module, "/market")

    assert "hidden" not in fields["asset"]
    assert "hidden" in fields["period"]
    assert groups == {"market": "filter-group", "evaluation": "filter-group hidden"}
    assert "hidden" in note
    assert app_module.mark_page("/market")[0][list(app_module.PAGES).index("/market")] == (
        "nav-item active"
    )


def test_a_page_without_global_controls_says_so(app_module):
    fields, groups, note = toolbar_classes(app_module, "/portfolio")

    assert all("hidden" in c for c in fields.values())
    assert all("hidden" in c for c in groups.values())
    assert note == "toolbar-note"


def test_selection_fills_blank_controls_with_defaults():
    chosen = selection("AAPL", None, None, None, "1W", None, None, None)

    assert chosen["asset"] == "AAPL"
    assert chosen["freq"] == "1W"
    assert chosen["strategy"] == defaults()["strategy"]


@pytest.mark.parametrize("name", PAGES)
def test_every_page_exports_a_table(name):
    filename, frame = page(name).export(defaults())

    assert re.fullmatch(r"[A-Za-z0-9_]+", filename)
    assert isinstance(frame, pd.DataFrame) and not frame.empty


def render_all_pages() -> str:
    """Every page callback with the default controls, as text."""
    values = [*control_values(), 0]
    parts = [
        page("01_overview").static_blocks(0),
        page("01_overview").kpis(*values),
        page("01_overview").charts_and_table(*values),
        page("02_market_data").update(*values),
        page("03_feature_research").research_blocks(0),
        page("03_feature_research").asset_series(*values),
        page("04_model_performance").update(*values),
        page("05_backtesting").update(*values),
        page("06_portfolio_risk").fixed_blocks(0),
        page("06_portfolio_risk").update("minimum_variance", "bootstrap", 0),
        page("07_experiments").fixed_blocks(0),
        page("07_experiments").table_rows(None, None, 0),
        page("07_experiments").inspect(None, None, 0),
    ]
    return "".join(text_of(p) for p in parts)


def test_every_page_renders_without_an_error_state():
    rendered = render_all_pages()

    assert "could not be built" not in rendered
    # The one expected gap is said plainly, not hidden.
    assert "Holdout forecasts were not recorded" in rendered


def test_the_verdict_and_dsr_shown_are_the_holdout_record():
    evaluation = data.holdout_evaluation()["primary"]
    overview = page("01_overview")
    rendered = text_of(overview.verdict()) + text_of(overview.dsr_card())

    assert evaluation["conclusion"] in rendered
    assert fmt.number(evaluation["dsr"], 3) in rendered


@pytest.mark.parametrize(
    ("period", "key"),
    [(data.DEVELOPMENT, "ml_long_short"), (data.HOLDOUT, "allocation_minimum_variance")],
)
def test_every_sharpe_card_carries_the_trial_count(period, key):
    c = {**defaults(), "period": period, "strategy": key}

    rendered = text_of(page("05_backtesting").kpis(c)) + text_of(
        page("01_overview").strategy_cards(key, c["benchmark"], period)
    )

    assert rendered.count("N = 98 trials") >= 2


# --- Integrity -------------------------------------------------------------------


def _fingerprints() -> dict[str, str]:
    root = PROJECT_ROOT / "reports"
    files = [
        root / "results" / "holdout_freeze.json",
        root / "results" / "holdout_declaration.json",
        root / "trial_log.csv",
        *sorted((root / "results" / "holdout").rglob("*")),
        *sorted((root / "results" / "holdout_artifacts").rglob("*")),
    ]
    return {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}


def test_the_dashboard_never_writes_results_logs_trials_or_opens_the_holdout(monkeypatch):
    from src.evaluation import trial_log
    from src.pipelines import holdout

    def forbidden(*args, **kwargs):
        raise AssertionError("the dashboard tried to write a result")

    monkeypatch.setattr(trial_log.TrialLog, "log_trial", forbidden)
    monkeypatch.setattr(holdout, "run_holdout", forbidden)
    monkeypatch.setattr(holdout, "load_holdout_panel", forbidden)
    monkeypatch.setattr(holdout, "prepare_artifacts", forbidden)
    before = _fingerprints()

    data.clear_cache()
    render_all_pages()
    for name in PAGES:
        page(name).export(defaults())

    assert _fingerprints() == before
    assert len(before) >= 10


def test_no_performance_number_is_hard_coded():
    headline = data.freeze()["headline"]
    primary = data.holdout_evaluation()["primary"]
    values = [headline["dsr"], headline["sharpe_annual"], primary["dsr"], primary["sharpe_annual"]]
    sources = "".join(
        p.read_text(encoding="utf-8") for p in (PROJECT_ROOT / "dashboard").rglob("*.py")
    )

    for value in values:
        for text in (f"{value:.3f}", f"{value:.2f}"):
            assert text not in sources, text
    assert "Path(" not in Path(theme.__file__).read_text(encoding="utf-8")
