"""Checks on the Makefile, the CI workflow and .gitignore.

GNU make is not available on every machine this runs on, so the Makefile is
checked by reading it: every required target exists, recipes are
tab-indented, and every script or module a target runs exists. CI also
dry-runs the targets with real make.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from src.config import PROJECT_ROOT

REQUIRED_TARGETS = (
    "install",
    "fetch",
    "research",
    "evaluate",
    "test",
    "lint",
    "format",
    "dashboard",
    "all",
)


def makefile_targets() -> dict[str, list[str]]:
    """Each target's recipe lines, from the Makefile."""
    targets: dict[str, list[str]] = {}
    current = None

    for line in (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([a-z-]+):", line)
        if match:
            current = match.group(1)
            targets[current] = []
        elif line.startswith("\t") and current:
            targets[current].append(line.strip())
        elif line.startswith("    ") and current:
            pytest.fail(f"recipe of '{current}' is indented with spaces, make needs a tab")

    return targets


def test_every_required_target_exists_and_does_something():
    targets = makefile_targets()

    for target in REQUIRED_TARGETS:
        assert target in targets, target
    for target in set(REQUIRED_TARGETS) - {"all"}:
        assert targets[target], f"'{target}' has no recipe"


def test_targets_run_scripts_and_modules_that_exist():
    for recipe in makefile_targets().values():
        for line in recipe:
            for script in re.findall(r"python (entrypoint/\w+\.py)", line):
                assert (PROJECT_ROOT / script).is_file(), script
            for module in re.findall(r"python -m ([\w.]+)", line):
                assert (PROJECT_ROOT / (module.replace(".", "/") + ".py")).is_file(), module
            for notebook in re.findall(r"(notebooks/\S+\.ipynb)", line):
                assert (PROJECT_ROOT / notebook).is_file(), notebook


def test_dashboard_says_it_is_not_built_and_fails():
    recipe = " ".join(makefile_targets()["dashboard"])

    assert "Phase 9" in recipe
    assert "exit 1" in recipe


def test_ci_never_runs_the_expensive_targets():
    targets = makefile_targets()

    assert not any("research" in line or "evaluate" in line for line in targets["ci"])


def test_ci_workflow_runs_ruff_format_check_and_pytest():
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/test.yml").read_text(encoding="utf-8")
    )
    commands = " ".join(step.get("run", "") for step in workflow["jobs"]["test"]["steps"])

    assert "ruff check" in commands
    assert "ruff format --check" in commands
    assert "pytest" in commands
    assert "uv sync --locked" in commands
    # No script that downloads data or fits models; make targets only dry-run (-n).
    assert "entrypoint/" not in commands and "python -m src.pipelines" not in commands
    assert re.findall(r"make (?!-n|dashboard)\S+", commands) == []


def test_gitignore_covers_generated_files():
    patterns = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    for required in (
        "mlflow.db",
        "mlruns/",
        ".ipynb_checkpoints/",
        "logs/",
        "models/*",
        ".pytest_cache/",
    ):
        assert required in patterns, required


def test_no_tracked_file_is_ignored():
    try:
        tracked_but_ignored = subprocess.run(
            ["git", "ls-files", "--cached", "--ignored", "--exclude-standard"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available")

    assert tracked_but_ignored == []


def test_the_results_notebook_imports_production_code():
    text = Path(PROJECT_ROOT / "notebooks/05_results.ipynb").read_text(encoding="utf-8")

    assert "from src." in text
