# Stock Return Predictor
#
# uv manages the environment, so every target runs through `uv run` and needs
# no virtualenv to be activated first. On Windows without make, run the
# command under each target directly; they are plain uv commands.
#
# `all` runs the whole development-period pipeline. It refits models (several
# minutes) and appends to the trial log, so it is never part of `ci`. Nothing
# in `all` or `ci` touches the 2020-2024 holdout: `holdout` opens it once, by
# hand, after `holdout-check` passes (reports/methodology/holdout_protocol.md).

.DEFAULT_GOAL := help
.PHONY: help install setup fetch eda research evaluate portfolio report freeze reconcile results \
	dashboard all lint format test ci lock clean holdout-check holdout

help:  ## Show the available targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Create the environment and install everything, from the lock file
	uv sync --locked

setup: install  ## Alias for install

fetch:  ## Pull and cache the price data (network)
	uv run python entrypoint/fetch.py

eda:  ## Rebuild the exploratory figures and the stationarity table
	uv run python -m src.reporting.figures
	uv run python -m src.reporting.stationarity

research:  ## CPCV research run: fits models, logs a trial (minutes)
	uv run python entrypoint/research.py

evaluate:  ## CPCV backtests of every strategy: fits models, logs trials (minutes)
	uv run python entrypoint/evaluate.py

portfolio:  ## Allocation, frontier and risk analysis on the development period
	uv run python -m src.pipelines.portfolio

report:  ## Write the research report from saved results and the trial log
	uv run python -m src.reporting.report

freeze:  ## Freeze N, V[SR] and the trial list before the holdout (needs committed code)
	uv run python -m src.reporting.report --freeze

reconcile:  ## Re-run notebook 4's code on development years and reconcile its costs
	uv run python reports/methodology/cost_reconciliation.py

results:  ## Execute notebooks/05_results.ipynb from a clean kernel, in place
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_results.ipynb

holdout-check:  ## Preflight for the holdout run: every safeguard, no holdout data read
	uv run python entrypoint/holdout.py check

holdout:  ## Open the 2020-2024 holdout ONCE (needs approval, prepared artifacts, clean commit)
	uv run python entrypoint/holdout.py run

dashboard:  ## The research dashboard on http://127.0.0.1:8050 (read-only view of saved results)
	uv run python -m dashboard.app

all: install research evaluate portfolio report reconcile results ## The whole development pipeline, end to end

lint:  ## Check style and formatting
	uv run ruff check .
	uv run ruff format --check .

format:  ## Reformat the code in place
	uv run ruff format .

test:  ## Run the test suite
	uv run pytest

ci:  ## What CI runs: lint, format check and tests, no data download or refit
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest -q

lock:  ## Refresh uv.lock after changing dependencies
	uv lock --upgrade

clean:  ## Remove caches and logs
	rm -rf .pytest_cache .ruff_cache logs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
