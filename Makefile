# Stock Return Predictor
#
# uv manages the environment, so every target runs through `uv run` and needs
# no virtualenv to be activated first.

.DEFAULT_GOAL := help
.PHONY: help setup fetch eda portfolio lint format test ci lock clean

help:  ## Show the available targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the environment and install everything
	uv sync

fetch:  ## Pull and cache the price data
	uv run python entrypoint/fetch.py

eda:  ## Rebuild the exploratory figures and the stationarity table
	uv run python -m src.reporting.figures
	uv run python -m src.reporting.stationarity

portfolio:  ## Allocation, frontier and risk analysis on the development period
	uv run python -m src.pipelines.portfolio

lint:  ## Check style and formatting
	uv run ruff check .
	uv run ruff format --check .

format:  ## Reformat the code in place
	uv run ruff format .

test:  ## Run the test suite
	uv run pytest

ci:  ## Everything the pipeline runs, locally
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest -q

lock:  ## Refresh uv.lock after changing dependencies
	uv lock --upgrade

clean:  ## Remove caches and logs
	rm -rf .pytest_cache .ruff_cache logs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +