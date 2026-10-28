"""The research dashboard: a read-only view of the saved results.

Every number it shows is read from a file under reports/results, data/ or the
trial log. It never fits a model, runs a backtest, logs a trial or opens the
holdout. Run it with `python -m dashboard.app`.
"""
