"""Mean-variance optimisation, the efficient frontier, and where it really lands.

The in-sample frontier always looks beautiful, because it was fitted to that
exact data. out_of_sample_frontier() is the function that matters: it fits
the frontier on a training window, then holds each frontier portfolio
through a later test window and records where it actually landed. The gap
between the curve and the realised points is the real lesson of Markowitz.

The inputs decide most of it:

- Expected returns are the noisiest estimate an optimiser takes. A few
  years of daily data cannot pin an asset's mean down to within several
  percent a year, and the optimiser bets hardest on exactly the errors.
  "shrunk" applies Jorion's (1986) Bayes-Stein estimator, pulling every mean
  toward the minimum-variance portfolio's return.
- The sample covariance of 11 assets is noisy enough to push the optimiser
  into two or three names. Ledoit and Wolf (2004) shrink it toward a
  structured target, which is better conditioned and gives weights that
  survive out of sample.

Everything is annualised with 252 trading days. Portfolios are long-only and
fully invested by default (config.portfolio.long_only); weights are capped at
config.backtest.max_position_size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.backtest.costs import TRADING_DAYS
from src.config import Config
from src.logger import get_logger
from src.reporting.figures import BLUE, DARK_GREY, GREY, ORANGE, _style

logger = get_logger(__name__)

EWMA_HALFLIFE = 63


class OptimisationError(Exception):
    """Raised when an optimisation cannot be solved as posed."""


@dataclass(frozen=True)
class Constraints:
    """What a portfolio may hold: weights sum to budget, each within bounds."""

    long_only: bool = True
    max_weight: float = 1.0
    budget: float = 1.0

    @classmethod
    def from_config(cls, config: Config) -> Constraints:
        """Long-only from config.portfolio, cap from config.backtest."""
        return cls(
            long_only=config.portfolio.long_only or not config.backtest.allow_short,
            max_weight=config.backtest.max_position_size,
        )

    def bounds(self, n: int) -> list[tuple[float, float]]:
        """Per-asset weight bounds for the solver."""
        low = 0.0 if self.long_only else -self.max_weight
        return [(low, self.max_weight)] * n


# Long-only, fully invested, no cap: the textbook problem.
DEFAULT_CONSTRAINTS = Constraints()


def expected_returns(
    returns: pd.DataFrame,
    method: str = "historical",
    days: int = TRADING_DAYS,
) -> pd.Series:
    """Annualised expected returns: the sample mean, or Bayes-Stein shrunk.

    The shrunk estimate is (1 - delta) * mean + delta * m0, where m0 is the
    minimum-variance portfolio's return and delta = (N + 2) / ((N + 2) +
    T (mean - m0)' Cov^-1 (mean - m0)): the noisier the means relative to
    their spread, the harder they are pulled together.
    """
    clean = returns.dropna()
    mean = clean.mean()

    if method == "historical":
        return mean * days
    if method != "shrunk":
        raise OptimisationError(f"unknown expected return method '{method}'")

    cov = clean.cov().to_numpy()
    inverse = np.linalg.pinv(cov)
    ones = np.ones(len(mean))
    w_mv = inverse @ ones / (ones @ inverse @ ones)
    m0 = float(w_mv @ mean.to_numpy())
    gap = mean.to_numpy() - m0
    n, t = len(mean), len(clean)
    delta = (n + 2) / ((n + 2) + t * gap @ inverse @ gap)

    return pd.Series((1 - delta) * mean.to_numpy() + delta * m0, index=mean.index) * days


def covariance_matrix(
    returns: pd.DataFrame,
    method: str = "sample",
    days: int = TRADING_DAYS,
    halflife: int = EWMA_HALFLIFE,
) -> pd.DataFrame:
    """Annualised covariance: sample, Ledoit-Wolf shrunk, or exponentially weighted."""
    clean = returns.dropna()

    if method == "sample":
        cov = clean.cov().to_numpy()
    elif method == "ledoit_wolf":
        cov = LedoitWolf().fit(clean.to_numpy()).covariance_
    elif method == "ewma":
        cov = clean.ewm(halflife=halflife).cov().iloc[-clean.shape[1] :].to_numpy()
    else:
        raise OptimisationError(f"unknown covariance method '{method}'")

    return pd.DataFrame(cov * days, index=clean.columns, columns=clean.columns)


def _solve(objective, n: int, constraints: Constraints, extra: list | None = None) -> np.ndarray:
    """Minimise an objective over portfolios that respect the constraints."""
    if constraints.max_weight * n < constraints.budget - 1e-12:
        raise OptimisationError("the weight cap makes the budget unreachable")

    rules = [{"type": "eq", "fun": lambda w: w.sum() - constraints.budget}, *(extra or [])]
    start = np.full(n, constraints.budget / n)
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=constraints.bounds(n),
        constraints=rules,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise OptimisationError(f"optimiser failed: {result.message}")

    return result.x


def portfolio_point(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame) -> tuple[float, float]:
    """Expected annual return and volatility of a portfolio."""
    w = weights.reindex(mu.index).fillna(0.0).to_numpy()

    return float(w @ mu.to_numpy()), float(np.sqrt(w @ cov.to_numpy() @ w))


def minimum_variance_portfolio(
    cov: pd.DataFrame,
    constraints: Constraints = DEFAULT_CONSTRAINTS,
) -> pd.Series:
    """The fully invested portfolio with the lowest variance."""
    sigma = cov.to_numpy()
    weights = _solve(lambda w: w @ sigma @ w, len(sigma), constraints)

    return pd.Series(weights, index=cov.index, name="minimum_variance")


def maximum_sharpe_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free: float = 0.0,
    constraints: Constraints = DEFAULT_CONSTRAINTS,
) -> pd.Series:
    """The tangency portfolio: the highest expected excess return per unit of risk."""
    m, sigma = mu.to_numpy(), cov.to_numpy()

    def negative_sharpe(w):
        return -(w @ m - risk_free) / np.sqrt(w @ sigma @ w)

    weights = _solve(negative_sharpe, len(m), constraints)

    return pd.Series(weights, index=mu.index, name="maximum_sharpe")


def _highest_return(mu: pd.Series, constraints: Constraints) -> float:
    """The largest expected return the constraints allow (fill the best names to the cap)."""
    remaining, total = constraints.budget, 0.0
    for value in mu.sort_values(ascending=False):
        take = min(constraints.max_weight, remaining)
        total += take * value
        remaining -= take
        if remaining <= 1e-12:
            break

    return total


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 30,
    constraints: Constraints = DEFAULT_CONSTRAINTS,
    risk_free: float = 0.0,
) -> pd.DataFrame:
    """The minimum-variance portfolio for each target return, low to high.

    Returns:
        One row per point: target_return, volatility, sharpe, then one weight
        column per asset. The first row is the minimum-variance portfolio.
    """
    sigma, m = cov.to_numpy(), mu.to_numpy()
    w_min = minimum_variance_portfolio(cov, constraints)
    low, _ = portfolio_point(w_min, mu, cov)
    high = _highest_return(mu, constraints)
    rows = []

    for target in np.linspace(low, high, n_points):
        hit = {"type": "eq", "fun": lambda w, target=target: w @ m - target}
        w = (
            w_min.to_numpy()
            if target == low
            else _solve(lambda w: w @ sigma @ w, len(m), constraints, [hit])
        )
        volatility = float(np.sqrt(w @ sigma @ w))
        rows.append(
            {
                "target_return": float(w @ m),
                "volatility": volatility,
                "sharpe": (float(w @ m) - risk_free) / volatility,
                **dict(zip(mu.index, w, strict=True)),
            }
        )

    return pd.DataFrame(rows)


@dataclass
class OutOfSampleFrontier:
    """A frontier fitted on one window and what its portfolios did in the next."""

    train_period: tuple[str, str]
    test_period: tuple[str, str]
    frontier: pd.DataFrame
    portfolios: pd.DataFrame
    assets: pd.DataFrame
    weights: dict[str, pd.Series] = field(default_factory=dict)


def _realised(weights: np.ndarray, returns: pd.DataFrame, days: int) -> tuple[float, float]:
    """Annual mean return and volatility of fixed weights, rebalanced daily, over a window."""
    series = returns.to_numpy() @ weights

    return float(series.mean() * days), float(series.std(ddof=1) * np.sqrt(days))


def out_of_sample_frontier(
    returns: pd.DataFrame,
    config: Config,
    split: str | pd.Timestamp | None = None,
    n_points: int = 20,
    days: int = TRADING_DAYS,
) -> OutOfSampleFrontier:
    """Fit on the training window, then see where the weights land in the test window.

    Both windows must lie before validation.test_start: this is research on
    the development period and may not touch the holdout.

    Args:
        returns: Daily asset returns.
        config: Supplies the estimation methods, constraints and holdout date.
        split: First test date; the middle of the data when None.
        n_points: Frontier portfolios.
        days: Periods per year.

    Returns:
        The training frontier with each point's realised test return and
        volatility, the same for equal weight, minimum variance and maximum
        Sharpe, and each asset's estimated and realised figures.

    Raises:
        OptimisationError: If the data reaches into the holdout.
    """
    clean = returns.dropna()
    holdout = pd.Timestamp(config.validation.test_start)
    if clean.index.max() >= holdout:
        raise OptimisationError(f"returns reach into the holdout starting {holdout.date()}")

    split = pd.Timestamp(split) if split is not None else clean.index[len(clean) // 2]
    train, test = clean[clean.index < split], clean[clean.index >= split]
    settings = config.portfolio
    constraints = Constraints.from_config(config)

    mu = expected_returns(train, settings.expected_return_method, days)
    cov = covariance_matrix(train, settings.covariance_method, days)
    frontier = efficient_frontier(mu, cov, n_points, constraints, settings.risk_free_rate)

    realised = [
        _realised(row[mu.index].to_numpy(dtype=float), test, days) for _, row in frontier.iterrows()
    ]
    frontier["realised_return"] = [r for r, _ in realised]
    frontier["realised_volatility"] = [v for _, v in realised]

    weights = {
        "equal_weight": pd.Series(1 / len(mu), index=mu.index),
        "minimum_variance": minimum_variance_portfolio(cov, constraints),
        "maximum_sharpe": maximum_sharpe_portfolio(mu, cov, settings.risk_free_rate, constraints),
    }
    rows = {}
    for name, w in weights.items():
        expected, volatility = portfolio_point(w, mu, cov)
        actual, actual_vol = _realised(w.to_numpy(), test, days)
        rows[name] = {
            "expected_return": expected,
            "expected_volatility": volatility,
            "realised_return": actual,
            "realised_volatility": actual_vol,
            "realised_sharpe": (actual - settings.risk_free_rate) / actual_vol,
            "largest_weight": float(w.max()),
            "names_above_5pct": int((w > 0.05).sum()),
        }

    assets = pd.DataFrame(
        {
            "expected_return": mu,
            "expected_volatility": np.sqrt(np.diag(cov.to_numpy())),
            "realised_return": test.mean() * days,
            "realised_volatility": test.std() * np.sqrt(days),
        }
    )
    logger.info(
        "Out-of-sample frontier: trained %s to %s, tested %s to %s",
        train.index[0].date(),
        train.index[-1].date(),
        test.index[0].date(),
        test.index[-1].date(),
    )

    return OutOfSampleFrontier(
        train_period=(str(train.index[0].date()), str(train.index[-1].date())),
        test_period=(str(test.index[0].date()), str(test.index[-1].date())),
        frontier=frontier,
        portfolios=pd.DataFrame(rows).T,
        assets=assets,
        weights=weights,
    )


def frontier_figure(
    frontier: pd.DataFrame,
    assets: pd.DataFrame,
    portfolios: pd.DataFrame | None = None,
    title: str = "Efficient frontier",
) -> Figure:
    """The frontier as fitted, and where its portfolios landed out of sample.

    Args:
        frontier: Output of efficient_frontier, optionally with realised_*
            columns from out_of_sample_frontier.
        assets: expected_return and expected_volatility per asset (and
            realised_* columns, which are not drawn, to keep the chart readable).
        portfolios: Named portfolios with expected_* and realised_* columns.
        title: Figure title.

    Returns:
        The figure, returns and volatility in percent a year.
    """
    with _style():
        figure = Figure(figsize=(8.5, 5.5))
        axis = figure.add_subplot()

        axis.plot(
            frontier["volatility"] * 100,
            frontier["target_return"] * 100,
            color=BLUE,
            linewidth=2,
            label="Frontier as fitted (training window)",
        )
        if "realised_return" in frontier:
            axis.plot(
                frontier["realised_volatility"] * 100,
                frontier["realised_return"] * 100,
                color=ORANGE,
                linewidth=2,
                marker="o",
                markersize=4,
                label="Same portfolios, test window",
            )

        axis.scatter(
            assets["expected_volatility"] * 100,
            assets["expected_return"] * 100,
            color=GREY,
            s=28,
            zorder=3,
            label="Single assets (training window)",
        )
        for ticker, row in assets.iterrows():
            axis.annotate(
                ticker,
                (row["expected_volatility"] * 100, row["expected_return"] * 100),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
                color=DARK_GREY,
            )

        if portfolios is not None:
            for number, (name, row) in enumerate(portfolios.iterrows()):
                label = name.replace("_", " ")
                axis.scatter(
                    row["expected_volatility"] * 100,
                    row["expected_return"] * 100,
                    color=BLUE,
                    s=44,
                    marker="D",
                    zorder=4,
                    label="Named portfolios, as fitted" if number == 0 else None,
                )
                axis.annotate(
                    label,
                    (row["expected_volatility"] * 100, row["expected_return"] * 100),
                    xytext=(6, 2),
                    textcoords="offset points",
                    fontsize=8,
                    color=BLUE,
                )
                if "realised_return" in row:
                    axis.scatter(
                        row["realised_volatility"] * 100,
                        row["realised_return"] * 100,
                        color=ORANGE,
                        s=44,
                        marker="D",
                        zorder=4,
                        label="Named portfolios, test window" if number == 0 else None,
                    )
                    axis.annotate(
                        label,
                        (row["realised_volatility"] * 100, row["realised_return"] * 100),
                        xytext=(5, -10),
                        textcoords="offset points",
                        fontsize=8,
                        color=DARK_GREY,
                    )

        axis.set_xlabel("Annual volatility (%)")
        axis.set_ylabel("Annual return (%)")
        axis.set_title(title)
        axis.legend(loc="lower right", fontsize=8)

    return figure
