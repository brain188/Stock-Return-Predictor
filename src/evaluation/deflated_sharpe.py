"""The probabilistic and deflated Sharpe ratios: how much a Sharpe ratio is worth.

A Sharpe ratio estimated from a backtest is inflated twice. Non-normal
returns (negative skew, fat tails) make it noisier than the textbook
formula assumes, and selection makes it biased: try enough strategies and
the best one looks good even if every one of them is worthless, because the
maximum of many noisy estimates is not near zero. This module corrects for
both, following Bailey and López de Prado:

- PSR, the probabilistic Sharpe ratio (Bailey and López de Prado, 2012):
  the probability that the true Sharpe ratio exceeds a benchmark, given the
  estimate, the sample length, and the skewness and kurtosis of the returns.
- E[max SR], the expected maximum Sharpe ratio of N independent trials whose
  true Sharpe ratio is zero, from extreme value theory (Bailey and López de
  Prado, 2014, "The Deflated Sharpe Ratio", Journal of Portfolio Management
  40(5)). This is the luck threshold: the bar a selected strategy must clear.
- DSR, the deflated Sharpe ratio (same paper): the PSR with the luck
  threshold as the benchmark.

Conventions, used by every function here:

- Sharpe ratios are per period, not annualised: the mean over the standard
  deviation of the returns at their own frequency (daily here). Annualised
  figures are converted with to_period_sharpe() and back with
  to_annual_sharpe(); the paper's own example uses 250 periods a year, this
  project 252.
- n (or T) is the number of return observations behind the selected Sharpe.
- skew is the sample skewness; kurtosis is the raw (non-excess) kurtosis, so
  a normal distribution has kurtosis 3. pandas' Series.kurt() returns excess
  kurtosis: add 3.
- The variance of Sharpe ratios is the variance across trials of their
  per-period Sharpe estimates.
- The benchmark is a per-period Sharpe ratio; 0 asks "better than nothing".

What the numbers mean: PSR and DSR are probabilities that the true Sharpe
ratio exceeded the benchmark over the sample, under the model's assumptions.
They are not probabilities of future profit.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

# Euler-Mascheroni constant, from the expected maximum of Gumbel-like extremes.
EULER_MASCHERONI = 0.5772156649015329


class SharpeError(ValueError):
    """Raised for inputs that make a Sharpe statistic undefined."""


def to_period_sharpe(annual: float, periods: int = 252) -> float:
    """An annualised Sharpe ratio converted to one period: annual / sqrt(periods)."""
    return annual / math.sqrt(periods)


def to_annual_sharpe(per_period: float, periods: int = 252) -> float:
    """A per-period Sharpe ratio annualised: per_period * sqrt(periods)."""
    return per_period * math.sqrt(periods)


def _sharpe_dispersion(sharpe: float, skew: float, kurtosis: float) -> float:
    """1 - skew SR + (kurtosis - 1) / 4 SR^2: the non-normal variance factor of the SR estimate.

    For normal returns (skew 0, kurtosis 3) it reduces to 1 + SR^2 / 2, Lo's
    (2002) result; negative skew and fat tails make it larger.
    """
    return 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2


def probabilistic_sharpe_ratio(
    sharpe: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark: float = 0.0,
) -> float:
    """Probability that the true Sharpe ratio exceeds the benchmark.

    PSR = Z[(SR - SR*) sqrt(n - 1) / sqrt(1 - skew SR + (kurtosis - 1)/4 SR^2)],
    with Z the standard normal distribution function (Bailey and López de
    Prado, 2012, eq. 2 of the 2014 paper). The sqrt(n - 1) is the
    finite-sample correction of the estimator's standard error.

    Args:
        sharpe: Estimated per-period Sharpe ratio.
        n: Number of returns behind it, at least 2.
        skew: Skewness of those returns.
        kurtosis: Raw kurtosis of those returns (3 for normal).
        benchmark: Per-period Sharpe ratio to beat.

    Returns:
        A probability in [0, 1].

    Raises:
        SharpeError: For n < 2, non-finite inputs, kurtosis below 1 (no
            distribution has that), or a non-positive variance factor.
    """
    values = (sharpe, skew, kurtosis, benchmark)
    if not all(math.isfinite(v) for v in values):
        raise SharpeError(f"inputs must be finite, got {values}")
    if n < 2:
        raise SharpeError(f"need at least 2 observations, got {n}")
    if kurtosis < 1:
        raise SharpeError(f"kurtosis is raw (normal = 3) and cannot be below 1, got {kurtosis}")

    dispersion = _sharpe_dispersion(sharpe, skew, kurtosis)
    if dispersion <= 0:
        raise SharpeError(
            f"the variance factor 1 - skew SR + (kurtosis - 1)/4 SR^2 = {dispersion:.4g} "
            "is not positive: the inputs are inconsistent"
        )

    z = (sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(dispersion)

    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe ratio of n_trials independent trials whose true SR is 0.

    E[max SR] ~ sqrt(V) [(1 - g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e))], with g
    the Euler-Mascheroni constant and Z^-1 the inverse standard normal
    distribution function (Bailey and López de Prado, 2014).

    Assumptions and limits of the approximation:

    - The trials are independent and their Sharpe estimates roughly normal
      with variance V. Correlated trials behave like fewer independent ones,
      so using the raw count overstates the bar (a conservative error).
    - It is an asymptotic extreme-value approximation, least accurate for
      small N; with one trial the expected maximum of one zero-mean draw is 0.
    - V is estimated from the same trials, so it is itself uncertain.

    Args:
        n_trials: Number of independent trials, at least 1.
        sharpe_variance: Variance across trials of their per-period Sharpe
            ratios (or annualised, if every other input is annualised too).

    Returns:
        The expected maximum, in the units of sqrt(sharpe_variance).

    Raises:
        SharpeError: For fewer than 1 trial or a negative or non-finite variance.
    """
    if n_trials < 1:
        raise SharpeError(f"need at least 1 trial, got {n_trials}")
    if not math.isfinite(sharpe_variance) or sharpe_variance < 0:
        raise SharpeError(f"variance must be finite and non-negative, got {sharpe_variance}")
    if n_trials == 1 or sharpe_variance == 0:
        return 0.0

    quantile = stats.norm.ppf
    bracket = (1 - EULER_MASCHERONI) * quantile(1 - 1 / n_trials) + EULER_MASCHERONI * quantile(
        1 - 1 / (n_trials * math.e)
    )

    return float(math.sqrt(sharpe_variance) * bracket)


def luck_threshold(n_trials: int, variance: float) -> float:
    """The Sharpe ratio the best of n_trials worthless strategies reaches on average.

    The same quantity as expected_max_sharpe: a selected strategy has to
    clear this bar, not zero, before its Sharpe ratio says anything.
    """
    return expected_max_sharpe(n_trials, variance)


def deflated_sharpe_ratio(
    sharpe: float,
    trials: int,
    n: int,
    skew: float,
    kurtosis: float,
    variance: float,
) -> float:
    """Probability that the selected strategy's true Sharpe beats the luck threshold.

    DSR = PSR with benchmark E[max SR] for `trials` trials of Sharpe variance
    `variance`. A Sharpe ratio at or below the luck threshold gives a DSR at
    or below about 0.5; far below it, close to 0.

    Args:
        sharpe: The selected strategy's per-period Sharpe ratio.
        trials: Number of independent trials searched.
        n: Number of returns behind the selected Sharpe.
        skew: Skewness of the selected strategy's returns.
        kurtosis: Raw kurtosis of the selected strategy's returns.
        variance: Variance across trials of the per-period Sharpe ratios.

    Returns:
        A probability in [0, 1].
    """
    return probabilistic_sharpe_ratio(
        sharpe, n, skew, kurtosis, benchmark=expected_max_sharpe(trials, variance)
    )


def minimum_track_record_length(
    sharpe: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    target: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed for the Sharpe ratio to beat `target` at `confidence`.

    MinTRL = 1 + (1 - skew SR + (kurtosis - 1)/4 SR^2) (Z^-1(confidence) / (SR - target))^2
    (Bailey and López de Prado, 2012). It is the n at which the PSR against
    `target` reaches `confidence`.

    Returns:
        The number of observations, or infinity when the Sharpe ratio does
        not exceed the target (no amount of data would make it significant).

    Raises:
        SharpeError: For a confidence outside (0, 1) or a non-positive
            variance factor.
    """
    if not 0 < confidence < 1:
        raise SharpeError(f"confidence must be between 0 and 1, got {confidence}")
    if sharpe <= target:
        return math.inf

    dispersion = _sharpe_dispersion(sharpe, skew, kurtosis)
    if dispersion <= 0:
        raise SharpeError("the variance factor is not positive: the inputs are inconsistent")

    return float(1 + dispersion * (stats.norm.ppf(confidence) / (sharpe - target)) ** 2)


def return_moments(returns: np.ndarray) -> tuple[float, int, float, float]:
    """Per-period Sharpe, n, skewness and raw kurtosis of a return series, NaNs dropped.

    Uses the sample standard deviation (ddof = 1) and the bias-corrected
    sample skewness and kurtosis, matching pandas.

    Raises:
        SharpeError: With fewer than 3 returns or no variation.
    """
    values = np.asarray(returns, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 3:
        raise SharpeError(f"need at least 3 returns, got {len(values)}")

    std = values.std(ddof=1)
    if std == 0:
        raise SharpeError("returns do not vary: the Sharpe ratio is undefined")

    skew = float(stats.skew(values, bias=False))
    kurtosis = float(stats.kurtosis(values, fisher=False, bias=False))

    return float(values.mean() / std), len(values), skew, kurtosis
