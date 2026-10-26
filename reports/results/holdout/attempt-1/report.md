# Holdout evaluation, attempt 1

Runner commit b3891469baf3cc9365890190a8e6a846fb5efdda, frozen code 20cdb7a, freeze commit 65263c7, declaration SHA-256 84732cdb2453d4585039ec1b4fbef9428a29da55b487f5b3cc77dbf44633c209, config f33d4859198d.

Window: decision date 2020-01-02; every book flat until 2020-01-03, when it trades to its target; first return earned 2020-01-06; last date 2024-12-30. Missing return cells set to zero: 0.

## Primary decision (predeclared)

Primary: `allocation_minimum_variance`. Rule: DSR above 0.95 using the frozen N = 98 and V[SR] = 0.1582 a year, 0.000628 a day (`variance_scenarios.strategies_and_allocations.variance_period`), and the paired 95% interval of its net Sharpe minus `equal_weight_hold`'s entirely above zero.

| primary | value |
|---|---|
| net Sharpe (annual) | 0.537 |
| PSR against 0 | 0.884 |
| luck threshold (annual) | 1.004 |
| DSR | 0.149 |
| Sharpe difference vs equal_weight_hold | -0.261 |
| 95% interval low | -0.737 |
| 95% interval high | 0.138 |

**Conclusion: no convincing evidence of an edge.**

The full frozen trial count N = 98 deflates the single predeclared primary. This is a deliberately conservative choice: the selection happened on development data, and the rule charges the holdout result for it again.

Neither the PSR nor the DSR is a probability of future profit. Each is the probability, under the method's assumptions, that the true Sharpe ratio over the holdout period exceeded a benchmark (zero, or the luck threshold for N trials).

## Every declared book (descriptive)

Secondary candidates and benchmarks are reported in full. They do not replace the primary or change the rule.

| book | role | days | net_sharpe | gross_sharpe | net_annual_return | net_annual_volatility | net_max_drawdown | net_sortino | sharpe_standard_error | psr_vs_zero | turnover_annual | cost_annual | exposure_net_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| allocation_minimum_variance | primary | 1256 | 0.537 | 0.543 | 0.078 | 0.165 | -0.254 | 0.770 | 0.449 | 0.884 | 1.977 | 0.001 | 1 |
| allocation_inverse_volatility | secondary | 1256 | 0.774 | 0.777 | 0.134 | 0.185 | -0.295 | 1.106 | 0.452 | 0.957 | 1.266 | 0.001 | 1 |
| allocation_risk_parity | secondary | 1256 | 0.790 | 0.792 | 0.137 | 0.184 | -0.298 | 1.130 | 0.452 | 0.960 | 0.794 | 0.000 | 1 |
| allocation_maximum_sharpe | secondary | 1256 | 0.975 | 0.985 | 0.229 | 0.241 | -0.304 | 1.438 | 0.451 | 0.985 | 4.993 | 0.002 | 1 |
| ml_long_short | secondary | 1256 | 0.753 | 0.879 | 0.113 | 0.160 | -0.196 | 1.060 | 0.457 | 0.950 | 39.150 | 0.020 | 0.787 |
| ml_long_only | secondary | 1256 | 0.834 | 0.895 | 0.129 | 0.161 | -0.255 | 1.182 | 0.457 | 0.966 | 19.672 | 0.010 | 0.873 |
| ml_thresholded | secondary | 1256 | 0.332 | 0.371 | 0.039 | 0.149 | -0.201 | 0.462 | 0.451 | 0.769 | 8.066 | 0.006 | 0.390 |
| momentum | secondary | 1256 | 0.229 | 0.262 | 0.023 | 0.143 | -0.348 | 0.318 | 0.449 | 0.695 | 4.908 | 0.005 | 0.134 |
| pairs | secondary | 1256 | 0.407 | 0.456 | 0.049 | 0.142 | -0.234 | 0.624 | 0.444 | 0.820 | 11.793 | 0.007 | -0.032 |
| buy_and_hold | benchmark | 1256 | 0.752 | 0.752 | 0.145 | 0.210 | -0.337 | 1.051 | 0.455 | 0.951 | 0.201 | 0.000 | 1 |
| equal_weight_hold | benchmark | 1256 | 0.798 | 0.798 | 0.148 | 0.198 | -0.314 | 1.132 | 0.453 | 0.961 | 0.201 | 0.000 | 1.000 |
| equal_weight | benchmark | 1256 | 0.772 | 0.773 | 0.143 | 0.199 | -0.319 | 1.098 | 0.453 | 0.956 | 0.712 | 0.000 | 1 |

## Paired comparisons with each benchmark

Only `allocation_minimum_variance` against `equal_weight_hold` enters the decision. Its comparison with `equal_weight` (the development reference) is descriptive.

| book | benchmark | role | difference | low | high | share_positive |
|---|---|---|---|---|---|---|
| allocation_minimum_variance | buy_and_hold | descriptive | -0.215 | -0.758 | 0.321 | 0.199 |
| allocation_minimum_variance | equal_weight_hold | decision | -0.261 | -0.737 | 0.138 | 0.104 |
| allocation_minimum_variance | equal_weight | development reference, descriptive | -0.235 | -0.738 | 0.202 | 0.145 |
| allocation_inverse_volatility | buy_and_hold | descriptive | 0.022 | -0.255 | 0.335 | 0.615 |
| allocation_inverse_volatility | equal_weight_hold | descriptive | -0.024 | -0.154 | 0.094 | 0.367 |
| allocation_inverse_volatility | equal_weight | descriptive | 0.002 | -0.127 | 0.148 | 0.544 |
| allocation_risk_parity | buy_and_hold | descriptive | 0.038 | -0.251 | 0.362 | 0.641 |
| allocation_risk_parity | equal_weight_hold | descriptive | -0.008 | -0.139 | 0.119 | 0.480 |
| allocation_risk_parity | equal_weight | descriptive | 0.018 | -0.095 | 0.146 | 0.642 |
| allocation_maximum_sharpe | buy_and_hold | descriptive | 0.223 | -0.335 | 0.769 | 0.777 |
| allocation_maximum_sharpe | equal_weight_hold | descriptive | 0.177 | -0.375 | 0.670 | 0.734 |
| allocation_maximum_sharpe | equal_weight | descriptive | 0.203 | -0.391 | 0.746 | 0.747 |
| ml_long_short | buy_and_hold | descriptive | 0.001 | -0.594 | 0.495 | 0.502 |
| ml_long_short | equal_weight_hold | descriptive | -0.045 | -0.584 | 0.382 | 0.410 |
| ml_long_short | equal_weight | descriptive | -0.019 | -0.563 | 0.406 | 0.444 |
| ml_long_only | buy_and_hold | descriptive | 0.082 | -0.266 | 0.414 | 0.677 |
| ml_long_only | equal_weight_hold | descriptive | 0.036 | -0.236 | 0.242 | 0.565 |
| ml_long_only | equal_weight | descriptive | 0.062 | -0.154 | 0.239 | 0.678 |
| ml_thresholded | buy_and_hold | descriptive | -0.420 | -1.140 | 0.258 | 0.113 |
| ml_thresholded | equal_weight_hold | descriptive | -0.466 | -1.179 | 0.197 | 0.080 |
| ml_thresholded | equal_weight | descriptive | -0.440 | -1.170 | 0.257 | 0.089 |
| momentum | buy_and_hold | descriptive | -0.523 | -1.768 | 0.688 | 0.218 |
| momentum | equal_weight_hold | descriptive | -0.569 | -1.866 | 0.629 | 0.189 |
| momentum | equal_weight | descriptive | -0.543 | -1.892 | 0.708 | 0.220 |
| pairs | buy_and_hold | descriptive | -0.345 | -1.501 | 0.614 | 0.251 |
| pairs | equal_weight_hold | descriptive | -0.391 | -1.536 | 0.496 | 0.210 |
| pairs | equal_weight | descriptive | -0.365 | -1.502 | 0.575 | 0.233 |
| buy_and_hold | equal_weight_hold | descriptive | -0.046 | -0.308 | 0.163 | 0.297 |
| buy_and_hold | equal_weight | descriptive | -0.020 | -0.315 | 0.267 | 0.402 |
| equal_weight_hold | buy_and_hold | descriptive | 0.046 | -0.163 | 0.308 | 0.703 |
| equal_weight_hold | equal_weight | descriptive | 0.026 | -0.105 | 0.163 | 0.669 |
| equal_weight | buy_and_hold | descriptive | 0.020 | -0.267 | 0.315 | 0.598 |
| equal_weight | equal_weight_hold | descriptive | -0.026 | -0.163 | 0.105 | 0.331 |

## Limitations

- **Survivorship.** The universe is 11 tickers chosen with hindsight about which companies survived (survivorship bias, acknowledged in config). The result applies to this universe only and is not evidence about an investable universe.

- **One period, one path.** Five years, one realisation; regimes in it may not recur.

- **Costs are a model.** 2 + 3 bps per side and 0.5% a year to borrow; real fills and borrow availability can differ.

- **Readiness.** allocation_minimum_variance: weights from the 2019-12-31 rebalance; allocation_inverse_volatility: weights from the 2019-12-31 rebalance; allocation_risk_parity: weights from the 2019-12-31 rebalance; allocation_maximum_sharpe: weights from the 2019-12-31 rebalance; ml_long_short: forecasts from D0; ml_long_only: forecasts from D0; ml_thresholded: forecasts from D0; momentum: 12-1 signal defined at D0; pairs: 2 pair(s); buy_and_hold: constant target; equal_weight_hold: constant target; equal_weight: weights from the 2019-12-31 rebalance

- **Earlier attempts.** None.
