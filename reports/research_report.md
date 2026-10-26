# Stock return predictor: development results

Generated 2026-10-26 12:25 UTC from saved development results. Code version 6433345-dirty, config f33d4859198d. Development period 2005-01-01 to the day before 2020-01-01. **The holdout has not been evaluated:** 2020-01-01 to 2024-12-31 is untouched.

## Findings

- **Forecast.** The pooled information coefficient averaged 0.0059 over 5 CPCV paths (positive on 4), and the within-date rank IC 0.0047 (t = 0.78). The hit rate beat always predicting up by 0.0003.
- **Best strategy.** ml_long_short, net Sharpe 0.76, against 0.86 for equal_weight_hold; 95% interval of the difference [-0.36, 0.16].
- **Nothing distinguishably beats equal weight.** No strategy's 95% interval for the Sharpe difference against equal_weight_hold lies above zero.
- **Distinguishably worse than equal weight:** momentum, pairs, pair_INTC_MSFT, pair_AAPL_INTC, pair_AAPL_SPY, pair_JNJ_SPY, pair_INTC_JPM, pair_GE_INTC, pair_GE_SPY, pair_PG_SPY, pair_INTC_SPY, pair_JPM_SPY, pair_KO_PG (whole interval below zero).
- **Exposure.** ml_long_short was net long 0.92 of capital on average, so much of its return is market exposure rather than forecasting skill.
- **Costs.** Costs took 0.02 to 0.07 of annual Sharpe, the most from pair_JPM_SPY.
- **Allocations.** The best, minimum_variance, had net Sharpe 0.79 against 0.72 for monthly equal weight; no allocation beats equal weight distinguishably.
- **Deflated Sharpe.** Pre-registered headline: allocation_minimum_variance (annual Sharpe 0.785) against a luck threshold of 1.004 for N = 98 trials and V[SR] = 0.1582 (strategies_and_allocations): DSR 0.207. Across 24 to 98 trials and every variance scenario, the DSR of the selected strategy ranges from 0.207 (allocation_minimum_variance, all_fingerprints, strategies_and_allocations) to 0.638 (allocation_minimum_variance, pipeline_only, every_backtest_fingerprint). None reaches 0.95, so no selected Sharpe ratio is distinguishable from the best of that many unskilled trials.

## Forecast quality (research_random_forest_regression_fixed_horizon, trial 107)

| metric | value |
|---|---|
| pooled IC, mean over paths | 0.0059 |
| pooled IC, sd over paths | 0.0050 |
| paths with positive IC | 4 |
| paths | 5 |
| t-stat of yearly IC | 3.0664 |
| within-date rank IC, mean | 0.0047 |
| within-date rank IC, t-stat | 0.7825 |
| hit rate | 0.5178 |
| hit-rate edge over always-up | 0.0003 |
| labelled rows | 39985 |

Against naive forecasts:

| forecast | ic | hit_rate | edge_over_always_up | cs_spearman_mean |
|---|---|---|---|---|
| research_random_forest_regression_fixed_horizon | 0.0059 | 0.5178 | 0.0003 | 0.0047 |
| always_up | n/a | 0.5175 | 0.0000 | n/a |
| random | 0.0049 | 0.5035 | -0.0141 | 0.0022 |
| last_sign | -0.0277 | 0.4876 | -0.0300 | -0.0047 |
| reverse_last_sign | 0.0277 | 0.5124 | -0.0051 | 0.0047 |

## Strategies and benchmarks (mean over CPCV paths, net of costs)

Intervals are 95% paired block-bootstrap intervals for the Sharpe difference against equal_weight_hold, averaged over paths.

| strategy | kind | Sharpe gross | Sharpe net | Sharpe net, sd over paths | vs equal weight | 95% low | 95% high | net exposure | turnover / yr | cost / yr | max drawdown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| equal_weight_hold | benchmark | 0.857 | 0.857 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.069 | 0.000 | -0.478 |
| ml_long_short | backtest | 0.817 | 0.759 | 0.086 | -0.098 | -0.361 | 0.158 | 0.924 | 17.710 | 0.009 | -0.345 |
| ml_long_short_same_close | diagnostic | 0.813 | 0.756 | 0.106 | -0.101 | -0.367 | 0.155 | 0.925 | 17.709 | 0.009 | -0.319 |
| ml_long_only | backtest | 0.775 | 0.748 | 0.056 | -0.109 | -0.330 | 0.104 | 0.958 | 8.890 | 0.004 | -0.401 |
| ml_thresholded | backtest | 0.717 | 0.701 | 0.063 | -0.156 | -0.435 | 0.124 | 0.696 | 3.339 | 0.002 | -0.392 |
| buy_and_hold | benchmark | 0.560 | 0.560 | 0.000 | -0.298 | -0.564 | -0.027 | 1.000 | 0.069 | 0.000 | -0.552 |
| pair_MSFT_SPY | backtest | 0.432 | 0.371 | n/a | -0.486 | -1.001 | 0.019 | 0.001 | 0.728 | 0.000 | -0.015 |
| pair_JPM_MSFT | backtest | 0.293 | 0.265 | 0.392 | -0.593 | -1.341 | 0.054 | 0.001 | 0.696 | 0.000 | -0.042 |
| pair_GE_INTC | backtest | 0.079 | 0.056 | 0.005 | -0.801 | -1.466 | -0.152 | 0.002 | 0.681 | 0.000 | -0.063 |
| momentum | backtest | 0.100 | 0.049 | 0.000 | -0.808 | -1.386 | -0.277 | 0.093 | 5.656 | 0.005 | -0.354 |
| pair_JPM_SPY | backtest | 0.122 | 0.049 | 0.194 | -0.808 | -1.534 | -0.140 | -0.002 | 1.342 | 0.001 | -0.048 |
| pair_GE_SPY | backtest | 0.076 | 0.039 | 0.129 | -0.818 | -1.520 | -0.115 | 0.004 | 0.686 | 0.000 | -0.052 |
| pair_INTC_SPY | backtest | 0.038 | -0.013 | n/a | -0.870 | -1.567 | -0.308 | -0.002 | 0.711 | 0.000 | -0.028 |
| pair_KO_PG | backtest | -0.016 | -0.057 | 0.502 | -0.914 | -1.641 | -0.170 | -0.002 | 0.594 | 0.000 | -0.043 |
| pair_INTC_MSFT | backtest | -0.041 | -0.079 | 0.240 | -0.936 | -1.654 | -0.277 | -0.022 | 3.314 | 0.002 | -0.226 |
| pairs | backtest | -0.101 | -0.172 | 0.182 | -1.029 | -1.751 | -0.368 | -0.031 | 6.940 | 0.004 | -0.304 |
| pair_INTC_JPM | backtest | -0.187 | -0.213 | 0.167 | -1.070 | -1.784 | -0.383 | -0.010 | 1.233 | 0.001 | -0.148 |
| pair_PG_SPY | backtest | -0.177 | -0.213 | 0.240 | -1.070 | -1.804 | -0.378 | -0.007 | 0.630 | 0.000 | -0.050 |
| pair_AAPL_INTC | backtest | -0.198 | -0.221 | 0.173 | -1.078 | -1.781 | -0.388 | 0.002 | 1.216 | 0.001 | -0.174 |
| pair_AAPL_SPY | backtest | -0.199 | -0.225 | 0.470 | -1.082 | -1.784 | -0.352 | 0.002 | 0.557 | 0.000 | -0.085 |
| pair_JNJ_SPY | backtest | -0.205 | -0.254 | 0.161 | -1.111 | -1.944 | -0.355 | -0.011 | 0.991 | 0.001 | -0.070 |

## Allocations (monthly rebalanced, one development path)

| allocation | Sharpe net | vs equal weight | 95% low | 95% high | share of resamples beating | max drawdown | turnover / yr |
|---|---|---|---|---|---|---|---|
| equal_weight | 0.724 | 0.000 | 0.000 | 0.000 | 0.000 | -0.460 | 0.498 |
| inverse_volatility | 0.741 | 0.017 | -0.044 | 0.082 | 0.718 | -0.425 | 1.119 |
| risk_parity | 0.749 | 0.025 | -0.031 | 0.083 | 0.822 | -0.420 | 0.584 |
| minimum_variance | 0.785 | 0.061 | -0.211 | 0.315 | 0.708 | -0.331 | 1.662 |
| maximum_sharpe | 0.604 | -0.120 | -0.414 | 0.137 | 0.176 | -0.402 | 5.695 |

## Trials

| kind | executions | unique_trials | names | counts_towards_N |
|---|---|---|---|---|
| backtest | 102 | 54 | 23 | yes |
| benchmark | 13 | 5 | 3 | no |
| diagnostic | 15 | 12 | 11 | no |
| notebook | 42 | 42 | 42 | yes |
| research | 3 | 2 | 1 | yes |

Where the selection trials came from:

| kind | code_version | executions |
|---|---|---|
| backtest | 6433345-dirty | 18 |
| backtest | 70fb742-dirty | 8 |
| backtest | c8ff091-dirty | 36 |
| backtest | e69b687-dirty | 4 |
| backtest | f80baef-dirty | 36 |
| notebook | notebook, not recorded | 42 |
| research | 6433345-dirty | 1 |
| research | unrecorded | 2 |

Defensible values of N:

| scenario | n_trials | pipeline | notebook | rule |
|---|---|---|---|---|
| all_fingerprints | 98 | 56 | 42 | every distinct (kind, fingerprint), including versions superseded by bug fixes; upper bound |
| distinct_choices | 66 | 24 | 42 | versions of one named trial merged: a corrected rerun is the same choice |
| distinct_less_overlap | 63 | 24 | 39 | as distinct_choices, less 3 notebook experiments the pipeline re-ran (notebook04_model_random_forest_spy, notebook04_rule_binary_long_short, notebook04_rule_long_only) |
| pipeline_only | 24 | 24 | 0 | pipeline trials only; ignores the notebook search, so understates N |

## Deflated Sharpe ratio

### Headline

Fixed in advance by reports/methodology/trial_accounting.md (rules 2 and 3): N counts every distinct fingerprint (all_fingerprints); V[SR] is the largest of the pre-registered options (the trial log's variance, the current candidates' with and without the allocations, and the selected strategy's own sampling variance), the one giving the highest luck threshold. A minimum track record of inf means the observed Sharpe is below the luck threshold, so no length of record at that Sharpe would clear it.

| item | value |
|---|---|
| selected | allocation_minimum_variance |
| n_trials | 98 |
| variance_strategies_annual | 0.1175 |
| variance_strategies_and_allocations_annual | 0.1582 |
| variance_every_backtest_fingerprint_annual | 0.1216 |
| variance_own_sampling_variance_annual | 0.0715 |
| variance_used | strategies_and_allocations |
| variance_used_annual | 0.1582 |
| sharpe_period | 0.0495 |
| n | 3504 |
| skew | 0.3668 |
| kurtosis | 20.8928 |
| paths | 1 |
| sharpe_annual | 0.7851 |
| luck_threshold_annual | 1.0036 |
| psr_vs_zero | 0.9983 |
| dsr | 0.2070 |
| min_track_record_years | inf |
| n_where_dsr_below_0.95 | 4 |

### Correlated trials

The DSR assumes independent trials. Clustering the current candidates' daily net returns (average linkage, cut at correlation 0.5) and the participation ratio of their correlation matrix estimate how many independent candidates they amount to. The other trials are kept at their counted value. This sits beside the headline and does not replace it.

| estimate | candidates | independent_candidates | other_trials | n_trials | sharpe_annual | luck_threshold_annual | psr_vs_zero | dsr | min_track_record_years |
|---|---|---|---|---|---|---|---|---|---|
| clusters | 22 | 15 | 41 | 56 | 0.785 | 0.922 | 0.998 | 0.304 | inf |
| participation | 22 | 8.200 | 41 | 49 | 0.785 | 0.902 | 0.998 | 0.331 | inf |

### Every scenario

Variance of annual net Sharpe ratios across candidates (benchmarks and notebook records excluded; notebook records have no verifiable development Sharpe):

| scenario | candidates | variance_annual | best | source |
|---|---|---|---|---|
| strategies | 18 | 0.1175 | ml_long_short | saved development results |
| strategies_and_allocations | 22 | 0.1582 | allocation_minimum_variance | saved development results |
| every_backtest_fingerprint | 54 | 0.1216 | allocation_minimum_variance | trial log, superseded versions included |

DSR of each variance scenario's best candidate, under each trial count. The luck threshold is the expected maximum annual Sharpe of N unskilled trials.

| variance / count scenario | selected | n_trials | sharpe_annual | luck_threshold_annual | psr_vs_zero | dsr |
|---|---|---|---|---|---|---|
| strategies / all_fingerprints | ml_long_short | 98 | 0.759 | 0.865 | 0.998 | 0.343 |
| strategies / distinct_choices | ml_long_short | 66 | 0.759 | 0.816 | 0.998 | 0.414 |
| strategies / distinct_less_overlap | ml_long_short | 63 | 0.759 | 0.810 | 0.998 | 0.423 |
| strategies / pipeline_only | ml_long_short | 24 | 0.759 | 0.679 | 0.998 | 0.620 |
| strategies_and_allocations / all_fingerprints | allocation_minimum_variance | 98 | 0.785 | 1.004 | 0.998 | 0.207 |
| strategies_and_allocations / distinct_choices | allocation_minimum_variance | 66 | 0.785 | 0.947 | 0.998 | 0.273 |
| strategies_and_allocations / distinct_less_overlap | allocation_minimum_variance | 63 | 0.785 | 0.940 | 0.998 | 0.281 |
| strategies_and_allocations / pipeline_only | allocation_minimum_variance | 24 | 0.785 | 0.787 | 0.998 | 0.497 |
| every_backtest_fingerprint / all_fingerprints | allocation_minimum_variance | 98 | 0.785 | 0.880 | 0.998 | 0.361 |
| every_backtest_fingerprint / distinct_choices | allocation_minimum_variance | 66 | 0.785 | 0.830 | 0.998 | 0.433 |
| every_backtest_fingerprint / distinct_less_overlap | allocation_minimum_variance | 63 | 0.785 | 0.824 | 0.998 | 0.442 |
| every_backtest_fingerprint / pipeline_only | allocation_minimum_variance | 24 | 0.785 | 0.690 | 0.998 | 0.638 |

Number of trials at which the DSR falls below each level:

| variance_scenario | selected | N where DSR < 0.95 | N where DSR < 0.5 |
|---|---|---|---|
| strategies | ml_long_short | 4 | 43 |
| strategies_and_allocations | allocation_minimum_variance | 4 | 24 |
| every_backtest_fingerprint | allocation_minimum_variance | 4 | 47 |

## Limitations

- **Survivorship.** The universe is 11 tickers (SPY, AAPL, MSFT, JNJ, JPM, XOM, PG, KO, WMT, GE, INTC) known to have survived to 2024-12-31; bias status in the config: acknowledged. Every backtest is flattered by it, the long books most.
- **One model, one universe.** The strategies share one random forest's forecasts, so their results are not independent evidence.
- **Trial dependence.** The DSR assumes independent trials. The trials here are strongly correlated (shared forecasts, shared pair legs, the same eleven stocks), so the effective N is smaller than any count above and those DSRs are conservative. The raw count is not evidence that the trials are independent.
- **Notebook trials.** The 42 notebook experiments are recorded from their saved outputs only. Their code and settings are not recoverable, they ran on data to 2024, and none has a development Sharpe ratio, so they enter N but not V[SR].
- **DSR meaning.** The DSR is the probability, under its assumptions, that the selected strategy's true Sharpe ratio over the development sample exceeded the luck threshold. It is not a probability of future profit.
- **Holdout.** Everything here uses data before 2020-01-01. The holdout has not been evaluated.
