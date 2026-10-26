# Trial accounting for the deflated Sharpe ratio

How Phase 8 will count trials and estimate the variance of their Sharpe ratios,
written down before any deflated Sharpe is computed so the rules cannot be bent
to fit a result.

## What the deflated Sharpe needs

The deflated Sharpe ratio (Bailey and López de Prado, 2014) asks how good the
best of N tries would look by luck alone. It needs:

- **N**, how many configurations were tried before the one being judged was chosen;
- **V[SR]**, the variance of the Sharpe ratios across those tries;
- **T, skew and kurtosis** of the chosen strategy's own returns.

Undercounting N, or understating V[SR], lowers the bar the strategy has to clear.
Every rule below is therefore set on the conservative side.

## The three identities in the trial log

`reports/trial_log.csv` keeps three things apart (`src/evaluation/trial_log.py`):

| Identity | Columns | Used for |
| --- | --- | --- |
| Execution | `trial_id`, `timestamp`, `code_version` | Auditing: every attempt, reruns and failures included |
| Config file | `config_hash` | Provenance only, never counting |
| Trial | `kind` + `fingerprint` | Counting: the resolved settings that decide a result |

A rerun with the same settings is a second execution of the same trial, not a
second trial. `unique_trials()` flags any trial whose executions disagree
(`consistent = False`), which catches code or data changes made without a
settings change.

## Inventory of everything tried so far

### Before the pipeline: the notebooks (not in the trial log)

Notebook 4 counted these, and they shaped the pipeline's features and settings:

| Source | Trials |
| --- | --- |
| Features tested one at a time (notebook 3) | 17 |
| Fractional differencing orders searched (notebook 2) | 21 |
| Position rules (notebook 4) | 3 |
| Model configurations (notebook 4) | 1 |
| **Total** | **42** |

These cannot be turned into log rows with Sharpe ratios. Only notebook 4's three
position rules have one (net 0.28, −0.10, 0.70, using data to 2024). They enter N
as a fixed count of 42 and are left out of V[SR].

### In the trial log (as of the Phase 7 audit: 106 executions)

| Kind | Unique trials | Counted in N? | Why |
| --- | --- | --- | --- |
| `research` | 1 (2 executions) | **Yes** | It is a model-selection step, judged on IC; it has no Sharpe, so it is not in V[SR] |
| `backtest` | 36: 18 from the first evaluation, 14 re-selected pairs trials after the gap fix, and 4 Phase 7 allocation rules | **Yes, all of them** | Every strategy, every traded pair and every allocation rule is a candidate |
| `benchmark` | 3: SPY hold, equal-weight hold, monthly equal weight | No | Fixed reference points, never candidates for selection |
| `diagnostic` | 6 (5 embargo variants, 1 same-close) | No | They check the method, not the strategy. The embargo was fixed at 36 before its check ran; the same-close run is a known look-ahead and can never be selected |

**Two kinds of Sharpe sit in the backtest trials.** The 32 Phase 6 trials report
the mean over 5 CPCV paths of the period 2005-07-22 to 2019-12-31. The 4 Phase 7
allocation trials report one backtest from 2006-01-31 to 2019-12-31. Both are
annualised and net of costs, but they are not measured identically. Phase 8
reports V[SR] with and without the allocation trials, and uses the larger luck
threshold.

**Dependence between trials.** Treating the 36 as independent overstates the
independent evidence, because:

- The three ML rules share one forecast.
- The 13 pairs reuse the same 11 legs, and the 14 re-selected pairs trials mostly
  repeat the 13 before the fix.
- The four allocation rules all hold the same 11 names long-only.
- Every trial covers the same market years.

**Superseded trials stay counted.** The pairs trials from before the gap fix
(selection differenced across CPCV gaps) remain in N. They are a real search
that was run and seen, and fixing a bug does not un-see the result.

**Failed executions are not counted.** A configuration that crashed before
producing a result could not have been selected.

### Selection decisions made outside the log

| Decision | Made before or after seeing results | How it is counted |
| --- | --- | --- |
| The 11-ticker universe | Before, but chosen with hindsight about which companies survived | Not a trial; a bias, documented in `phase6_verification.md` |
| Feature set and fracdiff grid | After the notebook research | Covered by the 17 + 21 notebook trials |
| Labelling method (fixed horizon), RF settings | Config defaults, not tuned in the pipeline | Part of the research trial |
| Embargo of 36 trading dates | After the leakage analysis, before the sensitivity check | Diagnostic; the check found no effect |
| Threshold multiple 1.0, momentum 12-1 with 3/3, pairs z of 2/0.5/4, windows 252/63, half-life cap 126 | Textbook defaults, set before the first evaluation and not changed after it | Part of each strategy's trial |
| Which strategy goes to the holdout | Not yet made | **This is the selection the DSR corrects. N is frozen before it is made.** |

## Rules Phase 8 will apply

1. **Sharpe units.** Annualised, net of all costs. A trial's Sharpe is the mean
   of its CPCV path Sharpes (the `sharpe` metric in the log).
2. **N.** 42 (notebooks) + unique `research` trials + unique `backtest` trials,
   counting only trials that produced a result, with superseded versions included:
   `42 + log.unique_trial_count({"kind": "research"}) + log.unique_trial_count({"kind": "backtest"})`.
3. **V[SR].** The sample variance (ddof = 1) of the annualised net Sharpe across
   unique `backtest` trials, one value per trial (`log.sharpe_variance(filters={"kind": "backtest"})`).
   As a cross-check, the variance of the chosen strategy's own Sharpe across
   resampled paths (Phase 7) is computed as well. **Whichever gives the higher
   luck threshold is used.**
4. **Correlated trials.** The three ML rules share one forecast, and many pairs
   share legs, so the raw N overstates how many independent tries there were.
   That makes the expected-maximum Sharpe, and so the bar, too high: conservative.
   Phase 8 reports the raw-N result as the headline, and alongside it an effective
   N from clustering the trials' daily net returns (saved in
   `reports/results/evaluation/path_returns.parquet`), without replacing the headline.
5. **T, skew, kurtosis.** From the chosen strategy's daily net returns on the
   development period, using the daily (not annualised) Sharpe in the PSR formula.
6. **Freezing.** Before the holdout is run, Phase 8 writes N, V[SR] and the trial
   list to a frozen file under `reports/results/`. The single holdout run is then
   judged against the threshold from that file. It is logged as one more trial,
   and nothing tried after it can be counted as out of sample.

## What this cannot fix

- **Notebook trials are a count, not a record.** Their Sharpe ratios mostly don't
  exist, so V[SR] comes from pipeline trials only.
- **Unlogged thinking cannot be counted.** Choices considered and dropped without
  being run are not in N.
- **The fingerprint covers settings, not code or data.** A change to how a feature
  is computed, with the config unchanged, keeps the fingerprint; the
  `consistent` flag and `code_version` make that visible but do not prevent it.
- **Every execution so far ran on uncommitted code** (`code_version` ends in
  `-dirty`). Before the holdout, the code is committed and the freeze file
  records a clean commit.

## Phase 8: how the rules were applied (2026-10-26)

The rules above were written before any DSR was computed and are left as
written. This section records what Phase 8 did with them, including where it
departed from them. The numbers come from `reports/research_report.md`, which is
rebuilt from the saved results and the trial log (`make report`). Rebuild it
rather than trusting the figures quoted here.

### Changes to the trial log

- **The boundary refit added 27 rows** (trial_id 107 to 133): 1 research, 18
  backtest, 2 benchmark and 6 diagnostic. All 27 fingerprints are new, because
  `research_settings()` now records `development_boundary`. Rows 1 to 106 are
  unchanged, byte for byte against the pre-refit copy, and against HEAD apart
  from line endings.
- **The 42 notebook experiments are now in the log** (trial_id 134 to 175, kind
  `notebook`, code_version `notebook, not recorded`), parsed from the
  notebooks' saved outputs by `src/evaluation/notebook_trials.py`. Their
  metrics are only what the outputs printed: ADF statistic and correlation for
  21 differencing orders, IC and p-value for 17 features, gross and net Sharpe
  for 3 position rules, and IC and hit rate for the model. Every metric name
  carries its period (for example `sharpe_net_2008_2024`). They ran on data to
  2024 and none is a development Sharpe ratio. They count in N and never enter
  V[SR].
- **The count is 42, not "notebooks 3 and 4".** Notebook 4's tally of 42
  includes notebook 2's 21 differencing orders. Notebooks 3 and 4 alone hold
  21 experiments (17 + 3 + 1).

### Rule 2: N

Four defensible counts are reported. The headline is the pre-registered one.

| Scenario | N | Rule |
| --- | --- | --- |
| `all_fingerprints` (headline) | 98 | Every distinct research, backtest and notebook fingerprint, superseded versions included (56 pipeline + 42 notebook) |
| `distinct_choices` | 66 | Versions of one named trial merged (24 + 42) |
| `distinct_less_overlap` | 63 | Less the 3 notebook experiments the pipeline re-ran: binary long/short (`ml_long_short`), long only (`ml_long_only`), and the random forest (the research trial) |
| `pipeline_only` | 24 | Ignores the notebook search; a lower bound that understates N |

Notebook 4's "scaled by conviction" rule, the 21 differencing orders and the 17
feature tests have no pipeline counterpart, so they are not treated as overlaps.

### Rule 3: V[SR]

| Option | Candidates | Variance (annual) |
| --- | --- | --- |
| Current strategies, no allocations | 18 | 0.1175 |
| Current strategies and allocations | 22 | 0.1582 |
| Every backtest fingerprint in the log (rule 3) | 54 | 0.1216 |
| Selected strategy's own sampling variance (rule 3 cross-check) | 1 | 0.0715 |

The rules gave two instructions that point different ways. Rule 3 names the
log-wide variance, and the inventory section says "uses the larger luck
threshold" across the with- and without-allocation options. The headline takes
the largest of all four options, 0.1582, which is the most conservative reading
consistent with both.

**Departure from the rule.** Rule 3's cross-check asked for the spread of the
selected strategy's Sharpe across Phase 7's resampled paths. The saved Monte
Carlo table keeps only quantile intervals, and only for the Phase 6
strategies, not the allocations. The cross-check therefore uses the analytic
sampling variance, the PSR denominator (Mertens). As a sanity check for
`ml_long_short`: its bootstrap 95% range of 0.31 to 1.26 implies a variance of
about 0.06, close to the analytic figure. It is the smallest option either way,
so it does not set the headline.

Superseded versions cannot be selected, so the headline's selected strategy is
the best current candidate: `allocation_minimum_variance`, with an annual net
Sharpe of 0.785. Its moments come from its saved net returns, 2006-01-31 to
2019-12-31.

### The headline

| N | V[SR] | Luck threshold (annual) | Selected Sharpe | PSR vs 0 | DSR |
| --- | --- | --- | --- | --- | --- |
| 98 | 0.1582 | 1.004 | 0.785 | 0.998 | 0.207 |

Across every count and variance scenario, the DSR ranges from 0.207 to 0.638.
For every selected strategy it falls below 0.95 from N = 4. **No development
Sharpe ratio is distinguishable from the best of that many unskilled trials.**
This is about the past sample under the model's assumptions. It is not a
probability of future profit, nor of holdout performance.

### Rule 4: correlated trials

The raw counts are not evidence that the trials are independent. The 22
current candidates' daily net returns were averaged over CPCV paths, aligned
on shared dates (2006-01-31 to 2019-12-30), and clustered (average linkage on
sqrt((1 - rho)/2), cut at correlation 0.5). This gives 15 clusters, and a
participation ratio of 8.2 for their correlation matrix. With the other 41
trials kept at their counted value, the effective N is 56 or 49, and the DSR
is 0.30 or 0.33.

Two limits on this estimate:

- **Notebook trials stay uncorrected.** The other 41 trials (pair_JNJ_PG from
  an earlier pairs selection, the research trial, and 39 notebook experiments)
  have no comparable daily returns, so their own dependence cannot be measured.
- **The threshold is a choice.** The correlation cut-off of 0.5 is
  conventional, not derived.

The adjustment moves the DSR by about 0.1 and changes no conclusion.

### Rule 6: freezing

`make freeze` (`python -m src.reporting.report --freeze`) writes
`reports/results/holdout_freeze.json`. It records N, every V[SR] option, the
headline, the config hash, the selection-trial list and a SHA-256 of the
trial log. It refuses to run on uncommitted code and never overwrites an
existing freeze.

**It has not been run.** Every execution so far is `-dirty`, so the freeze
must follow a commit. Declaring the holdout's primary candidate is left to a
person, as `holdout_protocol.md` requires.
