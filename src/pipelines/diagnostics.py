"""Does the residual feature overlap change the validation results?

With a 36 day embargo, training rows 37 to 139 trading dates after a test
block still hold test-period prices inside their slow features: the 63 day
windows (px_over_ma_63, vol_63d, mom_63d, dist_high_63, vol_ratio_10_63) up
to 63 dates out, and fracdiff, whose weights reach back 139 dates, all the
way. Their targets are later returns that belong to no test label, so this
is shared inputs, not leaked answers. The question is whether it matters.

The check refits the same model under several embargoes and compares the
out-of-sample scores on identical test rows:

- embargo_0, embargo_36, embargo_63, embargo_139: the gap after each test
  block. 139 removes every training row whose features contain any
  test-period price.
- control_139: the embargo_36 training set minus as many dates as
  embargo_139 removes, drawn at random from dates more than 139 away from
  every test block. It has the same amount of training data as embargo_139
  but none of its overlap removed.

Leakage through the overlap would show as embargo_36 scoring above
embargo_139 while control_139 scores like embargo_36. If embargo_139 and
control_139 agree, the difference is the cost of less data, not leakage.
Because the test rows are the same in every variant, each split's IC is
compared pairwise against embargo_36, which removes the test-set noise that
dominates any unpaired comparison.

The embargo is fixed at 36 before this runs; the result does not choose it.
Every variant is logged as kind "diagnostic", so it is recorded but not
counted among the trials a deflated Sharpe ratio corrects for.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, Config, load_config
from src.data.cleaning import load_clean_panel
from src.evaluation.trial_log import TrialLog, trial_fingerprint
from src.logger import get_logger, log_step, setup_logging
from src.pipelines.research import (
    ResearchData,
    fit_splits,
    prepare_research_data,
    research_settings,
)
from src.validation.cpcv import CombinatorialPurgedCV, Split, path_predictions
from src.validation.metrics import cross_sectional_score, information_coefficient, score
from src.validation.purged_cv import date_codes, row_dates

logger = get_logger(__name__)

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "results" / "diagnostics" / "embargo_sensitivity"

EMBARGOES = (0, 36, 63, 139)
BASELINE = 36
CONTROL_WIDTH = 139


@dataclass
class SensitivityResult:
    """Scores per variant, and the paired per-split comparison."""

    summary: pd.DataFrame
    splits: pd.DataFrame
    predictions: dict[str, pd.DataFrame]

    def save(self, out_dir: str | Path = DEFAULT_OUTPUT) -> list[Path]:
        """Write summary.csv and splits.csv."""
        folder = Path(out_dir)
        folder.mkdir(parents=True, exist_ok=True)
        files = [folder / "summary.csv", folder / "splits.csv"]
        self.summary.to_csv(files[0])
        self.splits.to_csv(files[1])
        logger.info("Saved the embargo sensitivity tables to %s", folder)

        return files


def with_embargo(config: Config, days: int) -> Config:
    """The config with only validation.embargo_days changed."""
    validation = dataclasses.replace(config.validation, embargo_days=days)

    return dataclasses.replace(config, validation=validation)


def _near_test(codes: np.ndarray, test: np.ndarray, width: int) -> np.ndarray:
    """Date positions within `width` dates before or after any test block."""
    test_dates = np.unique(codes[test])
    breaks = np.flatnonzero(np.diff(test_dates) > 1)
    firsts = test_dates[np.r_[0, breaks + 1]]
    lasts = test_dates[np.r_[breaks, len(test_dates) - 1]]

    near = [
        np.arange(first - width, last + width + 1)
        for first, last in zip(firsts, lasts, strict=True)
    ]

    return np.unique(np.concatenate(near))


def control_split(
    base: Split,
    matched: Split,
    codes: np.ndarray,
    width: int,
    rng: np.random.Generator,
) -> Split:
    """The base split with training dates removed far from the test set.

    Removes as many whole dates as `matched` has fewer than `base`, chosen at
    random among training dates more than `width` dates from every test
    block, so the control loses the same amount of data as the wide embargo
    without removing any of the rows the wide embargo targets.

    Args:
        base: The split at the baseline embargo.
        matched: The same split at the wide embargo, whose size to match.
        codes: Date position of every row.
        width: Distance from the test blocks the removed dates must keep.
        rng: Random generator, seeded by the caller.

    Returns:
        A Split whose train holds the base rows minus the removed dates.

    Raises:
        ValueError: If there are not enough far dates to remove.
    """
    base_dates = np.unique(codes[base.train])
    n_remove = len(base_dates) - len(np.unique(codes[matched.train]))
    far = np.setdiff1d(base_dates, _near_test(codes, base.test, width))

    if n_remove > len(far):
        raise ValueError(f"split {base.number}: need {n_remove} far dates, only {len(far)} exist")

    removed = rng.choice(far, n_remove, replace=False)
    train = base.train[~np.isin(codes[base.train], removed)]

    return dataclasses.replace(base, train=train)


def _variant_scores(
    name: str,
    data: ResearchData,
    split_predictions: list[pd.Series],
    cv: CombinatorialPurgedCV,
    report: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame, pd.Series]:
    """Path-level scores, the path predictions, and the IC of every split."""
    predictions = path_predictions(split_predictions, cv, data.X)
    paths = pd.DataFrame(
        {
            path: {
                **score(predictions[path], data.actual),
                **cross_sectional_score(predictions[path], data.actual),
            }
            for path in predictions
        }
    ).T
    split_ic = pd.Series(
        [information_coefficient(p, data.actual.reindex(p.index)) for p in split_predictions],
        name=name,
    )

    summary = {
        "train_rows_mean": float(report["train_rows"].mean()),
        "ic_mean": float(paths["ic"].mean()),
        "ic_path_std": float(paths["ic"].std()),
        "split_ic_mean": float(split_ic.mean()),
        "cs_spearman_mean": float(paths["cs_spearman_mean"].mean()),
        "hit_rate_mean": float(paths["hit_rate"].mean()),
        "edge_over_always_up_mean": float(paths["edge_over_always_up"].mean()),
    }

    return summary, predictions, split_ic


def embargo_sensitivity(
    config: Config,
    panel: pd.DataFrame | None = None,
    *,
    embargoes: Sequence[int] = EMBARGOES,
    control_width: int = CONTROL_WIDTH,
    trial_log: TrialLog | None = None,
) -> SensitivityResult:
    """Refit the configured model under each embargo and a matched control.

    Args:
        config: The research configuration; its embargo is the baseline.
        panel: Cleaned panel. Loaded from disk when None.
        embargoes: Embargo widths to try, in trading dates. Must include the
            configured one, against which the others are paired.
        control_width: Wide embargo the control is matched to; must be in
            embargoes.
        trial_log: Where to record the diagnostic runs. Defaults to the config's.

    Returns:
        Scores per variant and the per-split IC table.
    """
    baseline = config.validation.embargo_days
    if baseline not in embargoes or control_width not in embargoes:
        raise ValueError(f"embargoes {embargoes} must include {baseline} and {control_width}")

    trial_log = trial_log or TrialLog.from_config(config)
    panel = load_clean_panel(config) if panel is None else panel
    data = prepare_research_data(panel, config)
    touch = data.labels["touch_time"]
    codes, _ = date_codes(row_dates(data.X.index))

    splits_by_width = {
        days: list(
            CombinatorialPurgedCV.from_config(with_embargo(config, days), touch).iter_splits(data.X)
        )
        for days in embargoes
    }
    rng = np.random.default_rng(config.project.random_seed)
    control = [
        control_split(base, wide, codes, control_width, rng)
        for base, wide in zip(
            splits_by_width[baseline], splits_by_width[control_width], strict=True
        )
    ]

    # The control keeps the baseline embargo, so it is checked against that.
    variants = {
        f"embargo_{days}": (with_embargo(config, days), splits_by_width[days]) for days in embargoes
    }
    variants[f"control_{control_width}"] = (config, control)

    cv = CombinatorialPurgedCV.from_config(config, touch)
    rows, split_ics, predictions = {}, {}, {}

    for name, (variant_config, splits) in variants.items():
        settings = {
            **research_settings(variant_config),
            "diagnostic": "embargo_sensitivity",
            "variant": name,
        }
        with log_step(logger, f"Embargo sensitivity: {name}"):
            fits = fit_splits(data, variant_config, splits, label=name)
        rows[name], predictions[name], split_ics[name] = _variant_scores(
            name, data, fits.predictions, cv, fits.report
        )
        rows[name]["embargo_days"] = float(variant_config.validation.embargo_days)

        trial_log.log_trial(
            f"diagnostic_{name}",
            kind="diagnostic",
            fingerprint=trial_fingerprint(settings),
            status="ok",
            config=variant_config,
            features=list(data.X.columns),
            model_params=fits.models[0].params,
            label=config.labelling.method,
            metrics=rows[name],
            notes=(
                f"Residual feature overlap check. Embargo fixed at {baseline} beforehand; "
                "this measures, it does not select."
            ),
        )

    split_table = pd.DataFrame(split_ics)
    split_table.index.name = "split"
    base_name = f"embargo_{baseline}"

    summary = pd.DataFrame(rows).T
    for name in summary.index:
        diff = split_table[name] - split_table[base_name]
        summary.loc[name, "paired_diff_mean"] = diff.mean()
        summary.loc[name, "paired_diff_std"] = diff.std()
        summary.loc[name, "paired_t"] = (
            diff.mean() / (diff.std() / np.sqrt(len(diff))) if diff.std() > 0 else np.nan
        )
    summary.index.name = "variant"

    return SensitivityResult(summary=summary, splits=split_table, predictions=predictions)


def main() -> None:
    """Run the check with the project config and print the comparison."""
    setup_logging()
    config = load_config()
    result = embargo_sensitivity(config)
    result.save()

    columns = [
        "embargo_days",
        "train_rows_mean",
        "ic_mean",
        "ic_path_std",
        "cs_spearman_mean",
        "edge_over_always_up_mean",
        "paired_diff_mean",
        "paired_t",
    ]
    print(result.summary[columns].round(5).to_string())


if __name__ == "__main__":
    main()
