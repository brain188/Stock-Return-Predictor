"""Typed access to config/config.yaml.

Loads the file once, checks the values are consistent, and exposes them as
frozen dataclasses so a typo raises rather than returning None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

LABELLING_METHODS = {"fixed_horizon", "triple_barrier"}
ESTIMATORS = {"random_forest", "lightgbm", "lstm"}
TASKS = {"regression", "classification"}


class ConfigError(Exception):
    """Raised when the config is missing, malformed, or inconsistent."""


def _to_date(value: str, field_name: str) -> date:
    """Parse a YAML date string into a date object."""
    try:
        return pd.Timestamp(value).date()
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{field_name}: '{value}' is not a valid date") from exc


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project settings."""

    name: str
    random_seed: int


@dataclass(frozen=True)
class UniverseConfig:
    """Which instruments to trade, over what period."""

    tickers: list[str]
    benchmark: str
    start_date: date
    end_date: date
    survivorship_bias: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> UniverseConfig:
        """Build the config, parsing the date strings."""
        return cls(
            tickers=list(raw["tickers"]),
            benchmark=raw["benchmark"],
            start_date=_to_date(raw["start_date"], "universe.start_date"),
            end_date=_to_date(raw["end_date"], "universe.end_date"),
            survivorship_bias=raw["survivorship_bias"],
        )

    def validate(self) -> None:
        """Check the universe is usable."""
        if not self.tickers:
            raise ConfigError("universe.tickers is empty")

        if self.benchmark not in self.tickers:
            raise ConfigError(f"universe.benchmark '{self.benchmark}' must also appear in tickers")

        if self.start_date >= self.end_date:
            raise ConfigError("universe.start_date must come before end_date")


@dataclass(frozen=True)
class DataConfig:
    """Where price data comes from and where it is cached."""

    primary_source: str
    fallback_source: str
    cache_dir: Path
    bars_dir: Path
    features_dir: Path
    use_cache: bool
    price_column: str
    min_history_days: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any], root: Path) -> DataConfig:
        """Build the config with directories anchored to the project root."""
        return cls(
            primary_source=raw["primary_source"],
            fallback_source=raw["fallback_source"],
            cache_dir=(root / raw["cache_dir"]).resolve(),
            bars_dir=(root / raw["bars_dir"]).resolve(),
            features_dir=(root / raw["features_dir"]).resolve(),
            use_cache=raw["use_cache"],
            price_column=raw["price_column"],
            min_history_days=raw["min_history_days"],
        )

    def create_directories(self) -> None:
        """Create every directory the pipeline writes to."""
        for directory in (self.cache_dir, self.bars_dir, self.features_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CleaningConfig:
    """Rules for handling defects in the raw price series."""

    drop_zero_volume: bool
    max_abs_daily_return: float
    max_forward_fill_days: int

    def validate(self) -> None:
        """Check the thresholds are sane."""
        if not 0 < self.max_abs_daily_return < 1:
            raise ConfigError("cleaning.max_abs_daily_return must be between 0 and 1")


@dataclass(frozen=True)
class FracDiffConfig:
    """Settings for the fractional differentiation search."""

    enabled: bool
    search_grid: list[float]
    adf_confidence: float
    weight_threshold: float

    def validate(self) -> None:
        """Check the search grid and confidence level are usable."""
        if not self.search_grid:
            raise ConfigError("features.fracdiff.search_grid is empty")

        if not all(0.0 <= d <= 1.0 for d in self.search_grid):
            raise ConfigError("features.fracdiff.search_grid values must be in [0, 1]")

        if not 0.5 < self.adf_confidence < 1.0:
            raise ConfigError("features.fracdiff.adf_confidence must be between 0.5 and 1")


@dataclass(frozen=True)
class FeaturesConfig:
    """Which features to build."""

    lagged_returns: list[int]
    moving_averages: list[int]
    volatility_windows: list[int]
    rsi_window: int
    momentum_windows: list[int]
    high_window: int
    volume_window: int
    vol_ratio_windows: list[int]
    fracdiff: FracDiffConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FeaturesConfig:
        """Build the config, including the nested fracdiff block."""
        return cls(
            lagged_returns=list(raw["lagged_returns"]),
            moving_averages=list(raw["moving_averages"]),
            volatility_windows=list(raw["volatility_windows"]),
            rsi_window=raw["rsi_window"],
            momentum_windows=list(raw["momentum_windows"]),
            high_window=raw["high_window"],
            volume_window=raw["volume_window"],
            vol_ratio_windows=list(raw["vol_ratio_windows"]),
            fracdiff=FracDiffConfig(**raw["fracdiff"]),
        )

    @property
    def windows(self) -> list[int]:
        """Every window any feature uses, fracdiff aside."""
        return [
            *self.lagged_returns,
            *self.moving_averages,
            *self.volatility_windows,
            self.rsi_window,
            *self.momentum_windows,
            self.high_window,
            self.volume_window,
            *self.vol_ratio_windows,
        ]

    @property
    def max_lookback(self) -> int:
        """Longest window any feature needs, used to trim the warm-up period."""
        return max(self.windows)

    def validate(self) -> None:
        """Check every window is positive and the volatility ratio is short over long."""
        if any(w <= 0 for w in self.windows):
            raise ConfigError("features windows must all be positive")

        short, long = self.vol_ratio_windows
        if short >= long:
            raise ConfigError("features.vol_ratio_windows must be [short, long] with short < long")

        self.fracdiff.validate()


@dataclass(frozen=True)
class FixedHorizonConfig:
    """Settings for next-day return labelling."""

    horizon_days: int
    threshold: float


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Settings for path-dependent, volatility-scaled labelling."""

    profit_multiple: float
    stop_multiple: float
    max_holding_days: int
    volatility_window: int


@dataclass(frozen=True)
class LabellingConfig:
    """How the target is defined."""

    method: str
    fixed_horizon: FixedHorizonConfig
    triple_barrier: TripleBarrierConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LabellingConfig:
        """Build the config with both labelling blocks."""
        return cls(
            method=raw["method"],
            fixed_horizon=FixedHorizonConfig(**raw["fixed_horizon"]),
            triple_barrier=TripleBarrierConfig(**raw["triple_barrier"]),
        )

    @property
    def labels_overlap(self) -> bool:
        """Whether labels share returns, which decides if purging is mandatory."""
        if self.method == "triple_barrier":
            return True

        return self.fixed_horizon.horizon_days > 1

    def validate(self) -> None:
        """Check the method is one we implement."""
        if self.method not in LABELLING_METHODS:
            raise ConfigError(
                f"labelling.method must be one of {sorted(LABELLING_METHODS)}, got '{self.method}'"
            )


@dataclass(frozen=True)
class CPCVConfig:
    """Settings for combinatorial purged cross-validation."""

    n_groups: int
    n_test_groups: int

    def validate(self) -> None:
        """Check the group counts produce more than one path."""
        if self.n_test_groups >= self.n_groups:
            raise ConfigError("cpcv.n_test_groups must be smaller than cpcv.n_groups")

        if self.n_test_groups < 1:
            raise ConfigError("cpcv.n_test_groups must be at least 1")


@dataclass(frozen=True)
class ValidationConfig:
    """How the data is split for honest evaluation."""

    embargo_days: int
    n_splits: int
    cpcv: CPCVConfig
    test_start: date

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ValidationConfig:
        """Build the config, parsing the test start date."""
        return cls(
            embargo_days=raw["embargo_days"],
            n_splits=raw["n_splits"],
            cpcv=CPCVConfig(**raw["cpcv"]),
            test_start=_to_date(raw["test_start"], "validation.test_start"),
        )

    def validate(self) -> None:
        """Check the embargo and split counts are usable."""
        # bool is a subclass of int, and `embargo_days: yes` should not pass as 1.
        if (
            not isinstance(self.embargo_days, int)
            or isinstance(self.embargo_days, bool)
            or self.embargo_days < 0
        ):
            raise ConfigError(
                "validation.embargo_days must be a whole number of trading dates, 0 or more, "
                f"got {self.embargo_days!r}"
            )

        if self.n_splits < 2:
            raise ConfigError("validation.n_splits must be at least 2")

        self.cpcv.validate()


@dataclass(frozen=True)
class ModelConfig:
    """Estimator choice and its hyperparameters."""

    estimator: str
    task: str
    random_forest: dict[str, Any]
    lightgbm: dict[str, Any]
    lstm: dict[str, Any]

    @property
    def params(self) -> dict[str, Any]:
        """Hyperparameters for the selected estimator."""
        return getattr(self, self.estimator)

    def validate(self) -> None:
        """Check the estimator and task are ones we support."""
        if self.estimator not in ESTIMATORS:
            raise ConfigError(
                f"model.estimator must be one of {sorted(ESTIMATORS)}, got '{self.estimator}'"
            )

        if self.task not in TASKS:
            raise ConfigError(f"model.task must be one of {sorted(TASKS)}, got '{self.task}'")

        if not 0 < self.lstm["validation_fraction"] < 0.5:
            raise ConfigError("model.lstm.validation_fraction must be between 0 and 0.5")


@dataclass(frozen=True)
class BacktestConfig:
    """Trading costs and position rules."""

    commission_bps: float
    slippage_bps: float
    initial_capital: float
    max_position_size: float
    allow_short: bool
    rebalance: str

    @property
    def round_trip_cost(self) -> float:
        """Total cost of entering and exiting a position, as a fraction."""
        return 2 * (self.commission_bps + self.slippage_bps) / 10_000

    def validate(self) -> None:
        """Check costs are present and positions are bounded."""
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ConfigError("backtest costs cannot be negative")

        if not 0 < self.max_position_size <= 1:
            raise ConfigError("backtest.max_position_size must be between 0 and 1")


@dataclass(frozen=True)
class EvaluationConfig:
    """Settings for scoring and for correcting selection bias."""

    trial_log: Path
    monte_carlo_runs: int
    trading_days_per_year: int
    random_hit_rate: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any], root: Path) -> EvaluationConfig:
        """Build the config with the trial log anchored to the project root."""
        return cls(
            trial_log=(root / raw["trial_log"]).resolve(),
            monte_carlo_runs=raw["monte_carlo_runs"],
            trading_days_per_year=raw["trading_days_per_year"],
            random_hit_rate=raw["random_hit_rate"],
        )

    def validate(self) -> None:
        """Check the resampling count is high enough to be meaningful."""
        if self.monte_carlo_runs < 100:
            raise ConfigError("evaluation.monte_carlo_runs should be at least 100")


@dataclass(frozen=True)
class Config:
    """The whole configuration, validated and ready to use."""

    project: ProjectConfig
    universe: UniverseConfig
    data: DataConfig
    cleaning: CleaningConfig
    features: FeaturesConfig
    labelling: LabellingConfig
    validation: ValidationConfig
    model: ModelConfig
    backtest: BacktestConfig
    evaluation: EvaluationConfig
    source_file: Path = field(repr=False)

    def validate(self) -> None:
        """Run every section's own checks, then the cross-section ones."""
        self.universe.validate()
        self.cleaning.validate()
        self.features.validate()
        self.labelling.validate()
        self.validation.validate()
        self.model.validate()
        self.backtest.validate()
        self.evaluation.validate()

        if not self.universe.start_date < self.validation.test_start < self.universe.end_date:
            raise ConfigError("validation.test_start must fall inside the universe date range")


def load_config(path: str | Path | None = None, *, create_dirs: bool = True) -> Config:
    """Load, validate, and return the project configuration.

    Args:
        path: Config file to read. Defaults to config/config.yaml.
        create_dirs: Whether to create the output directories on load.

    Returns:
        A validated Config.

    Raises:
        ConfigError: If the file is missing, malformed, or fails validation.
    """
    config_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH

    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} is empty or not a mapping")

    required = {
        "project",
        "universe",
        "data",
        "cleaning",
        "features",
        "labelling",
        "validation",
        "model",
        "backtest",
        "evaluation",
    }
    if absent := required - raw.keys():
        raise ConfigError(f"config is missing sections: {sorted(absent)}")

    config = Config(
        project=ProjectConfig(**raw["project"]),
        universe=UniverseConfig.from_dict(raw["universe"]),
        data=DataConfig.from_dict(raw["data"], PROJECT_ROOT),
        cleaning=CleaningConfig(**raw["cleaning"]),
        features=FeaturesConfig.from_dict(raw["features"]),
        labelling=LabellingConfig.from_dict(raw["labelling"]),
        validation=ValidationConfig.from_dict(raw["validation"]),
        model=ModelConfig(**raw["model"]),
        backtest=BacktestConfig(**raw["backtest"]),
        evaluation=EvaluationConfig.from_dict(raw["evaluation"], PROJECT_ROOT),
        source_file=config_path,
    )

    config.validate()

    if create_dirs:
        config.data.create_directories()
        config.evaluation.trial_log.parent.mkdir(parents=True, exist_ok=True)

    return config
