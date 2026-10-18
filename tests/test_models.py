"""Checks on baselines, the estimator wrapper, the LSTM and persistence.

Every model shares one interface, so the same checks run against all of them:
reproducible with a seed, aligned with the input rows, and refusing inputs
that would give a wrong answer silently.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.features.build import feature_names
from src.models.baseline import (
    AlwaysUp,
    BuyAndHold,
    LastSignPersistence,
    RandomForecaster,
)
from src.models.estimator import (
    ModelError,
    ReturnPredictor,
    build_estimator,
    feature_importance,
    fit_with_weights,
)
from src.models.persistence import PersistenceError, load_model, save_model
from src.models.sequence import (
    LSTMPredictor,
    SequenceError,
    chronological_split,
    make_sequences,
)

TICKERS = ["AAA", "BBB"]
N_DATES = 400

SMALL_PARAMS = {
    "random_forest": {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 10, "n_jobs": 1},
    "lightgbm": {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 10, "verbose": -1},
}
LSTM_PARAMS = {
    "lookback": 5,
    "hidden_size": 4,
    "num_layers": 1,
    "dropout": 0.0,
    "learning_rate": 0.01,
    "batch_size": 64,
    "max_epochs": 5,
    "patience": 2,
    "validation_fraction": 0.2,
}


@pytest.fixture
def data(config) -> tuple[pd.DataFrame, pd.Series]:
    """Features and a target with a weak planted signal in ret_1d."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-04", periods=N_DATES, freq="B")
    index = pd.MultiIndex.from_product([dates, TICKERS], names=["date", "ticker"])
    names = feature_names(config)

    X = pd.DataFrame(rng.normal(size=(len(index), len(names))), index=index, columns=names)
    y = pd.Series(0.3 * X["ret_1d"] + rng.normal(size=len(index)), index=index) / 100

    return X, y


# --- Baselines --------------------------------------------------------------


def hit_rate(predicted: np.ndarray, actual: pd.Series) -> float:
    """Share of rows where the predicted direction matches the realised one."""
    return float((predicted == np.where(actual > 0, 1.0, -1.0)).mean())


def test_always_up_scores_the_up_rate_exactly(data):
    """Predicting up every day is right exactly as often as the market rises."""
    X, y = data

    assert hit_rate(AlwaysUp().fit(X, y).predict(X), y) == pytest.approx((y > 0).mean())


def test_buy_and_hold_forecasts_like_always_up(data):
    """The benchmark is long every day, so as a forecast it is AlwaysUp."""
    X, _ = data

    np.testing.assert_array_equal(BuyAndHold().predict(X), AlwaysUp().predict(X))


def test_random_forecaster_is_reproducible(data):
    """Same seed, same coin flips; another seed, other flips."""
    X, _ = data

    first = RandomForecaster(seed=7).predict(X)

    np.testing.assert_array_equal(first, RandomForecaster(seed=7).predict(X))
    assert not np.array_equal(first, RandomForecaster(seed=8).predict(X))
    assert set(np.unique(first)) == {-1.0, 1.0}


def test_random_forecaster_hits_about_half(data):
    """A fair coin is right about half the time, within sampling error."""
    X, y = data

    assert abs(hit_rate(RandomForecaster(seed=1).predict(X), y) - 0.5) < 0.05


def test_last_sign_persistence(data):
    """Tomorrow repeats today; reversed, it bets the opposite."""
    X, _ = data
    expected = np.where(X["ret_1d"] > 0, 1.0, -1.0)

    np.testing.assert_array_equal(LastSignPersistence().predict(X), expected)
    np.testing.assert_array_equal(LastSignPersistence(reverse=True).predict(X), -expected)


@pytest.mark.parametrize(
    "model",
    [AlwaysUp(), BuyAndHold(), RandomForecaster(), LastSignPersistence()],
    ids=lambda m: m.name,
)
def test_baselines_share_the_model_interface(model, data):
    """fit returns self, predict gives one value per row, proba is in [0, 1]."""
    X, y = data

    assert model.fit(X, y, sample_weight=np.ones(len(y))) is model
    assert model.predict(X).shape == (len(X),)
    assert ((model.predict_proba(X) >= 0) & (model.predict_proba(X) <= 1)).all()
    assert isinstance(model.params, dict)


# --- Estimator --------------------------------------------------------------


@pytest.mark.parametrize("estimator", ["random_forest", "lightgbm"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_same_seed_same_predictions(estimator, task, data):
    """Fitting twice with the same seed gives identical predictions."""
    X, y = data

    first = ReturnPredictor(estimator, task, SMALL_PARAMS[estimator], seed=3).fit(X, y)
    second = ReturnPredictor(estimator, task, SMALL_PARAMS[estimator], seed=3).fit(X, y)

    np.testing.assert_array_equal(first.predict(X), second.predict(X))


@pytest.mark.parametrize("estimator", ["random_forest", "lightgbm"])
def test_sample_weights_change_the_fit(estimator, data):
    """Weighting half the rows to near zero gives a different model."""
    X, y = data
    weights = np.where(np.arange(len(y)) % 2 == 0, 1.0, 0.01)

    plain = ReturnPredictor(estimator, "regression", SMALL_PARAMS[estimator]).fit(X, y)
    weighted = ReturnPredictor(estimator, "regression", SMALL_PARAMS[estimator]).fit(
        X, y, sample_weight=weights
    )

    assert not np.allclose(plain.predict(X), weighted.predict(X))


def test_classification_path(data):
    """Classification predicts +1 / -1 and a probability of up."""
    X, y = data
    model = ReturnPredictor("random_forest", "classification", SMALL_PARAMS["random_forest"])

    model.fit(X, y)

    assert set(np.unique(model.predict(X))) <= {-1.0, 1.0}
    proba = model.predict_proba(X)
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_array_equal(model.predict(X), np.where(proba > 0.5, 1.0, -1.0))


def test_regression_path(data):
    """Regression predicts returns on the target's scale, and has no probability."""
    X, y = data
    model = ReturnPredictor("random_forest", "regression", SMALL_PARAMS["random_forest"])

    predicted = model.fit(X, y).predict(X)

    assert predicted.std() < y.std()
    with pytest.raises(ModelError, match="classification"):
        model.predict_proba(X)


def test_wrong_columns_are_refused(data):
    """Scoring reordered or missing columns raises instead of predicting nonsense."""
    X, y = data
    model = ReturnPredictor("random_forest", "regression", SMALL_PARAMS["random_forest"])
    model.fit(X, y)

    with pytest.raises(ModelError, match="fitted on"):
        model.predict(X[X.columns[::-1]])


def test_unfitted_model_refuses_to_predict(data):
    """Predicting before fitting is an error, not a default."""
    X, _ = data

    with pytest.raises(ModelError, match="not been fitted"):
        ReturnPredictor("random_forest").predict(X)


def test_importance_names_line_up(data):
    """Importance rows are the input columns, and the planted signal ranks first."""
    X, y = data
    model = ReturnPredictor("random_forest", "regression", SMALL_PARAMS["random_forest"])
    model.fit(X, y)

    table = feature_importance(model, list(X.columns), X, y, n_repeats=2)

    assert sorted(table.index) == sorted(X.columns)
    assert table["split"].sum() == pytest.approx(1.0)
    assert table.index[0] == "ret_1d"
    assert table["permutation"].idxmax() == "ret_1d"


def test_importance_rejects_mismatched_names(data):
    """Names that do not match the fitted columns are refused."""
    X, y = data
    model = ReturnPredictor("random_forest", "regression", SMALL_PARAMS["random_forest"])
    model.fit(X, y)

    with pytest.raises(ModelError):
        feature_importance(model, list(X.columns)[::-1])


def test_fit_with_weights_aligns_on_index(data):
    """Shuffled weights are aligned by index, and NaN labels are dropped."""
    X, y = data
    y = y.copy()
    y.iloc[:10] = np.nan
    weights = pd.Series(1.0, index=X.index).sample(frac=1.0, random_state=0)
    model = ReturnPredictor("random_forest", "regression", SMALL_PARAMS["random_forest"])

    fit_with_weights(model, X, y, weights)
    reference = ReturnPredictor("random_forest", "regression", SMALL_PARAMS["random_forest"])
    reference.fit(X.iloc[10:], y.iloc[10:], sample_weight=np.ones(len(y) - 10))

    np.testing.assert_array_equal(model.predict(X), reference.predict(X))


def test_fit_with_weights_requires_coverage(data):
    """Weights missing for some rows are an error, not a silent drop."""
    X, y = data

    with pytest.raises(ModelError, match="weights"):
        fit_with_weights(ReturnPredictor("random_forest"), X, y, pd.Series(1.0, index=X.index[5:]))


def test_build_estimator_reads_config(config):
    """model.estimator and model.task choose the model."""
    model = build_estimator(config)

    assert model.name == f"{config.model.estimator}_{config.model.task}"

    lstm_config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, estimator="lstm")
    )
    assert isinstance(build_estimator(lstm_config), LSTMPredictor)


# --- Sequence model ---------------------------------------------------------


def test_sequence_shape(data):
    """One window per row after the first lookback - 1 of each ticker."""
    X, y = data
    lookback = 5

    sequences, targets, index = make_sequences(X, y, lookback)

    expected = len(TICKERS) * (N_DATES - lookback + 1)
    assert sequences.shape == (expected, lookback, X.shape[1])
    assert targets.shape == (expected,)
    assert len(index) == expected


def test_sequence_window_contents(data):
    """The window for row t holds rows t - lookback + 1 to t of the same ticker."""
    X, y = data
    lookback = 3

    sequences, targets, index = make_sequences(X, y, lookback)
    date, ticker = index[10]
    block = X.xs(ticker, level="ticker")
    end = block.index.get_loc(date)

    np.testing.assert_allclose(sequences[10], block.iloc[end - 2 : end + 1], rtol=1e-6)
    assert targets[10] == pytest.approx(y.loc[(date, ticker)], rel=1e-6)


def test_sequences_never_cross_the_split(data):
    """Built per block, no test window contains a date from the training block."""
    X, y = data
    dates = X.index.get_level_values("date")
    test_start = dates.unique()[300]

    _, _, test_index = make_sequences(X[dates >= test_start], y[dates >= test_start], 5)

    assert test_index.get_level_values("date").min() >= test_start + pd.offsets.BDay(4)


def test_chronological_split_puts_validation_last(data):
    """Every validation date is after every training date, for all tickers."""
    X, _ = data

    train, validation = chronological_split(X.index, 0.2)
    dates = X.index.get_level_values("date")

    assert dates[train].max() < dates[validation].min()
    assert validation.mean() == pytest.approx(0.2, abs=0.01)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_lstm_is_reproducible(task, data):
    """A fixed seed gives identical predictions."""
    X, y = data

    first = LSTMPredictor(LSTM_PARAMS, task, seed=5).fit(X, y).predict(X)
    second = LSTMPredictor(LSTM_PARAMS, task, seed=5).fit(X, y).predict(X)

    np.testing.assert_array_equal(first, second)


def test_lstm_output_aligns_with_rows(data):
    """One prediction per input row, NaN only in each ticker's warm-up."""
    X, y = data
    model = LSTMPredictor(LSTM_PARAMS, "regression", seed=5).fit(X, y)

    predicted = pd.Series(model.predict(X), index=X.index)

    assert len(predicted) == len(X)
    assert predicted.isna().sum() == len(TICKERS) * (LSTM_PARAMS["lookback"] - 1)
    assert model.history_


def test_lstm_classification_proba(data):
    """Classification gives directions and probabilities that agree."""
    X, y = data
    model = LSTMPredictor(LSTM_PARAMS, "classification", seed=5).fit(X, y)

    proba = model.predict_proba(X)
    direction = model.predict(X)
    valid = ~np.isnan(proba)

    assert ((proba[valid] > 0) & (proba[valid] < 1)).all()
    np.testing.assert_array_equal(direction[valid], np.where(proba[valid] > 0.5, 1.0, -1.0))


def test_lstm_refuses_before_fit(data):
    """Predicting before fitting is an error."""
    X, _ = data

    with pytest.raises(SequenceError, match="not been fitted"):
        LSTMPredictor(LSTM_PARAMS).predict(X)


# --- Persistence ------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        ReturnPredictor("random_forest", "regression", SMALL_PARAMS["random_forest"]),
        ReturnPredictor("lightgbm", "classification", SMALL_PARAMS["lightgbm"]),
        LSTMPredictor(LSTM_PARAMS, "regression"),
    ],
    ids=lambda m: m.name,
)
def test_round_trip_gives_identical_predictions(model, data, config, tmp_path):
    """Save then load: the same predictions, and the metadata says how it was made."""
    X, y = data
    model.fit(X, y)

    save_model(model, config, {"ic": 0.03}, tmp_path / "bundle", data_window=X.index)
    bundle = load_model(tmp_path / "bundle", config)

    np.testing.assert_array_equal(bundle.model.predict(X), model.predict(X))
    assert bundle.feature_names == list(X.columns)
    assert bundle.metadata["metrics"] == {"ic": 0.03}
    assert bundle.metadata["training_window"]["rows"] == len(X)
    assert bundle.metadata["training_window"]["start"] == "2010-01-04"
    assert bundle.metadata["libraries"]["scikit-learn"]
    assert "commit" in bundle.metadata["git"]


def test_load_refuses_a_changed_feature_list(data, config, tmp_path):
    """A bundle trained on other features raises instead of predicting."""
    X, y = data
    model = ReturnPredictor("random_forest", "regression", SMALL_PARAMS["random_forest"])
    save_model(model.fit(X, y), config, {}, tmp_path / "bundle", data_window=X.index)

    changed = dataclasses.replace(
        config,
        features=dataclasses.replace(config.features, lagged_returns=[1, 2, 3]),
    )

    with pytest.raises(PersistenceError, match="different features"):
        load_model(tmp_path / "bundle", changed)


def test_save_refuses_features_outside_config(data, config, tmp_path):
    """A model fitted on columns the config does not name cannot be saved."""
    X, y = data
    model = ReturnPredictor("random_forest", "regression", SMALL_PARAMS["random_forest"])
    model.fit(X.iloc[:, :5], y)

    with pytest.raises(PersistenceError, match="differ"):
        save_model(model, config, {}, tmp_path / "bundle", data_window=X.index)


def test_incomplete_bundle_raises(config, tmp_path):
    """A directory without both files is not a bundle."""
    with pytest.raises(PersistenceError, match="not a complete"):
        load_model(tmp_path, config)
