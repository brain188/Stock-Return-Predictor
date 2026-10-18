"""An LSTM on windows of past features, as a comparison rather than a centrepiece.

The honest expectation is that it does not beat the Random Forest. Roughly
5,000 daily bars per ticker is a tiny dataset for a recurrent network, and
daily returns have almost no autocorrelation for it to exploit. Running it
anyway and reporting the result beats skipping it, as long as it gets a fair
chance and counts as a trial: the network is small to match the data,
features are standardised with training statistics only, and early stopping
watches a validation slice taken from the end of the training period.

A sequence for row t holds the `lookback` rows ending at t, from the same
ticker. Sequences are built separately for each block of data (train,
validation, test), so a window never spans two blocks and test sequences
never contain training rows. The first lookback - 1 rows of each ticker in
each block have no full window and get no prediction.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.logger import get_logger
from src.models.baseline import Forecaster

logger = get_logger(__name__)


class SequenceError(Exception):
    """Raised when sequences cannot be built or a model is used before fitting."""


def make_sequences(
    X: pd.DataFrame,
    y: pd.Series | None,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray | None, pd.Index]:
    """Stack each row's last `lookback` rows, per ticker, into a 3D array.

    Call it once per block (train, validation or test). Windows never cross
    from one ticker into another, nor outside the frame given.

    Args:
        X: Features indexed by (date, ticker).
        y: Target on the same index, or None when predicting.
        lookback: Rows per window, the current row included.

    Returns:
        sequences of shape (n, lookback, n_features), targets of shape (n,)
        or None, and the (date, ticker) index of the row each window ends on.
        n is the sum over tickers of rows - lookback + 1.

    Raises:
        SequenceError: If lookback is below 1.
    """
    if lookback < 1:
        raise SequenceError(f"lookback must be at least 1, got {lookback}")

    windows, targets, index = [], [], []

    for ticker in X.index.unique("ticker"):
        block = X.xs(ticker, level="ticker", drop_level=False).sort_index()
        if len(block) < lookback:
            continue

        values = block.to_numpy(dtype=np.float32)
        view = np.lib.stride_tricks.sliding_window_view(values, lookback, axis=0)
        windows.append(view.transpose(0, 2, 1))
        index.append(block.index[lookback - 1 :])

        if y is not None:
            targets.append(y.reindex(block.index).to_numpy(dtype=np.float32)[lookback - 1 :])

    if not windows:
        empty = np.empty((0, lookback, X.shape[1]), dtype=np.float32)
        return empty, (np.empty(0, dtype=np.float32) if y is not None else None), X.index[:0]

    sequences = np.concatenate(windows)
    full_index = index[0].append(index[1:]) if len(index) > 1 else index[0]

    return sequences, (np.concatenate(targets) if y is not None else None), full_index


class LSTMNet(nn.Module):
    """LSTM over the window, then a linear head on the last hidden state."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)

        return self.head(output[:, -1, :]).squeeze(-1)


def build_lstm(n_features: int, params: dict[str, Any], seed: int = 42) -> LSTMNet:
    """A deliberately small network, seeded so it is reproducible.

    Args:
        n_features: Features per time step.
        params: The config.model.lstm block.
        seed: Seed for the weight initialisation.

    Returns:
        An untrained network.
    """
    torch.manual_seed(seed)

    return LSTMNet(n_features, params["hidden_size"], params["num_layers"], params["dropout"])


def chronological_split(index: pd.Index, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into train and validation by date, the last `fraction` of dates last.

    Splitting by date rather than by row keeps every ticker's validation rows
    after every ticker's training rows.

    Returns:
        Boolean masks for the training and validation rows.
    """
    dates = index.get_level_values("date")
    unique = dates.unique().sort_values()
    cut = unique[int(len(unique) * (1 - fraction))]
    validation = np.asarray(dates >= cut)

    return ~validation, validation


def fit_lstm(
    model: LSTMNet,
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray, np.ndarray],
    params: dict[str, Any],
    task: str,
    seed: int = 42,
) -> tuple[LSTMNet, list[dict[str, float]]]:
    """Train with early stopping on the validation slice.

    Args:
        model: Output of build_lstm.
        train: Sequences, targets and weights for training.
        validation: Sequences, targets and weights for early stopping, all
            later in time than the training rows.
        params: The config.model.lstm block.
        task: "regression" (mean squared error) or "classification"
            (binary cross entropy on up / not up).
        seed: Seed for batch shuffling.

    Returns:
        The model at its best validation epoch, and the loss per epoch.
    """
    generator = torch.Generator().manual_seed(seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = (
        nn.BCEWithLogitsLoss(reduction="none")
        if task == "classification"
        else nn.MSELoss(reduction="none")
    )

    def tensors(block):
        x, y, w = block
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)

    x_train, y_train, w_train = tensors(train)
    x_val, y_val, w_val = tensors(validation)

    def weighted_loss(x, y, w):
        return (criterion(model(x), y) * w).sum() / w.sum()

    best_loss, best_state, stale, history = np.inf, copy.deepcopy(model.state_dict()), 0, []

    for epoch in range(params["max_epochs"]):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)

        for start in range(0, len(order), params["batch_size"]):
            batch = order[start : start + params["batch_size"]]
            optimiser.zero_grad()
            loss = weighted_loss(x_train[batch], y_train[batch], w_train[batch])
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            train_loss = float(weighted_loss(x_train, y_train, w_train))
            val_loss = float(weighted_loss(x_val, y_val, w_val))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss - 1e-9:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= params["patience"]:
                break

    model.load_state_dict(best_state)
    logger.info("LSTM stopped after %s epochs, best validation loss %.6f", len(history), best_loss)

    return model, history


class LSTMPredictor(Forecaster):
    """The LSTM behind the same interface as every other model.

    predict returns one value per row of X, NaN for rows without a full
    window, so its output lines up with the tree models row for row.

    Args:
        params: The config.model.lstm block.
        task: "regression" or "classification".
        seed: Seed for initialisation and shuffling.
    """

    def __init__(self, params: dict[str, Any], task: str = "regression", seed: int = 42) -> None:
        self._params = dict(params)
        self.task = task
        self.seed = seed
        self.net: LSTMNet | None = None
        self.feature_names_: list[str] = []
        self.history_: list[dict[str, float]] = []

    @property
    def name(self) -> str:
        return f"lstm_{self.task}"

    @property
    def params(self) -> dict[str, Any]:
        return {"estimator": "lstm", "task": self.task, "seed": self.seed, **self._params}

    def _scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Standardise with the training mean and std, never the data's own."""
        return (X - self.mean_) / self.std_

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weight: pd.Series | np.ndarray | None = None,
    ) -> LSTMPredictor:
        """Fit on (date, ticker) indexed features, with early stopping."""
        lookback = self._params["lookback"]
        target = pd.Series(np.asarray(y, dtype=float), index=X.index)
        weight = pd.Series(
            1.0 if sample_weight is None else np.asarray(sample_weight, float), index=X.index
        )

        self.feature_names_ = list(X.columns)
        train_rows, val_rows = chronological_split(X.index, self._params["validation_fraction"])
        self.mean_ = X[train_rows].mean()
        self.std_ = X[train_rows].std().replace(0, 1.0)

        if self.task == "classification":
            target = (target > 0).astype(float)
        else:
            self.y_scale_ = float(target[train_rows].std()) or 1.0
            target = target / self.y_scale_

        blocks = []
        for rows in (train_rows, val_rows):
            x_seq, y_seq, index = make_sequences(self._scale(X[rows]), target[rows], lookback)
            w_seq = weight.reindex(index).to_numpy(dtype=np.float32)
            blocks.append((x_seq, y_seq, w_seq))

        if len(blocks[0][0]) == 0 or len(blocks[1][0]) == 0:
            raise SequenceError("not enough rows for training and validation sequences")

        self.net = build_lstm(X.shape[1], self._params, self.seed)
        self.net, self.history_ = fit_lstm(
            self.net, blocks[0], blocks[1], self._params, self.task, self.seed
        )

        return self

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        """Network output per row of X, NaN where there is no full window."""
        if self.net is None:
            raise SequenceError("LSTMPredictor has not been fitted")

        if list(X.columns) != self.feature_names_:
            raise SequenceError(f"fitted on {self.feature_names_}, got {list(X.columns)}")

        sequences, _, index = make_sequences(self._scale(X), None, self._params["lookback"])
        self.net.eval()
        with torch.no_grad():
            output = self.net(torch.from_numpy(sequences)).numpy().astype(float)

        return pd.Series(output, index=index).reindex(X.index).to_numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Forecast return (regression) or +1 / -1 (classification), NaN in warm-up."""
        raw = self._raw(X)

        if self.task == "classification":
            return np.where(np.isnan(raw), np.nan, np.where(raw > 0, 1.0, -1.0))

        return raw * self.y_scale_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of up, classification only."""
        if self.task != "classification":
            raise SequenceError("predict_proba needs model.task = classification")

        return 1 / (1 + np.exp(-self._raw(X)))
