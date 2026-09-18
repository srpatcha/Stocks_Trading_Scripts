"""
LSTM/GRU Price Predictor (PyTorch)
====================================

Usage:
    from shared.ml.deep_learning.lstm_predictor import LSTMPredictor, LSTMConfig
    predictor = LSTMPredictor(LSTMConfig(hidden_size=128))
    predictor.train(df)
    signals = predictor.predict(df)
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    logger.warning("PyTorch not installed. Install: pip install torch")


@dataclass
class LSTMConfig:
    """Configuration for LSTM/GRU predictor."""
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    seq_len: int = 60
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    use_gru: bool = False
    target: str = "return"
    device: str = "auto"
    patience: int = 5  # early stopping patience
    lr_scheduler_patience: int = 3  # ReduceLROnPlateau patience
    lr_scheduler_factor: float = 0.5  # LR reduction factor


if _HAS_TORCH:

    class _LSTMModel(nn.Module):
        def __init__(self, input_size: int, config: LSTMConfig):
            super().__init__()
            self.config = config
            rnn_cls = nn.GRU if config.use_gru else nn.LSTM
            self.rnn = rnn_cls(
                input_size=input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0,
                batch_first=True,
            )
            self.fc = nn.Linear(config.hidden_size, 1)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out).squeeze(-1)


class LSTMPredictor:
    """LSTM/GRU predictor for stock prices."""

    def __init__(self, config: Optional[LSTMConfig] = None):
        if not _HAS_TORCH:
            raise ImportError("PyTorch required. Install: pip install torch")
        self.config = config or LSTMConfig()
        self.model = None
        self.feature_cols = None
        self._scaler_mean = None
        self._scaler_std = None

        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the LSTM model on OHLCV data.

        Args:
            df: OHLCV DataFrame

        Returns:
            Dict with train_loss, val_loss
        """
        from shared.ml.deep_learning.feature_engineer import FeatureEngineer

        fe = FeatureEngineer()
        features = fe.compute_features(df)

        close = df["close"].reindex(features.index)
        if self.config.target == "direction":
            target = (close.pct_change().shift(-1) > 0).astype(float)
        else:
            target = close.pct_change().shift(-1)
        target = target.reindex(features.index).dropna()
        features = features.loc[target.index]

        self.feature_cols = list(features.columns)
        self._scaler_mean = features.mean()
        self._scaler_std = features.std().replace(0, 1)
        features_norm = (features - self._scaler_mean) / self._scaler_std

        X, y = fe.prepare_sequences(features_norm, target, self.config.seq_len)
        splits = fe.temporal_split(X, y)

        self.model = _LSTMModel(X.shape[2], self.config).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.config.lr_scheduler_patience,
            factor=self.config.lr_scheduler_factor,
        )

        train_ds = TensorDataset(
            torch.FloatTensor(splits["X_train"]),
            torch.FloatTensor(splits["y_train"]),
        )
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state_dict = None
        patience_counter = 0
        for epoch in range(self.config.epochs):
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_x = torch.FloatTensor(splits["X_val"]).to(self.device)
                val_y = torch.FloatTensor(splits["y_val"]).to(self.device)
                val_pred = self.model(val_x)
                val_loss = criterion(val_pred, val_y).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)",
                               epoch + 1, self.config.patience)
                    break

            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %d/%d - train_loss: %.6f, val_loss: %.6f",
                           epoch + 1, self.config.epochs, np.mean(train_losses), val_loss)

        # Restore best model weights
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        return {"train_loss": np.mean(train_losses), "val_loss": best_val_loss}

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate predictions for new data.

        Returns:
            Series of predicted values (returns or direction probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        from shared.ml.deep_learning.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        features = fe.compute_features(df)
        features = features[self.feature_cols]
        features_norm = (features - self._scaler_mean) / self._scaler_std

        X_vals = features_norm.values.astype(np.float32)
        if len(X_vals) < self.config.seq_len:
            raise ValueError(f"Need at least {self.config.seq_len} bars, got {len(X_vals)}")

        # Use last seq_len bars
        seq = X_vals[-self.config.seq_len:]
        x_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_tensor).cpu().numpy()

        idx = features.index[-1:]
        return pd.Series(pred, index=idx, name="prediction")

    def backtest(self, df: pd.DataFrame) -> Any:
        """Run walk-forward backtest. Returns BacktestResultV2."""
        from shared.ml.deep_learning.feature_engineer import FeatureEngineer
        from shared.backtesting.backtest_engine_v2 import BacktestEngineV2

        fe = FeatureEngineer()
        features = fe.compute_features(df)
        close = df["close"].reindex(features.index)
        target = close.pct_change().shift(-1).reindex(features.index).dropna()
        features = features.loc[target.index]

        self.feature_cols = list(features.columns)
        self._scaler_mean = features.mean()
        self._scaler_std = features.std().replace(0, 1)
        features_norm = (features - self._scaler_mean) / self._scaler_std

        X, y = fe.prepare_sequences(features_norm, target, self.config.seq_len)
        splits = fe.temporal_split(X, y, train_pct=0.6, val_pct=0.2)

        # Train on train+val
        self.model = _LSTMModel(X.shape[2], self.config).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        X_train = np.concatenate([splits["X_train"], splits["X_val"]])
        y_train = np.concatenate([splits["y_train"], splits["y_val"]])
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        for epoch in range(self.config.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Generate signals on test set
        self.model.eval()
        with torch.no_grad():
            test_x = torch.FloatTensor(splits["X_test"]).to(self.device)
            test_preds = self.model(test_x).cpu().numpy()

        # Convert predictions to signals for BacktestEngineV2
        test_start = len(X) - len(splits["X_test"])
        test_df = df.iloc[test_start + self.config.seq_len:]
        if len(test_df) > len(test_preds):
            test_df = test_df.iloc[:len(test_preds)]

        def strategy_fn(ctx):
            idx = ctx.bar_index
            if idx >= len(test_preds):
                return {}
            signal = 1 if test_preds[idx] > 0 else -1
            symbols = list(ctx.bars.keys())
            return {s: signal for s in symbols}

        engine = BacktestEngineV2()
        symbol = "STOCK"
        engine.load_data({symbol: test_df})
        return engine.run(strategy_fn)

    #: Bumped when the on-disk checkpoint layout changes. Version 2 stores only
    #: primitives and tensors so the file can be read with weights_only=True.
    CHECKPOINT_FORMAT = 2

    def save_model(self, path: str) -> None:
        """Save model weights and config.

        The config dataclass and the pandas scaler Series are written as plain
        dicts rather than pickled objects. A checkpoint containing arbitrary
        Python objects can only be read by unpickling it, and unpickling
        executes code — see ``load_model``.
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "format_version": self.CHECKPOINT_FORMAT,
            "model_state": self.model.state_dict(),
            "config": asdict(self.config),
            "feature_cols": list(self.feature_cols),
            # float()/str() coercion matters: dict(Series) keeps numpy
            # scalars, which are still pickled globals and would defeat
            # weights_only=True on read.
            "scaler_mean": {str(k): float(v) for k, v in self._scaler_mean.items()},
            "scaler_std": {str(k): float(v) for k, v in self._scaler_std.items()},
        }, path)

    def load_model(self, path: str, allow_unsafe_legacy: bool = False) -> None:
        """Load model weights and config.

        Reads with ``weights_only=True``. The previous default,
        ``weights_only=False``, unpickles the checkpoint, and unpickling runs
        arbitrary code embedded in the file — so a tampered or shared model
        file was remote code execution.

        Args:
            path: Checkpoint path.
            allow_unsafe_legacy: Read a pre-version-2 checkpoint by unpickling
                it. Only pass True for a file you produced yourself; it will
                execute whatever the file contains. Re-save afterwards to get a
                safe checkpoint.

        Raises:
            RuntimeError: For a legacy checkpoint when allow_unsafe_legacy is
                False.
        """
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as e:
            if not allow_unsafe_legacy:
                raise RuntimeError(
                    f"{path} is not a version-{self.CHECKPOINT_FORMAT} checkpoint and "
                    f"could not be read safely ({e}). Reading it requires unpickling, "
                    f"which executes code from the file. If you produced this file "
                    f"yourself, re-load with allow_unsafe_legacy=True and call "
                    f"save_model() to convert it."
                ) from e
            logger.warning(
                "Unpickling legacy checkpoint %s — this executes code from the file", path,
            )
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        config = checkpoint["config"]
        self.config = LSTMConfig(**config) if isinstance(config, dict) else config
        self.feature_cols = list(checkpoint["feature_cols"])
        self._scaler_mean = pd.Series(checkpoint["scaler_mean"])
        self._scaler_std = pd.Series(checkpoint["scaler_std"])
        self.model = _LSTMModel(len(self.feature_cols), self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])

    def optimize_hyperparams(self, df: pd.DataFrame, n_trials: int = 50) -> LSTMConfig:
        """Optimize hyperparameters using Optuna."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna required. Install: pip install optuna")

        from shared.ml.deep_learning.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        features = fe.compute_features(df)
        close = df["close"].reindex(features.index)
        target = close.pct_change().shift(-1).reindex(features.index).dropna()
        features = features.loc[target.index]

        scaler_mean = features.mean()
        scaler_std = features.std().replace(0, 1)
        features_norm = (features - scaler_mean) / scaler_std

        X, y = fe.prepare_sequences(features_norm, target, 60)
        splits = fe.temporal_split(X, y)

        def objective(trial):
            cfg = LSTMConfig(
                hidden_size=trial.suggest_int("hidden_size", 32, 256),
                num_layers=trial.suggest_int("num_layers", 1, 4),
                dropout=trial.suggest_float("dropout", 0.0, 0.5),
                seq_len=60,
                epochs=20,
                batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
                learning_rate=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                use_gru=trial.suggest_categorical("use_gru", [True, False]),
            )
            model = _LSTMModel(X.shape[2], cfg).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
            criterion = nn.MSELoss()
            train_ds = TensorDataset(
                torch.FloatTensor(splits["X_train"]),
                torch.FloatTensor(splits["y_train"]),
            )
            loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
            for _ in range(cfg.epochs):
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    loss = criterion(model(xb), yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                val_x = torch.FloatTensor(splits["X_val"]).to(self.device)
                val_y = torch.FloatTensor(splits["y_val"]).to(self.device)
                val_loss = criterion(model(val_x), val_y).item()
            return val_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        return LSTMConfig(
            hidden_size=best["hidden_size"],
            num_layers=best["num_layers"],
            dropout=best["dropout"],
            batch_size=best["batch_size"],
            learning_rate=best["lr"],
            use_gru=best["use_gru"],
        )
