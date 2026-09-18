"""
Transformer Price Predictor (PyTorch)
=======================================

Usage:
    from shared.ml.deep_learning.transformer_predictor import TransformerPredictor
    predictor = TransformerPredictor()
    predictor.train(df)
    signals = predictor.predict(df)
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

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


@dataclass
class TransformerConfig:
    """Configuration for Transformer predictor."""
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    seq_len: int = 60
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "auto"
    patience: int = 5  # early stopping patience
    lr_scheduler_patience: int = 3  # ReduceLROnPlateau patience
    lr_scheduler_factor: float = 0.5  # LR reduction factor


if _HAS_TORCH:

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding."""

        def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model > 1:
                pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)

    class TimeSeriesTransformer(nn.Module):
        """Transformer model for time series prediction."""

        def __init__(self, input_size: int, config: TransformerConfig):
            super().__init__()
            self.input_proj = nn.Linear(input_size, config.d_model)
            self.pos_enc = PositionalEncoding(config.d_model, config.seq_len, config.dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            self.fc = nn.Linear(config.d_model, 1)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_enc(x)
            x = self.transformer(x)
            x = x[:, -1, :]
            return self.fc(x).squeeze(-1)


class TransformerPredictor:
    """Transformer predictor for stock prices. Same interface as LSTMPredictor."""

    def __init__(self, config: Optional[TransformerConfig] = None):
        if not _HAS_TORCH:
            raise ImportError("PyTorch required. Install: pip install torch")
        self.config = config or TransformerConfig()
        self.model = None
        self.feature_cols = None
        self._scaler_mean = None
        self._scaler_std = None
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the Transformer model on OHLCV data."""
        from shared.ml.deep_learning.feature_engineer import FeatureEngineer
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
        splits = fe.temporal_split(X, y)

        self.model = TimeSeriesTransformer(X.shape[2], self.config).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.config.lr_scheduler_patience,
            factor=self.config.lr_scheduler_factor,
        )

        train_ds = TensorDataset(torch.FloatTensor(splits["X_train"]), torch.FloatTensor(splits["y_train"]))
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state_dict = None
        patience_counter = 0
        for epoch in range(self.config.epochs):
            self.model.train()
            losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_x = torch.FloatTensor(splits["X_val"]).to(self.device)
                val_y = torch.FloatTensor(splits["y_val"]).to(self.device)
                val_loss = criterion(self.model(val_x), val_y).item()

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
                logger.info("Epoch %d/%d - train: %.6f, val: %.6f", epoch + 1, self.config.epochs, np.mean(losses), val_loss)

        # Restore best model weights
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        return {"train_loss": np.mean(losses), "val_loss": best_val_loss}

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate prediction for the latest data point."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        from shared.ml.deep_learning.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        features = fe.compute_features(df)[self.feature_cols]
        features_norm = (features - self._scaler_mean) / self._scaler_std
        X_vals = features_norm.values.astype(np.float32)
        seq = X_vals[-self.config.seq_len:]
        x_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_tensor).cpu().numpy()
        return pd.Series(pred, index=features.index[-1:], name="prediction")

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
        import os
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
            # nosec B614 - reached only when the caller passes
            # allow_unsafe_legacy=True, having been told in the error message
            # above that this unpickles and therefore executes the file.
            checkpoint = torch.load(  # nosec B614
                path, map_location=self.device, weights_only=False,
            )

        config = checkpoint["config"]
        self.config = TransformerConfig(**config) if isinstance(config, dict) else config
        self.feature_cols = list(checkpoint["feature_cols"])
        self._scaler_mean = pd.Series(checkpoint["scaler_mean"])
        self._scaler_std = pd.Series(checkpoint["scaler_std"])
        self.model = TimeSeriesTransformer(len(self.feature_cols), self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
