"""
ML Regime Classifier using LightGBM
======================================

Machine learning approach to market regime classification.
Uses gradient boosting on engineered features from OHLCV data
to predict TRENDING, RANGING, or VOLATILE market regimes.

Auto-labels historical data using forward return statistics,
eliminating the need for manual labeling.

Usage:
    clf = MLRegimeClassifier()
    clf.fit(df_spy, lookforward=20)
    regime = clf.predict(df_latest)
    proba = clf.predict_proba(df_latest)
    clf.save_model("models/regime_lgbm.joblib")
"""

from __future__ import annotations

import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb  # type: ignore[import-untyped]

    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False
    logger.debug("lightgbm not installed — ML regime classification unavailable")

try:
    import joblib  # type: ignore[import-untyped]

    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


class MarketRegime(Enum):
    """Market regime labels."""

    TRENDING = 0
    RANGING = 1
    VOLATILE = 2


_REGIME_NAMES = {0: "TRENDING", 1: "RANGING", 2: "VOLATILE"}


class MLRegimeClassifier:
    """LightGBM-based market regime classifier.

    Features are auto-engineered from OHLCV data. Regimes are
    auto-labeled using forward return characteristics.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Learning rate for boosting.
        random_state: Random seed for reproducibility.
    """

    FEATURE_NAMES: List[str] = []

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        if not _HAS_LIGHTGBM:
            raise ImportError(
                "lightgbm is required for MLRegimeClassifier. "
                "Install with: pip install lightgbm"
            )

        self._model: Optional[lgb.LGBMClassifier] = None
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._random_state = random_state
        self._feature_names: List[str] = []
        self._is_fitted = False

    # ─── Feature Engineering ───

    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute the 26-feature set from OHLCV data.

        The count is asserted exactly in
        ``tests/test_functional_ai_pipeline.py::test_compute_features_all_present``.
        Update that test alongside any change here — every model in the repo is
        trained on this set, so adding or removing a column changes their
        inputs.

        Args:
            df: DataFrame with columns: open, high, low, close, volume.

        Returns:
            DataFrame with computed features (same index, NaN rows at start).
        """
        feat = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Returns
        feat["ret_1d"] = close.pct_change(1)
        feat["ret_5d"] = close.pct_change(5)
        feat["ret_10d"] = close.pct_change(10)
        feat["ret_20d"] = close.pct_change(20)
        feat["log_ret_1d"] = np.log(close / close.shift(1))

        # Volatility
        feat["vol_5d"] = feat["ret_1d"].rolling(5).std()
        feat["vol_10d"] = feat["ret_1d"].rolling(10).std()
        feat["vol_20d"] = feat["ret_1d"].rolling(20).std()

        # Garman-Klass volatility
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / df["open"]) ** 2
        feat["gk_vol"] = np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(20).mean())

        # Parkinson volatility
        feat["parkinson_vol"] = np.sqrt(
            (np.log(high / low) ** 2).rolling(20).mean() / (4 * np.log(2))
        )

        # Momentum
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss_val = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
        avg_loss = loss_val.ewm(alpha=1 / 14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        feat["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD histogram
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        feat["macd_hist"] = macd_line - signal_line

        # Stochastic %K
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        feat["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)

        # ROC
        feat["roc_10"] = (close - close.shift(10)) / close.shift(10).replace(0, np.nan) * 100
        feat["roc_20"] = (close - close.shift(20)) / close.shift(20).replace(0, np.nan) * 100

        # ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.ewm(alpha=1 / 14, min_periods=14).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, min_periods=14).mean() / atr_14)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, min_periods=14).mean() / atr_14)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        feat["adx"] = dx.ewm(alpha=1 / 14, min_periods=14).mean()

        # Trend: price vs SMAs (z-score)
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        std20 = close.rolling(20).std()
        feat["price_vs_sma20"] = (close - sma20) / std20.replace(0, np.nan)
        feat["price_vs_sma50"] = (close - sma50) / std20.replace(0, np.nan)
        feat["price_vs_sma200"] = (close - sma200) / std20.replace(0, np.nan)

        # EMA slope
        ema20 = close.ewm(span=20, adjust=False).mean()
        feat["ema_slope"] = ema20.diff(5) / ema20.shift(5).replace(0, np.nan)

        # Volume
        vol_20avg = volume.rolling(20).mean()
        feat["rel_volume"] = volume / vol_20avg.replace(0, np.nan)
        obv = (np.sign(close.diff()) * volume).cumsum()
        feat["obv_slope"] = obv.diff(5) / obv.shift(5).abs().replace(0, np.nan)
        feat["vol_change_rate"] = volume.pct_change(5)

        # Regime-specific features
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        feat["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
        feat["bb_width_z"] = (
            (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
        ).pipe(lambda x: (x - x.rolling(50).mean()) / x.rolling(50).std().replace(0, np.nan))

        # ATR percentile rank (vs 50-day)
        feat["atr_pct_rank"] = atr_14.rolling(50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False,
        )

        return feat

    # ─── Auto-Labeling ───

    @staticmethod
    def auto_label(df: pd.DataFrame, lookforward: int = 20) -> pd.Series:
        """Auto-label regimes using forward return characteristics.

        Labeling rules:
        - TRENDING: |forward_return| > 1.5 × stdev AND ADX > 20
        - RANGING: |forward_return| < 0.5 × stdev AND ADX < 25
        - VOLATILE: realized_vol > 90th percentile of 60-day rolling

        Args:
            df: OHLCV DataFrame.
            lookforward: Number of bars to look forward for return calc.

        Returns:
            Series of integer labels (0=TRENDING, 1=RANGING, 2=VOLATILE).
        """
        close = df["close"]
        ret_1d = close.pct_change()

        # Forward returns
        forward_ret = close.shift(-lookforward) / close - 1

        # Realized vol (5-day)
        vol_5d = ret_1d.rolling(5).std()
        vol_90pct = vol_5d.rolling(60).quantile(0.90)

        # Rolling stdev of returns
        ret_std = ret_1d.rolling(lookforward).std()

        # ADX
        high, low = df["high"], df["low"]
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.ewm(alpha=1 / 14, min_periods=14).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, min_periods=14).mean() / atr_14)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, min_periods=14).mean() / atr_14)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1 / 14, min_periods=14).mean()

        labels = pd.Series(1, index=df.index, dtype=int)  # default RANGING

        is_volatile = vol_5d > vol_90pct
        is_trending = (forward_ret.abs() > 1.5 * ret_std) & (adx > 20)
        is_ranging = (forward_ret.abs() < 0.5 * ret_std) & (adx < 25)

        labels[is_volatile] = 2  # VOLATILE
        labels[is_trending & ~is_volatile] = 0  # TRENDING
        labels[is_ranging & ~is_volatile] = 1  # RANGING

        return labels

    # ─── Training ───

    def fit(
        self,
        df: pd.DataFrame,
        lookforward: int = 20,
        test_size: float = 0.2,
        use_walk_forward: bool = True,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Train the regime classifier.

        Args:
            df: OHLCV DataFrame.
            lookforward: Bars to look forward for labeling.
            test_size: Fraction reserved for validation.
            use_walk_forward: If True, run walk-forward CV before final training.
            n_splits: Number of walk-forward splits.

        Returns:
            Dict with training metrics (accuracy, classification_report).
        """
        logger.info("Computing features...")
        features = self.compute_features(df)
        labels = self.auto_label(df, lookforward)

        # Drop NaN rows
        valid_mask = features.notna().all(axis=1) & labels.notna()
        features = features[valid_mask]
        labels = labels[valid_mask]

        # Remove last lookforward rows (labels use future data)
        if lookforward > 0:
            features = features.iloc[:-lookforward]
            labels = labels.iloc[:-lookforward]

        if len(features) < 100:
            raise ValueError(
                f"Insufficient data for training: {len(features)} rows "
                "(need at least 100 after feature computation)"
            )

        self._feature_names = list(features.columns)

        # LightGBM params
        self._lgb_params = {
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "learning_rate": self._learning_rate,
            "random_state": self._random_state,
            "num_class": 3,
            "objective": "multiclass",
            "metric": "multi_logloss",
            "verbosity": -1,
            "class_weight": "balanced",
        }

        # Walk-forward cross-validation (GAP 13)
        if use_walk_forward and len(features) > 200:
            cv_scores = self._walk_forward_cv(
                features.values, labels.values, n_splits=n_splits,
            )
            avg_cv = float(np.mean(cv_scores))
            logger.info(
                "Walk-forward CV (%d splits): scores=%s, avg=%.4f",
                n_splits,
                [round(s, 4) for s in cv_scores],
                avg_cv,
            )

        # Train/test split (temporal, not random)
        split_idx = int(len(features) * (1 - test_size))
        X_train = features.iloc[:split_idx]
        y_train = labels.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_test = labels.iloc[split_idx:]

        logger.info(
            "Training LightGBM: %d train, %d test, %d features",
            len(X_train), len(X_test), len(self._feature_names),
        )

        # Label distribution
        for regime_id, name in _REGIME_NAMES.items():
            count = (y_train == regime_id).sum()
            logger.info("  %s: %d samples (%.1f%%)", name, count, count / len(y_train) * 100)

        self._model = lgb.LGBMClassifier(**self._lgb_params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
            )

        self._is_fitted = True

        # Evaluate
        y_pred = self._model.predict(X_test)
        accuracy = float(np.mean(y_pred == y_test))

        report: Dict[str, Any] = {"accuracy": accuracy}
        if use_walk_forward and len(features) > 200:
            report["walk_forward_cv_avg"] = avg_cv
            report["walk_forward_cv_scores"] = cv_scores
        for regime_id, name in _REGIME_NAMES.items():
            mask = y_test == regime_id
            if mask.sum() > 0:
                regime_acc = float(np.mean(y_pred[mask] == regime_id))
                report[f"{name}_accuracy"] = regime_acc

        logger.info("Training complete. Test accuracy: %.2f%%", accuracy * 100)
        return report

    def _walk_forward_cv(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> List[float]:
        """Time-series walk-forward cross-validation.

        Each fold trains on all data up to the split point and tests
        on the next segment, preserving temporal order.

        Args:
            X: Feature matrix.
            y: Label array.
            n_splits: Number of CV folds.

        Returns:
            List of accuracy scores per fold.
        """
        fold_size = len(X) // (n_splits + 1)
        scores: List[float] = []
        for i in range(n_splits):
            train_end = fold_size * (i + 2)
            test_end = min(train_end + fold_size, len(X))
            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]

            if len(X_test) == 0:
                continue

            model = lgb.LGBMClassifier(**self._lgb_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            score = float(model.score(X_test, y_test))
            scores.append(score)
            logger.debug(
                "Walk-forward fold %d: train=%d, test=%d, accuracy=%.4f",
                i + 1, len(X_train), len(X_test), score,
            )
        return scores

    # ─── Prediction ───

    def predict(self, df: pd.DataFrame) -> MarketRegime:
        """Predict the current market regime.

        Args:
            df: OHLCV DataFrame (uses last row's features).

        Returns:
            MarketRegime enum value.
        """
        self._check_fitted()
        features = self.compute_features(df)
        features = features[self._feature_names]

        last_row = features.iloc[[-1]].fillna(0)
        pred = int(self._model.predict(last_row)[0])

        return MarketRegime(pred)

    def predict_proba(self, df: pd.DataFrame) -> Dict[str, float]:
        """Predict regime probabilities.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Dict mapping regime name to probability.
        """
        self._check_fitted()
        features = self.compute_features(df)
        features = features[self._feature_names]

        last_row = features.iloc[[-1]].fillna(0)
        proba = self._model.predict_proba(last_row)[0]

        return {
            _REGIME_NAMES[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        }

    # ─── Feature Importance ───

    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Get sorted feature importance scores.

        Returns:
            List of (feature_name, importance) tuples sorted descending.
        """
        self._check_fitted()
        importance = self._model.feature_importances_
        pairs = list(zip(self._feature_names, importance.tolist()))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    # ─── Persistence ───

    def save_model(self, path: str) -> None:
        """Save trained model to disk.

        Args:
            path: File path (e.g., "models/regime_lgbm.joblib").
        """
        if not _HAS_JOBLIB:
            raise ImportError("joblib is required to save models. Install with: pip install joblib")
        self._check_fitted()

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self._model,
            "feature_names": self._feature_names,
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "learning_rate": self._learning_rate,
        }
        joblib.dump(data, path)
        logger.info("Model saved to %s", path)

    def load_model(self, path: str) -> None:
        """Load a trained model from disk.

        Args:
            path: File path to load.
        """
        if not _HAS_JOBLIB:
            raise ImportError("joblib is required to load models. Install with: pip install joblib")

        data = joblib.load(path)
        self._model = data["model"]
        self._feature_names = data["feature_names"]
        self._n_estimators = data.get("n_estimators", 200)
        self._max_depth = data.get("max_depth", 6)
        self._learning_rate = data.get("learning_rate", 0.05)
        self._is_fitted = True
        logger.info("Model loaded from %s (%d features)", path, len(self._feature_names))

    # ─── Helpers ───

    def _check_fitted(self) -> None:
        """Raise if model has not been trained."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError(
                "Model not fitted. Call fit() or load_model() first."
            )
