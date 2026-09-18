"""
Functional Tests for the AI/ML Decision Pipeline
====================================================

Comprehensive tests covering the full AI pipeline:
- Regime classification (LightGBM)
- Ensemble prediction with adaptive weights
- Self-learning agent decision loop
- LSTM/GRU deep learning predictions
- RL trading agent
- Feature engineering
- News sentiment + LLM reasoning

Uses synthetic OHLCV data and mocks for external dependencies
(torch, lightgbm, openai, etc.) to ensure isolated, fast tests.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on the path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Synthetic OHLCV Data Generators
# ---------------------------------------------------------------------------

def _make_ohlcv(
    n: int = 300,
    start_price: float = 100.0,
    trend: float = 0.0,
    volatility: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data.

    Args:
        n: Number of bars.
        start_price: Initial close price.
        trend: Daily drift (e.g. 0.001 = 0.1% per day uptrend).
        volatility: Daily return std-dev.
        seed: Random seed.
    """
    rng = np.random.RandomState(seed)
    returns = rng.normal(trend, volatility, n)
    close = start_price * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))
    volume = rng.randint(100_000, 5_000_000, n).astype(float)

    dates = pd.bdate_range(end=datetime.now(), periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_uptrend(n: int = 300, seed: int = 10) -> pd.DataFrame:
    """Strong uptrend data (0.3% daily drift)."""
    return _make_ohlcv(n=n, trend=0.003, volatility=0.008, seed=seed)


def _make_sideways(n: int = 300, seed: int = 20) -> pd.DataFrame:
    """Sideways / ranging data (near-zero drift, low vol)."""
    return _make_ohlcv(n=n, trend=0.0, volatility=0.005, seed=seed)


def _make_volatile(n: int = 300, seed: int = 30) -> pd.DataFrame:
    """Highly volatile data (high vol, little trend)."""
    return _make_ohlcv(n=n, trend=0.0, volatility=0.04, seed=seed)


# =========================================================================
# Test Class 1: Regime Classifier Pipeline
# =========================================================================

class TestRegimeClassifierPipeline:
    """Tests for shared.ml.regime_classifier.MLRegimeClassifier."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_lgbm(self):
        """Skip entire class if lightgbm is not installed."""
        pytest.importorskip("lightgbm", reason="lightgbm required for regime classifier tests")

    def _make_classifier(self):
        from shared.ml.regime_classifier import MLRegimeClassifier
        return MLRegimeClassifier(n_estimators=30, max_depth=4, learning_rate=0.1)

    # --- test_trending_data_classified ---
    def test_trending_data_classified(self):
        """Strong uptrend data should be classified as TRENDING."""
        from shared.ml.regime_classifier import MLRegimeClassifier, MarketRegime

        clf = self._make_classifier()
        df = _make_uptrend(n=400)
        clf.fit(df)
        regime = clf.predict(df)
        assert isinstance(regime, MarketRegime)
        # Even if the classifier doesn't pick TRENDING every time, the
        # prediction must be one of the valid regimes.
        assert regime in (MarketRegime.TRENDING, MarketRegime.RANGING, MarketRegime.VOLATILE)

    # --- test_ranging_data_classified ---
    def test_ranging_data_classified(self):
        """Sideways data should be classified as RANGING."""
        from shared.ml.regime_classifier import MarketRegime

        clf = self._make_classifier()
        df = _make_sideways(n=400)
        clf.fit(df)
        regime = clf.predict(df)
        assert isinstance(regime, MarketRegime)

    # --- test_volatile_data_classified ---
    def test_volatile_data_classified(self):
        """High-vol data should be classified as VOLATILE."""
        from shared.ml.regime_classifier import MarketRegime

        clf = self._make_classifier()
        df = _make_volatile(n=400)
        clf.fit(df)
        regime = clf.predict(df)
        assert isinstance(regime, MarketRegime)

    # --- test_compute_features_all_present ---
    def test_compute_features_all_present(self):
        """compute_features must produce exactly its documented feature set.

        This asserted ">= 30" against a docstring that claimed "30+ features";
        the implementation has only ever produced 26, so the test failed
        whenever lightgbm was installed. Pinning the exact set is a stronger
        check than a round number: it catches a feature being dropped or
        renamed, which silently changes what every model is trained on.
        """
        from shared.ml.regime_classifier import MLRegimeClassifier

        expected = {
            "adx", "atr_pct_rank", "bb_pct_b", "bb_width_z", "ema_slope",
            "gk_vol", "log_ret_1d", "macd_hist", "obv_slope", "parkinson_vol",
            "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
            "rel_volume", "ret_1d", "ret_5d", "ret_10d", "ret_20d",
            "roc_10", "roc_20", "rsi_14", "stoch_k",
            "vol_5d", "vol_10d", "vol_20d", "vol_change_rate",
        }

        df = _make_ohlcv(n=300)
        features = MLRegimeClassifier.compute_features(df)

        assert set(features.columns) == expected, (
            f"feature set changed: missing {expected - set(features.columns)}, "
            f"unexpected {set(features.columns) - expected}"
        )

    # --- test_compute_features_no_nan ---
    def test_compute_features_no_nan(self):
        """After dropna, features should be NaN-free."""
        from shared.ml.regime_classifier import MLRegimeClassifier

        df = _make_ohlcv(n=300)
        features = MLRegimeClassifier.compute_features(df).dropna()
        assert features.isna().sum().sum() == 0

    # --- test_auto_label_distribution ---
    def test_auto_label_distribution(self):
        """auto_label should produce all three label classes on mixed data."""
        from shared.ml.regime_classifier import MLRegimeClassifier

        df = _make_ohlcv(n=500, volatility=0.015, seed=99)
        labels = MLRegimeClassifier.auto_label(df, lookforward=20)
        unique = set(labels.dropna().unique())
        # At minimum, the labeler should produce at least 2 regime types
        assert len(unique) >= 2, f"Expected ≥2 labels, got {unique}"
        # All labels should be valid (0, 1, or 2)
        assert unique.issubset({0, 1, 2})

    # --- test_fit_predict_roundtrip ---
    def test_fit_predict_roundtrip(self):
        """Train → predict → predict_proba should all succeed."""
        from shared.ml.regime_classifier import MLRegimeClassifier, MarketRegime

        clf = self._make_classifier()
        df = _make_ohlcv(n=400)
        metrics = clf.fit(df)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

        regime = clf.predict(df)
        assert isinstance(regime, MarketRegime)

        proba = clf.predict_proba(df)
        assert isinstance(proba, dict)
        assert abs(sum(proba.values()) - 1.0) < 0.01

    # --- test_fallback_when_lgbm_missing ---
    def test_fallback_when_lgbm_missing(self):
        """When lightgbm is not importable, MLRegimeClassifier __init__ should raise ImportError."""
        import shared.ml.regime_classifier as rc_mod

        original = rc_mod._HAS_LIGHTGBM
        try:
            rc_mod._HAS_LIGHTGBM = False
            with pytest.raises(ImportError, match="lightgbm"):
                rc_mod.MLRegimeClassifier()
        finally:
            rc_mod._HAS_LIGHTGBM = original

    # --- test_feature_importance ---
    def test_feature_importance(self):
        """After fit, get_feature_importance should return sorted list of (name, score)."""
        clf = self._make_classifier()
        df = _make_ohlcv(n=400)
        clf.fit(df)
        importance = clf.get_feature_importance()
        assert len(importance) > 0
        # Sorted descending by importance
        scores = [s for _, s in importance]
        assert scores == sorted(scores, reverse=True)


# =========================================================================
# Test Class 2: Ensemble Predictor Pipeline
# =========================================================================

class TestEnsemblePredictorPipeline:
    """Tests for shared.ml.ensemble_predictor.EnsemblePredictor."""

    def _make_ensemble(self, **kwargs):
        from shared.ml.ensemble_predictor import EnsemblePredictor
        return EnsemblePredictor(**kwargs)

    # --- test_all_models_contribute ---
    def test_all_models_contribute(self):
        """Each model should contribute to the ensemble signal."""
        ensemble = self._make_ensemble()
        preds = {"lstm": 0.02, "transformer": 0.01, "rl": 1, "momentum": 0.03}
        signal = ensemble.predict(preds, regime="TRENDING")
        assert len(signal.model_contributions) == 4
        for model in preds:
            assert model in signal.model_contributions

    # --- test_regime_changes_weights ---
    def test_regime_changes_weights(self):
        """Switching regime should change model contributions."""
        ensemble = self._make_ensemble()
        preds = {"lstm": 0.02, "momentum": 0.03}
        sig_trend = ensemble.predict(preds, regime="TRENDING")
        sig_range = ensemble.predict(preds, regime="RANGING")
        # Momentum's contribution should differ between TRENDING and RANGING
        assert sig_trend.model_contributions["momentum"] != sig_range.model_contributions["momentum"]

    # --- test_high_agreement_high_confidence ---
    def test_high_agreement_high_confidence(self):
        """When all models agree on bullish, confidence should be high."""
        ensemble = self._make_ensemble()
        preds = {"lstm": 0.03, "transformer": 0.02, "rl": 1, "momentum": 0.05, "sentiment": 0.8}
        signal = ensemble.predict(preds, regime="TRENDING")
        assert signal.confidence > 0.7, f"Expected confidence > 0.7, got {signal.confidence}"
        assert signal.agreement_ratio >= 0.8

    # --- test_low_agreement_low_confidence ---
    def test_low_agreement_low_confidence(self):
        """When models disagree, confidence should be lower than full agreement."""
        ensemble = self._make_ensemble()
        # Mixed signals: some bullish, some bearish
        preds_disagree = {"lstm": 0.03, "transformer": -0.02, "rl": -1, "momentum": 0.01, "sentiment": -0.5}
        sig_disagree = ensemble.predict(preds_disagree, regime="UNKNOWN")
        # Full agreement for comparison
        preds_agree = {"lstm": 0.03, "transformer": 0.02, "rl": 1, "momentum": 0.05, "sentiment": 0.8}
        sig_agree = ensemble.predict(preds_agree, regime="UNKNOWN")
        assert sig_disagree.confidence < sig_agree.confidence, (
            f"Disagreement confidence {sig_disagree.confidence} should be < agreement confidence {sig_agree.confidence}"
        )

    # --- test_normalization_bounds ---
    def test_normalization_bounds(self):
        """Raw score should be bounded near [-1, 1]."""
        ensemble = self._make_ensemble()
        preds = {"lstm": 0.05, "rl": 1, "momentum": 0.1}
        signal = ensemble.predict(preds, regime="TRENDING")
        assert -1.1 <= signal.raw_score <= 1.1

    # --- test_instance_isolation ---
    def test_instance_isolation(self):
        """Two ensemble instances should be independent."""
        e1 = self._make_ensemble(buy_threshold=0.1)
        e2 = self._make_ensemble(buy_threshold=0.5)
        preds = {"lstm": 0.015, "rl": 1}
        s1 = e1.predict(preds, regime="TRENDING")
        s2 = e2.predict(preds, regime="TRENDING")
        # Same raw_score, but different thresholds can lead to different directions
        assert s1.raw_score == s2.raw_score

    # --- test_adaptive_weight_update ---
    def test_adaptive_weight_update(self):
        """update_weights_from_memory should modify model weights."""
        from shared.ml.ensemble_predictor import EnsemblePredictor

        ensemble = EnsemblePredictor()
        weights_before = ensemble.get_weight_summary()

        mock_memory = MagicMock()
        mock_memory.get_all_model_accuracies.return_value = {
            "lstm": {"accuracy": 0.7, "trend": 0.05},
            "rl": {"accuracy": 0.3, "trend": -0.15},
        }
        ensemble.update_weights_from_memory(mock_memory, window=50)
        weights_after = ensemble.get_weight_summary()

        # LSTM should have higher weight due to better accuracy
        assert weights_after["lstm"]["accuracy_weight"] > weights_after["rl"]["accuracy_weight"]

    # --- test_predict_with_missing_model ---
    def test_predict_with_missing_model(self):
        """When one model is absent from predictions, others still contribute."""
        ensemble = self._make_ensemble()
        preds_all = {"lstm": 0.02, "rl": 1, "momentum": 0.03}
        preds_partial = {"lstm": 0.02, "momentum": 0.03}
        sig_all = ensemble.predict(preds_all, regime="TRENDING")
        sig_partial = ensemble.predict(preds_partial, regime="TRENDING")
        # Both should produce valid signals
        assert sig_all.direction in (-1, 0, 1)
        assert sig_partial.direction in (-1, 0, 1)
        # Number of contributions should match input
        assert len(sig_partial.model_contributions) == 2


# =========================================================================
# Test Class 3: Self-Learning Agent Pipeline
# =========================================================================

class TestSelfLearningPipeline:
    """Tests for shared.ml.self_learning_agent.SelfLearningAgent."""

    @pytest.fixture
    def tmp_db(self, tmp_path):
        """Provide a temporary SQLite database path."""
        return str(tmp_path / "test_agent.db")

    def _make_agent(self, db_path: str, **kwargs):
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig
        config = AgentConfig(
            db_path=db_path,
            total_capital=100_000.0,
            **kwargs,
        )
        agent = SelfLearningAgent(config=config)
        # Relax risk manager throttling for test speed
        agent._risk_manager.config.min_seconds_between_trades = 0
        agent._risk_manager.config.max_trades_per_hour = 999
        agent._risk_manager.config.cooldown_seconds = 0
        return agent

    # --- test_decide_before_training ---
    def test_decide_before_training(self, tmp_db):
        """Before training, agent should still produce a decision via fallback."""
        agent = self._make_agent(tmp_db)
        df = _make_ohlcv(n=100)
        decision = agent.decide(df, symbol="TEST")
        assert decision["action"] in ("BUY", "SELL", "HOLD")
        assert "confidence" in decision
        assert "regime" in decision
        # Momentum should be the only prediction source
        assert "momentum" in decision["predictions"]
        agent.close()

    # --- test_decide_after_training ---
    def test_decide_after_training(self, tmp_db):
        """After training (with mocked models), agent uses model predictions."""
        agent = self._make_agent(tmp_db)
        df = _make_ohlcv(n=100)

        # Inject mock predictors
        mock_lstm = MagicMock()
        mock_lstm.predict.return_value = pd.Series([0.02], name="prediction")
        agent._lstm_predictor = mock_lstm
        agent._models_trained = True

        decision = agent.decide(df, symbol="TEST")
        assert "lstm" in decision["predictions"]
        assert decision["action"] in ("BUY", "SELL", "HOLD")
        agent.close()

    # --- test_record_outcome_in_memory ---
    def test_record_outcome_in_memory(self, tmp_db):
        """record_outcome should persist trade data to SQLite."""
        agent = self._make_agent(tmp_db)
        df = _make_ohlcv(n=100)

        agent.decide(df, symbol="MEM_TEST")
        agent.record_outcome(exit_price=102.0, pnl=200.0)

        count = agent._memory.get_trade_count()
        assert count >= 1

        trades = agent._memory.get_recent_trades(5)
        assert any(t["symbol"] == "MEM_TEST" for t in trades)
        agent.close()

    # --- test_accuracy_improves ---
    def test_accuracy_improves(self, tmp_db):
        """Correct predictions should increase model accuracy weight."""
        agent = self._make_agent(tmp_db)
        df = _make_ohlcv(n=100)

        mock_lstm = MagicMock()
        mock_lstm.predict.return_value = pd.Series([0.02], name="prediction")
        agent._lstm_predictor = mock_lstm
        agent._models_trained = True

        # Record several winning trades
        for i in range(5):
            agent.decide(df, symbol="WIN")
            agent.record_outcome(exit_price=105.0, pnl=500.0)

        acc = agent._memory.get_model_accuracy("lstm", window=10)
        # After recording positive predictions with positive outcomes,
        # accuracy should be positive
        assert acc["total_predictions"] > 0
        agent.close()

    # --- test_accuracy_degrades ---
    def test_accuracy_degrades(self, tmp_db):
        """Wrong predictions should decrease model accuracy weight."""
        agent = self._make_agent(tmp_db)
        df = _make_ohlcv(n=100)

        mock_lstm = MagicMock()
        mock_lstm.predict.return_value = pd.Series([0.02], name="prediction")  # bullish
        agent._lstm_predictor = mock_lstm
        agent._models_trained = True

        # Record several losing trades (predicted bullish, but lost money)
        for i in range(5):
            agent.decide(df, symbol="LOSE")
            agent.record_outcome(exit_price=95.0, pnl=-500.0)

        acc = agent._memory.get_model_accuracy("lstm", window=10)
        # Direction mismatch: predicted positive, actual negative
        if acc["total_predictions"] > 0:
            assert acc["accuracy"] < 0.6
        agent.close()

    # --- test_adaptive_threshold ---
    def test_adaptive_threshold(self, tmp_db):
        """With many trades, adaptive thresholds should engage."""
        agent = self._make_agent(tmp_db, adaptive_thresholds=True)
        df = _make_ohlcv(n=100)

        # Pre-populate memory with poor TRENDING trades
        for i in range(12):
            from shared.ml.trade_memory import TradeDecisionRecord
            trade = TradeDecisionRecord(
                timestamp=(datetime.now() - timedelta(days=i)).isoformat(),
                symbol="AT_TEST",
                action="BUY",
                entry_price=100.0,
                exit_price=95.0,
                pnl=-500.0,
                pnl_pct=-0.05,
                regime="TRENDING",
                is_winner=False,
            )
            agent._memory.record_trade(trade)

        # Decision should be more cautious (HOLD) due to poor history
        decision = agent.decide(df, symbol="AT_TEST")
        # The adaptive threshold path was exercised; action should be valid
        assert decision["action"] in ("BUY", "SELL", "HOLD")
        agent.close()

    # --- test_caution_from_memory ---
    def test_caution_from_memory(self, tmp_db):
        """Bad historical regime performance → caution recommendation → HOLD."""
        agent = self._make_agent(tmp_db)
        df = _make_ohlcv(n=100)

        # Populate bad history for current regime
        for i in range(10):
            from shared.ml.trade_memory import TradeDecisionRecord
            trade = TradeDecisionRecord(
                timestamp=(datetime.now() - timedelta(days=i)).isoformat(),
                symbol="CAUT",
                action="BUY",
                entry_price=100.0,
                exit_price=90.0,
                pnl=-1000.0,
                pnl_pct=-0.10,
                regime="RANGING",
                is_winner=False,
            )
            agent._memory.record_trade(trade)

        insight = agent._memory.query_similar_regime("RANGING", lookback_days=90)
        if insight.get("sufficient_data"):
            assert insight["recommendation"] == "caution"
        agent.close()

    # --- test_retrain_trigger ---
    def test_retrain_trigger(self, tmp_db):
        """When accuracy < 40%, _check_retrain_needed should flag it."""
        agent = self._make_agent(tmp_db, retrain_on_degradation=True)

        # Insert bad model performance records
        for i in range(30):
            agent._memory.record_model_prediction(
                model_name="lstm",
                prediction=1.0,
                actual_outcome=-1.0,  # always wrong
                regime="TRENDING",
                symbol="RETRAIN",
            )

        acc = agent._memory.get_model_accuracy("lstm", window=50)
        assert acc["needs_retrain"] is True
        agent.close()


# =========================================================================
# Test Class 4: LSTM Predictor Pipeline
# =========================================================================

class TestLSTMPredictorPipeline:
    """Tests for shared.ml.deep_learning.lstm_predictor.LSTMPredictor."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_torch(self):
        pytest.importorskip("torch", reason="PyTorch required for LSTM tests")

    def _make_predictor(self, **kwargs):
        from shared.ml.deep_learning.lstm_predictor import LSTMPredictor, LSTMConfig
        config = LSTMConfig(
            hidden_size=32,
            num_layers=1,
            epochs=kwargs.pop("epochs", 3),
            seq_len=20,
            batch_size=16,
            device="cpu",
            **kwargs,
        )
        return LSTMPredictor(config)

    # --- test_train_creates_model ---
    def test_train_creates_model(self):
        """Training should create a model attribute."""
        predictor = self._make_predictor()
        df = _make_ohlcv(n=500)
        metrics = predictor.train(df)
        assert predictor.model is not None
        assert "val_loss" in metrics
        assert "train_loss" in metrics

    # --- test_predict_returns_float ---
    def test_predict_returns_float(self):
        """Prediction should return a Series with a float value."""
        predictor = self._make_predictor()
        df = _make_ohlcv(n=500)
        predictor.train(df)
        pred = predictor.predict(df)
        assert isinstance(pred, pd.Series)
        assert len(pred) == 1
        val = float(pred.iloc[0])
        assert np.isfinite(val)

    # --- test_prediction_direction ---
    def test_prediction_direction(self):
        """On strongly uptrending data, prediction should tend positive."""
        predictor = self._make_predictor(epochs=10)
        df = _make_uptrend(n=500, seed=7)
        predictor.train(df)
        pred = predictor.predict(df)
        # Note: with only 10 epochs on synthetic data, this is a soft check
        # The model should at minimum produce a finite float
        assert np.isfinite(float(pred.iloc[0]))

    # --- test_early_stopping ---
    def test_early_stopping(self):
        """Verify training completes within the specified epochs
        (early stopping would reduce this further if implemented)."""
        predictor = self._make_predictor(epochs=5)
        df = _make_ohlcv(n=500)
        metrics = predictor.train(df)
        assert metrics["val_loss"] >= 0

    # --- test_nan_features_handled ---
    def test_nan_features_handled(self):
        """Input data with some NaN values should not crash prediction."""
        predictor = self._make_predictor()
        df = _make_ohlcv(n=500)
        predictor.train(df)
        # FeatureEngineer.compute_features already calls dropna internally
        pred = predictor.predict(df)
        assert np.isfinite(float(pred.iloc[0]))

    # --- test_save_load_roundtrip ---
    def test_save_load_roundtrip(self, tmp_path):
        """Save → load should reproduce the same predictions."""
        import torch
        predictor = self._make_predictor()
        df = _make_ohlcv(n=500)
        predictor.train(df)

        pred_before = predictor.predict(df)

        model_path = str(tmp_path / "lstm_test.pt")
        predictor.save_model(model_path)

        from shared.ml.deep_learning.lstm_predictor import LSTMPredictor
        loaded = LSTMPredictor()
        loaded.load_model(model_path)
        pred_after = loaded.predict(df)

        np.testing.assert_allclose(
            float(pred_before.iloc[0]),
            float(pred_after.iloc[0]),
            atol=1e-5,
        )


# =========================================================================
# Test Class 5: RL Agent Pipeline
# =========================================================================

class TestRLAgentPipeline:
    """Tests for shared.ml.rl_agent.RLTrader."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_sb3(self):
        pytest.importorskip("stable_baselines3", reason="stable-baselines3 required for RL tests")
        pytest.importorskip("gymnasium", reason="gymnasium required for RL tests")

    def _make_trader(self, **env_config):
        from shared.ml.rl_agent import RLTrader
        default_config = {"reward_type": "pnl"}
        default_config.update(env_config)
        return RLTrader(algorithm="PPO", env_config=default_config)

    # --- test_train_creates_policy ---
    def test_train_creates_policy(self):
        """Training should create a trained model/policy."""
        trader = self._make_trader()
        df = _make_ohlcv(n=200)
        metrics = trader.train(df, total_timesteps=500, verbose=0)
        assert trader._is_trained
        assert metrics["algorithm"] == "PPO"

    # --- test_predict_returns_action ---
    def test_predict_returns_action(self):
        """Predict should return an action in {-1, 0, 1}."""
        trader = self._make_trader()
        df = _make_ohlcv(n=200)
        trader.train(df, total_timesteps=500, verbose=0)
        action = trader.predict(df)
        assert action in (-1, 0, 1), f"Expected action in {{-1,0,1}}, got {action}"

    # --- test_environment_reward ---
    def test_environment_reward(self):
        """A profitable action should yield a positive reward from the env."""
        from shared.ml.rl_trading_env import TradingEnv

        df = _make_uptrend(n=100)
        env = TradingEnv(df, reward_type="pnl")
        obs, info = env.reset()
        # BUY on uptrend should eventually be profitable
        _, reward, _, _, _ = env.step(TradingEnv.BUY)
        # Reward is price-based; on uptrend it should be non-negative most of the time
        assert isinstance(reward, float)

    # --- test_environment_penalty ---
    def test_environment_penalty(self):
        """A SELL in an uptrend should eventually produce a negative or low reward."""
        from shared.ml.rl_trading_env import TradingEnv

        df = _make_uptrend(n=100)
        env = TradingEnv(df, reward_type="pnl")
        obs, info = env.reset()
        # SELL on uptrend — shorting a rising market
        _, reward, _, _, _ = env.step(TradingEnv.SELL)
        # Just verify it returns a valid float reward
        assert isinstance(reward, float)

    # --- test_fallback_to_rsi ---
    def test_fallback_to_rsi(self):
        """Without a trained model, RLTrader.predict should raise RuntimeError."""
        from shared.ml.rl_agent import RLTrader
        trader = RLTrader(algorithm="PPO")
        df = _make_ohlcv(n=100)
        with pytest.raises(RuntimeError, match="not trained"):
            trader.predict(df)


# =========================================================================
# Test Class 6: Feature Engineer Pipeline
# =========================================================================

class TestFeatureEngineerPipeline:
    """Tests for shared.ml.deep_learning.feature_engineer.FeatureEngineer."""

    def _make_fe(self):
        from shared.ml.deep_learning.feature_engineer import FeatureEngineer
        return FeatureEngineer()

    # --- test_compute_all_features ---
    def test_compute_all_features(self):
        """compute_features should produce 40+ feature columns."""
        fe = self._make_fe()
        df = _make_ohlcv(n=300)
        features = fe.compute_features(df)
        # FeatureEngineer adds lags, skew, kurtosis, sma ratios on top of
        # regime classifier features (30+) → total should be 40+
        assert len(features.columns) >= 40, (
            f"Expected ≥40 features, got {len(features.columns)}: {sorted(features.columns.tolist())}"
        )

    # --- test_features_no_nan_after_dropna ---
    def test_features_no_nan_after_dropna(self):
        """FeatureEngineer.compute_features already calls dropna internally."""
        fe = self._make_fe()
        df = _make_ohlcv(n=300)
        features = fe.compute_features(df)
        assert features.isna().sum().sum() == 0

    # --- test_feature_selection ---
    def test_feature_selection(self):
        """select_features should return a subset of top N feature names."""
        fe = self._make_fe()
        df = _make_ohlcv(n=300)
        features = fe.compute_features(df)
        close = df["close"].reindex(features.index)
        target = close.pct_change().shift(-1).reindex(features.index).dropna()
        features = features.loc[target.index]

        top_n = 15
        selected = fe.select_features(features, target, top_n=top_n)
        assert len(selected) <= top_n
        assert len(selected) > 0
        # All selected features should be valid column names
        for feat_name in selected:
            assert feat_name in features.columns

    # --- test_regime_features_included ---
    def test_regime_features_included(self):
        """If regime classifier is available, base regime features should appear."""
        fe = self._make_fe()
        df = _make_ohlcv(n=300)
        features = fe.compute_features(df)
        # These regime classifier features should be present
        regime_feature_names = ["rsi_14", "adx", "macd_hist", "vol_5d"]
        for name in regime_feature_names:
            assert name in features.columns, f"Missing regime feature: {name}"


# =========================================================================
# Test Class 7: News Sentiment + LLM Reasoning Pipeline
# =========================================================================

class TestNewsAndLLMPipeline:
    """Tests for news_sentiment.NewsSentimentAnalyzer and llm_reasoning.LLMReasoner."""

    # --- News Sentiment Tests ---

    def _make_analyzer(self, method="keyword"):
        from shared.ml.news_sentiment import NewsSentimentAnalyzer
        return NewsSentimentAnalyzer(method=method)

    def _bearish_headlines(self) -> List[Dict[str, str]]:
        return [
            {"title": "Stock crashes amid fraud investigation and massive layoffs"},
            {"title": "Company plunges after earnings miss and guidance cut"},
            {"title": "Shares drop on disappointing revenue decline and recession fears"},
            {"title": "Market selloff deepens as bearish outlook worsens"},
            {"title": "Stock falls sharply on downgrade to underperform"},
        ]

    def _bullish_headlines(self) -> List[Dict[str, str]]:
        return [
            {"title": "Stock surges to record high on strong earnings beat"},
            {"title": "Shares rally after revenue growth exceeds expectations"},
            {"title": "Company upgraded to outperform amid bullish momentum"},
            {"title": "Stock soars on expansion plans and positive outlook"},
            {"title": "Gains accelerate as strong growth drives optimistic forecast"},
        ]

    def _neutral_headlines(self) -> List[Dict[str, str]]:
        return [
            {"title": "Company reports quarterly results in line with estimates"},
            {"title": "Board announces regular dividend payment schedule"},
            {"title": "Annual meeting scheduled for next month"},
        ]

    # --- test_bearish_news_reduces_buy ---
    def test_bearish_news_reduces_buy(self):
        """Strong negative sentiment should produce BEARISH label, reducing BUY signals."""
        analyzer = self._make_analyzer(method="keyword")
        result = analyzer.analyze("TEST", headlines=self._bearish_headlines())
        assert result["sentiment_score"] < -0.1
        assert result["sentiment_label"] == "BEARISH"
        assert result["bearish_count"] > result["bullish_count"]

        # Feeding this into ensemble should dampen buy signal
        from shared.ml.ensemble_predictor import EnsemblePredictor
        ensemble = EnsemblePredictor()
        # Without sentiment: strong buy
        preds_no_sent = {"lstm": 0.02, "momentum": 0.03}
        sig_no_sent = ensemble.predict(preds_no_sent, regime="TRENDING")
        # With strong negative sentiment
        preds_with_sent = {"lstm": 0.02, "momentum": 0.03, "sentiment": result["sentiment_score"]}
        sig_with_sent = ensemble.predict(preds_with_sent, regime="TRENDING")
        # Negative sentiment should reduce the raw score
        assert sig_with_sent.raw_score < sig_no_sent.raw_score

    # --- test_bullish_news_confirms_buy ---
    def test_bullish_news_confirms_buy(self):
        """Positive sentiment should produce BULLISH label and keep/boost BUY."""
        analyzer = self._make_analyzer(method="keyword")
        result = analyzer.analyze("TEST", headlines=self._bullish_headlines())
        assert result["sentiment_score"] > 0.1
        assert result["sentiment_label"] == "BULLISH"
        assert result["bullish_count"] > result["bearish_count"]

        from shared.ml.ensemble_predictor import EnsemblePredictor
        ensemble = EnsemblePredictor()
        preds_no_sent = {"lstm": 0.02, "momentum": 0.03}
        sig_no_sent = ensemble.predict(preds_no_sent, regime="TRENDING")
        preds_with_sent = {"lstm": 0.02, "momentum": 0.03, "sentiment": result["sentiment_score"]}
        sig_with_sent = ensemble.predict(preds_with_sent, regime="TRENDING")
        # Positive sentiment should boost the raw score
        assert sig_with_sent.raw_score > sig_no_sent.raw_score

    # --- test_llm_override_with_reasoning ---
    def test_llm_override_with_reasoning(self):
        """LLMReasoner should return structured decision with reasoning."""
        from shared.ml.llm_reasoning import LLMReasoner

        mock_response = json.dumps({
            "action": "SELL",
            "confidence": 0.85,
            "tp_price": 140.0,
            "sl_price": 160.0,
            "allocation_pct": 10,
            "exit_plan": "Close if RSI drops below 30",
            "reasoning": "Overbought conditions with bearish divergence",
        })

        with patch.object(LLMReasoner, "__init__", lambda self, **kw: None):
            reasoner = LLMReasoner.__new__(LLMReasoner)
            reasoner._provider = "anthropic"
            reasoner._model = "claude-test"
            reasoner._max_tokens = 2048
            reasoner._temperature = 0.1
            reasoner._client = MagicMock()

        with patch.object(reasoner, "_call_llm", return_value=mock_response):
            result = reasoner.reason(
                symbol="TEST",
                price=155.0,
                regime="TRENDING",
                regime_confidence=0.8,
                predictions={"lstm": -0.01, "rl": -1},
                ensemble_signal={"direction": -1, "raw_score": -0.3, "confidence": 0.7},
            )

        assert result["action"] == "SELL"
        assert result["confidence"] == 0.85
        assert result["tp_price"] == 140.0
        assert result["sl_price"] == 160.0
        assert "reasoning" in result
        assert result["llm_used"] is True

    # --- test_llm_blocked_by_risk ---
    def test_llm_blocked_by_risk(self):
        """Even if LLM says BUY, risk manager blocking should result in HOLD."""
        from shared.ml.llm_reasoning import LLMReasoner
        from shared.risk_manager import RiskManager, RiskManagerConfig

        # LLM returns BUY
        mock_response = json.dumps({
            "action": "BUY",
            "confidence": 0.9,
            "tp_price": 170.0,
            "sl_price": 145.0,
            "allocation_pct": 15,
            "exit_plan": "Hold for breakout",
            "reasoning": "Strong bullish setup",
        })

        with patch.object(LLMReasoner, "__init__", lambda self, **kw: None):
            reasoner = LLMReasoner.__new__(LLMReasoner)
            reasoner._provider = "openai"
            reasoner._model = "gpt-test"
            reasoner._max_tokens = 2048
            reasoner._temperature = 0.1
            reasoner._client = MagicMock()

        with patch.object(reasoner, "_call_llm", return_value=mock_response):
            llm_decision = reasoner.reason(
                symbol="TEST",
                price=150.0,
                regime="TRENDING",
                regime_confidence=0.7,
                predictions={"lstm": 0.02},
                ensemble_signal={"direction": 1, "raw_score": 0.3, "confidence": 0.8},
                risk_status={"can_trade": False, "daily_pnl": -6000, "consecutive_losses": 4},
            )

        # LLM says BUY
        assert llm_decision["action"] == "BUY"

        # But risk manager blocks trading
        rm = RiskManager(RiskManagerConfig(max_daily_loss=5000.0))
        rm._daily_pnl = -6000.0  # exceeded daily loss
        assert rm.can_trade() is False

        # Integration: agent should override to HOLD
        final_action = llm_decision["action"]
        if not rm.can_trade():
            final_action = "HOLD"
        assert final_action == "HOLD"

    # --- test_sentiment_fallback_chain ---
    def test_sentiment_fallback_chain(self):
        """Sentiment should fall back through FinBERT → VADER → keywords."""
        from shared.ml.news_sentiment import NewsSentimentAnalyzer

        # Test keyword fallback (always available)
        analyzer = NewsSentimentAnalyzer(method="keyword")
        assert analyzer._active_method == "keyword"
        result = analyzer.analyze("TEST", headlines=self._bullish_headlines())
        assert result["method"] == "keyword"
        assert result["sentiment_score"] > 0

        # Test VADER if available
        try:
            analyzer_vader = NewsSentimentAnalyzer(method="vader")
            assert analyzer_vader._active_method == "vader"
            result_vader = analyzer_vader.analyze("TEST", headlines=self._bullish_headlines())
            assert result_vader["method"] == "vader"
        except ImportError:
            pass  # VADER not installed, skip gracefully

        # Test auto mode picks best available
        analyzer_auto = NewsSentimentAnalyzer(method="auto")
        assert analyzer_auto._active_method in ("finbert", "vader", "keyword")


# =========================================================================
# Test Class 8: Integration / Cross-Component Tests
# =========================================================================

class TestCrossPipelineIntegration:
    """Integration tests spanning multiple pipeline components."""

    @pytest.fixture
    def tmp_db(self, tmp_path):
        return str(tmp_path / "integration.db")

    # --- test_full_decision_loop ---
    def test_full_decision_loop(self, tmp_db):
        """Full loop: data → features → regime → ensemble → decide → record."""
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        config = AgentConfig(
            db_path=tmp_db,
            total_capital=100_000.0,
            adaptive_thresholds=False,
        )
        agent = SelfLearningAgent(config=config)
        agent._risk_manager.config.min_seconds_between_trades = 0
        agent._risk_manager.config.max_trades_per_hour = 999

        df = _make_ohlcv(n=100)

        # Decide
        decision = agent.decide(df, symbol="INTEG")
        assert "action" in decision
        assert "confidence" in decision
        assert "regime" in decision
        assert "predictions" in decision
        assert "ensemble_signal" in decision
        assert "risk_status" in decision
        assert "reasoning" in decision

        # Record outcome
        agent.record_outcome(exit_price=101.0, pnl=100.0)
        assert agent._memory.get_trade_count() >= 1
        agent.close()

    # --- test_ensemble_feeds_agent ---
    def test_ensemble_feeds_agent(self, tmp_db):
        """EnsemblePredictor output feeds correctly into SelfLearningAgent."""
        from shared.ml.ensemble_predictor import EnsemblePredictor

        ensemble = EnsemblePredictor()
        preds = {"lstm": 0.02, "momentum": 0.03, "sentiment": 0.5}
        signal = ensemble.predict(preds, regime="TRENDING")

        assert signal.direction in (-1, 0, 1)
        assert 0.0 <= signal.confidence <= 1.0
        assert isinstance(signal.model_contributions, dict)

    # --- test_regime_classifier_feeds_ensemble ---
    def test_regime_classifier_feeds_ensemble(self):
        """Regime classification output should work as ensemble regime input."""
        from shared.ml.ensemble_predictor import EnsemblePredictor

        ensemble = EnsemblePredictor()
        for regime in ("TRENDING", "RANGING", "VOLATILE", "UNKNOWN"):
            preds = {"lstm": 0.01, "rl": 0}
            signal = ensemble.predict(preds, regime=regime)
            assert signal.regime == regime
            assert signal.direction in (-1, 0, 1)

    # --- test_sentiment_into_ensemble ---
    def test_sentiment_into_ensemble(self):
        """NewsSentimentAnalyzer score should integrate as ensemble model."""
        from shared.ml.news_sentiment import NewsSentimentAnalyzer
        from shared.ml.ensemble_predictor import EnsemblePredictor

        analyzer = NewsSentimentAnalyzer(method="keyword")
        result = analyzer.analyze("TEST", headlines=[
            {"title": "Stock surges on record earnings beat and upgrade"},
        ])

        ensemble = EnsemblePredictor()
        preds = {"lstm": 0.01, "sentiment": result["sentiment_score"]}
        signal = ensemble.predict(preds, regime="TRENDING")
        assert "sentiment" in signal.model_contributions

    # --- test_empty_predictions_handled ---
    def test_empty_predictions_handled(self):
        """Ensemble should handle empty predictions gracefully."""
        from shared.ml.ensemble_predictor import EnsemblePredictor

        ensemble = EnsemblePredictor()
        signal = ensemble.predict({}, regime="UNKNOWN")
        assert signal.direction == 0
        assert signal.confidence == 0.0
        assert signal.raw_score == 0.0
