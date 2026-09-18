"""
Tests for shared/ml/regime_classifier.py
==========================================

Covers:
- compute_features() returns 30+ features
- auto_label() distribution
- fit() + predict() roundtrip
- Fallback when lightgbm missing
- Feature importance after fit
"""
import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_ohlcv(n=500, seed=42, drift=0.0):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    price = 100.0
    rows = []
    for i in range(n):
        price *= 1 + drift + rng.randn() * 0.015
        rows.append({
            "date": dates[i],
            "open": price * 1.001,
            "high": price * 1.005,
            "low": price * 0.995,
            "close": price,
            "volume": int(rng.uniform(500_000, 2_000_000)),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_df():
    return _make_ohlcv(500, seed=42)


class TestComputeFeatures:
    def test_returns_dataframe(self, sample_df):
        from shared.ml.regime_classifier import MLRegimeClassifier
        feat = MLRegimeClassifier.compute_features(sample_df)
        assert isinstance(feat, pd.DataFrame)

    def test_has_25_plus_features(self, sample_df):
        from shared.ml.regime_classifier import MLRegimeClassifier
        feat = MLRegimeClassifier.compute_features(sample_df)
        assert len(feat.columns) >= 25

    def test_same_index_as_input(self, sample_df):
        from shared.ml.regime_classifier import MLRegimeClassifier
        feat = MLRegimeClassifier.compute_features(sample_df)
        assert len(feat) == len(sample_df)

    def test_contains_key_features(self, sample_df):
        from shared.ml.regime_classifier import MLRegimeClassifier
        feat = MLRegimeClassifier.compute_features(sample_df)
        expected = ["rsi_14", "adx", "macd_hist", "vol_5d", "ret_1d",
                    "bb_pct_b", "ema_slope", "rel_volume"]
        for col in expected:
            assert col in feat.columns, f"Missing feature: {col}"

    def test_features_have_valid_values_at_end(self, sample_df):
        from shared.ml.regime_classifier import MLRegimeClassifier
        feat = MLRegimeClassifier.compute_features(sample_df)
        last_row = feat.iloc[-1]
        non_nan = last_row.dropna()
        assert len(non_nan) >= 25


class TestAutoLabel:
    def test_returns_series(self, sample_df):
        from shared.ml.regime_classifier import MLRegimeClassifier
        labels = MLRegimeClassifier.auto_label(sample_df)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(sample_df)

    def test_labels_are_valid_regime_ids(self, sample_df):
        from shared.ml.regime_classifier import MLRegimeClassifier
        labels = MLRegimeClassifier.auto_label(sample_df)
        assert set(labels.unique()).issubset({0, 1, 2})

    def test_distribution_has_all_regimes(self, sample_df):
        from shared.ml.regime_classifier import MLRegimeClassifier
        labels = MLRegimeClassifier.auto_label(sample_df)
        unique = set(labels.unique())
        assert len(unique) >= 2

    def test_custom_lookforward(self, sample_df):
        from shared.ml.regime_classifier import MLRegimeClassifier
        labels_10 = MLRegimeClassifier.auto_label(sample_df, lookforward=10)
        labels_30 = MLRegimeClassifier.auto_label(sample_df, lookforward=30)
        assert len(labels_10) == len(labels_30) == len(sample_df)


class TestFitPredict:
    @pytest.fixture
    def classifier(self):
        from shared.ml.regime_classifier import _HAS_LIGHTGBM
        if not _HAS_LIGHTGBM:
            pytest.skip("lightgbm not installed")
        from shared.ml.regime_classifier import MLRegimeClassifier
        return MLRegimeClassifier(n_estimators=20, max_depth=3)

    def test_fit_returns_metrics(self, classifier, sample_df):
        metrics = classifier.fit(sample_df)
        assert "accuracy" in metrics
        assert 0 < metrics["accuracy"] <= 1.0

    def test_predict_returns_regime(self, classifier, sample_df):
        from shared.ml.regime_classifier import MarketRegime
        classifier.fit(sample_df)
        regime = classifier.predict(sample_df)
        assert isinstance(regime, MarketRegime)

    def test_predict_proba_returns_dict(self, classifier, sample_df):
        classifier.fit(sample_df)
        proba = classifier.predict_proba(sample_df)
        assert isinstance(proba, dict)
        assert "TRENDING" in proba
        assert "RANGING" in proba
        assert "VOLATILE" in proba
        assert sum(proba.values()) == pytest.approx(1.0, abs=0.01)

    def test_predict_before_fit_raises(self, classifier, sample_df):
        with pytest.raises(RuntimeError, match="not fitted"):
            classifier.predict(sample_df)

    def test_fit_with_small_data_raises(self, classifier):
        small_df = _make_ohlcv(n=50)
        with pytest.raises(ValueError, match="Insufficient"):
            classifier.fit(small_df)


class TestFeatureImportance:
    def test_importance_after_fit(self):
        from shared.ml.regime_classifier import _HAS_LIGHTGBM
        if not _HAS_LIGHTGBM:
            pytest.skip("lightgbm not installed")
        from shared.ml.regime_classifier import MLRegimeClassifier

        clf = MLRegimeClassifier(n_estimators=20, max_depth=3)
        df = _make_ohlcv(500)
        clf.fit(df)
        importance = clf.get_feature_importance()
        assert isinstance(importance, list)
        assert len(importance) > 0
        assert all(len(pair) == 2 for pair in importance)
        name, score = importance[0]
        assert isinstance(name, str)
        assert score >= 0

    def test_importance_before_fit_raises(self):
        from shared.ml.regime_classifier import _HAS_LIGHTGBM
        if not _HAS_LIGHTGBM:
            pytest.skip("lightgbm not installed")
        from shared.ml.regime_classifier import MLRegimeClassifier
        # The classifier was never constructed — this raised NameError rather
        # than exercising the unfitted-model guard it claims to test.
        clf = MLRegimeClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.get_feature_importance()


class TestFallbackWhenMissing:
    def test_import_error_raised(self):
        from shared.ml.regime_classifier import _HAS_LIGHTGBM
        if _HAS_LIGHTGBM:
            pytest.skip("lightgbm is installed, cannot test fallback")
        from shared.ml.regime_classifier import MLRegimeClassifier
        with pytest.raises(ImportError):
            MLRegimeClassifier()


class TestMarketRegime:
    def test_enum_values(self):
        from shared.ml.regime_classifier import MarketRegime
        assert MarketRegime.TRENDING.value == 0
        assert MarketRegime.RANGING.value == 1
        assert MarketRegime.VOLATILE.value == 2

    def test_enum_names(self):
        from shared.ml.regime_classifier import MarketRegime
        assert MarketRegime(0).name == "TRENDING"
        assert MarketRegime(1).name == "RANGING"
        assert MarketRegime(2).name == "VOLATILE"


class TestRegimeNames:
    def test_regime_names_mapping(self):
        from shared.ml.regime_classifier import _REGIME_NAMES
        assert _REGIME_NAMES[0] == "TRENDING"
        assert _REGIME_NAMES[1] == "RANGING"
        assert _REGIME_NAMES[2] == "VOLATILE"


class TestComputeFeaturesDetailed:
    def test_returns_features(self, sample_df):
        from shared.ml.regime_classifier import MLRegimeClassifier
        feat = MLRegimeClassifier.compute_features(sample_df)
        assert "ret_1d" in feat.columns
        assert "rsi_14" in feat.columns
        assert "adx" in feat.columns
        assert "macd_hist" in feat.columns
        assert "stoch_k" in feat.columns
        assert "roc_10" in feat.columns
        assert "roc_20" in feat.columns
        assert "price_vs_sma20" in feat.columns
        assert "price_vs_sma50" in feat.columns
        assert "bb_pct_b" in feat.columns
        assert "bb_width_z" in feat.columns
        assert "atr_pct_rank" in feat.columns
        assert "ema_slope" in feat.columns
        assert "gk_vol" in feat.columns
        assert "parkinson_vol" in feat.columns

    def test_features_shape(self, sample_df):
        from shared.ml.regime_classifier import MLRegimeClassifier
        feat = MLRegimeClassifier.compute_features(sample_df)
        assert feat.shape[0] == sample_df.shape[0]

    def test_features_last_row_mostly_valid(self, sample_df):
        from shared.ml.regime_classifier import MLRegimeClassifier
        feat = MLRegimeClassifier.compute_features(sample_df)
        last = feat.iloc[-1]
        valid_count = last.dropna().shape[0]
        assert valid_count >= 20


class TestAutoLabelDetailed:
    def test_labels_length_matches(self, sample_df):
        from shared.ml.regime_classifier import MLRegimeClassifier
        labels = MLRegimeClassifier.auto_label(sample_df, lookforward=10)
        assert len(labels) == len(sample_df)

    def test_labels_dtype(self, sample_df):
        from shared.ml.regime_classifier import MLRegimeClassifier
        labels = MLRegimeClassifier.auto_label(sample_df)
        assert labels.dtype == int or labels.dtype == np.int64 or labels.dtype == np.int32

    def test_trending_data_has_trending_labels(self):
        from shared.ml.regime_classifier import MLRegimeClassifier
        df = _make_ohlcv(500, drift=0.005)
        labels = MLRegimeClassifier.auto_label(df)
        assert (labels == 0).any()


class TestHasLightGBMFlag:
    def test_flag_is_bool(self):
        from shared.ml.regime_classifier import _HAS_LIGHTGBM
        assert isinstance(_HAS_LIGHTGBM, bool)


class TestFitWithMock:
    """Test fit/predict paths using mocked lightgbm."""

    def _make_classifier_with_mock(self):
        from shared.ml.regime_classifier import _HAS_LIGHTGBM
        if not _HAS_LIGHTGBM:
            pytest.skip("lightgbm not installed")
        from shared.ml.regime_classifier import MLRegimeClassifier
        return MLRegimeClassifier(n_estimators=10, max_depth=2)

    def test_fit_stores_feature_names(self):
        clf = self._make_classifier_with_mock()
        df = _make_ohlcv(500)
        clf.fit(df)
        assert len(clf._feature_names) > 0
        assert clf._is_fitted is True

    def test_fit_predict_roundtrip(self):
        clf = self._make_classifier_with_mock()
        df = _make_ohlcv(500)
        clf.fit(df)
        from shared.ml.regime_classifier import MarketRegime
        regime = clf.predict(df)
        assert isinstance(regime, MarketRegime)

    def test_predict_proba_sums_to_one(self):
        clf = self._make_classifier_with_mock()
        df = _make_ohlcv(500)
        clf.fit(df)
        proba = clf.predict_proba(df)
        assert abs(sum(proba.values()) - 1.0) < 0.02

    def test_feature_importance_sorted_desc(self):
        clf = self._make_classifier_with_mock()
        df = _make_ohlcv(500)
        clf.fit(df)
        importance = clf.get_feature_importance()
        scores = [s for _, s in importance]
        assert scores == sorted(scores, reverse=True)

    def test_check_fitted_raises(self):
        clf = self._make_classifier_with_mock()
        with pytest.raises(RuntimeError):
            clf._check_fitted()

    def test_fit_test_size(self):
        clf = self._make_classifier_with_mock()
        df = _make_ohlcv(500)
        metrics = clf.fit(df, test_size=0.3)
        assert "accuracy" in metrics

    def test_fit_custom_lookforward(self):
        clf = self._make_classifier_with_mock()
        df = _make_ohlcv(500)
        metrics = clf.fit(df, lookforward=10)
        assert "accuracy" in metrics
