"""Checkpoint save/load must not unpickle untrusted data.

torch.load(..., weights_only=False) unpickles the checkpoint, and unpickling
executes code embedded in the file. Both deep-learning predictors used that
default, so loading a tampered or shared model file was arbitrary code
execution. The checkpoint format now stores only primitives and tensors and is
read with weights_only=True.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

torch = pytest.importorskip("torch", reason="torch not installed")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from shared.ml.deep_learning.lstm_predictor import LSTMConfig, LSTMPredictor
from shared.ml.deep_learning.transformer_predictor import (
    TransformerConfig,
    TransformerPredictor,
)

FEATURES = ["f1", "f2", "f3"]


def _prepare(predictor, config_cls):
    """Give a predictor the minimum state save_model() requires."""
    predictor.feature_cols = list(FEATURES)
    predictor._scaler_mean = pd.Series({c: 1.5 for c in FEATURES})
    predictor._scaler_std = pd.Series({c: 0.5 for c in FEATURES})
    return predictor


@pytest.fixture(params=["lstm", "transformer"])
def predictor_pair(request):
    """(fresh_predictor, populated_predictor, config_cls) for both models."""
    if request.param == "lstm":
        from shared.ml.deep_learning.lstm_predictor import _LSTMModel

        cfg = LSTMConfig(hidden_size=8, num_layers=1)
        src = LSTMPredictor(config=cfg)
        src.model = _LSTMModel(len(FEATURES), cfg)
        return _prepare(src, LSTMConfig), LSTMPredictor(), LSTMConfig

    from shared.ml.deep_learning.transformer_predictor import TimeSeriesTransformer

    cfg = TransformerConfig(d_model=8, nhead=2, num_layers=1)
    src = TransformerPredictor(config=cfg)
    src.model = TimeSeriesTransformer(len(FEATURES), cfg)
    return _prepare(src, TransformerConfig), TransformerPredictor(), TransformerConfig


class TestCheckpointRoundTrip:
    def test_save_then_load_restores_state(self, predictor_pair, tmp_path):
        src, dst, config_cls = predictor_pair
        path = str(tmp_path / "model.pt")

        src.save_model(path)
        dst.load_model(path)

        assert isinstance(dst.config, config_cls)
        assert dst.feature_cols == FEATURES
        assert dst._scaler_mean.to_dict() == src._scaler_mean.to_dict()
        assert dst._scaler_std.to_dict() == src._scaler_std.to_dict()
        assert dst.model is not None

    def test_weights_survive_the_round_trip(self, predictor_pair, tmp_path):
        src, dst, _ = predictor_pair
        path = str(tmp_path / "model.pt")

        src.save_model(path)
        dst.load_model(path)

        for (ka, va), (kb, vb) in zip(
            src.model.state_dict().items(), dst.model.state_dict().items()
        ):
            assert ka == kb
            assert torch.allclose(va, vb)

    def test_checkpoint_is_readable_with_weights_only(self, predictor_pair, tmp_path):
        """The actual security property: no unpickling needed to read it."""
        src, _, _ = predictor_pair
        path = str(tmp_path / "model.pt")
        src.save_model(path)

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        assert checkpoint["format_version"] == 2
        assert isinstance(checkpoint["config"], dict)
        assert isinstance(checkpoint["feature_cols"], list)
        assert isinstance(checkpoint["scaler_mean"], dict)

    def test_checkpoint_holds_no_python_objects(self, predictor_pair, tmp_path):
        """Nothing in the file may require a class to reconstruct."""
        src, _, _ = predictor_pair
        path = str(tmp_path / "model.pt")
        src.save_model(path)

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        allowed = (dict, list, str, int, float, bool, type(None), torch.Tensor)

        def check(value, where):
            assert isinstance(value, allowed), f"{where} is {type(value)}"
            if isinstance(value, dict):
                for k, v in value.items():
                    check(v, f"{where}.{k}")
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    check(v, f"{where}[{i}]")

        check(checkpoint, "checkpoint")


class TestLegacyCheckpointsAreRefused:
    def test_pickled_checkpoint_is_rejected_by_default(self, predictor_pair, tmp_path):
        """A v1 checkpoint carries pickled objects and must not load silently."""
        src, dst, _ = predictor_pair
        path = str(tmp_path / "legacy.pt")

        # Exactly the old on-disk layout: a pickled config dataclass and
        # pickled pandas Series.
        torch.save({
            "model_state": src.model.state_dict(),
            "config": src.config,
            "feature_cols": src.feature_cols,
            "scaler_mean": src._scaler_mean,
            "scaler_std": src._scaler_std,
        }, path)

        with pytest.raises(RuntimeError, match="could not be read safely"):
            dst.load_model(path)

    def test_legacy_escape_hatch_is_explicit(self, predictor_pair, tmp_path):
        src, dst, config_cls = predictor_pair
        path = str(tmp_path / "legacy.pt")
        torch.save({
            "model_state": src.model.state_dict(),
            "config": src.config,
            "feature_cols": src.feature_cols,
            "scaler_mean": src._scaler_mean,
            "scaler_std": src._scaler_std,
        }, path)

        dst.load_model(path, allow_unsafe_legacy=True)

        assert isinstance(dst.config, config_cls)
        assert dst.feature_cols == FEATURES

    def test_converted_legacy_checkpoint_loads_safely(self, predictor_pair, tmp_path):
        """The documented migration path actually works."""
        src, dst, _ = predictor_pair
        legacy = str(tmp_path / "legacy.pt")
        converted = str(tmp_path / "converted.pt")
        torch.save({
            "model_state": src.model.state_dict(),
            "config": src.config,
            "feature_cols": src.feature_cols,
            "scaler_mean": src._scaler_mean,
            "scaler_std": src._scaler_std,
        }, legacy)

        dst.load_model(legacy, allow_unsafe_legacy=True)
        dst.save_model(converted)

        # No escape hatch needed the second time.
        dst.load_model(converted)
        assert dst.feature_cols == FEATURES
