"""Tests for the Meta Ensemble strategy.

449 lines with no test anywhere in the suite — it is the strategy the README
presents as the flagship (`meta_ensemble`), and it is the only one that does
its own enrichment via PublicDataFetcher rather than StrategyEnricher, so the
conftest stub does not cover it either.

The scoring helpers that reach the network (_score_fundamental,
_score_sentiment, _score_earnings) are stubbed per test so the decision logic
is what gets exercised.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.backtesting.backtest_engine_v2 import BacktestContext
from strategies.examples.meta_strategy import (
    MetaEnsembleConfig,
    MetaEnsembleStrategy,
)


def _bars(n=260, start=100.0, drift=0.004, vol=0.0):
    """Deterministic OHLCV; positive drift trends up, negative trends down."""
    idx = pd.bdate_range("2023-01-02", periods=n)
    closes, price = [], start
    for i in range(n):
        price *= 1 + drift + (vol * np.sin(i / 3.0))
        closes.append(price)
    close = pd.Series(closes, index=idx)
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": pd.Series([1_000_000] * n, index=idx, dtype=float),
    }, index=idx)


def _ctx(bars, positions=None):
    ctx = BacktestContext.__new__(BacktestContext)
    ctx.bars = bars
    ctx.positions = positions or {}
    return ctx


def _offline(strategy, *, fundamental=0.5, sentiment=0.5, earnings=0.5):
    """Pin the three network-backed scorers so decisions are deterministic."""
    strategy._score_fundamental = lambda symbol: fundamental
    strategy._score_sentiment = lambda symbol: sentiment
    strategy._score_earnings = lambda symbol: earnings
    return strategy


class TestConfigAndConstruction:
    def test_default_weights_sum_to_one(self):
        c = MetaEnsembleConfig()
        total = (c.technical_weight + c.fundamental_weight + c.sentiment_weight
                 + c.earnings_weight + c.regime_weight)
        assert total == pytest.approx(1.0)

    def test_exit_threshold_is_below_entry_threshold(self):
        c = MetaEnsembleConfig()
        assert c.exit_threshold < c.entry_threshold, "otherwise entry and exit oscillate"

    def test_from_params_overrides_defaults(self):
        s = MetaEnsembleStrategy.from_params({"entry_threshold": 0.9})
        assert s.config.entry_threshold == pytest.approx(0.9)

    def test_from_params_ignores_unknown_keys(self):
        s = MetaEnsembleStrategy.from_params({"not_a_real_option": 1})
        assert isinstance(s.config, MetaEnsembleConfig)

    def test_registered_under_meta_ensemble(self):
        from strategies import STRATEGY_REGISTRY
        assert STRATEGY_REGISTRY["meta_ensemble"] is MetaEnsembleStrategy


class TestScoringIsBounded:
    """Every component feeds a weighted sum, so each must stay in [0, 1]."""

    @pytest.mark.parametrize("drift", [0.01, -0.01, 0.0])
    def test_technical_score_in_range(self, drift):
        s = MetaEnsembleStrategy()
        assert 0.0 <= s._score_technical(_bars(drift=drift)) <= 1.0

    @pytest.mark.parametrize("drift", [0.01, -0.01, 0.0])
    def test_regime_score_in_range(self, drift):
        s = MetaEnsembleStrategy()
        assert 0.0 <= s._score_regime(_bars(drift=drift)) <= 1.0

    def test_uptrend_scores_above_downtrend(self):
        s = MetaEnsembleStrategy()
        assert s._score_technical(_bars(drift=0.004)) > s._score_technical(
            _bars(drift=-0.004))


class TestSignalGeneration:
    def test_insufficient_history_yields_flat(self):
        s = _offline(MetaEnsembleStrategy())
        out = s.generate_signals(_ctx({"AAPL": _bars(n=10)}))
        assert out["AAPL"] == 0

    def test_all_components_bullish_produces_a_buy(self):
        s = _offline(MetaEnsembleStrategy(), fundamental=1.0, sentiment=1.0,
                     earnings=1.0)
        out = s.generate_signals(_ctx({"AAPL": _bars(drift=0.004)}))
        assert out["AAPL"] == 1

    def test_all_components_bearish_does_not_buy(self):
        s = _offline(MetaEnsembleStrategy(), fundamental=0.0, sentiment=0.0,
                     earnings=0.0)
        out = s.generate_signals(_ctx({"AAPL": _bars(drift=-0.004)}))
        assert out["AAPL"] != 1

    def test_min_agreement_blocks_a_high_composite(self):
        """A strong score from too few components must not open a position."""
        bars = {"AAPL": _bars(drift=0.004)}
        permissive = _offline(
            MetaEnsembleStrategy(MetaEnsembleConfig(min_agreement=1)),
            fundamental=1.0, sentiment=1.0, earnings=1.0)
        strict = _offline(
            MetaEnsembleStrategy(MetaEnsembleConfig(min_agreement=5)),
            fundamental=1.0, sentiment=0.0, earnings=0.0)

        assert permissive.generate_signals(_ctx(bars))["AAPL"] == 1
        assert strict.generate_signals(_ctx(bars))["AAPL"] != 1

    def test_signals_are_returned_for_every_symbol(self):
        s = _offline(MetaEnsembleStrategy())
        bars = {"AAPL": _bars(), "MSFT": _bars(), "XOM": _bars(n=5)}
        out = s.generate_signals(_ctx(bars))
        assert set(out) == {"AAPL", "MSFT", "XOM"}

    def test_signal_values_are_only_minus_one_zero_or_one(self):
        s = _offline(MetaEnsembleStrategy(), fundamental=1.0, sentiment=0.0)
        out = s.generate_signals(_ctx({"AAPL": _bars(), "MSFT": _bars(drift=-0.004)}))
        assert set(out.values()) <= {-1, 0, 1}

    def test_entry_threshold_is_honoured(self):
        """An unreachable threshold must never produce an entry."""
        s = _offline(
            MetaEnsembleStrategy(MetaEnsembleConfig(entry_threshold=1.01)),
            fundamental=1.0, sentiment=1.0, earnings=1.0)
        assert s.generate_signals(_ctx({"AAPL": _bars(drift=0.004)}))["AAPL"] != 1


class TestTrailingStop:
    def test_trailing_stop_is_recorded_on_entry(self):
        s = _offline(MetaEnsembleStrategy(), fundamental=1.0, sentiment=1.0,
                     earnings=1.0)
        bars = {"AAPL": _bars(drift=0.004)}
        assert s.generate_signals(_ctx(bars))["AAPL"] == 1
        assert s._trailing_stops["AAPL"] > 0

    def test_trailing_stop_only_ratchets_up(self):
        s = _offline(MetaEnsembleStrategy(), fundamental=1.0, sentiment=1.0,
                     earnings=1.0)
        rising = _bars(drift=0.004)
        s.generate_signals(_ctx({"AAPL": rising}))
        first = s._trailing_stops["AAPL"]

        s.generate_signals(_ctx({"AAPL": rising}, positions={"AAPL": 1}))
        assert s._trailing_stops["AAPL"] >= first

    def test_breaching_the_trailing_stop_flattens(self):
        s = _offline(MetaEnsembleStrategy(), fundamental=1.0, sentiment=1.0,
                     earnings=1.0)
        bars = _bars(drift=0.004)
        s.generate_signals(_ctx({"AAPL": bars}))

        # Force the stop above the current price.
        s._trailing_stops["AAPL"] = float(bars["close"].iloc[-1]) * 1.5
        out = s.generate_signals(_ctx({"AAPL": bars}, positions={"AAPL": 1}))

        assert out["AAPL"] == 0
        assert "AAPL" not in s._trailing_stops


class TestRobustness:
    def test_flat_prices_do_not_raise(self):
        s = _offline(MetaEnsembleStrategy())
        flat = _bars(drift=0.0)
        assert s.generate_signals(_ctx({"AAPL": flat}))["AAPL"] in (-1, 0, 1)

    def test_empty_context_returns_empty(self):
        s = _offline(MetaEnsembleStrategy())
        assert s.generate_signals(_ctx({})) == {}

    def test_a_failing_scorer_propagates(self):
        """Documents current behaviour, which is not obviously right.

        A transient news-feed failure aborts the whole run rather than
        degrading that one component to neutral. Recorded here so the
        behaviour is visible and a deliberate change would show up as a test
        change, not a silent one.
        """
        s = MetaEnsembleStrategy()
        s._score_fundamental = lambda symbol: 0.5
        s._score_earnings = lambda symbol: 0.5

        def boom(symbol):
            raise RuntimeError("news feed down")

        s._score_sentiment = boom
        with pytest.raises(RuntimeError):
            s.generate_signals(_ctx({"AAPL": _bars()}))
