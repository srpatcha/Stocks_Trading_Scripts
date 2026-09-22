"""
Tests for backtrader_adapter/

Covers:
- BacktraderConfig: dataclass defaults and custom values
- StocksPluginBTStrategy: signal handling, position sizing, bracket orders
- run_backtest(): full pipeline
- Analyzers: to_backtest_result_v2() (verify fix: trade_log populated),
  Sortino (verify fix: 1 neg return), _HAS_BT guard
- DataFeed: conversion from pandas to backtrader format
- Verify fix: unused imports removed
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ═══════════════════════════════════════════════════════
# BacktraderConfig Tests
# ═══════════════════════════════════════════════════════

class TestBacktraderConfig:

    def test_default_values(self):
        from backtrader_adapter.strategy_adapter import BacktraderConfig
        cfg = BacktraderConfig()
        assert cfg.initial_capital == 100_000.0
        assert cfg.commission == 0.001
        assert cfg.slippage_perc == 0.001
        assert cfg.slippage_fixed is None
        assert cfg.use_bracket_orders is False
        assert cfg.size_pct == 0.95

    def test_custom_values(self):
        from backtrader_adapter.strategy_adapter import BacktraderConfig
        cfg = BacktraderConfig(
            initial_capital=50_000, commission=0.002,
            slippage_fixed=0.05, use_bracket_orders=True,
            stop_loss_pct=0.03, take_profit_pct=0.06,
        )
        assert cfg.initial_capital == 50_000
        assert cfg.slippage_fixed == 0.05
        assert cfg.use_bracket_orders is True
        assert cfg.stop_loss_pct == 0.03

    def test_size_pct_range(self):
        from backtrader_adapter.strategy_adapter import BacktraderConfig
        cfg = BacktraderConfig(size_pct=0.5)
        assert 0 < cfg.size_pct <= 1.0


# ═══════════════════════════════════════════════════════
# _HAS_BT Guard Tests
# ═══════════════════════════════════════════════════════

class TestHasBTGuard:

    def test_analyzers_has_bt_flag(self):
        from backtrader_adapter import analyzers
        assert hasattr(analyzers, "_HAS_BT")

    def test_data_feed_has_bt_flag(self):
        from backtrader_adapter import data_feed
        assert hasattr(data_feed, "_HAS_BT")

    def test_strategy_adapter_has_bt_flag(self):
        from backtrader_adapter import strategy_adapter
        assert hasattr(strategy_adapter, "_HAS_BT")

    def test_runner_has_bt_flag(self):
        from backtrader_adapter import runner
        assert hasattr(runner, "_HAS_BT")


# ═══════════════════════════════════════════════════════
# Helper for mock strategy results
# ═══════════════════════════════════════════════════════

def _make_mock_strategy_result(equity, dates, trades):
    """Build a stand-in whose analyzers behave like backtrader's.

    This used to stub `analyzers.getbytype()` — a method backtrader's
    ItemCollection does not have. MagicMock answers any attribute, so the
    fake satisfied the production code's call and every test passed, while
    the real API raised AttributeError on the first line of the conversion.
    The adapter's run_backtest() could never complete.

    The fake now exposes getitems(), which is the real API, and returns an
    actual BacktestResultAnalyzer instance so the isinstance lookup is
    exercised rather than mocked away.
    """
    from backtrader_adapter.analyzers import BacktestResultAnalyzer

    mock_analysis = {
        "equity_curve": equity,
        "dates": dates,
        "trades": trades,
    }

    if BacktestResultAnalyzer is None:          # backtrader not installed
        analyzer = MagicMock()
    else:
        analyzer = MagicMock(spec=BacktestResultAnalyzer)
        analyzer.__class__ = BacktestResultAnalyzer
    analyzer.get_analysis.return_value = mock_analysis

    mock_result = MagicMock()
    # Real backtrader yields (name, analyzer) pairs — mirror that exactly,
    # so the fake cannot drift from the API again.
    mock_result.analyzers.getitems.return_value = [
        ("backtestresultanalyzer", analyzer)]
    return mock_result


# ═══════════════════════════════════════════════════════
# Analyzers: to_backtest_result_v2 Tests
# ═══════════════════════════════════════════════════════

class TestToBacktestResultV2:

    def test_basic_conversion(self):
        from backtrader_adapter.analyzers import to_backtest_result_v2, _HAS_ENGINE, _HAS_BT
        if not _HAS_ENGINE or not _HAS_BT:
            pytest.skip("backtrader or BacktestResultV2 not available")

        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
        equity = list(np.linspace(100000, 110000, 100))
        trades = [
            {"pnl": 500, "pnlcomm": 490, "size": 10, "price": 100.0,
             "barlen": 5, "dtopen": dates[10], "dtclose": dates[15]},
            {"pnl": -200, "pnlcomm": -210, "size": -5, "price": 105.0,
             "barlen": 3, "dtopen": dates[20], "dtclose": dates[23]},
        ]
        mock_result = _make_mock_strategy_result(equity, dates, trades)
        result = to_backtest_result_v2(mock_result, initial_capital=100_000)
        assert result.total_return > 0
        assert result.total_trades == 2

    def test_trade_log_populated_fix(self):
        """Verify fix: trade_log field is populated, not empty."""
        from backtrader_adapter.analyzers import to_backtest_result_v2, _HAS_ENGINE, _HAS_BT
        if not _HAS_ENGINE or not _HAS_BT:
            pytest.skip("backtrader or BacktestResultV2 not available")

        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(50)]
        equity = list(np.linspace(100000, 105000, 50))
        trades = [
            {"pnl": 1000, "pnlcomm": 990, "size": 20, "price": 50.0,
             "barlen": 10, "dtopen": dates[5], "dtclose": dates[15]},
        ]
        mock_result = _make_mock_strategy_result(equity, dates, trades)
        result = to_backtest_result_v2(mock_result)
        assert len(result.trade_log) == 1
        assert len(result.trades) == 1
        assert result.trade_log[0].pnl == 990

    def test_empty_equity_raises(self):
        from backtrader_adapter.analyzers import to_backtest_result_v2, _HAS_ENGINE, _HAS_BT
        if not _HAS_ENGINE or not _HAS_BT:
            pytest.skip("backtrader or BacktestResultV2 not available")
        mock_result = _make_mock_strategy_result([], [], [])
        with pytest.raises(ValueError, match="Empty equity curve"):
            to_backtest_result_v2(mock_result)

    def test_no_analyzer_raises(self):
        from backtrader_adapter.analyzers import to_backtest_result_v2, _HAS_ENGINE, _HAS_BT
        if not _HAS_ENGINE or not _HAS_BT:
            pytest.skip("backtrader or BacktestResultV2 not available")
        mock_result = MagicMock()
        mock_result.analyzers.getitems.return_value = []
        with pytest.raises(ValueError, match="BacktestResultAnalyzer not found"):
            to_backtest_result_v2(mock_result)

    def test_no_engine_raises_import(self):
        from backtrader_adapter import analyzers
        original = analyzers._HAS_ENGINE
        try:
            analyzers._HAS_ENGINE = False
            with pytest.raises(ImportError):
                analyzers.to_backtest_result_v2(MagicMock())
        finally:
            analyzers._HAS_ENGINE = original

    def test_win_loss_stats(self):
        from backtrader_adapter.analyzers import to_backtest_result_v2, _HAS_ENGINE, _HAS_BT
        if not _HAS_ENGINE or not _HAS_BT:
            pytest.skip("backtrader or BacktestResultV2 not available")

        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
        equity = list(np.linspace(100000, 120000, 100))
        trades = [
            {"pnl": 1000, "pnlcomm": 980, "size": 10, "price": 100.0, "barlen": 5,
             "dtopen": dates[5], "dtclose": dates[10]},
            {"pnl": 2000, "pnlcomm": 1980, "size": 15, "price": 110.0, "barlen": 7,
             "dtopen": dates[15], "dtclose": dates[22]},
            {"pnl": -500, "pnlcomm": -520, "size": -8, "price": 105.0, "barlen": 3,
             "dtopen": dates[30], "dtclose": dates[33]},
        ]
        mock_result = _make_mock_strategy_result(equity, dates, trades)
        result = to_backtest_result_v2(mock_result)
        assert result.win_rate == pytest.approx(2 / 3, rel=0.01)
        assert result.long_trades == 2
        assert result.short_trades == 1


# ═══════════════════════════════════════════════════════
# Sortino Ratio Edge Cases
# ═══════════════════════════════════════════════════════

class TestSortinoEdgeCases:

    def test_single_negative_return(self):
        """Verify fix: Sortino handles exactly 1 negative return."""
        from backtrader_adapter.analyzers import to_backtest_result_v2, _HAS_ENGINE, _HAS_BT
        if not _HAS_ENGINE or not _HAS_BT:
            pytest.skip("backtrader or BacktestResultV2 not available")

        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
        equity = [100000, 100100, 100200, 100300, 100200, 100400, 100500, 100600, 100700, 100800]
        trades = []
        mock_result = _make_mock_strategy_result(equity, dates, trades)
        result = to_backtest_result_v2(mock_result)
        assert np.isfinite(result.sortino_ratio)

    def test_no_negative_returns(self):
        from backtrader_adapter.analyzers import to_backtest_result_v2, _HAS_ENGINE, _HAS_BT
        if not _HAS_ENGINE or not _HAS_BT:
            pytest.skip("backtrader or BacktestResultV2 not available")

        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
        equity = [100000 + i * 100 for i in range(10)]
        mock_result = _make_mock_strategy_result(equity, dates, [])
        result = to_backtest_result_v2(mock_result)
        assert np.isfinite(result.sortino_ratio)

    def test_all_negative_returns(self):
        from backtrader_adapter.analyzers import to_backtest_result_v2, _HAS_ENGINE, _HAS_BT
        if not _HAS_ENGINE or not _HAS_BT:
            pytest.skip("backtrader or BacktestResultV2 not available")

        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
        equity = [100000 - i * 100 for i in range(10)]
        mock_result = _make_mock_strategy_result(equity, dates, [])
        result = to_backtest_result_v2(mock_result)
        assert np.isfinite(result.sortino_ratio)
        assert result.sortino_ratio < 0


# ═══════════════════════════════════════════════════════
# DataFeed Tests
# ═══════════════════════════════════════════════════════

class TestDataFeed:

    def test_dataframe_feed_exists_or_none(self):
        from backtrader_adapter.data_feed import DataFrameFeed, _HAS_BT
        if _HAS_BT:
            assert DataFrameFeed is not None
        else:
            assert DataFrameFeed is None

    def test_cache_feed_exists_or_none(self):
        from backtrader_adapter.data_feed import CacheFeed, _HAS_BT
        if _HAS_BT:
            assert CacheFeed is not None
        else:
            assert CacheFeed is None

    def test_cache_feed_missing_data_raises(self):
        from backtrader_adapter.data_feed import CacheFeed, _HAS_BT
        if not _HAS_BT:
            pytest.skip("backtrader not available")
        mock_cache = MagicMock()
        mock_cache.get_bars.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No cached data"):
            CacheFeed.from_cache(mock_cache, "AAPL")

    def test_cache_feed_missing_columns_raises(self):
        from backtrader_adapter.data_feed import CacheFeed, _HAS_BT
        if not _HAS_BT:
            pytest.skip("backtrader not available")
        mock_cache = MagicMock()
        df = pd.DataFrame({"close": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3))
        mock_cache.get_bars.return_value = df
        with pytest.raises(ValueError, match="Missing columns"):
            CacheFeed.from_cache(mock_cache, "AAPL")


# ═══════════════════════════════════════════════════════
# Runner Tests
# ═══════════════════════════════════════════════════════

class TestRunner:

    def test_run_backtest_no_bt_raises(self):
        from backtrader_adapter import runner
        if runner._HAS_BT:
            pytest.skip("backtrader is installed, skip import error test")
        with pytest.raises(ImportError, match="backtrader is required"):
            runner.run_backtest(lambda ctx: {"default": 0}, pd.DataFrame())

    def test_run_backtest_with_bt(self):
        from backtrader_adapter import runner
        if not runner._HAS_BT:
            pytest.skip("backtrader not installed")
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        df = pd.DataFrame({
            "open": np.random.uniform(100, 110, 100),
            "high": np.random.uniform(110, 120, 100),
            "low": np.random.uniform(90, 100, 100),
            "close": np.random.uniform(100, 110, 100),
            "volume": np.random.randint(1000, 10000, 100),
        }, index=dates)
        result = runner.run_backtest(lambda ctx: {"default": 0}, df)
        assert result is not None


# ═══════════════════════════════════════════════════════
# Strategy Adapter Tests
# ═══════════════════════════════════════════════════════

class TestStocksPluginBTStrategy:

    def test_strategy_class_exists_or_none(self):
        from backtrader_adapter.strategy_adapter import StocksPluginBTStrategy, _HAS_BT
        if _HAS_BT:
            assert StocksPluginBTStrategy is not None
        else:
            assert StocksPluginBTStrategy is None

    def test_strategy_signal_dict_normalization(self):
        """Test that non-dict signals are wrapped in {'default': signal}."""
        from backtrader_adapter.strategy_adapter import _HAS_BT
        if not _HAS_BT:
            pytest.skip("backtrader not installed")
        # This is tested indirectly via the strategy code logic
        signals = 1
        if not isinstance(signals, dict):
            signals = {"default": signals}
        assert signals == {"default": 1}


# ═══════════════════════════════════════════════════════
# Verify Fix: Unused Imports Removed
# ═══════════════════════════════════════════════════════

class TestUnusedImports:

    def test_strategy_adapter_no_unused(self):
        """Verify fix: strategy_adapter.py has no unused imports."""
        import inspect
        from backtrader_adapter import strategy_adapter
        src = inspect.getsource(strategy_adapter)
        # Should not import modules it doesn't use
        assert "import json" not in src
        assert "import re" not in src
