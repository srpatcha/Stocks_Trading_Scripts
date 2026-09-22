"""
Tests for all 5 new book features:
    1. Elder Indicators (Force Index, Elder-ray, Elder Impulse)
    2. Cup and Handle pattern detection
    3. Fundamental data integration (fetch_fundamentals, fetch_earnings_dates)
    4. Pyramiding / adding to winners
    5. Monthly risk cap (Elder 6% rule)
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    price = 100.0
    rows = []
    for i in range(n):
        ret = 0.001 + rng.randn() * 0.015
        price *= 1 + ret
        high = price * (1 + abs(rng.randn()) * 0.005)
        low = price * (1 - abs(rng.randn()) * 0.005)
        op = price * (1 + rng.randn() * 0.002)
        vol = int(rng.uniform(500_000, 2_000_000))
        rows.append({"date": dates[i], "open": op, "high": high, "low": low, "close": price, "volume": vol})
    df = pd.DataFrame(rows).set_index("date")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Feature 1: Elder Indicators
# ══════════════════════════════════════════════════════════════════════════════

class TestForceIndex:
    """Tests for TechnicalIndicators.force_index()."""

    def test_returns_series(self):
        from shared.indicators.technical_indicators import TechnicalIndicators as TI
        df = _make_ohlcv()
        result = TI.force_index(df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        assert result.name == "FORCE_INDEX"

    def test_positive_and_negative_values(self):
        from shared.indicators.technical_indicators import TechnicalIndicators as TI
        df = _make_ohlcv()
        result = TI.force_index(df)
        non_nan = result.dropna()
        assert (non_nan > 0).any(), "Should have positive (bull) values"
        assert (non_nan < 0).any(), "Should have negative (bear) values"

    def test_custom_period(self):
        from shared.indicators.technical_indicators import TechnicalIndicators as TI
        df = _make_ohlcv()
        fi_2 = TI.force_index(df, period=2)
        fi_26 = TI.force_index(df, period=26)
        assert isinstance(fi_2, pd.Series)
        assert isinstance(fi_26, pd.Series)
        # Shorter period should be more volatile
        assert fi_2.std() > fi_26.std()


class TestElderRay:
    """Tests for TechnicalIndicators.elder_ray()."""

    def test_returns_dataframe(self):
        from shared.indicators.technical_indicators import TechnicalIndicators as TI
        df = _make_ohlcv()
        result = TI.elder_ray(df)
        assert isinstance(result, pd.DataFrame)
        assert "bull_power" in result.columns
        assert "bear_power" in result.columns
        assert len(result) == len(df)

    def test_bull_power_positive_bear_negative(self):
        from shared.indicators.technical_indicators import TechnicalIndicators as TI
        df = _make_ohlcv()
        result = TI.elder_ray(df)
        non_nan = result.dropna()
        # Bull power (high - EMA) should have some positive values
        assert (non_nan["bull_power"] > 0).any(), "Should have positive bull power values"
        # Bear power (low - EMA) should have some negative values
        assert (non_nan["bear_power"] < 0).any(), "Should have negative bear power values"


class TestElderImpulse:
    """Tests for TechnicalIndicators.elder_impulse()."""

    def test_returns_color_series(self):
        from shared.indicators.technical_indicators import TechnicalIndicators as TI
        df = _make_ohlcv()
        result = TI.elder_impulse(df)
        assert isinstance(result, pd.Series)
        assert result.name == "IMPULSE"
        valid_colors = {"green", "red", "blue"}
        unique_vals = set(result.dropna().unique())
        assert unique_vals.issubset(valid_colors), f"Unexpected values: {unique_vals - valid_colors}"

    def test_all_three_colors_present(self):
        from shared.indicators.technical_indicators import TechnicalIndicators as TI
        df = _make_ohlcv(n=500)
        result = TI.elder_impulse(df)
        colors = set(result.dropna().unique())
        assert "green" in colors
        assert "red" in colors
        assert "blue" in colors


# ══════════════════════════════════════════════════════════════════════════════
# Feature 2: Cup and Handle
# ══════════════════════════════════════════════════════════════════════════════

class TestCupAndHandle:
    """Tests for CandlestickPatterns.cup_and_handle()."""

    def test_returns_series(self):
        from shared.indicators.candlestick_patterns import CandlestickPatterns as CP
        df = _make_ohlcv()
        result = CP.cup_and_handle(df)
        assert isinstance(result, pd.Series)
        assert result.name == "CUP_AND_HANDLE"
        assert len(result) == len(df)

    def test_values_are_0_or_100(self):
        from shared.indicators.candlestick_patterns import CandlestickPatterns as CP
        df = _make_ohlcv()
        result = CP.cup_and_handle(df)
        assert set(result.unique()).issubset({0, 100})

    def test_short_data_returns_zeros(self):
        from shared.indicators.candlestick_patterns import CandlestickPatterns as CP
        df = _make_ohlcv(n=20)
        result = CP.cup_and_handle(df)
        assert (result == 0).all()

    def test_included_in_scan_all(self):
        from shared.indicators.candlestick_patterns import CandlestickPatterns as CP
        df = _make_ohlcv()
        result = CP.scan_all(df)
        assert "CUP_AND_HANDLE" in result.columns


# ══════════════════════════════════════════════════════════════════════════════
# Feature 3: Fundamental Data
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchFundamentals:
    """Tests for PublicDataFetcher.fetch_fundamentals()."""

    def test_returns_expected_keys(self):
        from shared.data.public_data_fetcher import PublicDataFetcher

        mock_ticker = MagicMock()
        mock_ticker.info = {
            "trailingPE": 25.0,
            "forwardPE": 22.0,
            "pegRatio": 1.5,
            "priceToBook": 3.2,
            "dividendYield": 0.006,
            "marketCap": 2_500_000_000_000,
            "totalRevenue": 380_000_000_000,
            "earningsGrowth": 0.15,
            "profitMargins": 0.25,
            "debtToEquity": 150.0,
            "currentRatio": 1.1,
            "bookValue": 4.3,
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }
        mock_ticker.institutional_holders = pd.DataFrame({"pctHeld": [0.30, 0.25]})

        mock_yf_module = MagicMock()
        mock_yf_module.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf_module}):
            fetcher = PublicDataFetcher(cache_enabled=False)
            result = fetcher.fetch_fundamentals("AAPL")

        assert result is not None
        expected_keys = {
            "pe_ratio", "forward_pe", "peg_ratio", "price_to_book",
            "dividend_yield", "market_cap", "revenue", "earnings_growth",
            "profit_margin", "debt_to_equity", "current_ratio", "book_value",
            "sector", "industry", "institutional_pct",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_returns_none_on_failure(self):
        from shared.data.public_data_fetcher import PublicDataFetcher

        mock_yf_module = MagicMock()
        mock_yf_module.Ticker.side_effect = Exception("API error")

        with patch.dict("sys.modules", {"yfinance": mock_yf_module}):
            fetcher = PublicDataFetcher(cache_enabled=False)
            result = fetcher.fetch_fundamentals("INVALID")

        assert result is None


class TestFetchEarningsDates:
    """Tests for PublicDataFetcher.fetch_earnings_dates()."""

    def test_returns_list(self):
        from shared.data.public_data_fetcher import PublicDataFetcher

        mock_ticker = MagicMock()
        mock_ticker.earnings_dates = pd.DataFrame(
            {
                "EPS Estimate": [1.5, 1.6],
                "Reported EPS": [1.55, 1.65],
                "Surprise(%)": [3.3, 3.1],
            },
            index=pd.to_datetime(["2024-01-25", "2024-04-25"]),
        )

        mock_yf_module = MagicMock()
        mock_yf_module.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf_module}):
            fetcher = PublicDataFetcher(cache_enabled=False)
            result = fetcher.fetch_earnings_dates("AAPL")

        assert isinstance(result, list)
        assert len(result) == 2
        assert "date" in result[0]
        assert "eps_estimate" in result[0]
        assert "eps_actual" in result[0]
        assert "surprise_pct" in result[0]

    def test_returns_empty_on_failure(self):
        from shared.data.public_data_fetcher import PublicDataFetcher

        mock_yf_module = MagicMock()
        mock_yf_module.Ticker.side_effect = Exception("API error")

        with patch.dict("sys.modules", {"yfinance": mock_yf_module}):
            fetcher = PublicDataFetcher(cache_enabled=False)
            result = fetcher.fetch_earnings_dates("INVALID")

        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# Feature 4: Pyramiding
# ══════════════════════════════════════════════════════════════════════════════

class TestPyramiding:
    """Tests for RiskManager pyramiding support."""

    def _make_rm(self, **kwargs):
        from shared.risk_manager import RiskManager, RiskManagerConfig
        config = RiskManagerConfig(enable_pyramiding=True, **kwargs)
        return RiskManager(config=config)

    def test_can_pyramid_enabled(self):
        rm = self._make_rm(pyramid_threshold_pct=2.0, max_pyramid_levels=3)
        # 5% profit on a $100 entry with pyramid_count=0 → threshold=2% → allowed
        assert rm.can_pyramid("AAPL", 105.0, 100.0, 0) is True

    def test_cannot_pyramid_disabled(self):
        from shared.risk_manager import RiskManager, RiskManagerConfig
        rm = RiskManager(config=RiskManagerConfig(enable_pyramiding=False))
        assert rm.can_pyramid("AAPL", 105.0, 100.0, 0) is False

    def test_cannot_pyramid_max_levels(self):
        rm = self._make_rm(max_pyramid_levels=2)
        assert rm.can_pyramid("AAPL", 110.0, 100.0, 2) is False

    def test_cannot_pyramid_insufficient_profit(self):
        rm = self._make_rm(pyramid_threshold_pct=5.0)
        # Only 1% profit, threshold = 5%
        assert rm.can_pyramid("AAPL", 101.0, 100.0, 0) is False

    def test_pyramid_threshold_scales_with_level(self):
        rm = self._make_rm(pyramid_threshold_pct=2.0)
        # Level 0: threshold = 2% → 3% profit should pass
        assert rm.can_pyramid("AAPL", 103.0, 100.0, 0) is True
        # Level 1: threshold = 4% → 3% profit should fail
        assert rm.can_pyramid("AAPL", 103.0, 100.0, 1) is False

    def test_calculate_pyramid_size(self):
        rm = self._make_rm(pyramid_scale_factor=0.5)
        assert rm.calculate_pyramid_size(100, 0) == 100
        assert rm.calculate_pyramid_size(100, 1) == 50
        assert rm.calculate_pyramid_size(100, 2) == 25

    def test_record_and_reset_pyramid(self):
        rm = self._make_rm()
        assert rm.get_pyramid_count("AAPL") == 0
        rm.record_pyramid("AAPL")
        assert rm.get_pyramid_count("AAPL") == 1
        rm.record_pyramid("AAPL")
        assert rm.get_pyramid_count("AAPL") == 2
        rm.reset_pyramid("AAPL")
        assert rm.get_pyramid_count("AAPL") == 0


# ══════════════════════════════════════════════════════════════════════════════
# Feature 5: Monthly Risk Cap (Elder 6% Rule)
# ══════════════════════════════════════════════════════════════════════════════

class TestMonthlyRiskCap:
    """Tests for RiskManager monthly loss cap."""

    def _make_rm(self, max_monthly_loss: float = 6000.0, **kwargs):
        from shared.risk_manager import RiskManager, RiskManagerConfig
        config = RiskManagerConfig(
            max_monthly_loss=max_monthly_loss,
            total_capital=100_000.0,
            max_daily_loss=50_000.0,  # set high so daily limit doesn't interfere
            max_consecutive_losses=100,  # set high so cooldown doesn't interfere
            min_seconds_between_trades=0.0,  # disable for test speed
            max_trades_per_hour=10000,  # disable for test speed
            **kwargs,
        )
        return RiskManager(config=config)

    def test_monthly_pnl_tracks(self):
        rm = self._make_rm()
        rm.record_trade("AAPL", pnl=-1000)
        rm.record_trade("MSFT", pnl=-2000)
        assert rm.get_monthly_pnl() == -3000.0

    def test_monthly_loss_blocks_trading(self):
        rm = self._make_rm(max_monthly_loss=3000.0)
        rm.record_trade("AAPL", pnl=-1500)
        assert rm.can_trade() is True
        rm.record_trade("MSFT", pnl=-1500)
        assert rm.can_trade() is False

    def test_monthly_cap_disabled(self):
        """Disabling now takes both knobs to zero.

        max_monthly_loss=0.0 used to mean "off". It now means "derive the cap
        from max_monthly_loss_pct", which defaults to Elder's 6% — the layer
        the README documents. Turning it off is explicit.
        """
        from shared.risk_manager import RiskManager, RiskManagerConfig
        rm = RiskManager(config=RiskManagerConfig(
            max_monthly_loss=0.0, max_monthly_loss_pct=0.0))
        rm.record_trade("AAPL", pnl=-50000)
        assert rm.get_monthly_pnl() == 0.0

    def test_monthly_cap_is_on_by_default_at_six_percent(self):
        """Elder's 6% rule is documented as an active layer; it shipped off."""
        from shared.risk_manager import RiskManager, RiskManagerConfig

        cfg = RiskManagerConfig(total_capital=100_000.0)
        assert cfg.monthly_loss_limit == pytest.approx(6_000.0)

        # Raise the daily limit and widen the drawdown breaker so the monthly
        # cap is unambiguously the gate under test.
        cfg = RiskManagerConfig(
            total_capital=100_000.0,
            max_daily_loss=1e9,
            max_drawdown_pct=100.0,
            max_consecutive_losses=1000,
            max_trades_per_hour=1000,
            min_seconds_between_trades=0,
        )
        rm = RiskManager(config=cfg)
        rm.record_trade("AAPL", pnl=-5_900.0)
        assert rm.can_trade() is True, "under the 6% cap, trading continues"
        rm.record_trade("MSFT", pnl=-200.0)
        assert rm.can_trade() is False, "over the 6% cap, trading stops"

    def test_monthly_pnl_is_tracked_when_the_cap_is_active(self):
        """_monthly_pnl was not even accumulated, so this reported a false $0."""
        from shared.risk_manager import RiskManager, RiskManagerConfig

        rm = RiskManager(config=RiskManagerConfig(total_capital=100_000.0))
        rm.record_trade("AAPL", pnl=-1_000.0)
        assert rm.get_monthly_pnl() == pytest.approx(-1_000.0)

    def test_explicit_dollar_cap_overrides_the_percentage(self):
        from shared.risk_manager import RiskManagerConfig

        cfg = RiskManagerConfig(total_capital=100_000.0, max_monthly_loss=2_500.0)
        assert cfg.monthly_loss_limit == pytest.approx(2_500.0)

    def test_monthly_pnl_includes_wins(self):
        rm = self._make_rm(max_monthly_loss=5000.0)
        rm.record_trade("AAPL", pnl=-3000)
        rm.record_trade("MSFT", pnl=2000)
        assert rm.get_monthly_pnl() == -1000.0
        assert rm.can_trade() is True


# ══════════════════════════════════════════════════════════════════════════════
# Trend Following Pyramiding Integration
# ══════════════════════════════════════════════════════════════════════════════

class TestTrendFollowingPyramiding:
    """Tests for pyramiding integration in TrendFollowingStrategy."""

    def test_config_has_pyramid_fields(self):
        from strategies.examples.trend_following import TrendFollowingConfig
        cfg = TrendFollowingConfig()
        assert hasattr(cfg, "enable_pyramiding")
        assert hasattr(cfg, "pyramid_threshold_pct")
        assert hasattr(cfg, "max_pyramid_adds")
        assert cfg.enable_pyramiding is False
        assert cfg.max_pyramid_adds == 3

    def test_strategy_instantiates_with_pyramiding(self):
        from strategies.examples.trend_following import (
            TrendFollowingConfig,
            TrendFollowingStrategy,
        )
        cfg = TrendFollowingConfig(enable_pyramiding=True, pyramid_threshold_pct=1.5)
        strategy = TrendFollowingStrategy(config=cfg)
        assert strategy.config.enable_pyramiding is True


# ══════════════════════════════════════════════════════════════════════════════
# Strategy Registration
# ══════════════════════════════════════════════════════════════════════════════

class TestStrategyRegistration:
    """Test that new strategies are properly registered."""

    def test_canslim_registered(self):
        import strategies.examples.canslim_strategy  # noqa: F401
        from strategies import STRATEGY_REGISTRY
        assert "canslim" in STRATEGY_REGISTRY

    def test_value_registered(self):
        import strategies.examples.value_strategy  # noqa: F401
        from strategies import STRATEGY_REGISTRY
        assert "value" in STRATEGY_REGISTRY
