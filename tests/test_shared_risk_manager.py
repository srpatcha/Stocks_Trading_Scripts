"""
Tests for shared.risk_manager — RiskManager, RiskManagerConfig, TradeRecord,
SizingMethod. Covers position sizing, daily P&L tracking, consecutive loss
cooldown, drawdown circuit breaker, trade frequency, portfolio heat, and
the bug fix: profitable days NOT blocked.
"""

import sys
import time
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.risk_manager import (
    RiskManager,
    RiskManagerConfig,
    SizingMethod,
    TradeRecord,
)
# trading_day(), not date.today(): the reset logic measures the session
# at the exchange, so on a UTC host the host date is a day ahead and
# stamping state with it looks like a brand new trading day.
from shared.utils.trading_clock import trading_day


# ── Fixtures ──


@pytest.fixture
def default_rm():
    """RiskManager with default config (100k capital)."""
    return RiskManager()


@pytest.fixture
def custom_rm():
    """RiskManager with tighter risk limits for easier testing."""
    config = RiskManagerConfig(
        total_capital=50000.0,
        max_daily_loss=1000.0,
        max_consecutive_losses=2,
        cooldown_seconds=60,
        max_trades_per_hour=5,
        min_seconds_between_trades=1.0,
        max_open_positions=3,
        max_drawdown_pct=5.0,
        circuit_breaker_pause_hours=1.0,
        max_portfolio_heat_pct=10.0,
    )
    return RiskManager(config=config)


# ── TradeRecord tests ──


class TestTradeRecord:
    """Tests for the TradeRecord dataclass."""

    def test_positive_pnl_is_win(self):
        """Validates is_win=True for positive P&L."""
        rec = TradeRecord(symbol="AAPL", pnl=500.0)
        assert rec.is_win is True

    def test_negative_pnl_is_loss(self):
        """Validates is_win=False for negative P&L."""
        rec = TradeRecord(symbol="TSLA", pnl=-200.0)
        assert rec.is_win is False

    def test_zero_pnl_is_loss(self):
        """Validates is_win=False for zero P&L (breakeven is not a win)."""
        rec = TradeRecord(symbol="MSFT", pnl=0.0)
        assert rec.is_win is False

    def test_timestamp_auto_populated(self):
        """Validates timestamp is set automatically."""
        rec = TradeRecord(symbol="GOOG", pnl=100.0)
        assert isinstance(rec.timestamp, datetime)


# ── SizingMethod tests ──


class TestSizingMethod:
    """Tests for the SizingMethod enum."""

    def test_all_enum_values(self):
        """Validates all expected sizing methods exist."""
        assert SizingMethod.FIXED_FRACTIONAL.value == "fixed_fractional"
        assert SizingMethod.KELLY.value == "kelly"
        assert SizingMethod.FIXED_SHARES.value == "fixed_shares"
        assert SizingMethod.FIXED_DOLLAR.value == "fixed_dollar"


# ── RiskManagerConfig defaults ──


class TestRiskManagerConfig:
    """Tests for default values in RiskManagerConfig."""

    def test_default_values(self):
        """Validates all default config values match expected defaults."""
        cfg = RiskManagerConfig()
        assert cfg.sizing_method == SizingMethod.FIXED_FRACTIONAL
        assert cfg.risk_per_trade_pct == 2.0
        assert cfg.fixed_shares == 100
        assert cfg.fixed_dollar_amount == 10000.0
        assert cfg.max_daily_loss == 5000.0
        assert cfg.max_consecutive_losses == 3
        assert cfg.cooldown_seconds == 1800
        assert cfg.max_open_positions == 10
        assert cfg.max_trades_per_hour == 10
        assert cfg.total_capital == 100000.0
        assert cfg.max_drawdown_pct == 10.0

    def test_custom_values(self):
        """Validates custom config overrides."""
        cfg = RiskManagerConfig(total_capital=250000, max_daily_loss=10000)
        assert cfg.total_capital == 250000
        assert cfg.max_daily_loss == 10000


# ── Position Sizing ──


class TestPositionSizing:
    """Tests for calculate_position_size() across all sizing methods."""

    def test_fixed_shares(self):
        """Validates FIXED_SHARES returns configured share count regardless of price."""
        cfg = RiskManagerConfig(sizing_method=SizingMethod.FIXED_SHARES, fixed_shares=50)
        rm = RiskManager(config=cfg)
        assert rm.calculate_position_size("AAPL", 150.0) == 50

    def test_fixed_shares_ignores_stop(self):
        """Validates FIXED_SHARES ignores stop_price parameter."""
        cfg = RiskManagerConfig(sizing_method=SizingMethod.FIXED_SHARES, fixed_shares=75)
        rm = RiskManager(config=cfg)
        assert rm.calculate_position_size("AAPL", 150.0, stop_price=140.0) == 75

    def test_fixed_dollar(self):
        """Validates FIXED_DOLLAR computes shares = dollar_amount / price."""
        cfg = RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_DOLLAR, fixed_dollar_amount=10000.0
        )
        rm = RiskManager(config=cfg)
        shares = rm.calculate_position_size("AAPL", 200.0)
        assert shares == 50

    def test_fixed_dollar_low_price(self):
        """Validates FIXED_DOLLAR with very low price is capped by max_shares_per_order."""
        cfg = RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_DOLLAR, fixed_dollar_amount=10000.0
        )
        rm = RiskManager(config=cfg)
        shares = rm.calculate_position_size("PENNY", 0.50)
        # Uncapped would be 20000 but max_shares_per_order=10000
        assert shares == 10000

    def test_fixed_dollar_zero_price_returns_zero(self):
        """Validates FIXED_DOLLAR returns 0 for zero entry price."""
        cfg = RiskManagerConfig(sizing_method=SizingMethod.FIXED_DOLLAR)
        rm = RiskManager(config=cfg)
        assert rm.calculate_position_size("X", 0.0) == 0

    def test_fixed_dollar_negative_price_returns_zero(self):
        """Validates FIXED_DOLLAR returns 0 for negative entry price."""
        cfg = RiskManagerConfig(sizing_method=SizingMethod.FIXED_DOLLAR)
        rm = RiskManager(config=cfg)
        assert rm.calculate_position_size("X", -10.0) == 0

    def test_fixed_fractional_with_stop(self):
        """Validates fixed fractional sizing with explicit stop price.
        risk = 100000 * 0.02 = 2000, risk_per_share = |150-140| = 10, shares = 200.
        Capped by 25% equity: 100000*0.25/150 = 166.
        """
        cfg = RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade_pct=2.0,
            total_capital=100000.0,
        )
        rm = RiskManager(config=cfg)
        shares = rm.calculate_position_size("AAPL", 150.0, stop_price=140.0)
        # 200 shares * $150 = $30k > 25% of $100k, capped to 166
        assert shares == 166

    def test_fixed_fractional_with_atr(self):
        """Validates fixed fractional sizing using ATR (risk_per_share = atr * 2).
        Capped by 25% equity.
        """
        cfg = RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade_pct=2.0,
            total_capital=100000.0,
        )
        rm = RiskManager(config=cfg)
        shares = rm.calculate_position_size("AAPL", 150.0, atr=5.0)
        # 200 shares * $150 = $30k > 25% cap, capped to 166
        assert shares == 166

    def test_fixed_fractional_no_stop_no_atr_defaults_to_2pct(self):
        """Validates default 2% of price used when no stop/atr provided.
        risk = 100000 * 0.02 = 2000, risk_per_share = 150 * 0.02 = 3.0, shares = 666.
        Capped by 25% equity: 100000*0.25/150 = 166.
        """
        cfg = RiskManagerConfig(total_capital=100000.0, risk_per_trade_pct=2.0)
        rm = RiskManager(config=cfg)
        shares = rm.calculate_position_size("AAPL", 150.0)
        # 666 * $150 = $99.9k > 25% cap, capped to 166
        assert shares == 166

    def test_fixed_fractional_stop_equals_entry_uses_default(self):
        """Validates when stop == entry, falls through to default 2% risk.
        Capped by 25% equity: 100000*0.25/100 = 250.
        """
        cfg = RiskManagerConfig(total_capital=100000.0, risk_per_trade_pct=2.0)
        rm = RiskManager(config=cfg)
        shares = rm.calculate_position_size("AAPL", 100.0, stop_price=100.0)
        # 1000 * $100 = $100k > 25% cap, capped to 250
        assert shares == 250

    def test_kelly_sizing(self):
        """Validates Kelly criterion position sizing with half-Kelly.
        w=0.55, r=1.5/1.0=1.5, f*=(0.55*1.5-0.45)/1.5=0.25, half=0.125
        dollar = 100000*0.125=12500, shares = 12500/100 = 125.
        """
        cfg = RiskManagerConfig(
            sizing_method=SizingMethod.KELLY,
            total_capital=100000.0,
            kelly_win_rate=0.55,
            kelly_avg_win=1.5,
            kelly_avg_loss=1.0,
            kelly_fraction=0.5,
        )
        rm = RiskManager(config=cfg)
        shares = rm.calculate_position_size("SPY", 100.0)
        assert shares == 125

    def test_kelly_zero_entry_returns_zero(self):
        """Validates Kelly returns 0 for zero entry price."""
        cfg = RiskManagerConfig(sizing_method=SizingMethod.KELLY)
        rm = RiskManager(config=cfg)
        assert rm.calculate_position_size("X", 0.0) == 0

    def test_kelly_zero_avg_loss(self):
        """Validates Kelly handles zero avg_loss (division by zero guard)."""
        cfg = RiskManagerConfig(
            sizing_method=SizingMethod.KELLY,
            kelly_avg_loss=0.0,
            total_capital=100000.0,
        )
        rm = RiskManager(config=cfg)
        shares = rm.calculate_position_size("SPY", 100.0)
        assert shares >= 1

    def test_kelly_losing_strategy_returns_min_1(self):
        """Validates Kelly with negative expectancy returns 1 share (min)."""
        cfg = RiskManagerConfig(
            sizing_method=SizingMethod.KELLY,
            kelly_win_rate=0.2,
            kelly_avg_win=0.5,
            kelly_avg_loss=1.0,
            kelly_fraction=0.5,
            total_capital=100000.0,
        )
        rm = RiskManager(config=cfg)
        shares = rm.calculate_position_size("SPY", 100.0)
        assert shares == 1


# ── can_trade() and Trade Validation ──


class TestCanTrade:
    """Tests for can_trade() — all risk gates."""

    def test_can_trade_default(self, default_rm):
        """Validates trading is allowed with fresh default RiskManager."""
        assert default_rm.can_trade() is True

    def test_daily_loss_limit_blocks(self, custom_rm):
        """Validates can_trade returns False when daily loss exceeds limit."""
        custom_rm._daily_pnl = -1000.0
        custom_rm._daily_pnl_date = trading_day()
        assert custom_rm.can_trade() is False

    def test_daily_loss_at_exact_limit_blocks(self, custom_rm):
        """Validates can_trade blocks at exactly the loss limit (<=)."""
        custom_rm._daily_pnl = -1000.0
        custom_rm._daily_pnl_date = trading_day()
        assert custom_rm.can_trade() is False

    def test_daily_loss_just_under_limit_allows(self, custom_rm):
        """Validates can_trade allows when just under the daily loss limit."""
        custom_rm._daily_pnl = -999.99
        custom_rm._daily_pnl_date = trading_day()
        assert custom_rm.can_trade() is True

    def test_profitable_day_not_blocked(self, custom_rm):
        """BUG FIX VERIFICATION: Profitable daily P&L should NOT block trading."""
        custom_rm._daily_pnl = 5000.0
        custom_rm._daily_pnl_date = trading_day()
        assert custom_rm.can_trade() is True

    def test_cooldown_blocks_trading(self, custom_rm):
        """Validates cooldown timer blocks trading."""
        custom_rm._cooldown_until = time.time() + 600
        assert custom_rm.can_trade() is False

    def test_cooldown_expired_allows(self, custom_rm):
        """Validates expired cooldown allows trading."""
        custom_rm._cooldown_until = time.time() - 1
        assert custom_rm.can_trade() is True

    def test_circuit_breaker_blocks(self, custom_rm):
        """Validates circuit breaker timer blocks trading."""
        custom_rm._circuit_breaker_until = time.time() + 3600
        assert custom_rm.can_trade() is False

    def test_circuit_breaker_expired_allows(self, custom_rm):
        """Validates expired circuit breaker allows trading."""
        custom_rm._circuit_breaker_until = time.time() - 1
        assert custom_rm.can_trade() is True

    def test_trade_frequency_limit(self, custom_rm):
        """Validates can_trade blocks when max_trades_per_hour exceeded."""
        now = time.time()
        custom_rm._trade_timestamps = [now - i for i in range(5)]
        assert custom_rm.can_trade() is False

    def test_min_seconds_between_trades(self, custom_rm):
        """Validates minimum time between trades is enforced."""
        custom_rm._last_trade_time = time.time()
        assert custom_rm.can_trade() is False

    def test_max_open_positions_blocks(self, custom_rm):
        """Validates can_trade blocks when max open positions reached."""
        for i in range(3):
            custom_rm._open_positions[f"SYM{i}"] = 100.0
        assert custom_rm.can_trade() is False

    def test_can_trade_with_positions_below_max(self, custom_rm):
        """Validates can_trade allows when under max position count."""
        custom_rm._open_positions["AAPL"] = 100.0
        assert custom_rm.can_trade() is True


# ── record_trade() ──


class TestRecordTrade:
    """Tests for record_trade() — P&L tracking, consecutive losses, equity updates."""

    def test_record_winning_trade_updates_pnl(self, default_rm):
        """Validates daily P&L is updated on winning trade."""
        default_rm.record_trade("AAPL", 500.0)
        assert default_rm._daily_pnl == 500.0
        assert default_rm._daily_trade_count == 1

    def test_record_losing_trade_updates_pnl(self, default_rm):
        """Validates daily P&L is decremented on losing trade."""
        default_rm.record_trade("TSLA", -300.0)
        assert default_rm._daily_pnl == -300.0

    def test_equity_increases_on_win(self, default_rm):
        """Validates current equity increases after winning trade."""
        initial = default_rm._current_equity
        default_rm.record_trade("GOOG", 1000.0)
        assert default_rm._current_equity == initial + 1000.0

    def test_equity_decreases_on_loss(self, default_rm):
        """Validates current equity decreases after losing trade."""
        initial = default_rm._current_equity
        default_rm.record_trade("GOOG", -500.0)
        assert default_rm._current_equity == initial - 500.0

    def test_peak_equity_updated_on_new_high(self, default_rm):
        """Validates peak equity is updated when equity reaches new high."""
        default_rm.record_trade("SPY", 5000.0)
        assert default_rm._peak_equity == 105000.0

    def test_peak_equity_not_updated_on_loss(self, default_rm):
        """Validates peak equity stays unchanged on a losing trade."""
        default_rm.record_trade("SPY", -500.0)
        assert default_rm._peak_equity == 100000.0

    def test_consecutive_losses_incremented(self, default_rm):
        """Validates consecutive loss counter increments on each loss."""
        default_rm.record_trade("A", -100.0)
        assert default_rm._consecutive_losses == 1
        default_rm.record_trade("B", -100.0)
        assert default_rm._consecutive_losses == 2

    def test_consecutive_losses_reset_on_win(self, default_rm):
        """Validates consecutive loss counter resets to 0 after a win."""
        default_rm.record_trade("A", -100.0)
        default_rm.record_trade("B", -100.0)
        assert default_rm._consecutive_losses == 2
        default_rm.record_trade("C", 200.0)
        assert default_rm._consecutive_losses == 0

    def test_cooldown_triggered_after_max_consecutive_losses(self, custom_rm):
        """Validates cooldown is set after max consecutive losses reached."""
        custom_rm.record_trade("A", -100.0)
        custom_rm.record_trade("B", -100.0)
        assert custom_rm._cooldown_until > time.time()

    def test_trade_history_appended(self, default_rm):
        """Validates trade is appended to history list."""
        default_rm.record_trade("AAPL", 100.0)
        default_rm.record_trade("TSLA", -50.0)
        assert len(default_rm._trade_history) == 2
        assert default_rm._trade_history[0].symbol == "AAPL"
        assert default_rm._trade_history[1].symbol == "TSLA"

    def test_trade_timestamps_tracked(self, default_rm):
        """Validates trade timestamps are added for frequency tracking."""
        default_rm.record_trade("SPY", 10.0)
        assert len(default_rm._trade_timestamps) == 1

    def test_daily_trade_count_incremented(self, default_rm):
        """Validates daily trade count increments on each trade."""
        default_rm.record_trade("A", 10.0)
        default_rm.record_trade("B", -10.0)
        assert default_rm._daily_trade_count == 2

    def test_circuit_breaker_triggered_on_large_drawdown(self):
        """Validates circuit breaker is triggered when drawdown exceeds max."""
        cfg = RiskManagerConfig(
            total_capital=100000.0,
            max_drawdown_pct=5.0,
            circuit_breaker_pause_hours=2.0,
        )
        rm = RiskManager(config=cfg)
        rm.record_trade("CRASH", -5000.0)
        assert rm._circuit_breaker_until > time.time()

    def test_no_circuit_breaker_below_threshold(self):
        """Validates circuit breaker is NOT triggered when drawdown is under threshold."""
        cfg = RiskManagerConfig(
            total_capital=100000.0,
            max_drawdown_pct=10.0,
        )
        rm = RiskManager(config=cfg)
        rm.record_trade("DIP", -1000.0)
        assert rm._circuit_breaker_until == 0.0

    def test_profitable_trade_after_losses_still_allows_trading(self, custom_rm):
        """BUG FIX: After consecutive losses and a winning trade, trading resumes."""
        custom_rm.record_trade("A", -100.0)
        custom_rm.record_trade("B", -100.0)
        # Cooldown was triggered
        assert custom_rm._cooldown_until > time.time()
        # Manually expire cooldown for test
        custom_rm._cooldown_until = 0.0
        custom_rm._last_trade_time = 0.0
        custom_rm.record_trade("C", 500.0)
        assert custom_rm._consecutive_losses == 0
        assert custom_rm._daily_pnl == 300.0


# ── Daily Reset ──


class TestDailyReset:
    """Tests for _reset_daily_if_needed() — day rollover logic."""

    def test_reset_on_new_day(self, default_rm):
        """Validates counters reset when a new day starts."""
        default_rm._daily_pnl = -3000.0
        default_rm._daily_trade_count = 15
        default_rm._daily_pnl_date = date(2020, 1, 1)
        default_rm._reset_daily_if_needed()
        assert default_rm._daily_pnl == 0.0
        assert default_rm._daily_trade_count == 0
        assert default_rm._daily_pnl_date == trading_day()

    def test_no_reset_same_day(self, default_rm):
        """Validates counters are NOT reset on the same day."""
        default_rm._daily_pnl = -1000.0
        default_rm._daily_trade_count = 5
        default_rm._daily_pnl_date = trading_day()
        default_rm._reset_daily_if_needed()
        assert default_rm._daily_pnl == -1000.0
        assert default_rm._daily_trade_count == 5


# ── Portfolio Heat ──


class TestPortfolioHeat:
    """Tests for check_portfolio_heat() — exposure limits."""

    def test_within_limits(self, custom_rm):
        """Validates check passes when total heat is within limits."""
        custom_rm._open_positions["AAPL"] = 1000.0
        assert custom_rm.check_portfolio_heat(additional_risk=500.0) is True

    def test_exceeds_limit(self, custom_rm):
        """Validates check fails when heat would exceed max_portfolio_heat_pct."""
        custom_rm._open_positions["AAPL"] = 4000.0
        assert custom_rm.check_portfolio_heat(additional_risk=2000.0) is False

    def test_zero_equity_returns_false(self):
        """Validates check returns False when current equity is zero."""
        cfg = RiskManagerConfig(total_capital=0.0)
        rm = RiskManager(config=cfg)
        assert rm.check_portfolio_heat(additional_risk=100.0) is False

    def test_no_positions_allows(self, custom_rm):
        """Validates check passes with no existing positions and small risk."""
        assert custom_rm.check_portfolio_heat(additional_risk=100.0) is True


# ── Position Tracking ──


class TestPositionTracking:
    """Tests for add_position() and remove_position()."""

    def test_add_position(self, default_rm):
        """Validates position is added to tracking dict."""
        default_rm.add_position("AAPL", 500.0)
        assert "AAPL" in default_rm._open_positions
        assert default_rm._open_positions["AAPL"] == 500.0

    def test_add_position_overwrites(self, default_rm):
        """Validates adding same symbol overwrites risk amount."""
        default_rm.add_position("AAPL", 500.0)
        default_rm.add_position("AAPL", 750.0)
        assert default_rm._open_positions["AAPL"] == 750.0

    def test_remove_position(self, default_rm):
        """Validates position is removed from tracking."""
        default_rm.add_position("AAPL", 500.0)
        default_rm.remove_position("AAPL")
        assert "AAPL" not in default_rm._open_positions

    def test_remove_nonexistent_position_no_error(self, default_rm):
        """Removing an unknown symbol must be a no-op, not just non-raising."""
        default_rm.add_position("AAPL", 500.0)

        default_rm.remove_position("NONEXISTENT")

        assert default_rm.get_status()["open_positions"] == 1
        assert default_rm.get_status()["portfolio_heat_pct"] > 0


# ── get_status() ──


class TestGetStatus:
    """Tests for get_status() — returns risk metrics dict."""

    def test_status_keys(self, default_rm):
        """Validates get_status returns all expected keys."""
        status = default_rm.get_status()
        expected_keys = {
            "can_trade", "current_equity", "peak_equity", "daily_pnl",
            "daily_trade_count", "consecutive_losses", "cooldown_active",
            "cooldown_remaining_s", "circuit_breaker_active", "drawdown_pct",
            "open_positions", "portfolio_heat_pct", "trades_last_hour",
            "total_trades", "monthly_pnl", "pyramid_positions",
            "max_position_pct_equity", "max_shares_per_order", "enforce_market_hours",
        }
        assert expected_keys == set(status.keys())

    def test_initial_status_values(self, default_rm):
        """Validates initial status for a fresh RiskManager."""
        status = default_rm.get_status()
        assert status["current_equity"] == 100000.0
        assert status["peak_equity"] == 100000.0
        assert status["daily_pnl"] == 0.0
        assert status["consecutive_losses"] == 0
        assert status["cooldown_active"] is False
        assert status["circuit_breaker_active"] is False
        assert status["open_positions"] == 0
        assert status["total_trades"] == 0

    def test_status_after_trades(self, default_rm):
        """Validates status reflects trade activity."""
        default_rm.record_trade("AAPL", 1000.0)
        default_rm.record_trade("TSLA", -200.0)
        default_rm.add_position("GOOG", 300.0)
        status = default_rm.get_status()
        assert status["daily_pnl"] == 800.0
        assert status["total_trades"] == 2
        assert status["open_positions"] == 1


# ── __repr__() ──


class TestRepr:
    """Tests for __repr__() string representation."""

    def test_repr_contains_key_info(self, default_rm):
        """Validates repr contains equity, daily_pnl, positions, and can_trade."""
        r = repr(default_rm)
        assert "RiskManager" in r
        assert "equity" in r
        assert "daily_pnl" in r
        assert "can_trade" in r
