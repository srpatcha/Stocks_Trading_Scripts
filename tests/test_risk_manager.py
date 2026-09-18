"""Tests for the standalone RiskManager module."""

import pytest
import time
from unittest.mock import patch

from shared.risk_manager import (
    RiskManager,
    RiskManagerConfig,
    SizingMethod,
    TradeRecord,
)


# ─── Fixtures ───


@pytest.fixture
def default_rm():
    """RiskManager with default configuration."""
    return RiskManager()


@pytest.fixture
def custom_rm():
    """RiskManager with tight limits for easier testing."""
    config = RiskManagerConfig(
        total_capital=100000.0,
        max_daily_loss=1000.0,
        max_consecutive_losses=2,
        cooldown_seconds=60,
        max_trades_per_hour=5,
        min_seconds_between_trades=0.0,  # disable for test speed
        max_open_positions=3,
        max_portfolio_heat_pct=10.0,
        max_drawdown_pct=5.0,
        circuit_breaker_pause_hours=1.0,
        risk_per_trade_pct=2.0,
    )
    return RiskManager(config=config)


# ─── TradeRecord Tests ───


class TestTradeRecord:
    """Tests for the TradeRecord dataclass."""

    def test_winning_trade_flagged(self):
        record = TradeRecord(symbol="AAPL", pnl=500.0)
        assert record.is_win is True
        assert record.symbol == "AAPL"
        assert record.pnl == 500.0

    def test_losing_trade_flagged(self):
        record = TradeRecord(symbol="MSFT", pnl=-200.0)
        assert record.is_win is False

    def test_breakeven_trade_not_win(self):
        record = TradeRecord(symbol="GOOG", pnl=0.0)
        assert record.is_win is False

    def test_timestamp_auto_set(self):
        record = TradeRecord(symbol="TSLA", pnl=100.0)
        assert record.timestamp is not None


# ─── RiskManagerConfig Tests ───


class TestRiskManagerConfig:
    """Tests for the RiskManagerConfig dataclass."""

    def test_default_config_values(self):
        config = RiskManagerConfig()
        assert config.sizing_method == SizingMethod.FIXED_FRACTIONAL
        assert config.total_capital == 100000.0
        assert config.max_daily_loss == 5000.0
        assert config.max_consecutive_losses == 3
        assert config.cooldown_seconds == 1800
        assert config.max_drawdown_pct == 10.0

    def test_custom_config(self):
        config = RiskManagerConfig(
            sizing_method=SizingMethod.KELLY,
            total_capital=50000.0,
            max_daily_loss=2000.0,
        )
        assert config.sizing_method == SizingMethod.KELLY
        assert config.total_capital == 50000.0
        assert config.max_daily_loss == 2000.0


# ─── Initialization Tests ───


class TestRiskManagerInit:
    """Tests for RiskManager initialization."""

    def test_default_initialization(self, default_rm):
        assert default_rm.config.total_capital == 100000.0
        assert default_rm._current_equity == 100000.0
        assert default_rm._peak_equity == 100000.0
        assert default_rm._consecutive_losses == 0
        assert default_rm._daily_pnl == 0.0

    def test_custom_initialization(self, custom_rm):
        assert custom_rm.config.max_daily_loss == 1000.0
        assert custom_rm.config.max_consecutive_losses == 2

    def test_repr_string(self, default_rm):
        repr_str = repr(default_rm)
        assert "RiskManager" in repr_str
        assert "equity" in repr_str
        assert "can_trade" in repr_str


# ─── Position Sizing Tests ───


class TestPositionSizing:
    """Tests for calculate_position_size with different methods.

    calculate_position_size() applies _apply_position_caps() to every method,
    and max_position_pct_equity defaults to 25%. With $100k of equity at $150 a
    share that ceiling is 166 shares, which is below every raw size exercised
    here, so these cases previously asserted the uncapped arithmetic and failed
    against a correctly-capped implementation:

        assert size == 400
        E  assert 166 == 400

    The cap is a deliberate risk control, not a bug, so these tests lift it to
    100% to isolate the sizing arithmetic they exist to check. TestPositionCaps
    below covers the capping behaviour itself, which nothing tested before.
    """

    def test_fixed_fractional_with_stop(self):
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade_pct=2.0,
            total_capital=100000.0,
            max_position_pct_equity=100.0,
        ))
        # Risk $2000, stop $5 away → 400 shares
        size = rm.calculate_position_size("AAPL", 150.0, stop_price=145.0)
        assert size == 400

    def test_fixed_fractional_with_atr(self):
        rm = RiskManager(config=RiskManagerConfig(
            risk_per_trade_pct=2.0,
            total_capital=100000.0,
            max_position_pct_equity=100.0,
        ))
        # Risk $2000, ATR=2.5 → risk_per_share=5.0 → 400 shares
        size = rm.calculate_position_size("AAPL", 150.0, atr=2.5)
        assert size == 400

    def test_fixed_fractional_default_risk(self):
        rm = RiskManager(config=RiskManagerConfig(
            risk_per_trade_pct=2.0,
            total_capital=100000.0,
            max_position_pct_equity=100.0,
        ))
        # No stop or ATR → default 2% of price → risk_per_share=3.0 → 666 shares
        size = rm.calculate_position_size("AAPL", 150.0)
        assert size == 666

    def test_fixed_shares_method(self):
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_SHARES,
            fixed_shares=200,
            max_position_pct_equity=100.0,
        ))
        size = rm.calculate_position_size("AAPL", 150.0)
        assert size == 200


class TestPositionCaps:
    """The safety caps _apply_position_caps() enforces on every sizing method.

    These are the controls that stop a sizing formula from putting the whole
    account into one name. None of them had a test.
    """

    def test_equity_pct_cap_binds(self):
        # Raw fixed-fractional size is 400; 25% of $100k at $150 is 166.
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade_pct=2.0,
            total_capital=100000.0,
            max_position_pct_equity=25.0,
        ))
        assert rm.calculate_position_size("AAPL", 150.0, stop_price=145.0) == 166

    def test_notional_cap_binds(self):
        # $15k notional at $150 is 100 shares, tighter than the 25% equity cap.
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade_pct=2.0,
            total_capital=100000.0,
            max_position_pct_equity=100.0,
            max_position_notional=15000.0,
        ))
        assert rm.calculate_position_size("AAPL", 150.0, stop_price=145.0) == 100

    def test_fat_finger_cap_binds(self):
        # A fixed-shares request far above max_shares_per_order is clamped.
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_SHARES,
            fixed_shares=1_000_000,
            total_capital=100_000_000.0,
            max_position_pct_equity=100.0,
            max_shares_per_order=10000,
        ))
        assert rm.calculate_position_size("AAPL", 150.0) == 10000

    def test_returns_zero_when_one_share_breaches_the_cap(self):
        """A $150 share cannot fit a 25% cap on a $100 account — refuse it.

        The old behaviour floored every result at 1 share, which turned a
        rejection into a 150%-of-equity position.
        """
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade_pct=2.0,
            total_capital=100.0,
            max_position_pct_equity=25.0,
        ))
        assert rm.calculate_position_size("AAPL", 150.0, stop_price=145.0) == 0

    def test_equity_cap_holds_for_high_priced_instruments(self):
        """The cap must bind when it rounds down to zero whole shares.

        Previously the cap was skipped entirely in that case, so a $30k
        instrument on a $100k account returned 3 shares — $90k, 90% of equity,
        against a documented 25% cap.
        """
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade_pct=2.0,
            total_capital=100_000.0,
            max_position_pct_equity=25.0,
        ))
        for price in (30_000.0, 60_000.0):
            shares = rm.calculate_position_size("BRK.A", price, stop_price=price * 0.98)
            assert shares * price <= 25_000.0, (
                f"{shares} shares @ ${price} = ${shares * price} exceeds the 25% cap"
            )

    def test_notional_cap_holds_when_it_rounds_to_zero(self):
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_SHARES,
            fixed_shares=10,
            total_capital=10_000_000.0,
            max_position_pct_equity=100.0,
            max_position_notional=5_000.0,
        ))
        assert rm.calculate_position_size("BRK.A", 30_000.0) == 0

    def test_fixed_dollar_method(self):
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_DOLLAR,
            fixed_dollar_amount=10000.0,
        ))
        # $10000 / $150 = 66 shares
        size = rm.calculate_position_size("AAPL", 150.0)
        assert size == 66

    def test_fixed_dollar_zero_price(self):
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_DOLLAR,
        ))
        size = rm.calculate_position_size("AAPL", 0.0)
        assert size == 0

    def test_kelly_sizing(self):
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.KELLY,
            kelly_win_rate=0.60,
            kelly_avg_win=2.0,
            kelly_avg_loss=1.0,
            kelly_fraction=0.5,
            total_capital=100000.0,
        ))
        size = rm.calculate_position_size("AAPL", 150.0)
        # Kelly: f* = (0.6 * 2 - 0.4) / 2 = 0.4, half = 0.2 → $20k / $150 = 133
        assert size > 0
        assert size == 133

    def test_kelly_zero_price(self):
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.KELLY,
        ))
        size = rm.calculate_position_size("AAPL", 0.0)
        assert size == 0

    def test_unaffordable_share_is_refused_not_floored_to_one(self):
        """A $500k share on a $1k account must return 0, not 1.

        The old code floored every result at one share, so this returned 1 —
        a $500,000 position on a $1,000 account, 500x equity.
        """
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade_pct=0.01,  # very small risk
            total_capital=1000.0,
        ))
        size = rm.calculate_position_size("BRK.A", 500000.0, stop_price=499000.0)
        assert size == 0

    def test_one_share_allowed_when_it_fits_the_caps(self):
        """The floor is only removed where it was unsafe — 1 share still works."""
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade_pct=0.01,
            total_capital=100_000.0,
            max_position_pct_equity=25.0,
        ))
        assert rm.calculate_position_size("AAPL", 150.0, stop_price=145.0) >= 1


# ─── can_trade() Tests ───


class TestCanTrade:
    """Tests for the can_trade() risk gate checks."""

    def test_can_trade_initially(self, custom_rm):
        assert custom_rm.can_trade() is True

    def test_blocked_by_daily_loss(self, custom_rm):
        custom_rm._daily_pnl = -1000.0  # hit the $1000 limit
        assert custom_rm.can_trade() is False

    def test_blocked_by_cooldown(self, custom_rm):
        custom_rm._cooldown_until = time.time() + 3600
        custom_rm._consecutive_losses = 2
        assert custom_rm.can_trade() is False

    def test_blocked_by_circuit_breaker(self, custom_rm):
        custom_rm._circuit_breaker_until = time.time() + 3600
        assert custom_rm.can_trade() is False

    def test_blocked_by_max_positions(self, custom_rm):
        # max_open_positions=3
        custom_rm.add_position("AAPL", 500)
        custom_rm.add_position("MSFT", 500)
        custom_rm.add_position("GOOG", 500)
        assert custom_rm.can_trade() is False

    def test_blocked_by_trade_frequency(self, custom_rm):
        # max_trades_per_hour=5
        now = time.time()
        custom_rm._trade_timestamps = [now - i for i in range(5)]
        custom_rm._last_trade_time = 0  # disable min_seconds check
        assert custom_rm.can_trade() is False

    def test_allowed_after_removing_position(self, custom_rm):
        custom_rm.add_position("AAPL", 500)
        custom_rm.add_position("MSFT", 500)
        custom_rm.add_position("GOOG", 500)
        assert custom_rm.can_trade() is False
        custom_rm.remove_position("GOOG")
        assert custom_rm.can_trade() is True


# ─── record_trade() Tests ───


class TestRecordTrade:
    """Tests for trade recording and its side effects."""

    def test_record_winning_trade(self, custom_rm):
        custom_rm.record_trade("AAPL", pnl=500.0)
        assert custom_rm._daily_pnl == 500.0
        assert custom_rm._current_equity == 100500.0
        assert custom_rm._consecutive_losses == 0
        assert len(custom_rm._trade_history) == 1

    def test_record_losing_trade(self, custom_rm):
        custom_rm.record_trade("AAPL", pnl=-300.0)
        assert custom_rm._daily_pnl == -300.0
        assert custom_rm._current_equity == 99700.0
        assert custom_rm._consecutive_losses == 1

    def test_consecutive_losses_trigger_cooldown(self, custom_rm):
        # max_consecutive_losses=2
        custom_rm.record_trade("AAPL", pnl=-100.0)
        assert custom_rm._consecutive_losses == 1
        assert custom_rm.can_trade() is True

        custom_rm._last_trade_time = 0  # reset to allow next trade
        custom_rm.record_trade("AAPL", pnl=-100.0)
        assert custom_rm._consecutive_losses == 2
        assert custom_rm._cooldown_until > time.time()
        assert custom_rm.can_trade() is False

    def test_win_resets_loss_streak(self, custom_rm):
        custom_rm.record_trade("AAPL", pnl=-100.0)
        assert custom_rm._consecutive_losses == 1
        custom_rm._last_trade_time = 0
        custom_rm.record_trade("AAPL", pnl=200.0)
        assert custom_rm._consecutive_losses == 0

    def test_equity_peak_tracking(self, custom_rm):
        custom_rm.record_trade("AAPL", pnl=1000.0)
        assert custom_rm._peak_equity == 101000.0
        custom_rm._last_trade_time = 0
        custom_rm.record_trade("AAPL", pnl=-500.0)
        # Peak should not decrease
        assert custom_rm._peak_equity == 101000.0
        assert custom_rm._current_equity == 100500.0

    def test_drawdown_circuit_breaker(self):
        config = RiskManagerConfig(
            total_capital=100000.0,
            max_drawdown_pct=5.0,
            circuit_breaker_pause_hours=1.0,
            min_seconds_between_trades=0.0,
            max_consecutive_losses=100,  # disable cooldown interference
        )
        rm = RiskManager(config=config)
        # Record a large loss that triggers 5%+ drawdown
        rm.record_trade("AAPL", pnl=-5500.0)
        assert rm._circuit_breaker_until > time.time()
        assert rm.can_trade() is False

    def test_daily_pnl_accumulates(self, custom_rm):
        custom_rm.record_trade("AAPL", pnl=200.0)
        custom_rm._last_trade_time = 0
        custom_rm.record_trade("MSFT", pnl=-100.0)
        assert custom_rm._daily_pnl == 100.0
        assert custom_rm._daily_trade_count == 2


# ─── Portfolio Heat Tests ───


class TestPortfolioHeat:
    """Tests for portfolio heat calculations."""

    def test_heat_within_limit(self, custom_rm):
        # max_portfolio_heat_pct=10.0, equity=100k → max heat = $10k
        assert custom_rm.check_portfolio_heat(additional_risk=5000.0) is True

    def test_heat_exceeds_limit(self, custom_rm):
        custom_rm.add_position("AAPL", 5000)
        # Adding 6000 more → total 11000 → 11% > 10%
        assert custom_rm.check_portfolio_heat(additional_risk=6000.0) is False

    def test_heat_at_limit_boundary(self, custom_rm):
        # 10% of 100k = 10k exactly
        assert custom_rm.check_portfolio_heat(additional_risk=10000.0) is True

    def test_heat_just_over_limit(self, custom_rm):
        custom_rm.add_position("A", 5000)
        # Adding 5001 more → total 10001 → 10.001% > 10% → rejected
        assert custom_rm.check_portfolio_heat(additional_risk=5001.0) is False

    def test_short_risk_does_not_cancel_long_risk(self, custom_rm):
        """Shorts are stored negative; heat must sum magnitudes, not net.

        A market-neutral book still has both legs at risk simultaneously, so
        netting them reported 0% heat and waved through every later position.
        """
        custom_rm.add_position("LONG1", 5000.0)
        custom_rm.add_position("SHORT1", -5000.0)

        # Gross heat is $10k = 10% of $100k equity, already at the limit.
        assert custom_rm.get_status()["portfolio_heat_pct"] == pytest.approx(10.0)
        assert custom_rm.check_portfolio_heat(additional_risk=1.0) is False

    def test_additional_short_risk_counted_as_exposure(self, custom_rm):
        """A proposed short adds heat; it must not subtract from it."""
        custom_rm.add_position("LONG1", 9000.0)
        assert custom_rm.check_portfolio_heat(additional_risk=-5000.0) is False


# ─── Position Tracking Tests ───


class TestPositionTracking:
    """Tests for add_position and remove_position."""

    def test_add_position(self, custom_rm):
        custom_rm.add_position("AAPL", 1000.0)
        assert "AAPL" in custom_rm._open_positions
        assert custom_rm._open_positions["AAPL"] == 1000.0

    def test_remove_position(self, custom_rm):
        custom_rm.add_position("AAPL", 1000.0)
        custom_rm.remove_position("AAPL")
        assert "AAPL" not in custom_rm._open_positions

    def test_remove_nonexistent_position(self, custom_rm):
        # Should not raise
        custom_rm.remove_position("NONEXISTENT")
        assert len(custom_rm._open_positions) == 0

    def test_overwrite_position(self, custom_rm):
        custom_rm.add_position("AAPL", 1000.0)
        custom_rm.add_position("AAPL", 2000.0)
        assert custom_rm._open_positions["AAPL"] == 2000.0


# ─── Status Tests ───


class TestGetStatus:
    """Tests for the get_status() method."""

    def test_status_structure(self, default_rm):
        status = default_rm.get_status()
        expected_keys = {
            "can_trade", "current_equity", "peak_equity", "daily_pnl",
            "daily_trade_count", "consecutive_losses", "cooldown_active",
            "cooldown_remaining_s", "circuit_breaker_active", "drawdown_pct",
            "open_positions", "portfolio_heat_pct", "trades_last_hour",
            "total_trades",
        }
        assert expected_keys.issubset(set(status.keys()))

    def test_initial_status_values(self, default_rm):
        status = default_rm.get_status()
        assert status["current_equity"] == 100000.0
        assert status["peak_equity"] == 100000.0
        assert status["daily_pnl"] == 0.0
        assert status["consecutive_losses"] == 0
        assert status["cooldown_active"] is False
        assert status["circuit_breaker_active"] is False
        assert status["open_positions"] == 0
        assert status["total_trades"] == 0

    def test_status_reflects_trades(self, custom_rm):
        custom_rm.record_trade("AAPL", pnl=-200.0)
        custom_rm._last_trade_time = 0
        custom_rm.record_trade("MSFT", pnl=300.0)
        status = custom_rm.get_status()
        assert status["daily_pnl"] == 100.0
        assert status["total_trades"] == 2
        assert status["current_equity"] == 100100.0
        assert status["consecutive_losses"] == 0
