"""Tests for TradeStation risk controls — AccountMonitor and OrderRouter.

Covers: AccountMonitor blocks trading on drawdown, OrderRouter checks
monitor before orders, daily loss limit blocks orders, cooldown after
3 losses, max position count enforced, unblock after 24 hours,
RiskManager integration.

15+ tests total.
"""

import os
import sys
import time
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, PropertyMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tradestation.api.account_monitor import AccountMonitor
from tradestation.api.order_router import (
    TradeStationOrderRouter,
    TradingBlockedError,
    TradeStationAPIError,
)


# ─── Helpers ───

def _mock_auth_response():
    """Mock successful OAuth2 token response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "access_token": "test_token",
        "expires_in": 1200,
        "refresh_token": "test_refresh",
    }
    return resp


def _make_router(account_monitor=None, risk_manager=None, **config_overrides):
    """Create a TradeStationOrderRouter with mocked authentication."""
    base_config = {
        "client_id": "test_id",
        "client_secret": "test_secret",
        "redirect_uri": "http://localhost",
        "refresh_token": "test_refresh",
        "max_daily_loss": 5000.0,
        "max_consecutive_losses": 3,
        "cooldown_minutes": 30,
        "max_positions": 10,
    }
    base_config.update(config_overrides)

    with patch.object(
        TradeStationOrderRouter, "_authenticate", return_value=None
    ):
        router = TradeStationOrderRouter(
            config=base_config,
            account_monitor=account_monitor,
            risk_manager=risk_manager,
        )
        router.access_token = "test_token"
        router.token_expiry = time.time() + 3600
    return router


def _make_monitor(order_router=None, **config_overrides):
    """Create an AccountMonitor with a mock router."""
    router = order_router or MagicMock()
    base_config = {
        "max_drawdown_pct": 5.0,
        "margin_warning_pct": 80.0,
        "position_concentration_pct": 25.0,
        "auto_unblock_after_hours": 24,
    }
    base_config.update(config_overrides)
    return AccountMonitor(router, base_config)


# ═══════════════════════════════════════════════════════════════════════
#  1. AccountMonitor blocks trading on drawdown
# ═══════════════════════════════════════════════════════════════════════


class TestAccountMonitorDrawdown:

    def test_not_blocked_initially(self):
        monitor = _make_monitor()
        blocked, reason = monitor.is_trading_blocked()
        assert blocked is False
        assert reason == ""

    def test_drawdown_triggers_block(self):
        monitor = _make_monitor(max_drawdown_pct=5.0)
        monitor._peak_equity = 100_000
        balances = {"equity": 94_000, "margin_used": 0}
        monitor._check_drawdown(balances)
        blocked, reason = monitor.is_trading_blocked()
        assert blocked is True
        assert "drawdown" in reason

    def test_small_drawdown_no_block(self):
        monitor = _make_monitor(max_drawdown_pct=5.0)
        monitor._peak_equity = 100_000
        balances = {"equity": 97_000, "margin_used": 0}
        monitor._check_drawdown(balances)
        blocked, _ = monitor.is_trading_blocked()
        assert blocked is False

    def test_margin_triggers_block(self):
        monitor = _make_monitor(margin_warning_pct=80.0)
        balances = {"equity": 100_000, "margin_used": 85_000}
        monitor._check_margin(balances)
        blocked, reason = monitor.is_trading_blocked()
        assert blocked is True
        assert "margin" in reason


# ═══════════════════════════════════════════════════════════════════════
#  2. OrderRouter checks monitor before orders
# ═══════════════════════════════════════════════════════════════════════


class TestOrderRouterChecksMonitor:

    def test_pre_order_passes_without_monitor(self):
        router = _make_router()
        # Should not raise
        router._pre_order_checks("AAPL")

    def test_pre_order_blocked_by_monitor(self):
        monitor = MagicMock()
        monitor.is_trading_blocked.return_value = (True, "drawdown exceeded")
        router = _make_router(account_monitor=monitor)
        with pytest.raises(TradingBlockedError, match="AccountMonitor"):
            router._pre_order_checks("AAPL")

    def test_pre_order_allowed_by_monitor(self):
        monitor = MagicMock()
        monitor.is_trading_blocked.return_value = (False, "")
        router = _make_router(account_monitor=monitor)
        router._pre_order_checks("AAPL")  # should not raise


# ═══════════════════════════════════════════════════════════════════════
#  3. Daily loss limit blocks orders
# ═══════════════════════════════════════════════════════════════════════


class TestDailyLossLimit:

    def test_allows_trade_under_limit(self):
        router = _make_router(max_daily_loss=5000.0)
        router.record_trade_pnl(-3000)
        assert router.can_trade() is True

    def test_blocks_trade_at_limit(self):
        router = _make_router(max_daily_loss=5000.0)
        router.record_trade_pnl(-5000)
        assert router.can_trade() is False

    def test_blocks_via_pre_order_checks(self):
        router = _make_router(max_daily_loss=1000.0)
        router.record_trade_pnl(-1500)
        with pytest.raises(TradingBlockedError, match="daily loss"):
            router._pre_order_checks("AAPL")

    def test_daily_reset_clears_pnl(self):
        router = _make_router(max_daily_loss=5000.0)
        router.record_trade_pnl(-4000)
        router.reset_daily()
        assert router._daily_pnl == 0.0
        assert router.can_trade() is True

    def test_auto_reset_on_new_day(self):
        router = _make_router(max_daily_loss=5000.0)
        router.record_trade_pnl(-4000)
        router._daily_pnl_reset_date = "2020-01-01"
        assert router.can_trade() is True
        assert router._daily_pnl == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  4. Cooldown after 3 losses
# ═══════════════════════════════════════════════════════════════════════


class TestCooldownAfterLosses:

    def test_no_cooldown_initially(self):
        router = _make_router(max_consecutive_losses=3)
        assert router.is_in_cooldown() is False

    def test_single_loss_no_cooldown(self):
        router = _make_router(max_consecutive_losses=3)
        router.record_trade_result(won=False)
        assert router.is_in_cooldown() is False

    def test_three_losses_triggers_cooldown(self):
        router = _make_router(max_consecutive_losses=3, cooldown_minutes=30)
        for _ in range(3):
            router.record_trade_result(won=False)
        assert router.is_in_cooldown() is True

    def test_win_resets_streak(self):
        router = _make_router(max_consecutive_losses=3)
        router.record_trade_result(won=False)
        router.record_trade_result(won=False)
        router.record_trade_result(won=True)
        assert router._consecutive_losses == 0
        router.record_trade_result(won=False)
        assert router.is_in_cooldown() is False

    def test_cooldown_blocks_pre_order(self):
        router = _make_router(max_consecutive_losses=2, cooldown_minutes=30)
        router.record_trade_result(won=False)
        router.record_trade_result(won=False)
        with pytest.raises(TradingBlockedError, match="cooldown"):
            router._pre_order_checks("AAPL")

    def test_cooldown_expires(self):
        router = _make_router(max_consecutive_losses=2, cooldown_minutes=30)
        router.record_trade_result(won=False)
        router.record_trade_result(won=False)
        assert router.is_in_cooldown() is True
        router._cooldown_until = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert router.is_in_cooldown() is False


# ═══════════════════════════════════════════════════════════════════════
#  5. Max position count enforced
# ═══════════════════════════════════════════════════════════════════════


class TestMaxPositionCount:

    def test_allows_under_max(self):
        router = _make_router(max_positions=3)
        router.add_position("AAPL")
        router.add_position("MSFT")
        router._pre_order_checks("GOOG")  # should not raise

    def test_blocks_at_max_new_symbol(self):
        router = _make_router(max_positions=2)
        router.add_position("AAPL")
        router.add_position("MSFT")
        with pytest.raises(TradingBlockedError, match="max open positions"):
            router._pre_order_checks("GOOG")

    def test_allows_existing_symbol_at_max(self):
        router = _make_router(max_positions=2)
        router.add_position("AAPL")
        router.add_position("MSFT")
        # Adding to existing position should be allowed
        router._pre_order_checks("AAPL")  # should not raise

    def test_remove_position_allows_new(self):
        router = _make_router(max_positions=2)
        router.add_position("AAPL")
        router.add_position("MSFT")
        router.remove_position("MSFT")
        router._pre_order_checks("GOOG")  # should not raise


# ═══════════════════════════════════════════════════════════════════════
#  6. Unblock after 24 hours
# ═══════════════════════════════════════════════════════════════════════


class TestUnblockAfter24Hours:

    def test_auto_unblock(self):
        monitor = _make_monitor(auto_unblock_after_hours=24)
        monitor._block_trading("test drawdown")
        blocked, _ = monitor.is_trading_blocked()
        assert blocked is True

        monitor._blocked_at = datetime.now(timezone.utc) - timedelta(hours=25)
        blocked, _ = monitor.is_trading_blocked()
        assert blocked is False

    def test_manual_unblock(self):
        monitor = _make_monitor()
        monitor._block_trading("test margin")
        blocked, _ = monitor.is_trading_blocked()
        assert blocked is True

        monitor.unblock_trading()
        blocked, _ = monitor.is_trading_blocked()
        assert blocked is False


# ═══════════════════════════════════════════════════════════════════════
#  7. RiskManager integration
# ═══════════════════════════════════════════════════════════════════════


class TestRiskManagerIntegration:

    def test_risk_manager_blocks_pre_order(self):
        rm = MagicMock()
        rm.can_trade.return_value = False
        router = _make_router(risk_manager=rm)
        with pytest.raises(TradingBlockedError, match="RiskManager"):
            router._pre_order_checks("AAPL")

    def test_risk_manager_allows_pre_order(self):
        rm = MagicMock()
        rm.can_trade.return_value = True
        router = _make_router(risk_manager=rm)
        router._pre_order_checks("AAPL")  # should not raise
        rm.can_trade.assert_called()

    def test_risk_manager_error_blocks_trade(self):
        """A gate that errors has an unknown answer — that is not "yes".

        This used to log a warning and let the order through. Signature drift
        between a caller and RiskManager has already happened elsewhere in this
        codebase, so the branch is reachable in practice.
        """
        rm = MagicMock()
        rm.can_trade.side_effect = Exception("RM unavailable")
        router = _make_router(risk_manager=rm)
        with pytest.raises(TradingBlockedError, match="RiskManager check errored"):
            router._pre_order_checks("AAPL")

    def test_local_daily_loss_limit_applies_even_with_a_risk_manager(self):
        """Attaching a RiskManager must not disable the router's own stop.

        can_trade() returned the RiskManager's verdict directly, skipping the
        local check. Since record_trade_pnl only updates the local counter,
        attaching the "better" risk manager made the router strictly less safe:
        a $6,000 local loss against a $5,000 limit still returned True.
        """
        rm = MagicMock()
        rm.can_trade.return_value = True
        router = _make_router(risk_manager=rm)
        router._max_daily_loss = 5000.0
        router.record_trade_pnl(-6000.0)

        assert router.can_trade() is False
        with pytest.raises(TradingBlockedError, match="daily loss limit"):
            router._pre_order_checks("AAPL")
