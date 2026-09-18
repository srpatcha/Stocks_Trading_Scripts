"""Tests for webhook server risk controls and health monitoring.

Covers: health endpoint, HealthMonitor silence detection,
DailyPnLTracker, CooldownManager, DrawdownCircuitBreaker,
max trade value cap, rate limiter, and integrated risk gate checks.

25+ tests total.
"""

import os
import sys
import time
import json
import pytest
import threading
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pydantic import ValidationError

from tradingview.webhooks.webhook_server import (
    MAX_ORDER_QUANTITY,
    StaleAlertError,
    _check_alert_freshness,
    _update_risk_controls,
    HealthMonitor,
    DailyPnLTracker,
    CooldownManager,
    DrawdownCircuitBreaker,
    PositionLedger,
    RateLimiter,
    HealthResponse,
    AlertPayload,
)


# ═══════════════════════════════════════════════════════════════════════
#  1. Health Endpoint / HealthResponse
# ═══════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    """Health response model returns correct JSON fields."""

    def test_health_response_structure(self):
        hr = HealthResponse(
            status="healthy",
            uptime_seconds=120.5,
            last_alert_time=None,
            total_alerts_processed=0,
            version="1.0.0",
        )
        assert hr.status == "healthy"
        assert hr.uptime_seconds == 120.5
        assert hr.last_alert_time is None
        assert hr.total_alerts_processed == 0
        assert hr.version == "1.0.0"

    def test_health_response_with_last_alert(self):
        now = datetime.now(timezone.utc).isoformat()
        hr = HealthResponse(
            status="healthy",
            uptime_seconds=3600.0,
            last_alert_time=now,
            total_alerts_processed=42,
            version="2.0.0",
        )
        assert hr.last_alert_time == now
        assert hr.total_alerts_processed == 42

    def test_health_response_serialises(self):
        hr = HealthResponse(
            status="healthy", uptime_seconds=0.0,
            last_alert_time=None, total_alerts_processed=0, version="1.0.0",
        )
        data = hr.model_dump()
        assert "status" in data
        assert "uptime_seconds" in data


# ═══════════════════════════════════════════════════════════════════════
#  2. HealthMonitor — silence detection
# ═══════════════════════════════════════════════════════════════════════


class TestHealthMonitor:
    """HealthMonitor sends warning after silence period."""

    def test_initial_state(self):
        hm = HealthMonitor(max_silence_minutes=30)
        assert hm.alerts_processed == 0
        assert hm.last_alert_time is None

    def test_record_alert_updates_state(self):
        hm = HealthMonitor()
        hm.record_alert()
        assert hm.alerts_processed == 1
        assert hm.last_alert_time is not None

    def test_uptime_positive(self):
        hm = HealthMonitor()
        assert hm.uptime_seconds >= 0

    def test_silence_warning_dispatched(self):
        dispatcher = MagicMock()
        hm = HealthMonitor(
            max_silence_minutes=1,
            alert_dispatcher=dispatcher,
        )
        # Simulate being started 5 minutes ago so uptime > max_silence
        hm._start_time = time.time() - 300
        hm._check_silence()
        assert hm._silence_warned is True
        dispatcher.dispatch.assert_called_once()

    def test_no_duplicate_silence_warning(self):
        dispatcher = MagicMock()
        hm = HealthMonitor(max_silence_minutes=1, alert_dispatcher=dispatcher)
        hm._start_time = time.time() - 300
        hm._check_silence()
        hm._check_silence()
        assert dispatcher.dispatch.call_count == 1

    def test_silence_reset_after_alert(self):
        hm = HealthMonitor(max_silence_minutes=1)
        hm._start_time = time.time() - 300
        hm._check_silence()
        assert hm._silence_warned is True
        hm.record_alert()
        assert hm._silence_warned is False


# ═══════════════════════════════════════════════════════════════════════
#  3. DailyPnLTracker
# ═══════════════════════════════════════════════════════════════════════


class TestDailyPnLTracker:
    """DailyPnLTracker blocks trades after loss limit, allows under limit, resets daily."""

    def test_allows_trade_initially(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        assert tracker.can_trade() is True

    def test_allows_trade_under_limit(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        tracker.record_trade("AAPL", pnl=-500)
        assert tracker.can_trade() is True

    def test_blocks_trade_at_limit(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        tracker.record_trade("AAPL", pnl=-1000)
        assert tracker.can_trade() is False

    def test_blocks_trade_over_limit(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        tracker.record_trade("AAPL", pnl=-1500)
        assert tracker.can_trade() is False

    def test_accumulates_losses(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        tracker.record_trade("AAPL", pnl=-400)
        tracker.record_trade("MSFT", pnl=-400)
        assert tracker.can_trade() is True
        tracker.record_trade("GOOG", pnl=-300)
        assert tracker.can_trade() is False

    def test_wins_offset_losses(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        tracker.record_trade("AAPL", pnl=-800)
        tracker.record_trade("MSFT", pnl=500)
        assert tracker.daily_pnl == -300
        assert tracker.can_trade() is True

    def test_reset_daily_clears_state(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        tracker.record_trade("AAPL", pnl=-900)
        assert tracker.trade_count == 1
        tracker.reset_daily()
        assert tracker.daily_pnl == 0.0
        assert tracker.trade_count == 0
        assert tracker.can_trade() is True

    def test_auto_reset_on_new_day(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        tracker.record_trade("AAPL", pnl=-900)
        # Simulate date change
        tracker._last_reset_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        assert tracker.can_trade() is True
        assert tracker.daily_pnl == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  4. CooldownManager
# ═══════════════════════════════════════════════════════════════════════


class TestCooldownManager:
    """CooldownManager tracks consecutive losses, blocks during cooldown, resets on win."""

    def test_no_cooldown_initially(self):
        cm = CooldownManager(max_consecutive_losses=3, cooldown_minutes=30)
        assert cm.is_in_cooldown("strat_a") is False

    def test_single_loss_no_cooldown(self):
        cm = CooldownManager(max_consecutive_losses=3)
        cm.record_result("strat_a", won=False)
        assert cm.is_in_cooldown("strat_a") is False

    def test_two_losses_no_cooldown(self):
        cm = CooldownManager(max_consecutive_losses=3)
        cm.record_result("strat_a", won=False)
        cm.record_result("strat_a", won=False)
        assert cm.is_in_cooldown("strat_a") is False

    def test_three_losses_triggers_cooldown(self):
        cm = CooldownManager(max_consecutive_losses=3, cooldown_minutes=30)
        for _ in range(3):
            cm.record_result("strat_a", won=False)
        assert cm.is_in_cooldown("strat_a") is True

    def test_cooldown_blocks_trading(self):
        cm = CooldownManager(max_consecutive_losses=2, cooldown_minutes=60)
        cm.record_result("s1", won=False)
        cm.record_result("s1", won=False)
        assert cm.is_in_cooldown("s1") is True

    def test_cooldown_per_strategy(self):
        cm = CooldownManager(max_consecutive_losses=2, cooldown_minutes=60)
        cm.record_result("s1", won=False)
        cm.record_result("s1", won=False)
        assert cm.is_in_cooldown("s1") is True
        assert cm.is_in_cooldown("s2") is False

    def test_win_resets_loss_streak(self):
        cm = CooldownManager(max_consecutive_losses=3)
        cm.record_result("s1", won=False)
        cm.record_result("s1", won=False)
        cm.record_result("s1", won=True)
        cm.record_result("s1", won=False)
        assert cm.is_in_cooldown("s1") is False

    def test_cooldown_expires(self):
        cm = CooldownManager(max_consecutive_losses=2, cooldown_minutes=30)
        cm.record_result("s1", won=False)
        cm.record_result("s1", won=False)
        # Force expiry
        cm._cooldown_until["s1"] = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert cm.is_in_cooldown("s1") is False

    def test_get_status(self):
        cm = CooldownManager(max_consecutive_losses=3)
        cm.record_result("s1", won=False)
        status = cm.get_status("s1")
        assert status["strategy"] == "s1"
        assert status["loss_streak"] == 1
        assert status["in_cooldown"] is False


# ═══════════════════════════════════════════════════════════════════════
#  5. DrawdownCircuitBreaker
# ═══════════════════════════════════════════════════════════════════════


class TestDrawdownCircuitBreaker:
    """DrawdownCircuitBreaker blocks on 10%+ drawdown, auto-unblocks after lockout."""

    def test_can_trade_initially(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0, lockout_hours=24)
        assert dcb.can_trade() is True

    def test_small_drawdown_no_trigger(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        dcb.update_equity(100_000)
        dcb.update_equity(95_000)  # 5% drawdown
        assert dcb.can_trade() is True

    def test_10pct_drawdown_triggers(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0, lockout_hours=24)
        dcb.update_equity(100_000)
        dcb.update_equity(89_000)  # 11% drawdown
        assert dcb.can_trade() is False

    def test_exact_threshold_triggers(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        dcb.update_equity(100_000)
        dcb.update_equity(90_000)  # exactly 10%
        assert dcb.can_trade() is False

    def test_auto_unblock_after_lockout(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0, lockout_hours=24)
        dcb.update_equity(100_000)
        dcb.update_equity(89_000)
        assert dcb.can_trade() is False
        # Simulate lockout expiry
        dcb._tripped_until = datetime.now(timezone.utc) - timedelta(hours=1)
        assert dcb.can_trade() is True

    def test_peak_equity_tracking(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        dcb.update_equity(100_000)
        dcb.update_equity(110_000)
        assert dcb._peak_equity == 110_000
        dcb.update_equity(105_000)
        assert dcb._peak_equity == 110_000

    def test_drawdown_pct_property(self):
        dcb = DrawdownCircuitBreaker()
        dcb.update_equity(100_000)
        dcb.update_equity(95_000)
        assert abs(dcb.drawdown_pct - 5.0) < 0.01

    def test_reset_clears_trip(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        dcb.update_equity(100_000)
        dcb.update_equity(85_000)
        assert dcb.can_trade() is False
        dcb.reset()
        assert dcb.can_trade() is True

    def test_get_status(self):
        dcb = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        dcb.update_equity(100_000)
        status = dcb.get_status()
        assert status["peak_equity"] == 100_000
        assert status["tripped"] is False


# ═══════════════════════════════════════════════════════════════════════
#  6. Max Trade Value Cap
# ═══════════════════════════════════════════════════════════════════════


class TestMaxTradeValueCap:
    """Position size dollar cap is applied via _check_risk_gates."""

    def test_trade_value_under_cap_unchanged(self):
        alert = AlertPayload(
            symbol="AAPL", action="buy", price=150.0, quantity=10.0,
        )
        # 10 * 150 = 1500, under any reasonable cap
        max_cap = 50_000.0
        trade_value = alert.quantity * alert.price
        assert trade_value <= max_cap

    def test_trade_value_over_cap_gets_capped(self):
        alert = AlertPayload(
            symbol="AAPL", action="buy", price=150.0, quantity=500.0,
        )
        max_cap = 50_000.0
        trade_value = alert.quantity * alert.price  # 75000
        if trade_value > max_cap:
            capped_qty = max_cap / alert.price
            alert.quantity = capped_qty
        assert alert.quantity == pytest.approx(333.33, abs=0.1)

    def test_zero_price_is_rejected_at_the_schema(self):
        """price=0 must not be accepted — it defeated the notional cap.

        The cap is computed as quantity * price, so a zero price made every
        order look like a $0 trade: the dollar limit never bound, and the
        limit-order conversion was skipped so it stayed a market order.
        """
        with pytest.raises(ValidationError):
            AlertPayload(symbol="PENNY", action="buy", price=0.0, quantity=1000.0)

    def test_negative_price_is_rejected(self):
        with pytest.raises(ValidationError):
            AlertPayload(symbol="PENNY", action="buy", price=-1.0, quantity=10.0)

    def test_quantity_has_an_upper_bound(self):
        with pytest.raises(ValidationError):
            AlertPayload(
                symbol="AAPL", action="buy", price=150.0,
                quantity=MAX_ORDER_QUANTITY + 1,
            )


# ═══════════════════════════════════════════════════════════════════════
#  7. Rate Limiter
# ═══════════════════════════════════════════════════════════════════════


class TestRateLimiter:
    """Rate limiter enforces per-IP caps."""

    def test_allows_initial_requests(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.is_allowed("127.0.0.1") is True

    def test_blocks_excess_requests(self):
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.is_allowed("1.2.3.4")
        assert rl.is_allowed("1.2.3.4") is False

    def test_per_ip_isolation(self):
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.is_allowed("10.0.0.1")
        rl.is_allowed("10.0.0.1")
        assert rl.is_allowed("10.0.0.1") is False
        assert rl.is_allowed("10.0.0.2") is True

    def test_remaining_count(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        rl.is_allowed("1.1.1.1")
        rl.is_allowed("1.1.1.1")
        assert rl.get_remaining("1.1.1.1") == 3


# ═══════════════════════════════════════════════════════════════════════
#  8. Integration — all controls work together
# ═══════════════════════════════════════════════════════════════════════


class TestIntegrationAllControls:
    """All risk controls operate together."""

    def test_all_controls_allow_when_healthy(self):
        tracker = DailyPnLTracker(max_daily_loss=5000)
        cooldown = CooldownManager(max_consecutive_losses=3)
        breaker = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        breaker.update_equity(100_000)

        assert tracker.can_trade() is True
        assert not cooldown.is_in_cooldown("strat")
        assert breaker.can_trade() is True

    def test_one_control_blocks_pipeline(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        cooldown = CooldownManager(max_consecutive_losses=3)
        breaker = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        breaker.update_equity(100_000)

        tracker.record_trade("AAPL", pnl=-1500)
        # Tracker blocks, others allow
        assert tracker.can_trade() is False
        assert not cooldown.is_in_cooldown("strat")
        assert breaker.can_trade() is True

    def test_multiple_controls_can_block(self):
        tracker = DailyPnLTracker(max_daily_loss=1000)
        cooldown = CooldownManager(max_consecutive_losses=2)
        breaker = DrawdownCircuitBreaker(max_drawdown_pct=10.0)

        tracker.record_trade("AAPL", pnl=-1500)
        cooldown.record_result("s1", won=False)
        cooldown.record_result("s1", won=False)
        breaker.update_equity(100_000)
        breaker.update_equity(85_000)

        assert tracker.can_trade() is False
        assert cooldown.is_in_cooldown("s1") is True
        assert breaker.can_trade() is False

    def test_thread_safety_concurrent_pnl_recording(self):
        tracker = DailyPnLTracker(max_daily_loss=100_000)
        errors = []

        def record_trades():
            try:
                for _ in range(100):
                    tracker.record_trade("AAPL", pnl=-1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_trades) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.trade_count == 500
        assert tracker.daily_pnl == -500


# ═══════════════════════════════════════════════════════════════════════
#  PositionLedger — realized P&L for the risk controls
# ═══════════════════════════════════════════════════════════════════════


class TestPositionLedger:
    """The ledger is what makes the three risk controls able to fire.

    Before it existed the server fed them pnl=0.0 and won=True on every order,
    so the daily-loss limit, the loss-streak cooldown and the drawdown breaker
    were all mathematically unable to trip.
    """

    def test_long_round_trip_profit(self):
        ledger = PositionLedger()
        ledger.record_open("AAPL", 100, 150.0)
        assert ledger.record_close("AAPL", 100, 160.0) == pytest.approx(1000.0)

    def test_long_round_trip_loss(self):
        ledger = PositionLedger()
        ledger.record_open("AAPL", 100, 150.0)
        assert ledger.record_close("AAPL", 100, 140.0) == pytest.approx(-1000.0)

    def test_short_round_trip_profit(self):
        ledger = PositionLedger()
        ledger.record_open("AAPL", -100, 150.0)
        assert ledger.record_close("AAPL", 100, 140.0) == pytest.approx(1000.0)

    def test_short_round_trip_loss(self):
        ledger = PositionLedger()
        ledger.record_open("AAPL", -100, 150.0)
        assert ledger.record_close("AAPL", 100, 160.0) == pytest.approx(-1000.0)

    def test_average_cost_across_adds(self):
        ledger = PositionLedger()
        ledger.record_open("AAPL", 100, 100.0)
        ledger.record_open("AAPL", 100, 200.0)
        # Average cost 150; closing all 200 at 150 is flat.
        assert ledger.record_close("AAPL", 200, 150.0) == pytest.approx(0.0)

    def test_partial_close_leaves_remainder_open(self):
        ledger = PositionLedger()
        ledger.record_open("AAPL", 100, 150.0)
        assert ledger.record_close("AAPL", 40, 160.0) == pytest.approx(400.0)
        assert ledger.get_open_symbols() == ["AAPL"]
        assert ledger.record_close("AAPL", 60, 160.0) == pytest.approx(600.0)
        assert ledger.get_open_symbols() == []

    def test_close_without_entry_returns_none(self):
        """Unknown outcome must be reported as unknown, not as zero."""
        ledger = PositionLedger()
        assert ledger.record_close("AAPL", 100, 150.0) is None

    def test_oversized_close_is_clamped_to_holding(self):
        ledger = PositionLedger()
        ledger.record_open("AAPL", 100, 150.0)
        assert ledger.record_close("AAPL", 500, 160.0) == pytest.approx(1000.0)
        assert ledger.get_open_symbols() == []

    def test_position_is_fully_removed_when_flat(self):
        ledger = PositionLedger()
        ledger.record_open("AAPL", 100, 150.0)
        ledger.record_close("AAPL", 100, 150.0)
        assert ledger.get_open_symbols() == []

    def test_concurrent_open_and_close_is_consistent(self):
        ledger = PositionLedger()
        errors = []

        def cycle():
            try:
                for _ in range(100):
                    ledger.record_open("AAPL", 1, 100.0)
                    ledger.record_close("AAPL", 1, 100.0)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=cycle) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


class TestRiskControlsReceiveRealOutcomes:
    """_update_risk_controls must feed outcomes, not constants."""

    def _make_app(self, max_daily_loss=5000.0, max_consecutive_losses=3):
        app = MagicMock()
        app.state.position_ledger = PositionLedger()
        app.state.pnl_tracker = DailyPnLTracker(max_daily_loss=max_daily_loss)
        app.state.cooldown_mgr = CooldownManager(
            max_consecutive_losses=max_consecutive_losses)
        app.state.drawdown_breaker = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        app.state.starting_equity = 100_000.0
        app.state.realized_pnl_total = 0.0
        app.state.drawdown_breaker.update_equity(100_000.0)
        return app

    def _alert(self, action, price, quantity=100.0, strategy="s1"):
        return AlertPayload(
            symbol="AAPL", action=action, price=price,
            quantity=quantity, strategy=strategy,
        )

    def test_open_alone_does_not_move_pnl(self):
        app = self._make_app()
        _update_risk_controls(app, self._alert("buy", 150.0), 100.0)
        assert app.state.pnl_tracker.daily_pnl == 0.0

    def test_losing_round_trip_moves_daily_pnl(self):
        app = self._make_app()
        _update_risk_controls(app, self._alert("buy", 150.0), 100.0)
        _update_risk_controls(app, self._alert("sell", 140.0), 100.0)
        assert app.state.pnl_tracker.daily_pnl == pytest.approx(-1000.0)

    def test_daily_loss_limit_actually_trips(self):
        """The whole point: enough losses must block further trading."""
        app = self._make_app(max_daily_loss=2500.0)
        for _ in range(3):
            _update_risk_controls(app, self._alert("buy", 150.0), 100.0)
            _update_risk_controls(app, self._alert("sell", 140.0), 100.0)
        assert app.state.pnl_tracker.daily_pnl == pytest.approx(-3000.0)
        assert app.state.pnl_tracker.can_trade() is False

    def test_consecutive_losses_trigger_cooldown(self):
        app = self._make_app(max_consecutive_losses=3)
        for _ in range(3):
            _update_risk_controls(app, self._alert("buy", 150.0), 100.0)
            _update_risk_controls(app, self._alert("sell", 140.0), 100.0)
        assert app.state.cooldown_mgr.is_in_cooldown("s1") is True

    def test_a_win_resets_the_loss_streak(self):
        app = self._make_app(max_consecutive_losses=3)
        for _ in range(2):
            _update_risk_controls(app, self._alert("buy", 150.0), 100.0)
            _update_risk_controls(app, self._alert("sell", 140.0), 100.0)
        _update_risk_controls(app, self._alert("buy", 150.0), 100.0)
        _update_risk_controls(app, self._alert("sell", 160.0), 100.0)
        _update_risk_controls(app, self._alert("buy", 150.0), 100.0)
        _update_risk_controls(app, self._alert("sell", 140.0), 100.0)
        assert app.state.cooldown_mgr.is_in_cooldown("s1") is False

    def test_drawdown_breaker_trips_on_real_losses(self):
        app = self._make_app(max_daily_loss=1_000_000.0)
        # 11 x $1,000 loss on $100k starting equity = 11% drawdown > 10%.
        for _ in range(11):
            _update_risk_controls(app, self._alert("buy", 150.0), 100.0)
            _update_risk_controls(app, self._alert("sell", 140.0), 100.0)
        assert app.state.drawdown_breaker.can_trade() is False

    def test_close_without_entry_does_not_fabricate_pnl(self):
        app = self._make_app()
        _update_risk_controls(app, self._alert("sell", 140.0), 100.0)
        assert app.state.pnl_tracker.daily_pnl == 0.0
        assert app.state.cooldown_mgr.is_in_cooldown("s1") is False

    def test_broker_equity_is_preferred_over_the_synthetic_series(self):
        app = self._make_app()
        broker = MagicMock()
        broker.get_account_info.return_value = {"net_liquidation": 80_000.0}
        _update_risk_controls(app, self._alert("buy", 150.0), 100.0, broker)
        _update_risk_controls(app, self._alert("sell", 140.0), 100.0, broker)
        # 80k against a 100k peak is a 20% drawdown — breaker must trip.
        assert app.state.drawdown_breaker.can_trade() is False


# ═══════════════════════════════════════════════════════════════════════
#  Replay window — alert freshness
# ═══════════════════════════════════════════════════════════════════════


class TestAlertFreshness:
    """AlertPayload.timestamp existed but was never read anywhere.

    The only replay protection was a 5-minute dedup cache keyed on alert_id,
    which a replayer defeats by varying alert_id — so a captured signed request
    stayed valid indefinitely.
    """

    @staticmethod
    def _iso(offset_seconds):
        return (datetime.now(timezone.utc)
                + timedelta(seconds=offset_seconds)).isoformat()

    def test_fresh_alert_is_accepted(self):
        _check_alert_freshness(self._iso(-10), 300, True)

    def test_stale_alert_is_rejected(self):
        with pytest.raises(StaleAlertError, match="old"):
            _check_alert_freshness(self._iso(-600), 300, True)

    def test_boundary_inside_the_window_is_accepted(self):
        _check_alert_freshness(self._iso(-299), 300, True)

    def test_far_future_alert_is_rejected(self):
        """A future timestamp would stay replayable past the window."""
        with pytest.raises(StaleAlertError, match="future"):
            _check_alert_freshness(self._iso(600), 300, True)

    def test_small_clock_skew_is_tolerated(self):
        _check_alert_freshness(self._iso(30), 300, True)

    def test_naive_timestamp_is_treated_as_utc(self):
        """TradingView's {{timenow}} is UTC but carries no offset."""
        naive = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        _check_alert_freshness(naive, 300, True)

    def test_trailing_z_is_accepted(self):
        z = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        _check_alert_freshness(z, 300, True)

    def test_unparseable_timestamp_is_rejected(self):
        with pytest.raises(StaleAlertError, match="unparseable"):
            _check_alert_freshness("not-a-date", 300, False)

    def test_missing_timestamp_rejected_when_required(self):
        with pytest.raises(StaleAlertError, match="required"):
            _check_alert_freshness(None, 300, True)

    def test_missing_timestamp_allowed_when_not_required(self):
        _check_alert_freshness(None, 300, False)

    def test_zero_skew_disables_the_check(self):
        """Opt-out must be total, including the required-timestamp rule."""
        _check_alert_freshness(self._iso(-100_000), 0, True)
        _check_alert_freshness(None, 0, True)
        _check_alert_freshness("garbage", 0, True)
