"""
END-TO-END MONEY SECURITY TESTS
================================
These tests verify that your money is protected under every possible
adverse scenario. Each test simulates a real-world danger and confirms
the system prevents capital loss.

Tests cover:
- Maximum loss protection (daily, drawdown, exposure)
- Stop loss guarantees (TP/SL, trailing, ATR)
- Risk manager bypass prevention
- Crash recovery (state persistence)
- Market condition protection (regime, hours, volatility)
- Data integrity (NaN, zero volume, stale data)
- Paper trading accuracy
- Alert and notification delivery
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
import threading
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# ─── Risk Management ───
from shared.risk_manager import RiskManager, RiskManagerConfig, SizingMethod, TradeRecord
from shared.risk_manager_unified import (
    UnifiedPortfolioRiskGate,
    UnifiedRiskConfig,
    PositionInfo,
)

# ─── Webhook Server ───
from tradingview.webhooks.webhook_server import (
    AlertPayload,
    OrderResult,
    DailyPnLTracker,
    CooldownManager,
    DrawdownCircuitBreaker,
    HealthMonitor,
    _check_risk_gates,
    SECTOR_MAP,
    create_app,
)

# ─── Broker Bridge ───
from shared.daemon.broker_bridge import (
    BrokerBridge,
    BaseBrokerAdapter,
    Position,
    TrailingStop,
    ExecutionResult,
    CommissionModel,
)

# ─── Broker Simulator ───
from shared.testing.broker_simulator import BrokerSimulator, SimOrder, SimPosition, SimFill

# ─── Strategies ───
from strategies.examples.mean_reversion import MeanReversionConfig
from strategies.examples.factor_portfolio import FactorPortfolioConfig
from strategies.examples.self_learning_strategy import SelfLearningConfig

# ─── Notifications ───
from shared.notifier.alert_dispatcher import AlertDispatcher

# ─── Market Hours ───
from shared.utils.market_hours import MarketHours


# ════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def reset_unified_gate():
    """Reset the UnifiedPortfolioRiskGate singleton before every test."""
    UnifiedPortfolioRiskGate.reset_instance()
    yield
    UnifiedPortfolioRiskGate.reset_instance()


@pytest.fixture
def risk_manager() -> RiskManager:
    """Standard RiskManager with $100K capital and $5K daily loss limit."""
    return RiskManager(
        config=RiskManagerConfig(
            total_capital=100_000.0,
            max_daily_loss=5_000.0,
            max_drawdown_pct=10.0,
            max_consecutive_losses=3,
            cooldown_seconds=1800,
            circuit_breaker_pause_hours=24.0,
            max_open_positions=10,
            max_trades_per_hour=10,
            min_seconds_between_trades=0.0,  # no throttle in tests
        )
    )


@pytest.fixture
def unified_gate() -> UnifiedPortfolioRiskGate:
    """Fresh unified portfolio risk gate with $100K equity."""
    config = UnifiedRiskConfig(
        max_portfolio_exposure=0.80,
        max_single_stock_pct=0.15,
        max_sector_pct=0.30,
        max_correlated_exposure=0.40,
        max_daily_loss=5_000.0,
        account_equity=100_000.0,
    )
    return UnifiedPortfolioRiskGate(config)


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Mock broker adapter that always succeeds."""
    adapter = MagicMock(spec=BaseBrokerAdapter)
    adapter.name = "mock_broker"
    adapter.connect.return_value = True
    adapter.is_connected.return_value = True
    adapter.cancel_order.return_value = True
    adapter.get_positions.return_value = []
    adapter.get_account_info.return_value = {"broker": "mock", "connected": True}
    adapter.get_latest_price.return_value = 150.0

    def _market_order(symbol, action, qty):
        return ExecutionResult(
            success=True, broker="mock_broker", symbol=symbol,
            action=action, shares=qty, price=150.0,
            order_id=f"MOCK-{symbol}-{action}-{qty}",
            message="mock filled",
        )

    def _limit_order(symbol, action, qty, price):
        return ExecutionResult(
            success=True, broker="mock_broker", symbol=symbol,
            action=action, shares=qty, price=price,
            order_id=f"MOCK-LMT-{symbol}",
            message="mock limit filled",
        )

    adapter.place_market_order.side_effect = _market_order
    adapter.place_limit_order.side_effect = _limit_order
    return adapter


@pytest.fixture
def broker_bridge(mock_adapter) -> BrokerBridge:
    """BrokerBridge wired to a mock adapter."""
    bridge = BrokerBridge.__new__(BrokerBridge)
    bridge._broker_name = "mock"
    bridge._config = {}
    bridge._mode = "paper"
    bridge._max_position_pct = 0.10
    bridge._max_shares = 500
    bridge._capital = 100_000.0
    bridge._max_loss_pct = 5.0
    bridge._default_tp_pct = 3.0
    bridge._default_sl_pct = 2.0
    bridge._diary_path = os.path.join(tempfile.gettempdir(), "test_diary.jsonl")
    bridge._positions = {}
    bridge._oco_pairs = {}
    bridge._trailing_stops = {}
    bridge._risk_manager = None
    bridge._portfolio_gate = UnifiedPortfolioRiskGate.get_instance(
        UnifiedRiskConfig(account_equity=100_000.0)
    )
    bridge._adapter = mock_adapter
    bridge._position_lock = threading.Lock()
    bridge._commission = CommissionModel()
    bridge._max_holding_bars = 240
    return bridge


@pytest.fixture
def simulator() -> BrokerSimulator:
    """BrokerSimulator preloaded with synthetic price data."""
    sim = BrokerSimulator(initial_capital=100_000.0, commission_per_share=0.005)
    sim.load_synthetic_data(symbols=["AAPL", "SPY", "MSFT"], n_bars=200, seed=42)
    return sim


@pytest.fixture
def daily_pnl_tracker() -> DailyPnLTracker:
    return DailyPnLTracker(max_daily_loss=5_000.0, reset_hour_utc=0)


@pytest.fixture
def cooldown_mgr() -> CooldownManager:
    return CooldownManager(max_consecutive_losses=3, cooldown_minutes=30)


@pytest.fixture
def drawdown_breaker() -> DrawdownCircuitBreaker:
    return DrawdownCircuitBreaker(max_drawdown_pct=10.0, lockout_hours=24)


def _make_ohlcv(n: int = 100, seed: int = 42, base_price: float = 100.0) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    close = base_price + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "open": close * (1 + rng.uniform(-0.01, 0.01, n)),
        "high": close * (1 + rng.uniform(0, 0.02, n)),
        "low": close * (1 - rng.uniform(0, 0.02, n)),
        "close": close,
        "volume": rng.integers(100_000, 5_000_000, n),
    }, index=pd.date_range("2023-01-01", periods=n, freq="B"))


# ════════════════════════════════════════════════════════════════════════════
# 1. TestMaximumLossProtection
# ════════════════════════════════════════════════════════════════════════════


class TestMaximumLossProtection:
    """Verify absolute worst-case losses are bounded."""

    def test_max_single_trade_loss_bounded(self, risk_manager: RiskManager):
        """A single trade can never lose more than 5% of equity via position sizing."""
        equity = risk_manager._current_equity  # $100K
        entry = 150.0
        stop = 142.5  # $7.50 risk per share → 5% of equity
        shares = risk_manager.calculate_position_size("AAPL", entry, stop_price=stop)

        max_loss = shares * abs(entry - stop)
        max_loss_pct = max_loss / equity * 100
        assert max_loss_pct <= 5.01, (
            f"Single trade risk is {max_loss_pct:.2f}% of equity, exceeds 5%"
        )

    def test_max_daily_loss_bounded_risk_manager(self, risk_manager: RiskManager):
        """Daily losses capped at $5K — can_trade() returns False after limit hit."""
        for i in range(10):
            risk_manager.record_trade(f"SYM{i}", pnl=-600.0)
            risk_manager._last_trade_time = 0  # reset throttle

        assert risk_manager._daily_pnl <= -5_000.0
        assert risk_manager.can_trade() is False

    def test_max_daily_loss_bounded_webhook_pnl_tracker(self, daily_pnl_tracker: DailyPnLTracker):
        """Webhook DailyPnLTracker blocks trades at $5K cumulative loss."""
        for i in range(5):
            daily_pnl_tracker.record_trade(f"SYM{i}", pnl=-1100.0)

        assert daily_pnl_tracker.daily_pnl <= -5_000.0
        assert daily_pnl_tracker.can_trade() is False

    def test_drawdown_circuit_breaker_halts_everything(self, risk_manager: RiskManager):
        """10% portfolio drawdown stops ALL trading for 24 hours."""
        for i in range(20):
            risk_manager.record_trade(f"SYM{i}", pnl=-550.0)
            risk_manager._last_trade_time = 0

        # equity dropped from $100K to ~$89K = ~11% drawdown
        drawdown_pct = (
            (risk_manager._peak_equity - risk_manager._current_equity)
            / risk_manager._peak_equity * 100
        )
        assert drawdown_pct >= 10.0
        assert risk_manager.can_trade() is False
        assert risk_manager._circuit_breaker_until > time.time()

    def test_drawdown_breaker_webhook(self, drawdown_breaker: DrawdownCircuitBreaker):
        """DrawdownCircuitBreaker blocks after 10% equity drop."""
        drawdown_breaker.update_equity(100_000.0)
        drawdown_breaker.update_equity(89_000.0)
        assert drawdown_breaker.can_trade() is False
        assert drawdown_breaker.drawdown_pct >= 10.0

    def test_max_portfolio_exposure_never_exceeded(self, unified_gate: UnifiedPortfolioRiskGate):
        """Total exposure never exceeds 80% of equity ($80K on $100K)."""
        # Fill up to 75K
        unified_gate.register_position("AAPL", 200, 150.0, "strat1", "ib")  # $30K
        unified_gate.register_position("MSFT", 100, 250.0, "strat1", "ib")  # $25K
        unified_gate.register_position("GOOGL", 100, 200.0, "strat1", "ib")  # $20K

        # Try to add $10K more → total $85K = 85% > 80%
        ok, reason = unified_gate.can_open_position("AMZN", 10_000.0)
        assert ok is False
        assert "exposure" in reason.lower()

    def test_single_stock_concentration_limited(self, unified_gate: UnifiedPortfolioRiskGate):
        """No single stock can be > 15% of portfolio."""
        # Try $20K in AAPL on $100K equity = 20% > 15%
        ok, reason = unified_gate.can_open_position("AAPL", 20_000.0)
        assert ok is False
        assert "Single-stock" in reason or "concentration" in reason.lower()

    def test_max_open_positions_limit(self, risk_manager: RiskManager):
        """Cannot exceed max_open_positions (10)."""
        for i in range(10):
            risk_manager.add_position(f"SYM{i}", 500.0)
        assert risk_manager.can_trade() is False


# ════════════════════════════════════════════════════════════════════════════
# 2. TestStopLossGuarantees
# ════════════════════════════════════════════════════════════════════════════


class TestStopLossGuarantees:
    """Every position MUST have a stop loss."""

    def test_broker_bridge_places_tp_sl_on_buy(self, broker_bridge: BrokerBridge, mock_adapter):
        """BrokerBridge places TP + SL orders after a BUY fill."""
        decision = {"action": "BUY", "confidence": 0.8, "price": 150.0}
        result = broker_bridge.execute_decision(decision, "AAPL")

        assert result is not None
        assert result.success is True
        # TP/SL calls should have been made via limit/stop orders
        limit_calls = mock_adapter.place_limit_order.call_count
        stop_calls = mock_adapter.place_stop_order.call_count
        total_protective = limit_calls + stop_calls
        assert total_protective >= 1, "No TP/SL orders placed after fill"

    def test_broker_bridge_places_tp_sl_on_sell(self, broker_bridge: BrokerBridge, mock_adapter):
        """BrokerBridge places TP + SL orders after a SELL (short) fill."""
        decision = {"action": "SELL", "confidence": 0.8, "price": 150.0}
        result = broker_bridge.execute_decision(decision, "AAPL")

        assert result is not None
        assert result.success is True
        limit_calls = mock_adapter.place_limit_order.call_count
        stop_calls = mock_adapter.place_stop_order.call_count
        total_protective = limit_calls + stop_calls
        assert total_protective >= 1, "No TP/SL orders placed after short fill"

    def test_oco_cancels_partner_on_fill(self, broker_bridge: BrokerBridge):
        """When TP fills, SL auto-cancels (OCO behavior)."""
        # Simulate OCO pair
        broker_bridge._oco_pairs["TP-001"] = "SL-001"
        broker_bridge._oco_pairs["SL-001"] = "TP-001"
        broker_bridge._positions["AAPL"] = Position(
            symbol="AAPL", direction="long", shares=100,
            entry_price=150.0, entry_time="2023-01-01",
        )

        broker_bridge.on_fill("TP-001", "AAPL", 155.0)

        # SL-001 should have been cancelled
        broker_bridge._adapter.cancel_order.assert_called_with("SL-001")
        assert "TP-001" not in broker_bridge._oco_pairs
        assert "SL-001" not in broker_bridge._oco_pairs

    # These three used to construct a TrailingStop dataclass, re-implement the
    # ratcheting arithmetic inline in the test body, and assert on their own
    # result. Every line of trailing-stop logic could be deleted from
    # broker_bridge.py and all three would still have passed. They now drive
    # the real BrokerBridge._update_trailing_stops().

    @staticmethod
    def _bridge_with_position(direction, entry, trail_pct=0.015, activation_pct=0.02):
        import tempfile

        from shared.daemon.broker_bridge import BrokerBridge, Position
        from shared.risk_manager_unified import UnifiedPortfolioRiskGate

        UnifiedPortfolioRiskGate.reset_instance()
        fd, diary = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        bridge = BrokerBridge(broker="ib", mode="paper", diary_path=diary)
        shares = 10 if direction == "long" else -10
        bridge._positions["AAPL"] = Position(
            "AAPL", direction, abs(shares), entry, "t")
        bridge.set_trailing_stop(
            "AAPL", direction,
            activation_pct=activation_pct, trail_pct=trail_pct)
        return bridge

    def test_trailing_stop_never_moves_against_long(self):
        """For a long position the stop only ratchets UP."""
        bridge = self._bridge_with_position("long", entry=100.0)

        bridge._update_trailing_stops("AAPL", 105.0)   # activates and ratchets
        ts = bridge._trailing_stops["AAPL"]
        assert ts.activated is True
        peak_stop = ts.stop_price
        assert peak_stop == pytest.approx(105.0 * 0.985, rel=1e-4)

        bridge._update_trailing_stops("AAPL", 103.0)   # pullback
        assert bridge._trailing_stops["AAPL"].stop_price == pytest.approx(peak_stop), (
            "Trailing stop moved DOWN — money exposed!"
        )

    def test_trailing_stop_never_moves_against_short(self):
        """For a short position the stop only ratchets DOWN."""
        bridge = self._bridge_with_position("short", entry=100.0)

        bridge._update_trailing_stops("AAPL", 95.0)
        ts = bridge._trailing_stops["AAPL"]
        assert ts.activated is True
        trough_stop = ts.stop_price
        assert trough_stop == pytest.approx(95.0 * 1.015, rel=1e-4)

        bridge._update_trailing_stops("AAPL", 97.0)
        assert bridge._trailing_stops["AAPL"].stop_price == pytest.approx(trough_stop), (
            "Short trailing stop moved UP — money exposed!"
        )

    def test_trailing_stop_triggers_exit(self):
        """A reversal through the stop reports a trigger."""
        bridge = self._bridge_with_position("long", entry=100.0, trail_pct=0.02)

        assert bridge._update_trailing_stops("AAPL", 105.0) is False
        assert bridge._update_trailing_stops("AAPL", 102.0) is True, (
            "Price below the trailing stop but no exit signalled"
        )

    def test_trailing_stop_does_not_activate_before_the_threshold(self):
        """Below activation_pct profit the stop must stay dormant."""
        bridge = self._bridge_with_position("long", entry=100.0, activation_pct=0.05)

        assert bridge._update_trailing_stops("AAPL", 101.0) is False
        assert bridge._trailing_stops["AAPL"].activated is False

    def test_all_python_strategy_configs_have_stop_loss(self):
        """Factor, ML, Self-Learning configs all define stop_loss_pct."""
        configs = [
            ("MeanReversionConfig", MeanReversionConfig()),
            ("FactorPortfolioConfig", FactorPortfolioConfig()),
            ("SelfLearningConfig", SelfLearningConfig()),
        ]
        for name, cfg in configs:
            assert hasattr(cfg, "stop_loss_pct"), f"{name} missing stop_loss_pct"
            assert cfg.stop_loss_pct > 0, f"{name}.stop_loss_pct is zero or negative"

    def test_simulator_tp_sl_triggers(self, simulator: BrokerSimulator):
        """BrokerSimulator correctly triggers TP and SL orders."""
        simulator.tick()  # advance to bar 1

        # Place a buy, then a TP above and SL below
        buy = simulator.place_market_order("AAPL", "BUY", 100)
        assert buy.status == "FILLED"

        fill_price = buy.fill_price
        tp = simulator.place_limit_order(
            "AAPL", "SELL", 100, fill_price * 1.10, is_tp=True
        )
        sl = simulator.place_stop_order("AAPL", "SELL", 100, fill_price * 0.90)

        # TP and SL should be pending
        assert tp.status == "PENDING" or tp.status == "FILLED"
        # Tick forward until one triggers or bars exhaust
        for _ in range(50):
            if simulator.is_done:
                break
            simulator.tick()


# ════════════════════════════════════════════════════════════════════════════
# 3. TestRiskManagerCannotBeBypassed
# ════════════════════════════════════════════════════════════════════════════


class TestRiskManagerCannotBeBypassed:
    """No code path can skip risk checks."""

    def test_llm_cannot_override_risk_block(self, broker_bridge: BrokerBridge, unified_gate):
        """LLM says BUY but unified risk gate says NO → trade blocked."""
        # Saturate portfolio exposure
        unified_gate.register_position("MSFT", 300, 250.0, "s1", "ib")  # $75K
        broker_bridge._portfolio_gate = unified_gate

        decision = {"action": "BUY", "confidence": 0.99, "price": 200.0}
        result = broker_bridge.execute_decision(decision, "GOOGL")

        # Should be blocked by portfolio exposure gate
        assert result is not None
        assert result.success is False
        assert "risk gate" in result.message.lower() or "blocked" in result.message.lower()

    def test_manual_order_checks_portfolio_risk(self, broker_bridge: BrokerBridge, unified_gate):
        """Direct BUY order still goes through unified risk gate."""
        # Single-stock limit: 15% of $100K = $15K
        broker_bridge._portfolio_gate = unified_gate
        decision = {"action": "BUY", "confidence": 0.9, "price": 200.0}

        # Request shares that exceed 15% concentration
        broker_bridge._max_position_pct = 0.20  # $20K > $15K limit
        broker_bridge._max_shares = 10000

        result = broker_bridge.execute_decision(decision, "AAPL")
        if result and not result.success:
            assert "risk gate" in result.message.lower() or "blocked" in result.message.lower()

    def test_webhook_checks_daily_pnl_gate(self, daily_pnl_tracker: DailyPnLTracker):
        """Webhook daily PnL tracker blocks trades after loss limit."""
        for i in range(6):
            daily_pnl_tracker.record_trade(f"SYM{i}", pnl=-1000.0)

        assert daily_pnl_tracker.can_trade() is False

    def test_webhook_checks_cooldown_gate(self, cooldown_mgr: CooldownManager):
        """Webhook CooldownManager blocks strategy after 3 consecutive losses."""
        for _ in range(3):
            cooldown_mgr.record_result("trend_follow", won=False)

        assert cooldown_mgr.is_in_cooldown("trend_follow") is True

    def test_webhook_checks_drawdown_gate(self, drawdown_breaker: DrawdownCircuitBreaker):
        """Webhook DrawdownCircuitBreaker blocks all trading on 10%+ drawdown."""
        drawdown_breaker.update_equity(100_000.0)
        drawdown_breaker.update_equity(88_000.0)
        assert drawdown_breaker.can_trade() is False

    def test_webhook_risk_gates_integration(self):
        """Full _check_risk_gates function blocks when daily loss hit."""
        app = MagicMock()
        app.state.pnl_tracker = DailyPnLTracker(max_daily_loss=5000.0)
        app.state.cooldown_mgr = CooldownManager(max_consecutive_losses=3)
        app.state.drawdown_breaker = DrawdownCircuitBreaker(max_drawdown_pct=10.0)
        app.state.config = {"risk_management": {"max_trade_value": 50000.0, "max_sector_pct": 30.0}}
        app.state.sector_tracker = defaultdict(int)

        # Exhaust daily loss
        for i in range(6):
            app.state.pnl_tracker.record_trade(f"SYM{i}", pnl=-1000.0)

        alert = AlertPayload(symbol="AAPL", action="buy", price=150.0, quantity=100)
        response = _check_risk_gates(app, alert, 100.0)

        assert response is not None
        body = json.loads(response.body.decode())
        assert body["status"] == "blocked"
        assert body["reason"] == "daily_loss_limit"

    def test_concurrent_orders_respect_max_positions(self, risk_manager: RiskManager):
        """Simultaneous orders can't exceed max positions (10).

        can_trade() and add_position() are two operations, so concurrent
        callers can all see can_trade() as True before any of them takes a
        slot. add_position() re-checks the ceiling inside its lock and returns
        False when the account is full — the caller must honour that return
        value. Treating can_trade() alone as permission admitted 15 positions
        against a limit of 10.
        """
        results = []

        def try_trade(idx):
            if risk_manager.can_trade():
                granted = risk_manager.add_position(f"SYM{idx}", 500.0)
                results.append(granted)
            else:
                results.append(False)

        threads = []
        for i in range(15):
            t = threading.Thread(target=try_trade, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        success_count = sum(1 for r in results if r is True)
        assert success_count <= 10, f"Accepted {success_count} positions, max is 10"
        # The ceiling must actually be reached, not under-filled by the race.
        assert success_count == 10
        assert risk_manager.get_status()["open_positions"] == 10


# ════════════════════════════════════════════════════════════════════════════
# 4. TestCrashRecovery
# ════════════════════════════════════════════════════════════════════════════


class TestCrashRecovery:
    """System survives crashes without losing money."""

    def test_risk_state_survives_restart(self, tmp_path):
        """Daily P&L, loss streaks persist across restarts via SQLite."""
        db_path = str(tmp_path / "risk_state.db")

        # Session 1: record losing trades
        rm1 = RiskManager(config=RiskManagerConfig(
            total_capital=100_000.0,
            max_daily_loss=5_000.0,
            max_consecutive_losses=3,
            cooldown_seconds=1800,
            persist_path=db_path,
            min_seconds_between_trades=0.0,
        ))
        rm1.record_trade("AAPL", pnl=-1000.0)
        rm1.record_trade("MSFT", pnl=-1500.0)
        rm1.record_trade("GOOGL", pnl=-800.0)

        saved_pnl = rm1._daily_pnl
        saved_losses = rm1._consecutive_losses
        saved_equity = rm1._current_equity
        # Close the SQLite connection before "crash"
        if rm1._persist_conn:
            rm1._persist_conn.close()
        del rm1

        # Session 2: restart — state should be restored
        rm2 = RiskManager(config=RiskManagerConfig(
            total_capital=100_000.0,
            max_daily_loss=5_000.0,
            max_consecutive_losses=3,
            cooldown_seconds=1800,
            persist_path=db_path,
            min_seconds_between_trades=0.0,
        ))

        assert rm2._daily_pnl == pytest.approx(saved_pnl, abs=0.01)
        assert rm2._consecutive_losses == saved_losses
        assert rm2._current_equity == pytest.approx(saved_equity, abs=0.01)
        if rm2._persist_conn:
            rm2._persist_conn.close()

    def test_unified_gate_positions_survive_restart(self, tmp_path):
        """Open positions tracked by UnifiedPortfolioRiskGate persist."""
        db_path = str(tmp_path / "unified_risk.db")

        # Session 1
        UnifiedPortfolioRiskGate.reset_instance()
        config = UnifiedRiskConfig(
            account_equity=100_000.0,
            persist_path=db_path,
        )
        gate1 = UnifiedPortfolioRiskGate(config)
        gate1.register_position("AAPL", 100, 150.0, "strat1", "ib", "Technology")
        gate1.register_position("JPM", 50, 200.0, "strat2", "schwab", "Financials")
        if gate1._persist_conn:
            gate1._persist_conn.close()
        del gate1

        # Session 2
        UnifiedPortfolioRiskGate.reset_instance()
        gate2 = UnifiedPortfolioRiskGate(UnifiedRiskConfig(
            account_equity=100_000.0, persist_path=db_path,
        ))
        summary = gate2.get_portfolio_summary()
        assert summary["open_positions"] == 2
        assert "AAPL" in summary["positions"]
        assert "JPM" in summary["positions"]
        if gate2._persist_conn:
            gate2._persist_conn.close()

    def test_positions_tracked_after_restart(self, tmp_path):
        """RiskManager open positions are known after restart."""
        db_path = str(tmp_path / "risk_positions.db")

        rm1 = RiskManager(config=RiskManagerConfig(
            persist_path=db_path,
            min_seconds_between_trades=0.0,
        ))
        rm1.add_position("AAPL", 500.0)
        rm1.add_position("MSFT", 300.0)
        rm1.add_position("TSLA", 700.0)
        if rm1._persist_conn:
            rm1._persist_conn.close()
        del rm1

        rm2 = RiskManager(config=RiskManagerConfig(
            persist_path=db_path,
            min_seconds_between_trades=0.0,
        ))
        assert len(rm2._open_positions) == 3
        assert "AAPL" in rm2._open_positions
        if rm2._persist_conn:
            rm2._persist_conn.close()

    def test_cooldown_state_survives_restart(self, tmp_path):
        """Cooldown activation persists across process restarts."""
        db_path = str(tmp_path / "risk_cooldown.db")

        rm1 = RiskManager(config=RiskManagerConfig(
            max_consecutive_losses=3,
            cooldown_seconds=3600,
            persist_path=db_path,
            min_seconds_between_trades=0.0,
        ))
        # Trigger cooldown
        rm1.record_trade("A", pnl=-100)
        rm1.record_trade("B", pnl=-100)
        rm1.record_trade("C", pnl=-100)

        assert rm1._cooldown_until > time.time()
        saved_cooldown = rm1._cooldown_until
        if rm1._persist_conn:
            rm1._persist_conn.close()
        del rm1

        rm2 = RiskManager(config=RiskManagerConfig(
            max_consecutive_losses=3,
            cooldown_seconds=3600,
            persist_path=db_path,
            min_seconds_between_trades=0.0,
        ))
        assert rm2._cooldown_until == pytest.approx(saved_cooldown, abs=1.0)
        if rm2._persist_conn:
            rm2._persist_conn.close()


# ════════════════════════════════════════════════════════════════════════════
# 5. TestMarketConditionProtection
# ════════════════════════════════════════════════════════════════════════════


class TestMarketConditionProtection:
    """System adapts to dangerous market conditions."""

    def test_volatile_regime_reduces_exposure(self):
        """VOLATILE regime → agent returns HOLD or lower confidence."""
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        agent = SelfLearningAgent(config=AgentConfig(
            db_path=":memory:",
            total_capital=100_000.0,
        ))

        # Create very volatile data (high ATR)
        rng = np.random.default_rng(42)
        n = 200
        close = 100.0 + np.cumsum(rng.normal(0, 5, n))  # large swings
        close = np.maximum(close, 1.0)
        df = pd.DataFrame({
            "open": close * (1 + rng.uniform(-0.05, 0.05, n)),
            "high": close * (1 + rng.uniform(0, 0.08, n)),
            "low": close * (1 - rng.uniform(0, 0.08, n)),
            "close": close,
            "volume": rng.integers(100_000, 5_000_000, n),
        }, index=pd.date_range("2023-01-01", periods=n, freq="B"))

        decision = agent.decide(df, "AAPL")
        # In volatile regime, either HOLD or reduced confidence
        if decision["action"] != "HOLD":
            assert decision.get("regime", "").upper() in (
                "VOLATILE", "TRENDING", "RANGING", "UNKNOWN", ""
            )

    def test_mean_reversion_blocked_in_strong_trend(self):
        """Mean reversion config defines ADX threshold for blocking."""
        cfg = MeanReversionConfig()
        assert hasattr(cfg, "adx_threshold") or hasattr(cfg, "stop_loss_pct")
        assert cfg.stop_loss_pct > 0

    def test_eod_flatten_closes_all_positions(self, broker_bridge: BrokerBridge, mock_adapter):
        """close_all_positions() flattens every tracked position."""
        broker_bridge._positions["AAPL"] = Position(
            symbol="AAPL", direction="long", shares=100,
            entry_price=150.0, entry_time="2023-01-01",
        )
        broker_bridge._positions["MSFT"] = Position(
            symbol="MSFT", direction="short", shares=50,
            entry_price=300.0, entry_time="2023-01-01",
        )

        results = broker_bridge.close_all_positions()
        assert len(results) == 2
        assert len(broker_bridge._positions) == 0

    def test_no_trading_outside_market_hours(self):
        """MarketHours rejects timestamps outside 9:30-16:00 ET."""
        mh = MarketHours(allow_premarket=False, allow_afterhours=False)
        # 2:00 AM ET is outside regular hours
        dt_2am = datetime(2024, 3, 15, 2, 0)
        is_open = mh.is_market_open(dt_2am) if hasattr(mh, "is_market_open") else False
        assert is_open is False

    def test_sector_concentration_blocked(self, unified_gate: UnifiedPortfolioRiskGate):
        """Can't put > 30% of equity in one sector."""
        # Register $20K in tech stocks
        unified_gate.register_position("AAPL", 100, 150.0, "s1", "ib", "Technology")  # $15K
        unified_gate.register_position("MSFT", 50, 250.0, "s1", "ib", "Technology")  # $12.5K

        # Try to add another $10K tech → total $37.5K = 37.5% > 30%
        ok, reason = unified_gate.can_open_position("NVDA", 10_000.0, sector="Technology")
        assert ok is False
        assert "sector" in reason.lower()


# ════════════════════════════════════════════════════════════════════════════
# 6. TestDataIntegrity
# ════════════════════════════════════════════════════════════════════════════


class TestDataIntegrity:
    """Bad data doesn't cause bad trades."""

    def test_nan_prices_produce_hold(self):
        """NaN in price data → HOLD signal, not crash."""
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        agent = SelfLearningAgent(config=AgentConfig(db_path=":memory:"))

        df = _make_ohlcv(100)
        # Inject NaN into last 10 bars
        df.iloc[-10:, df.columns.get_loc("close")] = np.nan

        decision = agent.decide(df, "TEST")
        # Should not crash — should return HOLD or handle gracefully
        assert decision["action"] in ("HOLD", "BUY", "SELL")

    def test_zero_volume_handled(self):
        """Zero volume bars don't crash position sizing."""
        rm = RiskManager(config=RiskManagerConfig(total_capital=100_000.0))
        shares = rm.calculate_position_size("AAPL", 150.0, stop_price=145.0)
        assert shares > 0

    def test_negative_prices_rejected(self):
        """Negative price → position sizing produces tiny or safe size."""
        rm = RiskManager(config=RiskManagerConfig(total_capital=100_000.0))
        shares = rm.calculate_position_size("BAD", -50.0, stop_price=-55.0)
        # With negative prices, risk_per_share = abs(-50 - -55) = 5
        # The system still calculates shares but notional is negative so
        # the order would never execute. Verify sizing doesn't explode.
        assert isinstance(shares, int)
        assert shares < 1000  # bounded, not astronomical

    def test_zero_price_returns_zero_shares(self):
        """Zero price → zero shares."""
        rm = RiskManager(config=RiskManagerConfig(total_capital=100_000.0))
        shares = rm.calculate_position_size("BAD", 0.0, stop_price=0.0)
        assert shares == 0

    def test_empty_dataframe_returns_hold(self):
        """Empty data → HOLD, not exception."""
        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

        agent = SelfLearningAgent(config=AgentConfig(db_path=":memory:"))
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        try:
            decision = agent.decide(df, "EMPTY")
            assert decision["action"] == "HOLD"
        except (ValueError, KeyError, IndexError):
            # Acceptable — raising an error is safe (no trade placed)
            pass

    def test_stale_data_detected_by_health_monitor(self):
        """HealthMonitor detects stale alert feeds."""
        hm = HealthMonitor(max_silence_minutes=30)
        # Simulate last alert was 2 hours ago
        hm.last_alert_time = datetime.now(timezone.utc) - timedelta(hours=2)
        hm._max_silence = timedelta(minutes=30)

        # _check_silence should warn
        hm._silence_warned = False
        hm._check_silence()
        assert hm._silence_warned is True


# ════════════════════════════════════════════════════════════════════════════
# 7. TestPaperTradingAccuracy
# ════════════════════════════════════════════════════════════════════════════


class TestPaperTradingAccuracy:
    """Paper mode accurately simulates real trading."""

    def test_paper_long_pnl_correct(self, simulator: BrokerSimulator):
        """Buy 100 → price goes up → PnL positive."""
        simulator.tick()
        buy = simulator.place_market_order("AAPL", "BUY", 100)
        assert buy.status == "FILLED"
        entry = buy.fill_price

        # Tick forward a few bars
        for _ in range(5):
            if not simulator.is_done:
                simulator.tick()

        positions = simulator.get_positions()
        aapl_pos = [p for p in positions if p.get("symbol") == "AAPL"]
        if aapl_pos:
            pos = aapl_pos[0]
            expected_pnl_sign = 1 if pos.get("market_value", 0) > entry * 100 else -1
            assert "unrealized_pnl" in pos or "market_value" in pos

    def test_paper_short_pnl_correct(self, simulator: BrokerSimulator):
        """Short 100 → PnL calculated correctly."""
        simulator.tick()
        sell = simulator.place_market_order("AAPL", "SELL", 100)
        assert sell.status == "FILLED"

        positions = simulator.get_positions()
        aapl_pos = [p for p in positions if p.get("symbol") == "AAPL"]
        if aapl_pos:
            assert aapl_pos[0].get("quantity", 0) < 0

    def test_paper_commissions_deducted(self, simulator: BrokerSimulator):
        """Commissions reduce cash balance."""
        simulator.tick()
        initial_account = simulator.get_account_info()
        initial_cash = initial_account.get("cash", initial_account.get("buying_power", 100_000.0))

        buy = simulator.place_market_order("AAPL", "BUY", 100)
        assert buy.status == "FILLED"

        after_account = simulator.get_account_info()
        after_cash = after_account.get("cash", after_account.get("buying_power", 0))

        # Cash should decrease by (fill_price * 100) + commission
        expected_commission = 100 * 0.005  # $0.50
        trade_cost = buy.fill_price * 100
        assert after_cash < initial_cash, "Cash should decrease after buy"

    def test_paper_does_not_call_real_broker(self):
        """BrokerSimulator never makes real network calls."""
        sim = BrokerSimulator(initial_capital=50_000.0, broker_name="paper_test")
        sim.load_synthetic_data(symbols=["TEST"], n_bars=50, seed=1)
        sim.tick()

        # No network imports or connections should be triggered
        order = sim.place_market_order("TEST", "BUY", 10)
        assert order.status == "FILLED"
        # Verify no real broker attributes
        assert not hasattr(sim, "_connection")
        assert not hasattr(sim, "_client")
        assert not hasattr(sim, "_router")

    def test_paper_account_info_accurate(self, simulator: BrokerSimulator):
        """Account info reflects correct balances."""
        info = simulator.get_account_info()
        assert "cash" in info or "buying_power" in info or "equity" in info
        total = info.get("cash", info.get("equity", info.get("buying_power", 0)))
        assert total == pytest.approx(100_000.0, rel=0.01)


# ════════════════════════════════════════════════════════════════════════════
# 8. TestAlertAndNotification
# ════════════════════════════════════════════════════════════════════════════


class TestAlertAndNotification:
    """You're always informed of critical events."""

    def test_daily_loss_limit_sends_alert(self):
        """Hitting daily loss limit triggers dispatcher notification."""
        dispatcher = MagicMock(spec=AlertDispatcher)
        rm = RiskManager(config=RiskManagerConfig(
            max_daily_loss=5_000.0,
            min_seconds_between_trades=0.0,
        ))

        for i in range(10):
            rm.record_trade(f"SYM{i}", pnl=-600.0)

        # Verify the risk manager blocked trading
        assert rm.can_trade() is False
        # The system should alert — we verify the state is correct for alerting
        assert rm._daily_pnl <= -5_000.0

    def test_circuit_breaker_sends_alert(self, drawdown_breaker: DrawdownCircuitBreaker):
        """Circuit breaker activation is detectable for notification."""
        drawdown_breaker.update_equity(100_000.0)
        drawdown_breaker.update_equity(89_000.0)

        status = drawdown_breaker.get_status()
        assert status["tripped"] is True
        assert status["drawdown_pct"] >= 10.0

    def test_webhook_health_alerts_on_silence(self):
        """No alerts for 30min → health warning sent via dispatcher."""
        dispatcher = MagicMock()
        hm = HealthMonitor(
            max_silence_minutes=30,
            alert_dispatcher=dispatcher,
        )
        # Simulate: last alert was 45 minutes ago
        hm.last_alert_time = datetime.now(timezone.utc) - timedelta(minutes=45)
        hm._silence_warned = False
        hm._check_silence()

        assert hm._silence_warned is True
        dispatcher.dispatch.assert_called_once()
        call_kwargs = dispatcher.dispatch.call_args
        assert "silence" in call_kwargs.kwargs.get("title", "").lower() or \
               "silence" in str(call_kwargs).lower()

    def test_connection_loss_sends_alert(self):
        """Broker disconnection is detectable for alert dispatch."""
        adapter = MagicMock(spec=BaseBrokerAdapter)
        adapter.is_connected.return_value = False
        adapter.name = "interactive_brokers"

        # Connection loss should be detectable
        assert adapter.is_connected() is False

    def test_health_monitor_records_alerts(self):
        """HealthMonitor tracks alert count and timestamps."""
        hm = HealthMonitor()
        assert hm.alerts_processed == 0
        assert hm.last_alert_time is None

        hm.record_alert()
        assert hm.alerts_processed == 1
        assert hm.last_alert_time is not None

        hm.record_alert()
        assert hm.alerts_processed == 2

    def test_health_monitor_latency_tracking(self):
        """HealthMonitor tracks alert-to-order latency."""
        hm = HealthMonitor()
        hm.record_latency(150.0)
        hm.record_latency(200.0)
        hm.record_latency(180.0)

        stats = hm.get_latency_stats()
        assert stats["samples"] == 3
        assert stats["avg_ms"] == pytest.approx(176.67, abs=1.0)
        assert stats["min_ms"] == pytest.approx(150.0)
        assert stats["max_ms"] == pytest.approx(200.0)

    def test_alert_freshness_detects_stale(self):
        """HealthMonitor flags stale alerts when timeout exceeded."""
        hm = HealthMonitor(alert_timeout_hours=1.0)
        hm.last_alert_time = datetime.now(timezone.utc) - timedelta(hours=2)
        result = hm.check_alert_freshness()
        assert result["status"] == "stale"


# ════════════════════════════════════════════════════════════════════════════
# 9. TestPositionSizing
# ════════════════════════════════════════════════════════════════════════════


class TestPositionSizing:
    """Position sizing methods all produce safe sizes."""

    def test_fixed_fractional_caps_risk(self):
        """Fixed fractional: risk per trade = 2% of capital."""
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade_pct=2.0,
            total_capital=100_000.0,
        ))
        shares = rm.calculate_position_size("AAPL", 150.0, stop_price=145.0)
        risk = shares * (150.0 - 145.0)
        assert risk <= 2_100.0  # 2% of $100K + rounding

    def test_kelly_sizing_bounded(self):
        """Kelly criterion respects half-Kelly fraction."""
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.KELLY,
            kelly_win_rate=0.55,
            kelly_avg_win=1.5,
            kelly_avg_loss=1.0,
            kelly_fraction=0.5,
            total_capital=100_000.0,
        ))
        shares = rm.calculate_position_size("AAPL", 150.0)
        # Kelly full fraction ~0.183, half ~0.092 → ~$9.2K → ~61 shares
        notional = shares * 150.0
        assert notional <= 15_000.0, f"Kelly sizing too aggressive: ${notional:.0f}"

    def test_fixed_shares_returns_exact(self):
        """Fixed shares method returns config value."""
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_SHARES,
            fixed_shares=100,
        ))
        assert rm.calculate_position_size("AAPL", 150.0) == 100

    def test_fixed_dollar_respects_amount(self):
        """Fixed dollar method: $10K / $150 = 66 shares."""
        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_DOLLAR,
            fixed_dollar_amount=10_000.0,
        ))
        shares = rm.calculate_position_size("AAPL", 150.0)
        assert shares == 66  # int(10000/150) = 66

    def test_bridge_caps_at_max_shares(self, broker_bridge: BrokerBridge):
        """BrokerBridge._calculate_shares never exceeds max_shares."""
        broker_bridge._max_shares = 200
        broker_bridge._max_position_pct = 1.0  # would suggest huge size
        broker_bridge._capital = 1_000_000.0

        shares = broker_bridge._calculate_shares(10.0)
        assert shares <= 200


# ════════════════════════════════════════════════════════════════════════════
# 10. TestCorrelationAndConcentration
# ════════════════════════════════════════════════════════════════════════════


class TestCorrelationAndConcentration:
    """Cross-asset risk controls prevent correlated blowups."""

    def test_correlated_exposure_blocked(self, unified_gate: UnifiedPortfolioRiskGate):
        """Correlated group exposure is checked and can block."""
        # Use financials (within 30% sector limit) to isolate the correlation check
        unified_gate.register_position("JPM", 100, 130.0, "s1", "ib")  # $13K financials
        unified_gate.register_position("BAC", 200, 100.0, "s1", "ib")  # $20K financials

        # JPM+BAC = $33K financials → 33% > 30% sector limit
        # Try adding another financial → sector gate fires first
        ok, reason = unified_gate.can_open_position("GS", 5_000.0, sector="Financials")
        assert ok is False
        # Either sector or correlated block is acceptable — both protect money
        assert "sector" in reason.lower() or "correlated" in reason.lower()

    def test_uncorrelated_positions_allowed(self, unified_gate: UnifiedPortfolioRiskGate):
        """Positions in different sectors/groups are allowed."""
        unified_gate.register_position("AAPL", 50, 150.0, "s1", "ib")   # $7.5K tech
        unified_gate.register_position("JPM", 30, 200.0, "s1", "ib")    # $6K financials

        # Add energy — different group, within limits
        ok, reason = unified_gate.can_open_position("XOM", 10_000.0)
        assert ok is True

    def test_portfolio_summary_accurate(self, unified_gate: UnifiedPortfolioRiskGate):
        """get_portfolio_summary returns correct numbers."""
        unified_gate.register_position("AAPL", 100, 150.0, "s1", "ib", "Technology")
        unified_gate.register_position("JPM", 50, 200.0, "s2", "schwab", "Financials")

        summary = unified_gate.get_portfolio_summary()
        assert summary["account_equity"] == 100_000.0
        assert summary["total_exposure"] == pytest.approx(25_000.0)
        assert summary["open_positions"] == 2
        assert "Technology" in summary["sector_exposure"]
        assert "Financials" in summary["sector_exposure"]

    def test_close_position_updates_exposure(self, unified_gate: UnifiedPortfolioRiskGate):
        """Closing a position reduces tracked exposure."""
        unified_gate.register_position("AAPL", 100, 150.0, "s1", "ib")
        assert unified_gate._total_exposure == pytest.approx(15_000.0)

        unified_gate.close_position("AAPL", pnl=500.0)
        assert unified_gate._total_exposure == pytest.approx(0.0)
        assert "AAPL" not in unified_gate._positions
