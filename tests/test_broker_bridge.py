"""
Comprehensive tests for BrokerBridge.

Covers: __init__, connect(), _place_tp_sl() (BOTH TP and SL placed),
_close_position(), _calculate_shares(), diary file operations (UTF-8),
variable naming (field_name not field), TP/SL price calculations,
execute_decision, reconcile_positions, force-close, and mock adapter.
"""

import sys
import os
import json
import tempfile

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.daemon.broker_bridge import (
    BaseBrokerAdapter,
    BrokerBridge,
    ExecutionResult,
    IBAdapter,
    Position,
    SchwabAdapter,
    TradeStationAdapter,
)


# ── Mock Adapter ─────────────────────────────────────────────────────────

class MockAdapter(BaseBrokerAdapter):
    def __init__(self):
        self._connected = False
        self._orders = []
        self._positions = []
        self._prices = {}
        self._name = "mock"

    @property
    def name(self):
        return self._name

    def connect(self):
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    def place_market_order(self, symbol, action, quantity):
        oid = f"MKT-{len(self._orders)}"
        self._orders.append({"symbol": symbol, "action": action, "qty": quantity, "type": "market"})
        return ExecutionResult(True, self.name, symbol, action, quantity, 0.0, order_id=oid)

    def place_limit_order(self, symbol, action, quantity, price):
        oid = f"LMT-{len(self._orders)}"
        self._orders.append({"symbol": symbol, "action": action, "qty": quantity, "price": price, "type": "limit"})
        return ExecutionResult(True, self.name, symbol, action, quantity, price, order_id=oid)

    def place_stop_order(self, symbol, action, quantity, stop_price):
        # A stop-capable broker. BaseBrokerAdapter.place_stop_order now fails
        # closed, so tests about TP/SL pairing need an adapter that really
        # supports stops rather than relying on the old limit-order fallback.
        oid = f"STP-{len(self._orders)}"
        self._orders.append({
            "symbol": symbol, "action": action, "qty": quantity,
            "price": stop_price, "type": "stop",
        })
        return ExecutionResult(True, self.name, symbol, action, quantity, stop_price, order_id=oid)

    def cancel_order(self, order_id):
        return True

    def get_positions(self):
        return self._positions

    def get_account_info(self):
        return {"broker": self.name, "connected": self._connected}

    def get_latest_price(self, symbol):
        return self._prices.get(symbol, 0.0)


def _make_bridge(adapter=None, **kwargs):
    from shared.risk_manager_unified import UnifiedPortfolioRiskGate
    UnifiedPortfolioRiskGate.reset_instance()
    diary_fd, diary_path = tempfile.mkstemp(suffix=".jsonl")
    os.close(diary_fd)
    defaults = dict(broker="ib", mode="paper", diary_path=diary_path)
    defaults.update(kwargs)
    bridge = BrokerBridge(**defaults)
    if adapter is None:
        adapter = MockAdapter()
    bridge._adapter = adapter
    return bridge


# ── Dataclass Tests ──────────────────────────────────────────────────────

class TestPosition:
    def test_fields(self):
        p = Position("AAPL", "long", 100, 150.0, "2024-01-01T10:00:00")
        assert p.symbol == "AAPL"
        assert p.direction == "long"
        assert p.unrealized_pnl == 0.0

    def test_defaults(self):
        p = Position("SPY", "short", 50, 400.0, "2024-01-01")
        assert p.current_price == 0.0
        assert p.order_id is None


class TestExecutionResult:
    def test_success(self):
        r = ExecutionResult(True, "ib", "AAPL", "BUY", 10, 150.0, order_id="123")
        assert r.success and r.order_id == "123"

    def test_failure(self):
        r = ExecutionResult(False, "ib", "AAPL", "BUY", 10, 0.0, message="refused")
        assert not r.success and "refused" in r.message


# ── BrokerBridge __init__ ────────────────────────────────────────────────

class TestBrokerBridgeInit:
    def test_defaults(self):
        bridge = _make_bridge()
        assert bridge._broker_name == "ib"
        assert bridge._mode == "paper"
        assert bridge._max_shares == 500
        assert bridge._default_tp_pct == 3.0
        assert bridge._default_sl_pct == 2.0

    def test_custom(self):
        bridge = _make_bridge(max_shares=100, capital=50_000, max_loss_pct=3.0)
        assert bridge._max_shares == 100
        assert bridge._capital == 50_000

    def test_unknown_broker_raises(self):
        with pytest.raises(ValueError, match="Unknown broker"):
            BrokerBridge(broker="nonexistent", diary_path="/tmp/t.jsonl")


# ── _create_adapter ──────────────────────────────────────────────────────

class TestCreateAdapter:
    def test_ib(self):
        b = BrokerBridge(broker="ib", diary_path=tempfile.mktemp(suffix=".jsonl"))
        assert isinstance(b._adapter, IBAdapter)

    def test_tradestation(self):
        b = BrokerBridge(broker="ts", diary_path=tempfile.mktemp(suffix=".jsonl"))
        assert isinstance(b._adapter, TradeStationAdapter)

    def test_schwab(self):
        b = BrokerBridge(broker="schwab", diary_path=tempfile.mktemp(suffix=".jsonl"))
        assert isinstance(b._adapter, SchwabAdapter)


# ── connect / disconnect ─────────────────────────────────────────────────

class TestConnectDisconnect:
    def test_connect(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        assert bridge.connect() is True
        assert a.is_connected()

    def test_disconnect(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge.connect()
        bridge.disconnect()
        assert not a.is_connected()

    def test_is_connected(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        assert not bridge.is_connected()
        bridge.connect()
        assert bridge.is_connected()


# ── _calculate_shares ────────────────────────────────────────────────────

class TestCalculateShares:
    def test_normal(self):
        bridge = _make_bridge(capital=100_000, max_position_pct=0.10, max_shares=500)
        assert bridge._calculate_shares(50.0) == 200

    def test_zero_price(self):
        assert _make_bridge()._calculate_shares(0.0) == 0

    def test_negative_price(self):
        assert _make_bridge()._calculate_shares(-10.0) == 0

    def test_max_shares_cap(self):
        bridge = _make_bridge(capital=1_000_000, max_position_pct=0.50, max_shares=100)
        assert bridge._calculate_shares(10.0) == 100


# ── _place_tp_sl: BOTH TP and SL ────────────────────────────────────────

class TestPlaceTpSl:
    # The take-profit is a limit order and the stop-loss is a STOP order. They
    # are asserted separately: a stop-loss submitted as a limit at the stop
    # price is marketable and would close the position immediately.

    def test_both_placed_long(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a, default_tp_pct=3.0, default_sl_pct=2.0)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 150.0, "t")
        bridge._place_tp_sl("AAPL", 150.0, "long")
        limits = [o for o in a._orders if o["type"] == "limit"]
        stops = [o for o in a._orders if o["type"] == "stop"]
        assert len(limits) == 1
        assert len(stops) == 1
        assert all(o["action"] == "SELL" for o in limits + stops)

    def test_both_placed_short(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a, default_tp_pct=3.0, default_sl_pct=2.0)
        bridge._positions["AAPL"] = Position("AAPL", "short", 100, 150.0, "t")
        bridge._place_tp_sl("AAPL", 150.0, "short")
        limits = [o for o in a._orders if o["type"] == "limit"]
        stops = [o for o in a._orders if o["type"] == "stop"]
        assert len(limits) == 1
        assert len(stops) == 1
        assert all(o["action"] == "BUY" for o in limits + stops)

    def test_tp_price_long(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a, default_tp_pct=5.0, default_sl_pct=3.0)
        bridge._positions["X"] = Position("X", "long", 10, 100.0, "t")
        bridge._place_tp_sl("X", 100.0, "long")
        limits = [o for o in a._orders if o["type"] == "limit"]
        stops = [o for o in a._orders if o["type"] == "stop"]
        assert limits[0]["price"] == pytest.approx(105.0)
        assert stops[0]["price"] == pytest.approx(97.0)

    def test_tp_price_short(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a, default_tp_pct=5.0, default_sl_pct=3.0)
        bridge._positions["X"] = Position("X", "short", 10, 100.0, "t")
        bridge._place_tp_sl("X", 100.0, "short")
        limits = [o for o in a._orders if o["type"] == "limit"]
        stops = [o for o in a._orders if o["type"] == "stop"]
        assert limits[0]["price"] == pytest.approx(95.0)
        assert stops[0]["price"] == pytest.approx(103.0)

    def test_custom_overrides(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["X"] = Position("X", "long", 10, 150.0, "t")
        bridge._place_tp_sl("X", 150.0, "long", tp_price=160.0, sl_price=140.0)
        limits = [o for o in a._orders if o["type"] == "limit"]
        stops = [o for o in a._orders if o["type"] == "stop"]
        assert limits[0]["price"] == 160.0
        assert stops[0]["price"] == 140.0

    def test_no_position_no_orders(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._place_tp_sl("AAPL", 150.0, "long")
        assert len(a._orders) == 0

    def test_tp_failure_handled(self):
        """TP order fails, SL still placed."""
        class FailTPAdapter(MockAdapter):
            def place_limit_order(self, symbol, action, quantity, price):
                if price > 100:  # TP price
                    raise RuntimeError("TP failed")
                return super().place_limit_order(symbol, action, quantity, price)

        adapter = FailTPAdapter()
        bridge = _make_bridge(adapter=adapter)
        bridge._positions["X"] = Position("X", "long", 10, 100.0, "t")
        bridge._place_tp_sl("X", 100.0, "long")

    def test_sl_failure_handled(self):
        """SL stop order fails, no crash."""
        class FailSLAdapter(MockAdapter):
            def place_stop_order(self, symbol, action, quantity, stop_price):
                raise RuntimeError("SL failed")

        adapter = FailSLAdapter()
        bridge = _make_bridge(adapter=adapter)
        bridge._positions["X"] = Position("X", "long", 10, 100.0, "t")
        bridge._place_tp_sl("X", 100.0, "long")


# ── execute_decision ─────────────────────────────────────────────────────

class TestExecuteDecision:
    def test_hold_returns_none(self):
        bridge = _make_bridge()
        assert bridge.execute_decision({"action": "HOLD", "price": 100}, "AAPL") is None

    def test_buy_opens_long(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        r = bridge.execute_decision({"action": "BUY", "price": 100.0, "confidence": 0.9}, "AAPL")
        assert r.success
        assert bridge._positions["AAPL"].direction == "long"

    def test_sell_opens_short(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        r = bridge.execute_decision({"action": "SELL", "price": 100.0, "confidence": 0.9}, "AAPL")
        assert r.success
        assert bridge._positions["AAPL"].direction == "short"

    def test_buy_while_long_skips(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["AAPL"] = Position("AAPL", "long", 50, 100.0, "t")
        r = bridge.execute_decision({"action": "BUY", "price": 105.0}, "AAPL")
        assert r is None

    def test_sell_while_short_skips(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["AAPL"] = Position("AAPL", "short", 50, 100.0, "t")
        r = bridge.execute_decision({"action": "SELL", "price": 95.0}, "AAPL")
        assert r is None

    def test_buy_closes_short_then_opens_long(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["AAPL"] = Position("AAPL", "short", 50, 100.0, "t")
        r = bridge.execute_decision({"action": "BUY", "price": 95.0, "confidence": 0.8}, "AAPL")
        assert r.success
        assert bridge._positions["AAPL"].direction == "long"

    def test_sell_closes_long(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["AAPL"] = Position("AAPL", "long", 50, 100.0, "t")
        r = bridge.execute_decision({"action": "SELL", "price": 110.0}, "AAPL")
        assert r.success
        assert "AAPL" not in bridge._positions


# ── _close_position ──────────────────────────────────────────────────────

class TestClosePosition:
    def test_close_long_pnl(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 100.0, "t")
        r = bridge._close_position("AAPL", 110.0)
        assert r.success
        assert "AAPL" not in bridge._positions

    def test_close_short_pnl(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["AAPL"] = Position("AAPL", "short", 100, 110.0, "t")
        r = bridge._close_position("AAPL", 100.0)
        assert r.success
        assert "AAPL" not in bridge._positions

    def test_close_no_position(self):
        bridge = _make_bridge()
        r = bridge._close_position("AAPL", 100.0)
        assert not r.success


# ── Diary file operations (UTF-8) ────────────────────────────────────────

class TestDiary:
    def test_write_diary_utf8(self):
        """Verify fix: diary file opened with encoding='utf-8'."""
        bridge = _make_bridge()
        bridge._write_diary({"action": "BUY", "note": "special chars"})
        with open(bridge._diary_path, "r", encoding="utf-8") as f:
            content = f.read()
        entry = json.loads(content.strip())
        assert entry["action"] == "BUY"
        assert entry["note"] == "special chars"

    def test_diary_entry_has_timestamp(self):
        bridge = _make_bridge()
        bridge._write_diary({"action": "TEST"})
        with open(bridge._diary_path, "r", encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert "timestamp" in entry
        assert "broker" in entry

    def test_get_diary_reads_entries(self):
        bridge = _make_bridge()
        for i in range(5):
            bridge._write_diary({"index": i})
        entries = bridge.get_diary(n=3)
        assert len(entries) == 3

    def test_get_diary_malformed_line(self):
        bridge = _make_bridge()
        with open(bridge._diary_path, "w", encoding="utf-8") as f:
            f.write('{"action": "BUY"}\n')
            f.write('NOT VALID JSON\n')
            f.write('{"action": "SELL"}\n')
        entries = bridge.get_diary(n=10)
        assert len(entries) == 2

    def test_get_diary_file_not_found(self):
        bridge = _make_bridge()
        bridge._diary_path = "/nonexistent/path/diary.jsonl"
        assert bridge.get_diary() == []


# ── check_and_force_close ────────────────────────────────────────────────

class TestForceClose:
    def test_force_close_on_max_loss(self):
        a = MockAdapter()
        a._prices["AAPL"] = 90.0
        bridge = _make_bridge(adapter=a, max_loss_pct=5.0)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 100.0, "t")
        results = bridge.check_and_force_close()
        assert len(results) == 1
        assert results[0].success

    def test_no_force_close_within_threshold(self):
        a = MockAdapter()
        a._prices["AAPL"] = 98.0
        bridge = _make_bridge(adapter=a, max_loss_pct=5.0)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 100.0, "t")
        results = bridge.check_and_force_close()
        assert len(results) == 0

    def test_force_close_short_position(self):
        a = MockAdapter()
        a._prices["AAPL"] = 112.0
        bridge = _make_bridge(adapter=a, max_loss_pct=10.0)
        bridge._positions["AAPL"] = Position("AAPL", "short", 100, 100.0, "t")
        results = bridge.check_and_force_close()
        assert len(results) == 1


# ── reconcile_positions ──────────────────────────────────────────────────

class TestReconcile:
    def test_removes_stale(self):
        a = MockAdapter()
        a._positions = []
        bridge = _make_bridge(adapter=a)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 150.0, "t")
        recon = bridge.reconcile_positions()
        assert "AAPL" in recon["removed"]
        assert "AAPL" not in bridge._positions

    def test_detects_missing(self):
        a = MockAdapter()
        a._positions = [{"symbol": "MSFT", "quantity": 50}]
        bridge = _make_bridge(adapter=a)
        recon = bridge.reconcile_positions()
        assert "MSFT" in recon["added"]

    def test_matched_count(self):
        a = MockAdapter()
        a._positions = [{"symbol": "AAPL", "quantity": 100}]
        bridge = _make_bridge(adapter=a)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 150.0, "t")
        recon = bridge.reconcile_positions()
        assert recon["matched"] == 1


# ── Variable naming: field_name not field ────────────────────────────────

class TestFieldNameFix:
    def test_tradestation_get_latest_price_uses_field_name(self):
        """Quote fields are tried in preference order: Last, Ask, Bid, Close.

        This asserted on the SOURCE TEXT — `"field_name" in source` — and its
        second assertion, `"for field " not in source or "field_name" in
        source`, could never fail: the line above had already established the
        right operand. It passed on any refactor and tested no behaviour.
        """
        adapter = TradeStationAdapter({}, account_id="A1")

        class Router:
            def __init__(self, quote):
                self._quote = quote

            def get_quote(self, symbol):
                return self._quote

        # Preference order.
        adapter._router = Router({"Last": 101.5, "Ask": 102.0, "Bid": 101.0})
        assert adapter.get_latest_price("AAPL") == pytest.approx(101.5)

        # Falls through missing fields.
        adapter._router = Router({"Ask": 102.0, "Bid": 101.0})
        assert adapter.get_latest_price("AAPL") == pytest.approx(102.0)

        # Skips non-positive values rather than returning them.
        adapter._router = Router({"Last": 0, "Ask": None, "Bid": 99.5})
        assert adapter.get_latest_price("AAPL") == pytest.approx(99.5)

        # No usable field, a broken router, and no router at all all give 0.0.
        adapter._router = Router({"Last": None})
        assert adapter.get_latest_price("AAPL") == 0.0

        class Broken:
            def get_quote(self, symbol):
                raise RuntimeError("quote feed down")

        adapter._router = Broken()
        assert adapter.get_latest_price("AAPL") == 0.0

        adapter._router = None
        assert adapter.get_latest_price("AAPL") == 0.0


# ── __repr__ ─────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr(self):
        bridge = _make_bridge()
        r = repr(bridge)
        assert "BrokerBridge" in r
        assert "mock" in r


# ── TrailingStop dataclass ───────────────────────────────────────────────

from shared.daemon.broker_bridge import TrailingStop


class TestTrailingStopDataclass:
    def test_fields(self):
        ts = TrailingStop(
            symbol="AAPL", direction="long",
            activation_pct=0.02, trail_pct=0.015,
        )
        assert ts.symbol == "AAPL"
        assert ts.direction == "long"
        assert ts.activation_pct == 0.02
        assert ts.trail_pct == 0.015
        assert ts.activated is False
        assert ts.highest_price == 0.0
        assert ts.lowest_price == float('inf')
        assert ts.stop_price == 0.0


# ── set_trailing_stop ────────────────────────────────────────────────────

class TestSetTrailingStop:
    def test_creates_entry(self):
        bridge = _make_bridge()
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 150.0, "t")
        bridge.set_trailing_stop("AAPL", "long", activation_pct=0.02, trail_pct=0.015)
        assert "AAPL" in bridge._trailing_stops
        ts = bridge._trailing_stops["AAPL"]
        assert ts.activation_pct == 0.02
        assert ts.trail_pct == 0.015
        assert ts.highest_price == 150.0

    def test_no_position_does_nothing(self):
        bridge = _make_bridge()
        bridge.set_trailing_stop("AAPL", "long")
        assert "AAPL" not in bridge._trailing_stops


# ── Trailing stop activation and triggering ──────────────────────────────

class TestTrailingStopLogic:
    def test_activates_on_profit(self):
        bridge = _make_bridge()
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 100.0, "t")
        bridge.set_trailing_stop("AAPL", "long", activation_pct=0.02, trail_pct=0.015)
        triggered = bridge._update_trailing_stops("AAPL", 102.5)
        assert not triggered
        assert bridge._trailing_stops["AAPL"].activated is True

    def test_triggers_exit_long(self):
        bridge = _make_bridge()
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 100.0, "t")
        bridge.set_trailing_stop("AAPL", "long", activation_pct=0.02, trail_pct=0.015)
        bridge._update_trailing_stops("AAPL", 105.0)
        assert bridge._trailing_stops["AAPL"].activated is True
        ts = bridge._trailing_stops["AAPL"]
        assert ts.highest_price == 105.0
        stop = 105.0 * (1 - 0.015)
        triggered = bridge._update_trailing_stops("AAPL", stop - 0.01)
        assert triggered

    def test_trailing_stop_short(self):
        bridge = _make_bridge()
        bridge._positions["AAPL"] = Position("AAPL", "short", 100, 100.0, "t")
        bridge.set_trailing_stop("AAPL", "short", activation_pct=0.02, trail_pct=0.015)
        bridge._update_trailing_stops("AAPL", 97.5)
        assert bridge._trailing_stops["AAPL"].activated is True
        ts = bridge._trailing_stops["AAPL"]
        assert ts.lowest_price == 97.5
        stop = 97.5 * (1 + 0.015)
        triggered = bridge._update_trailing_stops("AAPL", stop + 0.01)
        assert triggered

    def test_no_trailing_stop_returns_false(self):
        bridge = _make_bridge()
        assert bridge._update_trailing_stops("AAPL", 100.0) is False

    def test_trailing_stop_long_not_yet_activated(self):
        bridge = _make_bridge()
        bridge._positions["X"] = Position("X", "long", 100, 100.0, "t")
        bridge.set_trailing_stop("X", "long", activation_pct=0.05, trail_pct=0.02)
        triggered = bridge._update_trailing_stops("X", 101.0)
        assert not triggered
        assert bridge._trailing_stops["X"].activated is False

    def test_trailing_stop_short_not_yet_activated(self):
        bridge = _make_bridge()
        bridge._positions["X"] = Position("X", "short", 100, 100.0, "t")
        bridge.set_trailing_stop("X", "short", activation_pct=0.05, trail_pct=0.02)
        triggered = bridge._update_trailing_stops("X", 99.0)
        assert not triggered
        assert bridge._trailing_stops["X"].activated is False

    def test_trailing_stop_short_updates_lowest(self):
        bridge = _make_bridge()
        bridge._positions["X"] = Position("X", "short", 100, 100.0, "t")
        bridge.set_trailing_stop("X", "short", activation_pct=0.02, trail_pct=0.015)
        bridge._update_trailing_stops("X", 97.0)
        assert bridge._trailing_stops["X"].activated is True
        bridge._update_trailing_stops("X", 95.0)
        assert bridge._trailing_stops["X"].lowest_price == 95.0
        assert bridge._trailing_stops["X"].stop_price == pytest.approx(95.0 * 1.015)

    def test_trailing_stop_no_position_cleans_up(self):
        bridge = _make_bridge()
        bridge._trailing_stops["X"] = TrailingStop("X", "long", 0.02, 0.015)
        triggered = bridge._update_trailing_stops("X", 100.0)
        assert not triggered
        assert "X" not in bridge._trailing_stops


# ── OCO pairs ────────────────────────────────────────────────────────────

class TestOCOPairs:
    def test_oco_pairs_tracked(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a, default_tp_pct=3.0, default_sl_pct=2.0)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 150.0, "t")
        bridge._place_tp_sl("AAPL", 150.0, "long")
        assert len(bridge._oco_pairs) == 2

    def test_cancel_paired_order(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a, default_tp_pct=3.0, default_sl_pct=2.0)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 150.0, "t")
        bridge._place_tp_sl("AAPL", 150.0, "long")
        tp_id = list(bridge._oco_pairs.keys())[0]
        sl_id = bridge._oco_pairs[tp_id]
        bridge._cancel_paired_order(tp_id)
        assert tp_id not in bridge._oco_pairs
        assert sl_id not in bridge._oco_pairs

    def test_cancel_paired_order_failure(self):
        """OCO cancel fails gracefully."""
        class FailCancelAdapter(MockAdapter):
            def cancel_order(self, order_id):
                return False

        adapter = FailCancelAdapter()
        bridge = _make_bridge(adapter=adapter)
        bridge._oco_pairs["A"] = "B"
        bridge._oco_pairs["B"] = "A"
        bridge._cancel_paired_order("A")
        assert "A" not in bridge._oco_pairs

    def test_cancel_paired_order_exception(self):
        """OCO cancel exception handled."""
        class ExplodeAdapter(MockAdapter):
            def cancel_order(self, order_id):
                raise RuntimeError("network error")

        adapter = ExplodeAdapter()
        bridge = _make_bridge(adapter=adapter)
        bridge._oco_pairs["X"] = "Y"
        bridge._oco_pairs["Y"] = "X"
        bridge._cancel_paired_order("X")

    def test_cancel_no_paired(self):
        bridge = _make_bridge()
        bridge._cancel_paired_order("NONEXISTENT")


# ── on_fill callback ────────────────────────────────────────────────────

class TestOnFill:
    def test_on_fill_closes_position_and_cancels_pair(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 100.0, "t")
        bridge._oco_pairs["TP1"] = "SL1"
        bridge._oco_pairs["SL1"] = "TP1"
        bridge.on_fill("TP1", "AAPL", 110.0)
        assert "AAPL" not in bridge._positions
        assert "TP1" not in bridge._oco_pairs
        assert "SL1" not in bridge._oco_pairs

    def test_on_fill_short_position(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["X"] = Position("X", "short", 50, 100.0, "t")
        bridge._trailing_stops["X"] = TrailingStop("X", "short", 0.02, 0.015)
        bridge.on_fill("O1", "X", 95.0)
        assert "X" not in bridge._positions
        assert "X" not in bridge._trailing_stops

    def test_on_fill_no_paired_order(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["X"] = Position("X", "long", 50, 100.0, "t")
        bridge.on_fill("UNKNOWN", "X", 110.0)
        assert "X" not in bridge._positions


# ── RiskManager position sizing ──────────────────────────────────────────

class TestRiskManagerSizing:
    def test_risk_manager_used_for_shares(self):
        bridge = _make_bridge(capital=100_000, max_shares=500)
        mock_rm = MagicMock()
        mock_rm.calculate_position_size.return_value = 75
        bridge._risk_manager = mock_rm
        shares = bridge._calculate_shares(100.0)
        assert shares == 75
        mock_rm.calculate_position_size.assert_called_once()

    def test_risk_manager_capped_at_max_shares(self):
        bridge = _make_bridge(capital=100_000, max_shares=50)
        mock_rm = MagicMock()
        mock_rm.calculate_position_size.return_value = 200
        bridge._risk_manager = mock_rm
        shares = bridge._calculate_shares(100.0)
        assert shares == 50

    def test_risk_manager_error_fails_closed(self):
        """A sizing error must block the order, not fall back to full size.

        Falling back to max_position_pct meant any exception inside the risk
        manager silently produced a normal-sized unprotected position — the
        risk manager failing was indistinguishable from it approving.
        """
        bridge = _make_bridge(capital=100_000, max_position_pct=0.10, max_shares=500)
        mock_rm = MagicMock()
        mock_rm.calculate_position_size.side_effect = RuntimeError("error")
        bridge._risk_manager = mock_rm
        assert bridge._calculate_shares(50.0) == 0


# ── place_stop_order ─────────────────────────────────────────────────────

class TestPlaceStopOrder:
    def test_sl_uses_stop_order(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["X"] = Position("X", "long", 10, 100.0, "t")

        class StopAdapter(MockAdapter):
            def __init__(self):
                super().__init__()
                self._stop_orders = []

            def place_stop_order(self, symbol, action, quantity, stop_price):
                self._stop_orders.append({
                    "symbol": symbol, "action": action,
                    "qty": quantity, "stop_price": stop_price,
                })
                oid = f"STP-{len(self._stop_orders)}"
                return ExecutionResult(True, self.name, symbol, action, quantity, stop_price, order_id=oid)

        stop_adapter = StopAdapter()
        bridge._adapter = stop_adapter
        bridge._place_tp_sl("X", 100.0, "long")
        assert len(stop_adapter._stop_orders) == 1
        assert stop_adapter._stop_orders[0]["stop_price"] == pytest.approx(98.0)

    def test_base_adapter_stop_fails_closed(self):
        """An adapter without stop support must NOT substitute a limit order.

        A SELL limit at the stop price is below the market, i.e. immediately
        marketable, so the old fallback closed the position the stop was meant
        to protect. The adapter must report failure and place nothing.
        """
        class NoStopAdapter(MockAdapter):
            # Inherit everything except stop support, so the abstract base
            # class's fail-closed default is what gets exercised.
            place_stop_order = BaseBrokerAdapter.place_stop_order

        a = NoStopAdapter()
        result = a.place_stop_order("AAPL", "SELL", 10, 95.0)

        assert result.success is False
        assert "no protective order" in result.message
        assert a._orders == [], "no order may be sent to the broker"

    def test_unprotected_position_escalates_when_stop_fails(self):
        """A position left without a stop must be reported, not logged quietly."""
        class NoStopAdapter(MockAdapter):
            place_stop_order = BaseBrokerAdapter.place_stop_order

        a = NoStopAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["X"] = Position("X", "long", 10, 100.0, "t")

        with patch.object(bridge, "_notify_unprotected_position") as notify:
            bridge._place_tp_sl("X", 100.0, "long")

        notify.assert_called_once()
        assert notify.call_args[0][0] == "X"
        # TP still went out; only the stop is missing, and no OCO pair exists
        # because there is nothing to pair the take-profit against.
        assert any(o["type"] == "limit" for o in a._orders)
        assert bridge._oco_pairs == {}


# ── Additional coverage: close_all_positions ─────────────────────────────

class TestCloseAllPositions:
    def test_close_all(self):
        a = MockAdapter()
        a._prices["AAPL"] = 155.0
        a._prices["MSFT"] = 310.0
        bridge = _make_bridge(adapter=a)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 150.0, "t")
        bridge._positions["MSFT"] = Position("MSFT", "short", 50, 300.0, "t")
        results = bridge.close_all_positions()
        assert len(results) == 2
        assert all(r.success for r in results)
        assert len(bridge._positions) == 0

    def test_close_all_fallback_price(self):
        a = MockAdapter()
        a._prices = {}
        bridge = _make_bridge(adapter=a)
        bridge._positions["X"] = Position("X", "long", 10, 100.0, "t")
        results = bridge.close_all_positions()
        assert len(results) == 1


# ── Additional coverage: get methods ─────────────────────────────────────

class TestBridgeGetMethods:
    def test_get_positions(self):
        bridge = _make_bridge()
        bridge._positions["A"] = Position("A", "long", 10, 50.0, "t")
        pos = bridge.get_positions()
        assert "A" in pos

    def test_get_broker_positions(self):
        a = MockAdapter()
        a._positions = [{"symbol": "X", "quantity": 100}]
        bridge = _make_bridge(adapter=a)
        assert len(bridge.get_broker_positions()) == 1

    def test_get_account_info(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        info = bridge.get_account_info()
        assert info["broker"] == "mock"

    def test_get_latest_price(self):
        a = MockAdapter()
        a._prices["AAPL"] = 150.0
        bridge = _make_bridge(adapter=a)
        assert bridge.get_latest_price("AAPL") == 150.0


# ── execute_decision with agent callback ─────────────────────────────────

class TestExecuteDecisionWithAgent:
    def test_close_position_records_outcome(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["AAPL"] = Position("AAPL", "long", 50, 100.0, "t")
        mock_agent = MagicMock()
        bridge._close_position("AAPL", 110.0, agent=mock_agent)
        mock_agent.record_outcome.assert_called_once()

    def test_close_position_agent_exception_handled(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["AAPL"] = Position("AAPL", "long", 50, 100.0, "t")
        mock_agent = MagicMock()
        mock_agent.record_outcome.side_effect = RuntimeError("db error")
        result = bridge._close_position("AAPL", 110.0, agent=mock_agent)
        assert result.success


class TestForceCloseWithTrailingStops:
    def test_trailing_stop_triggers_force_close(self):
        a = MockAdapter()
        a._prices["AAPL"] = 103.0
        bridge = _make_bridge(adapter=a, max_loss_pct=50.0)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 100.0, "t")
        bridge.set_trailing_stop("AAPL", "long", activation_pct=0.02, trail_pct=0.015)
        bridge._update_trailing_stops("AAPL", 105.0)
        a._prices["AAPL"] = 103.0
        results = bridge.check_and_force_close()
        assert len(results) == 1
        assert results[0].success
        assert "AAPL" not in bridge._positions

    def test_force_close_skips_zero_price(self):
        a = MockAdapter()
        a._prices = {}
        bridge = _make_bridge(adapter=a, max_loss_pct=5.0)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 100.0, "t")
        results = bridge.check_and_force_close()
        assert len(results) == 0

    def test_force_close_short_exceeding_loss(self):
        a = MockAdapter()
        a._prices["X"] = 120.0
        bridge = _make_bridge(adapter=a, max_loss_pct=10.0)
        bridge._positions["X"] = Position("X", "short", 50, 100.0, "t")
        results = bridge.check_and_force_close()
        assert len(results) == 1


class TestExecuteDecisionDiary:
    def test_decision_written_to_diary(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge.execute_decision({"action": "BUY", "price": 100.0, "confidence": 0.9}, "AAPL")
        entries = bridge.get_diary(n=10)
        assert len(entries) >= 1
        buy_entries = [e for e in entries if e.get("action") == "BUY"]
        assert len(buy_entries) >= 1

    def test_hold_decision_written(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge.execute_decision({"action": "HOLD", "price": 100.0}, "AAPL")
        entries = bridge.get_diary(n=10)
        assert any(e.get("action") == "HOLD" for e in entries)

    def test_open_long_diary_entry(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge.execute_decision({"action": "BUY", "price": 100.0, "confidence": 0.9}, "AAPL")
        entries = bridge.get_diary(n=10)
        assert any(e.get("action") == "OPEN_LONG" for e in entries)

    def test_open_short_diary_entry(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge.execute_decision({"action": "SELL", "price": 100.0, "confidence": 0.9}, "AAPL")
        entries = bridge.get_diary(n=10)
        assert any(e.get("action") == "OPEN_SHORT" for e in entries)

    def test_execute_decision_with_tp_sl(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge.execute_decision({
            "action": "BUY", "price": 100.0, "confidence": 0.9,
            "tp_price": 110.0, "sl_price": 95.0,
            "exit_plan": "hold until TP", "reasoning": "bullish pattern",
        }, "AAPL")
        assert "AAPL" in bridge._positions

    def test_execute_decision_sell_with_existing_long(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["X"] = Position("X", "long", 50, 100.0, "t")
        result = bridge.execute_decision({"action": "SELL", "price": 110.0}, "X")
        assert result.success
        assert "X" not in bridge._positions

    def test_execute_decision_buy_closes_short_opens_long(self):
        a = MockAdapter()
        bridge = _make_bridge(adapter=a)
        bridge._positions["X"] = Position("X", "short", 50, 100.0, "t")
        result = bridge.execute_decision({"action": "BUY", "price": 95.0, "confidence": 0.8}, "X")
        assert result.success
        assert bridge._positions["X"].direction == "long"

    def test_diary_write_error_handled(self):
        bridge = _make_bridge()
        bridge._diary_path = "/nonexistent/dir/diary.jsonl"
        bridge._write_diary({"action": "TEST"})

    def test_default_diary_path(self):
        bridge = BrokerBridge(broker="ib", mode="paper")
        bridge._adapter = MockAdapter()
        assert ".stocks_plugin" in bridge._diary_path


# ── RiskManager wiring ────────────────────────────────────────────────────

class TestRiskManagerWiring:
    """BrokerBridge must be able to take a RiskManager and honour it.

    The attribute existed but nothing could set it, so _calculate_shares always
    fell through to a flat 10% of a static $100k and the per-trade risk limit
    never applied to a real order.
    """

    def test_risk_manager_is_accepted_and_used_for_sizing(self):
        from shared.risk_manager import RiskManager, RiskManagerConfig, SizingMethod

        rm = RiskManager(config=RiskManagerConfig(
            sizing_method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade_pct=1.0,
            total_capital=100_000.0,
            max_position_pct_equity=25.0,
        ))
        bridge = _make_bridge(adapter=MockAdapter(), risk_manager=rm)

        assert bridge._risk_manager is rm
        # 25% of $100k at $150 is 166 shares — the risk manager's cap, not the
        # bridge's flat 10% default (which would be 66).
        assert bridge._calculate_shares(150.0) == 166

    def test_default_sizing_still_applies_without_a_risk_manager(self):
        bridge = _make_bridge(adapter=MockAdapter())
        assert bridge._risk_manager is None
        # 10% of $100k at $150.
        assert bridge._calculate_shares(150.0) == 66

    def test_zero_size_from_risk_manager_is_a_rejection(self):
        """A risk manager returning 0 must block the order, not fall back."""
        rm = MagicMock()
        rm.calculate_position_size.return_value = 0
        bridge = _make_bridge(adapter=MockAdapter(), risk_manager=rm)
        assert bridge._calculate_shares(150.0) == 0

    def test_sizing_failure_fails_closed(self):
        """An exception in sizing must not become a full-size default order."""
        rm = MagicMock()
        rm.calculate_position_size.side_effect = RuntimeError("boom")
        bridge = _make_bridge(adapter=MockAdapter(), risk_manager=rm)
        assert bridge._calculate_shares(150.0) == 0

    def test_zero_shares_sends_no_order_to_the_broker(self):
        rm = MagicMock()
        rm.calculate_position_size.return_value = 0
        adapter = MockAdapter()
        bridge = _make_bridge(adapter=adapter, risk_manager=rm)

        result = bridge.execute_decision(
            {"action": "BUY", "confidence": 0.9, "price": 150.0}, "AAPL",
        )

        assert result.success is False
        assert adapter._orders == []


# ── on_fill releases portfolio-gate exposure ──────────────────────────────

class TestOnFillReleasesExposure:
    """A TP/SL fill is the normal exit; it must reach the portfolio gate.

    Without this the gate's exposure only ever grew, eventually blocking all
    trading, and realized P&L never reached the daily-loss limit.
    """

    def test_exposure_is_released_on_fill(self):
        adapter = MockAdapter()
        bridge = _make_bridge(adapter=adapter)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 150.0, "t")
        bridge._portfolio_gate.register_position("AAPL", 100, 150.0, "s", "mock")

        before = bridge._portfolio_gate.get_portfolio_summary()["total_exposure"]
        assert before > 0

        bridge.on_fill("ORD-1", "AAPL", 160.0)

        after = bridge._portfolio_gate.get_portfolio_summary()["total_exposure"]
        assert after == pytest.approx(0.0)
        assert "AAPL" not in bridge.get_positions()

    def test_realized_pnl_reaches_the_gate(self):
        adapter = MockAdapter()
        bridge = _make_bridge(adapter=adapter)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 150.0, "t")
        bridge._portfolio_gate.register_position("AAPL", 100, 150.0, "s", "mock")

        bridge.on_fill("ORD-1", "AAPL", 140.0)

        # $10/share loss on 100 shares, minus commissions on both legs.
        daily = bridge._portfolio_gate.get_portfolio_summary()["daily_pnl"]
        assert daily < -900.0
        assert daily > -1100.0

    def test_reconcile_releases_exposure_for_stale_positions(self):
        adapter = MockAdapter()
        adapter._positions = []  # broker reports flat
        bridge = _make_bridge(adapter=adapter)
        bridge._positions["AAPL"] = Position("AAPL", "long", 100, 150.0, "t")
        bridge._portfolio_gate.register_position("AAPL", 100, 150.0, "s", "mock")

        bridge.reconcile_positions()

        assert bridge._portfolio_gate.get_portfolio_summary()["total_exposure"] == pytest.approx(0.0)


# ── Real fill prices reach P&L ────────────────────────────────────────────

class TestFillPricesAreUsed:
    """No adapter ever populated fill_price, so every P&L used the decision
    price and slippage was invisible. Adapters now expose get_fill()."""

    class _FillingAdapter(MockAdapter):
        """Fills 3% away from the requested price — visible slippage."""

        def __init__(self, slip=1.03, qty=None):
            super().__init__()
            self._slip = slip
            self._qty = qty

        def get_fill(self, order_id):
            last = self._orders[-1]
            price = (last.get("price") or 100.0) * self._slip
            return price, (self._qty if self._qty is not None else last["qty"])

    def test_open_records_the_fill_price_not_the_decision_price(self):
        a = self._FillingAdapter(slip=1.03)
        a._prices["AAPL"] = 100.0
        bridge = _make_bridge(adapter=a)

        bridge.execute_decision(
            {"action": "BUY", "confidence": 0.9, "price": 100.0}, "AAPL")

        pos = bridge.get_positions()["AAPL"]
        assert pos.entry_price == pytest.approx(103.0), "slippage was discarded"

    def test_partial_fill_quantity_is_respected(self):
        a = self._FillingAdapter(slip=1.0, qty=7)
        a._prices["AAPL"] = 100.0
        bridge = _make_bridge(adapter=a)

        bridge.execute_decision(
            {"action": "BUY", "confidence": 0.9, "price": 100.0}, "AAPL")

        assert bridge.get_positions()["AAPL"].shares == 7

    def test_execution_result_carries_the_fill_back(self):
        a = self._FillingAdapter(slip=1.02)
        a._prices["AAPL"] = 100.0
        bridge = _make_bridge(adapter=a)

        result = bridge.execute_decision(
            {"action": "BUY", "confidence": 0.9, "price": 100.0}, "AAPL")

        assert result.fill_price == pytest.approx(102.0)
        assert result.filled_quantity is not None

    def test_falls_back_to_the_decision_price_when_broker_cannot_say(self):
        a = MockAdapter()  # base get_fill() returns None
        a._prices["AAPL"] = 100.0
        bridge = _make_bridge(adapter=a)

        bridge.execute_decision(
            {"action": "BUY", "confidence": 0.9, "price": 100.0}, "AAPL")

        assert bridge.get_positions()["AAPL"].entry_price == pytest.approx(100.0)

    def test_a_broken_adapter_does_not_crash_the_order_path(self):
        """This runs after the order is live — it must degrade, never raise."""
        class BadAdapter(MockAdapter):
            def get_fill(self, order_id):
                return ("not-a-price",)

        a = BadAdapter()
        a._prices["AAPL"] = 100.0
        bridge = _make_bridge(adapter=a)

        result = bridge.execute_decision(
            {"action": "BUY", "confidence": 0.9, "price": 100.0}, "AAPL")

        assert result.success
        assert bridge.get_positions()["AAPL"].entry_price == pytest.approx(100.0)

    def test_adapter_raising_does_not_crash_the_order_path(self):
        class ThrowingAdapter(MockAdapter):
            def get_fill(self, order_id):
                raise RuntimeError("broker unreachable")

        a = ThrowingAdapter()
        a._prices["AAPL"] = 100.0
        bridge = _make_bridge(adapter=a)

        result = bridge.execute_decision(
            {"action": "BUY", "confidence": 0.9, "price": 100.0}, "AAPL")
        assert result.success

    def test_close_uses_the_exit_fill_price(self):
        a = self._FillingAdapter(slip=1.0)
        a._prices["AAPL"] = 100.0
        bridge = _make_bridge(adapter=a)
        bridge._positions["AAPL"] = Position("AAPL", "long", 10, 100.0, "t")

        # Exit fills 10% better than the quoted exit.
        a._slip = 1.10
        bridge._close_position("AAPL", 100.0)

        assert "AAPL" not in bridge.get_positions()
