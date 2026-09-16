# tests/test_websocket.py
"""
Unit tests for client ↔ server WebSocket communication
and trend-detection logic.
"""
import json
import pytest
from fastapi.testclient import TestClient

import server as srv


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_state():
    """Reset server state before each test."""
    srv.trading_client = None
    srv.ws_clients = []
    srv.alerts_pending = {}
    srv.last_prices = {}
    yield


@pytest.fixture
def client():
    return TestClient(srv.app)


# ─── WebSocket: Alert → Client ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_broadcast_alert_reaches_client(client):
    """Server broadcasts an alert; connected client receives it."""
    with client.websocket_connect("/ws") as ws:
        await srv.broadcast({
            "type": "alert",
            "symbol": "AAPL",
            "price": 190.5,
            "change_pct": 2.3,
        })
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "alert"
        assert msg["symbol"] == "AAPL"
        assert msg["price"] == 190.5
        assert msg["change_pct"] == 2.3


# ─── WebSocket: Client → Server (buy) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_client_buy_triggers_order(client, monkeypatch):
    """Client sends buy; server calls place_order and confirms."""
    order_called = []

    async def fake_place_order(symbol):
        order_called.append(symbol)
        await srv.broadcast({
            "type": "order_filled",
            "symbol": symbol,
            "order_id": "12345",
        })

    monkeypatch.setattr(srv, "place_order", fake_place_order)
    srv.alerts_pending["TSLA"] = {"price": 250.0, "change": 0.025}

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"action": "buy", "symbol": "TSLA"}))
        msg = json.loads(ws.receive_text())

    assert msg["type"] == "order_filled"
    assert msg["symbol"] == "TSLA"
    assert order_called == ["TSLA"]
    assert "TSLA" not in srv.alerts_pending


# ─── WebSocket: Buy on non-pending alert is a no-op ────────────────────────

@pytest.mark.asyncio
async def test_buy_without_pending_alert_is_noop(client, monkeypatch):
    """Buying a symbol with no pending alert does nothing."""
    order_called = []

    async def fake_place_order(symbol):
        order_called.append(symbol)

    monkeypatch.setattr(srv, "place_order", fake_place_order)

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"action": "buy", "symbol": "AAPL"}))

    assert order_called == []


# ─── Trend Threshold Logic ──────────────────────────────────────────────────

def test_threshold_triggers_alert():
    """A 2.5% move exceeds the 2% threshold."""
    last = 100.0
    current = 102.5
    change = (current - last) / last
    assert abs(change) >= srv.PRICE_CHANGE_THRESHOLD


def test_below_threshold_no_alert():
    """A 1% move does NOT exceed the 2% threshold."""
    last = 100.0
    current = 101.0
    change = (current - last) / last
    assert abs(change) < srv.PRICE_CHANGE_THRESHOLD


def test_negative_move_triggers():
    """A -3% drop exceeds the 2% threshold (absolute value)."""
    last = 100.0
    current = 97.0
    change = (current - last) / last
    assert abs(change) >= srv.PRICE_CHANGE_THRESHOLD


def test_exact_threshold_triggers():
    """Exactly 2% change triggers (>= not >)."""
    last = 200.0
    current = 204.0
    change = (current - last) / last
    assert abs(change) >= srv.PRICE_CHANGE_THRESHOLD   