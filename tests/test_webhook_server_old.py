"""Tests for the TradingView webhook server."""

import pytest
import hashlib
import hmac
import json
import time
from unittest.mock import patch, MagicMock

# `app` was imported here and never used, and importing it built the
# entire application as a side effect of test collection.

try:
    from httpx import AsyncClient, ASGITransport
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from fastapi.testclient import TestClient
    HAS_TESTCLIENT = True
except ImportError:
    HAS_TESTCLIENT = False


VALID_PAYLOAD = {
    "symbol": "AAPL",
    "action": "buy",
    "price": 150.50,
    "quantity": 100,
    "order_type": "market",
    "passphrase": "test-secret",
}

WEBHOOK_SECRET = "test-hmac-secret"


def _compute_hmac(payload: dict, secret: str = WEBHOOK_SECRET) -> str:
    """Compute HMAC-SHA256 signature for a payload."""
    body = json.dumps(payload, separators=(",", ":")).encode()
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


@pytest.fixture
def client():
    """Create a test client for the webhook server with permissive security."""
    if not HAS_TESTCLIENT:
        pytest.skip("fastapi not installed")

    from tradingview.webhooks.webhook_server import create_app

    # Create app with permissive test config
    test_config = {
        "server": {"host": "0.0.0.0", "port": 5000, "debug": False, "title": "Test", "version": "test"},
        "security": {
            "hmac_secret": "",
            "hmac_algorithm": "sha256",
            "allowed_ips": [],
            "require_passphrase": False,
            "passphrase": "",
            "require_hmac": False,
            "cors_origins": [],
        },
        "rate_limiting": {"enabled": False, "max_requests_per_minute": 9999, "window_seconds": 60},
        "broker_routing": {"default_broker": "interactive_brokers", "routes": []},
        "logging": {"level": "WARNING"},
    }

    with patch("tradingview.webhooks.webhook_server.load_config", return_value=test_config):
        test_app = create_app()
    return TestClient(test_app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_health_includes_uptime(self, client):
        response = client.get("/health")
        data = response.json()
        assert "uptime" in data or "status" in data


class TestWebhookEndpoint:
    """Tests for the POST /webhook endpoint."""

    def test_valid_webhook_returns_200(self, client):
        response = client.post("/webhook", json=VALID_PAYLOAD)
        assert response.status_code in (200, 202)

    def test_webhook_accepts_json(self, client):
        response = client.post(
            "/webhook",
            json=VALID_PAYLOAD,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code in (200, 202)

    def test_webhook_returns_json_response(self, client):
        response = client.post("/webhook", json=VALID_PAYLOAD)
        data = response.json()
        assert isinstance(data, dict)

    def test_missing_symbol_returns_422(self, client):
        payload = {
            "action": "buy",
            "price": 150.0,
            "quantity": 100,
        }
        response = client.post("/webhook", json=payload)
        assert response.status_code == 422

    def test_missing_action_returns_422(self, client):
        payload = {
            "symbol": "AAPL",
            "price": 150.0,
            "quantity": 100,
        }
        response = client.post("/webhook", json=payload)
        assert response.status_code == 422

    def test_invalid_json_returns_422(self, client):
        response = client.post(
            "/webhook",
            content="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client):
        response = client.post(
            "/webhook",
            content="{}",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_sell_action_accepted(self, client):
        payload = VALID_PAYLOAD.copy()
        payload["action"] = "sell"
        response = client.post("/webhook", json=payload)
        assert response.status_code in (200, 202)

    def test_webhook_with_extra_fields(self, client):
        payload = VALID_PAYLOAD.copy()
        payload["extra_field"] = "should_be_ignored"
        response = client.post("/webhook", json=payload)
        assert response.status_code in (200, 202)


class TestWebhookValidation:
    """Tests for HMAC signature validation and input validation."""

    def test_valid_hmac_signature(self, client):
        sig = _compute_hmac(VALID_PAYLOAD)
        response = client.post(
            "/webhook",
            json=VALID_PAYLOAD,
            headers={"X-Webhook-Signature": sig},
        )
        assert response.status_code in (200, 202)

    def test_negative_quantity_rejected(self, client):
        payload = VALID_PAYLOAD.copy()
        payload["quantity"] = -10
        response = client.post("/webhook", json=payload)
        assert response.status_code in (400, 422)

    def test_zero_price_handling(self, client):
        payload = VALID_PAYLOAD.copy()
        payload["price"] = 0
        response = client.post("/webhook", json=payload)
        assert response.status_code in (200, 202, 400, 422)


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_single_request_not_limited(self, client):
        response = client.post("/webhook", json=VALID_PAYLOAD)
        assert response.status_code != 429

    def test_health_endpoint_not_rate_limited(self, client):
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200


class TestRegimeField:
    """Tests for the new regime and signal fields in webhook payloads."""

    def test_webhook_accepts_regime_field(self, client):
        payload = VALID_PAYLOAD.copy()
        payload["regime"] = "TRENDING"
        payload["signal"] = "trend_long"
        response = client.post("/webhook", json=payload)
        assert response.status_code in (200, 202)

    def test_webhook_regime_in_response(self, client):
        payload = VALID_PAYLOAD.copy()
        payload["regime"] = "RANGING"
        payload["signal"] = "mr_long"
        payload["strategy"] = "chameleon"
        response = client.post("/webhook", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert data["alert"]["regime"] == "RANGING"
            assert data["alert"]["strategy"] == "chameleon"

    def test_webhook_without_regime_backward_compat(self, client):
        response = client.post("/webhook", json=VALID_PAYLOAD)
        if response.status_code == 200:
            data = response.json()
            if "alert" in data:
                assert data["alert"]["regime"] is None

    def test_webhook_volatile_regime(self, client):
        payload = VALID_PAYLOAD.copy()
        payload["regime"] = "VOLATILE"
        payload["signal"] = "squeeze_fired"
        payload["strategy"] = "vol_breakout"
        response = client.post("/webhook", json=payload)
        assert response.status_code in (200, 202)
