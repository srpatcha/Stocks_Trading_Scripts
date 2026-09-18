"""Authentication and information-disclosure tests for /status and /health.

/status returns adapter.get_account_info() for every broker — net liquidation,
buying power and the full brokerage account identifier — and had no
authentication of any kind. /health reported daily P&L, equity and drawdown to
anonymous callers.
"""

import hashlib
import hmac
import os
import sys

import pytest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from fastapi.testclient import TestClient
    HAS_TESTCLIENT = True
except ImportError:
    HAS_TESTCLIENT = False

pytestmark = pytest.mark.skipif(not HAS_TESTCLIENT, reason="fastapi not installed")

SECRET = "a-real-secret-not-the-placeholder"

# Anything an unauthenticated caller must never see.
FINANCIAL_KEYS = {
    "daily_pnl", "net_liquidation", "NetLiquidation", "buying_power",
    "BuyingPower", "equity", "cash_balance", "account_id", "drawdown",
    "brokers", "risk",
}


def _config(hmac_secret="", allowed_ips=None):
    return {
        "server": {"host": "127.0.0.1", "port": 5000, "debug": False,
                   "title": "Test", "version": "test"},
        "security": {
            "hmac_secret": hmac_secret,
            "hmac_algorithm": "sha256",
            "allowed_ips": allowed_ips if allowed_ips is not None else [],
            "require_passphrase": False,
            "passphrase": "",
            "require_hmac": False,
            "cors_origins": [],
        },
        "rate_limiting": {"enabled": False, "max_requests_per_minute": 9999,
                          "window_seconds": 60},
        "broker_routing": {"default_broker": "interactive_brokers", "routes": []},
        "logging": {"level": "WARNING"},
    }


def _make_client(config):
    from tradingview.webhooks.webhook_server import create_app
    with patch("tradingview.webhooks.webhook_server.load_config", return_value=config):
        app = create_app()
    return TestClient(app), app


def _sign(path):
    return hmac.new(SECRET.encode(), path.encode(), hashlib.sha256).hexdigest()


class TestHealthCarriesNoFinancials:
    """/health stays open for liveness probes, so it must be boring."""

    def test_health_is_reachable_without_auth(self):
        client, _ = _make_client(_config())
        assert client.get("/health").status_code == 200

    def test_health_reports_liveness(self):
        client, _ = _make_client(_config())
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "uptime" in body
        assert "version" in body

    def test_health_leaks_no_financial_fields(self):
        client, app = _make_client(_config())
        app.state.pnl_tracker.record_trade("AAPL", -4321.0)

        body = client.get("/health").json()

        leaked = FINANCIAL_KEYS & set(body)
        assert not leaked, f"/health disclosed {leaked}"
        assert "4321" not in client.get("/health").text


class TestStatusRequiresAuth:
    def test_status_open_when_no_secret_is_configured(self):
        """With no secret and no allowlist there is nothing to check against."""
        client, _ = _make_client(_config())
        assert client.get("/status").status_code == 200

    def test_status_rejects_unsigned_request(self):
        client, _ = _make_client(_config(hmac_secret=SECRET))
        resp = client.get("/status")
        assert resp.status_code == 401
        assert "account" not in resp.text.lower()

    def test_status_rejects_a_wrong_signature(self):
        client, _ = _make_client(_config(hmac_secret=SECRET))
        resp = client.get("/status", headers={"X-Webhook-Signature": "deadbeef"})
        assert resp.status_code == 401

    def test_status_accepts_a_valid_signature(self):
        client, _ = _make_client(_config(hmac_secret=SECRET))
        resp = client.get(
            "/status", headers={"X-Webhook-Signature": _sign("/status")},
        )
        assert resp.status_code == 200
        assert resp.json()["server"] == "running"

    def test_placeholder_secret_does_not_enable_the_gate(self):
        """The shipped placeholder must behave like 'unconfigured', as /webhook does."""
        client, _ = _make_client(
            _config(hmac_secret="CHANGE_ME_TO_A_SECURE_SECRET_KEY"))
        assert client.get("/status").status_code == 200

    def test_status_enforces_the_ip_allowlist(self):
        client, _ = _make_client(_config(allowed_ips=["10.0.0.1"]))
        assert client.get("/status").status_code == 403

    def test_signature_from_another_path_is_rejected(self):
        """Signing over the path stops a captured signature being reused."""
        client, _ = _make_client(_config(hmac_secret=SECRET))
        resp = client.get(
            "/status", headers={"X-Webhook-Signature": _sign("/health")},
        )
        assert resp.status_code == 401


class TestStatusDoesNotLeakBrokerErrors:
    def test_broker_exception_text_is_not_returned(self):
        client, app = _make_client(_config())
        broken = MagicMock()
        broken.get_account_info.side_effect = RuntimeError(
            "API error 401: {'access_token': 'super-secret-token'}"
        )
        app.state.broker_router.get_all_brokers = lambda: {"ib": broken}

        resp = client.get("/status")

        assert resp.status_code == 200
        assert "super-secret-token" not in resp.text
        assert resp.json()["brokers"]["ib"] == {"error": "unavailable"}
