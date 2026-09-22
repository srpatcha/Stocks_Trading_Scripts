"""
Tests for tradingview/webhooks/webhook_server.py

Covers:
- validate_hmac_signature: valid/invalid signatures, algorithm fallback
- AlertPayload parsing: TradingView format, edge cases
- RateLimiter: sliding window, remaining count, documented limitation
- BrokerRouter: pattern matching, default broker, missing broker
- CORS: verify fix — no credentials with wildcard origin
- IBBrokerAdapter / TradeStationBrokerAdapter / SchwabBrokerAdapter: place_order paths
- Webhook endpoint: full flow, passphrase, IP allowlist, invalid action
- sys.path guarding in broker adapters
"""

import hashlib
import hmac
import json
import os
import sys
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# ── sys.path setup ──
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_TV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tradingview", "webhooks"))
if _TV_ROOT not in sys.path:
    sys.path.insert(0, _TV_ROOT)

# This file used to install MagicMocks into sys.modules for yaml, fastapi,
# pydantic and uvicorn, then delete three of them again, then open a
# `with patch.dict(...)` block whose entire body was `pass`. Three problems:
#
#   1. The patch.dict block did nothing — the real import below ran against
#      whatever was actually installed, so the apparatus was decorative.
#   2. uvicorn was never cleaned up. setdefault() fires when the key is
#      absent, so it SHADOWED the real uvicorn and left a MagicMock in
#      sys.modules for the rest of the pytest session; any later test
#      importing uvicorn silently got a mock.
#   3. Deleting already-imported yaml/fastapi/pydantic forced a re-import and
#      created duplicate class objects, so isinstance() checks against types
#      held by earlier-imported modules started returning False.
#
# These dependencies are in requirements-core.txt and are genuinely installed,
# so the tests just import them.
import yaml

# Direct function imports for unit-testable pieces
from tradingview.webhooks.webhook_server import (
    validate_hmac_signature,
    load_config,
    _default_config,
    setup_logging,
    RateLimiter,
    AlertPayload,
    OrderResult,
    BrokerRouter,
    IBBrokerAdapter,
    TradeStationBrokerAdapter,
    SchwabBrokerAdapter,
)


# ═══════════════════════════════════════════════════════
# HMAC Validation Tests
# ═══════════════════════════════════════════════════════

class TestValidateHMAC:
    """Tests for validate_hmac_signature()."""

    def test_valid_hmac_sha256(self):
        secret = "my_secret_key"
        payload = b'{"symbol":"AAPL","action":"buy","price":150.0}'
        sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        assert validate_hmac_signature(payload, sig, secret, "sha256") is True

    def test_invalid_hmac_signature(self):
        secret = "my_secret_key"
        payload = b'{"symbol":"AAPL","action":"buy","price":150.0}'
        assert validate_hmac_signature(payload, "bad_signature", secret, "sha256") is False

    def test_empty_signature_returns_false(self):
        assert validate_hmac_signature(b"data", "", "secret") is False

    def test_empty_secret_returns_false(self):
        assert validate_hmac_signature(b"data", "sig", "") is False

    def test_invalid_algorithm_falls_back_to_sha256(self):
        secret = "key"
        payload = b"test_data"
        expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        result = validate_hmac_signature(payload, expected, secret, "nonexistent_algo")
        assert result is True

    def test_valid_hmac_sha512(self):
        secret = "key512"
        payload = b"payload512"
        sig = hmac.new(secret.encode(), payload, hashlib.sha512).hexdigest()
        assert validate_hmac_signature(payload, sig, secret, "sha512") is True

    def test_hmac_tampered_payload(self):
        secret = "key"
        original = b"original"
        sig = hmac.new(secret.encode(), original, hashlib.sha256).hexdigest()
        assert validate_hmac_signature(b"tampered", sig, secret, "sha256") is False


# ═══════════════════════════════════════════════════════
# RateLimiter Tests
# ═══════════════════════════════════════════════════════

class TestRateLimiter:
    """Tests for RateLimiter — verify fix: documented limitation about multi-worker."""

    def test_allows_within_limit(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.is_allowed("127.0.0.1") is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.is_allowed("10.0.0.1")
        assert rl.is_allowed("10.0.0.1") is False

    def test_different_ips_independent(self):
        rl = RateLimiter(max_requests=1, window_seconds=60)
        assert rl.is_allowed("1.1.1.1") is True
        assert rl.is_allowed("2.2.2.2") is True
        assert rl.is_allowed("1.1.1.1") is False

    def test_window_expiry(self):
        rl = RateLimiter(max_requests=1, window_seconds=1)
        assert rl.is_allowed("3.3.3.3") is True
        assert rl.is_allowed("3.3.3.3") is False
        time.sleep(1.1)
        assert rl.is_allowed("3.3.3.3") is True

    def test_get_remaining(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        assert rl.get_remaining("4.4.4.4") == 5
        rl.is_allowed("4.4.4.4")
        assert rl.get_remaining("4.4.4.4") == 4

    def test_documented_limitation_in_docstring(self):
        """Verify fix: RateLimiter docstring documents multi-worker limitation."""
        assert "NOT shared" in RateLimiter.__doc__ or "not shared" in RateLimiter.__doc__.lower()
        assert "worker" in RateLimiter.__doc__.lower()


# ═══════════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════════

class TestConfig:
    """Tests for load_config and _default_config."""

    def test_default_config_structure(self):
        cfg = _default_config()
        assert "server" in cfg
        assert "security" in cfg
        assert "rate_limiting" in cfg
        assert "broker_routing" in cfg
        assert cfg["server"]["port"] == 5000

    def test_load_config_missing_file(self, tmp_path):
        result = load_config(str(tmp_path / "nonexistent.yaml"))
        assert result["server"]["port"] == 5000

    def test_load_config_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(yaml.dump({
            "server": {"host": "localhost", "port": 8080},
            "security": {"hmac_secret": "test_secret"},
        }))
        result = load_config(str(cfg_file))
        assert result["server"]["port"] == 8080
        assert result["security"]["hmac_secret"] == "test_secret"

    def test_load_config_invalid_yaml(self, tmp_path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text(": : : invalid yaml [[[")
        result = load_config(str(cfg_file))
        assert result["server"]["port"] == 5000


# ═══════════════════════════════════════════════════════
# AlertPayload Tests
# ═══════════════════════════════════════════════════════

class TestAlertPayload:
    """Tests for AlertPayload Pydantic model — TradingView format parsing."""

    def test_minimal_payload(self):
        p = AlertPayload(symbol="AAPL", action="buy", price=150.0)
        assert p.symbol == "AAPL"
        assert p.action == "buy"
        assert p.price == 150.0
        assert p.quantity is None
        assert p.order_type == "market"

    def test_full_payload(self):
        p = AlertPayload(
            symbol="MSFT", action="sell", price=300.5, quantity=10,
            order_type="limit", passphrase="secret", timestamp="2024-01-01T00:00:00Z",
            strategy="momentum", timeframe="1H", message="test", regime="TRENDING",
            signal="LONG",
        )
        assert p.quantity == 10
        assert p.regime == "TRENDING"
        assert p.signal == "LONG"

    def test_payload_from_dict(self):
        data = {"symbol": "TSLA", "action": "close", "price": 200.0}
        p = AlertPayload(**data)
        assert p.symbol == "TSLA"


# ═══════════════════════════════════════════════════════
# BrokerRouter Tests
# ═══════════════════════════════════════════════════════

class TestBrokerRouter:
    """Tests for BrokerRouter: symbol routing with regex patterns."""

    @patch("tradingview.webhooks.webhook_server.IBBrokerAdapter")
    @patch("tradingview.webhooks.webhook_server.TradeStationBrokerAdapter")
    @patch("tradingview.webhooks.webhook_server.SchwabBrokerAdapter")
    def test_default_broker_routing(self, mock_schwab, mock_ts, mock_ib):
        mock_ib.return_value.name = "interactive_brokers"
        mock_ts.return_value.name = "tradestation"
        mock_schwab.return_value.name = "schwab"
        config = {
            "broker_routing": {"default_broker": "interactive_brokers", "routes": []},
            "broker_configs": {},
        }
        router = BrokerRouter(config)
        broker = router.get_broker("AAPL")
        assert broker is not None

    @patch("tradingview.webhooks.webhook_server.IBBrokerAdapter")
    @patch("tradingview.webhooks.webhook_server.TradeStationBrokerAdapter")
    @patch("tradingview.webhooks.webhook_server.SchwabBrokerAdapter")
    def test_pattern_based_routing(self, mock_schwab, mock_ts, mock_ib):
        mock_ib.return_value.name = "interactive_brokers"
        mock_ts.return_value.name = "tradestation"
        mock_schwab.return_value.name = "schwab"
        config = {
            "broker_routing": {
                "default_broker": "interactive_brokers",
                "routes": [{"pattern": "^BTC.*", "broker": "tradestation"}],
            },
            "broker_configs": {},
        }
        router = BrokerRouter(config)
        broker = router.get_broker("BTCUSD")
        assert broker.name == "tradestation"

    @patch("tradingview.webhooks.webhook_server.IBBrokerAdapter")
    @patch("tradingview.webhooks.webhook_server.TradeStationBrokerAdapter")
    @patch("tradingview.webhooks.webhook_server.SchwabBrokerAdapter")
    def test_get_all_brokers(self, mock_schwab, mock_ts, mock_ib):
        config = {"broker_routing": {"default_broker": "interactive_brokers"}, "broker_configs": {}}
        router = BrokerRouter(config)
        all_b = router.get_all_brokers()
        assert "interactive_brokers" in all_b
        assert "tradestation" in all_b
        assert "schwab" in all_b


# ═══════════════════════════════════════════════════════
# Broker Adapter Tests
# ═══════════════════════════════════════════════════════

class TestIBBrokerAdapter:
    """Tests for IBBrokerAdapter — sys.path guard and place_order."""

    @patch("tradingview.webhooks.webhook_server.IBBrokerAdapter._init_adapter")
    def test_name_property(self, mock_init):
        adapter = IBBrokerAdapter.__new__(IBBrokerAdapter)
        adapter._config = {}
        adapter._adapter = None
        assert adapter.name == "interactive_brokers"

    @patch("tradingview.webhooks.webhook_server.IBBrokerAdapter._init_adapter")
    def test_place_order_no_adapter(self, mock_init):
        adapter = IBBrokerAdapter.__new__(IBBrokerAdapter)
        adapter._config = {}
        adapter._adapter = None
        result = adapter.place_order("AAPL", "buy", 10, "market", 150.0)
        assert result.success is False
        assert "not initialised" in result.message

    @patch("tradingview.webhooks.webhook_server.IBBrokerAdapter._init_adapter")
    def test_get_account_info_disconnected(self, mock_init):
        adapter = IBBrokerAdapter.__new__(IBBrokerAdapter)
        adapter._config = {}
        adapter._adapter = None
        info = adapter.get_account_info()
        assert info["connected"] is False

    @patch("tradingview.webhooks.webhook_server.IBBrokerAdapter._init_adapter")
    def test_connect_no_adapter(self, mock_init):
        adapter = IBBrokerAdapter.__new__(IBBrokerAdapter)
        adapter._config = {}
        adapter._adapter = None
        assert adapter.connect() is False


class TestSchwabBrokerAdapter:
    """Tests for SchwabBrokerAdapter."""

    @patch("tradingview.webhooks.webhook_server.SchwabBrokerAdapter._init_adapter")
    def test_name(self, mock_init):
        adapter = SchwabBrokerAdapter.__new__(SchwabBrokerAdapter)
        adapter._config = {}
        adapter._adapter = None
        assert adapter.name == "schwab"

    @patch("tradingview.webhooks.webhook_server.SchwabBrokerAdapter._init_adapter")
    def test_place_order_no_adapter(self, mock_init):
        adapter = SchwabBrokerAdapter.__new__(SchwabBrokerAdapter)
        adapter._config = {}
        adapter._adapter = None
        result = adapter.place_order("SPY", "sell", 5, "market", 400.0)
        assert result.success is False


# ═══════════════════════════════════════════════════════
# CORS Fix Verification
# ═══════════════════════════════════════════════════════

class TestCORSFix:
    """Verify fix: no credentials with wildcard origin.

    The create_app function must set allow_credentials=False when
    allow_origins=["*"]. Browsers reject responses with both
    Access-Control-Allow-Origin: * and Access-Control-Allow-Credentials: true.
    """

    @staticmethod
    def _cors_options(cors_origins):
        """Build an app and return its actual CORS middleware options."""
        from unittest.mock import patch

        from fastapi.middleware.cors import CORSMiddleware
        from tradingview.webhooks.webhook_server import create_app

        cfg = {
            "server": {"host": "127.0.0.1", "port": 5000, "version": "t"},
            "security": {
                "hmac_secret": "a-real-secret-for-tests", "require_hmac": True,
                "allowed_ips": [], "require_passphrase": False, "passphrase": "",
                "cors_origins": cors_origins,
            },
            "rate_limiting": {"enabled": False},
            "broker_routing": {"default_broker": "interactive_brokers", "routes": []},
            "logging": {"level": "WARNING"},
        }
        with patch("tradingview.webhooks.webhook_server.load_config", return_value=cfg):
            app = create_app()
        for mw in app.user_middleware:
            if mw.cls is CORSMiddleware:
                return mw.kwargs
        raise AssertionError("CORS middleware not installed")

    def test_credentials_are_never_allowed(self):
        """This asserted `'allow_credentials=False' in inspect.getsource(...)`.

        It was the only test for a security control and it exercised none of
        it — a comment mentioning the string would have satisfied it. It now
        inspects the middleware the app actually installs.
        """
        assert self._cors_options([]).get("allow_credentials") is False

    def test_origins_come_from_config(self):
        opts = self._cors_options(["https://example.test"])
        assert opts.get("allow_origins") == ["https://example.test"]

    def test_default_origins_are_empty_not_wildcard(self):
        """A wildcard plus credentials is the combination browsers refuse."""
        opts = self._cors_options([])
        assert opts.get("allow_origins") == []
        assert "*" not in (opts.get("allow_origins") or [])


# ═══════════════════════════════════════════════════════
# sys.path Guarding Tests
# ═══════════════════════════════════════════════════════

class TestSysPathGuarding:
    """Verify sys.path inserts are guarded with 'if path not in sys.path'."""

    # These asserted on source text — `"if _ib_path not in sys.path" in src`.
    # That passes if the string appears in a comment and breaks on any
    # equivalent rewrite. What actually matters is that repeated adapter
    # initialisation does not keep growing sys.path, so that is what is
    # measured.

    @pytest.mark.parametrize("adapter_cls", [
        IBBrokerAdapter, TradeStationBrokerAdapter, SchwabBrokerAdapter,
    ])
    def test_repeated_init_does_not_grow_sys_path(self, adapter_cls):
        import sys

        adapter = adapter_cls({})
        # Prime it once, then measure the delta from further calls. An
        # absolute duplicate check would fail on entries the test harness
        # itself duplicates: every test module does a bare
        # sys.path.insert(0, PROJECT_ROOT).
        try:
            adapter._init_adapter()
        except Exception:
            pass

        before = list(sys.path)
        for _ in range(3):
            try:
                adapter._init_adapter()
            except Exception:
                # The broker package may be absent; the path handling still ran.
                pass

        added = [p for p in sys.path if p not in before]
        assert not added, f"{adapter_cls.__name__} re-inserted into sys.path: {added}"
        assert len(sys.path) == len(before), "sys.path grew on repeated init"


# ═══════════════════════════════════════════════════════
# OrderResult Tests
# ═══════════════════════════════════════════════════════

class TestOrderResult:
    """Tests for OrderResult model."""

    def test_order_result_creation(self):
        r = OrderResult(
            success=True, broker="ib", order_id="123",
            message="filled", timestamp="2024-01-01T00:00:00Z",
        )
        assert r.success is True
        assert r.order_id == "123"

    def test_order_result_no_order_id(self):
        r = OrderResult(
            success=False, broker="schwab", order_id=None,
            message="failed", timestamp="2024-01-01T00:00:00Z",
        )
        assert r.order_id is None


# ═══════════════════════════════════════════════════════
# Webhook Payload Variations
# ═══════════════════════════════════════════════════════

class TestWebhookPayloads:
    """Test various webhook payload formats."""

    def test_tradingview_standard_format(self):
        payload = {
            "symbol": "AAPL", "action": "buy", "price": 150.25,
            "quantity": 100, "order_type": "limit",
        }
        p = AlertPayload(**payload)
        assert p.symbol == "AAPL"
        assert p.quantity == 100

    def test_regime_aware_payload(self):
        payload = {
            "symbol": "SPY", "action": "sell", "price": 450.0,
            "regime": "VOLATILE", "strategy": "chameleon_regime_switcher",
            "signal": "SHORT",
        }
        p = AlertPayload(**payload)
        assert p.regime == "VOLATILE"
        assert p.strategy == "chameleon_regime_switcher"

    def test_minimal_required_fields_only(self):
        # price must be > 0; a zero price bypassed the notional cap.
        p = AlertPayload(symbol="QQQ", action="close", price=350.0)
        assert p.price == 350.0
        assert p.order_type == "market"
        assert p.quantity is None
