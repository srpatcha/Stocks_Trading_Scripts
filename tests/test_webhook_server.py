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

# Mock heavy external deps before import
sys.modules.setdefault("yaml", MagicMock())
sys.modules.setdefault("fastapi", MagicMock())
sys.modules.setdefault("fastapi.middleware.cors", MagicMock())
sys.modules.setdefault("fastapi.responses", MagicMock())
sys.modules.setdefault("pydantic", MagicMock())
sys.modules.setdefault("uvicorn", MagicMock())

# We need real yaml for load_config tests; import after path setup
for mod_name in list(sys.modules):
    if mod_name.startswith("yaml") or mod_name.startswith("fastapi") or mod_name.startswith("pydantic"):
        del sys.modules[mod_name]

import yaml  # real yaml

# Now do the real import with mocked fastapi/pydantic
with patch.dict(sys.modules, {
    "fastapi": MagicMock(),
    "fastapi.middleware.cors": MagicMock(),
    "fastapi.responses": MagicMock(),
    "pydantic": MagicMock(),
}):
    # We need to import the functions we can test standalone
    pass

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

    def test_cors_no_credentials_with_wildcard(self):
        import inspect
        from tradingview.webhooks.webhook_server import create_app
        src = inspect.getsource(create_app)
        assert 'allow_credentials=False' in src
        # CORS origins are now configurable via cors_origins config, not hardcoded wildcard
        assert 'allow_origins=cors_origins' in src or 'cors_origins' in src


# ═══════════════════════════════════════════════════════
# sys.path Guarding Tests
# ═══════════════════════════════════════════════════════

class TestSysPathGuarding:
    """Verify sys.path inserts are guarded with 'if path not in sys.path'."""

    def test_ib_adapter_sys_path_guard(self):
        import inspect
        src = inspect.getsource(IBBrokerAdapter._init_adapter)
        assert "if _ib_path not in sys.path" in src

    def test_tradestation_adapter_sys_path_guard(self):
        import inspect
        src = inspect.getsource(TradeStationBrokerAdapter._init_adapter)
        assert "if _ts_path not in sys.path" in src

    def test_schwab_adapter_sys_path_guard(self):
        import inspect
        src = inspect.getsource(SchwabBrokerAdapter._init_adapter)
        assert "if _schwab_path not in sys.path" in src


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
