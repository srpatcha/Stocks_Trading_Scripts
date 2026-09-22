"""Client identity, allowlist ranges, body limits and startup guards.

Three defects motivated these:

- The IP allowlist and the rate limiter both keyed on request.client.host, the
  immediate peer. Behind the reverse proxy this server needs anyway (plaintext
  HTTP, passphrase in the body) that is the PROXY — so the allowlist admitted
  everyone or no one, and all clients shared one rate-limit bucket.
- The allowlist was exact string matching, so a CIDR could not be expressed at
  all, while TradingView publishes ranges.
- `await request.body()` buffered the whole payload before the HMAC check,
  i.e. while the caller was still unauthenticated.
"""

import os
import sys

import pytest
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tradingview.webhooks.webhook_server import (
    MAX_BODY_BYTES,
    client_ip_of,
    ip_allowed,
)


def _request(peer, headers=None):
    r = MagicMock()
    r.client.host = peer
    r.headers = headers or {}
    return r


class TestIpAllowlist:
    def test_empty_allowlist_permits_everything(self):
        assert ip_allowed("203.0.113.9", []) is True

    def test_exact_address_still_works(self):
        assert ip_allowed("127.0.0.1", ["127.0.0.1"]) is True
        assert ip_allowed("127.0.0.2", ["127.0.0.1"]) is False

    def test_cidr_range_is_honoured(self):
        """Exact string matching could not express a range at all."""
        assert ip_allowed("52.89.214.200", ["52.89.0.0/16"]) is True
        assert ip_allowed("53.89.214.200", ["52.89.0.0/16"]) is False

    def test_ipv6_is_supported(self):
        assert ip_allowed("::1", ["::1"]) is True
        assert ip_allowed("2001:db8::5", ["2001:db8::/32"]) is True

    def test_malformed_entry_is_ignored_not_fatal(self):
        assert ip_allowed("127.0.0.1", ["not-an-ip", "127.0.0.1"]) is True

    def test_unparseable_client_is_refused_when_a_list_exists(self):
        assert ip_allowed("unknown", ["127.0.0.1"]) is False


class TestClientIdentity:
    def test_peer_is_used_when_no_proxy_is_trusted(self):
        cfg = {"security": {"trusted_proxies": []}}
        req = _request("10.1.2.3", {"X-Forwarded-For": "203.0.113.7"})
        assert client_ip_of(req, cfg) == "10.1.2.3"

    def test_forwarded_header_from_an_untrusted_peer_is_ignored(self):
        """Otherwise anyone could spoof past the allowlist with a header."""
        cfg = {"security": {"trusted_proxies": ["10.0.0.0/8"]}}
        req = _request("203.0.113.1", {"X-Forwarded-For": "127.0.0.1"})
        assert client_ip_of(req, cfg) == "203.0.113.1"

    def test_forwarded_header_from_a_trusted_proxy_is_used(self):
        cfg = {"security": {"trusted_proxies": ["10.0.0.0/8"]}}
        req = _request("10.1.2.3", {"X-Forwarded-For": "203.0.113.7"})
        assert client_ip_of(req, cfg) == "203.0.113.7"

    def test_leftmost_entry_is_the_original_client(self):
        cfg = {"security": {"trusted_proxies": ["10.0.0.0/8"]}}
        req = _request("10.1.2.3",
                       {"X-Forwarded-For": "203.0.113.7, 10.9.9.9, 10.1.2.3"})
        assert client_ip_of(req, cfg) == "203.0.113.7"

    def test_malformed_forwarded_value_falls_back_to_the_peer(self):
        cfg = {"security": {"trusted_proxies": ["10.0.0.0/8"]}}
        req = _request("10.1.2.3", {"X-Forwarded-For": "garbage"})
        assert client_ip_of(req, cfg) == "10.1.2.3"

    def test_missing_client_does_not_raise(self):
        cfg = {"security": {"trusted_proxies": []}}
        r = MagicMock()
        r.client = None
        r.headers = {}
        assert client_ip_of(r, cfg) == "unknown"


class TestBodyLimit:
    def test_limit_is_small_enough_to_matter(self):
        assert MAX_BODY_BYTES <= 1024 * 1024

    def test_a_normal_alert_fits_comfortably(self):
        import json
        payload = json.dumps({
            "symbol": "AAPL", "action": "buy", "price": 150.0,
            "quantity": 100, "passphrase": "x" * 64,
            "strategy": "trend_following", "timestamp": "2026-01-01T00:00:00Z",
        })
        assert len(payload.encode()) < MAX_BODY_BYTES


class TestStartupGuards:
    def _cfg(self, **security):
        base = {
            "server": {"host": "127.0.0.1", "port": 5000, "version": "t"},
            "security": {
                "hmac_secret": "a-real-secret-value-for-tests",
                "require_hmac": True, "allowed_ips": [], "cors_origins": [],
                "require_passphrase": False, "passphrase": "",
            },
            "rate_limiting": {"enabled": False},
            "broker_routing": {"default_broker": "interactive_brokers", "routes": []},
            "logging": {"level": "WARNING"},
        }
        base["security"].update(security)
        return base

    def _build(self, cfg):
        from unittest.mock import patch
        from tradingview.webhooks.webhook_server import create_app
        with patch("tradingview.webhooks.webhook_server.load_config", return_value=cfg):
            return create_app()

    def test_placeholder_passphrase_refuses_to_start(self):
        """The shipped placeholder is public, so it authenticates nobody."""
        cfg = self._cfg(require_passphrase=True,
                        passphrase="CHANGE_ME_WEBHOOK_PASSPHRASE")
        with pytest.raises(RuntimeError, match="placeholder"):
            self._build(cfg)

    def test_empty_passphrase_refuses_to_start(self):
        cfg = self._cfg(require_passphrase=True, passphrase="")
        with pytest.raises(RuntimeError, match="placeholder|empty"):
            self._build(cfg)

    def test_a_real_passphrase_starts(self):
        cfg = self._cfg(require_passphrase=True,
                        passphrase="a-genuinely-configured-passphrase")
        assert self._build(cfg) is not None

    def test_placeholder_is_tolerated_when_not_required(self):
        cfg = self._cfg(require_passphrase=False,
                        passphrase="CHANGE_ME_WEBHOOK_PASSPHRASE")
        assert self._build(cfg) is not None


class TestNoImportSideEffects:
    """Importing the module must not build the application.

    `app = create_app()` at module scope meant that merely collecting tests
    read config.yaml from disk, instantiated broker adapters, started the
    health monitor and enforced the startup guards.
    """

    def test_importing_does_not_construct_the_app(self):
        """Checked in a subprocess, deliberately.

        importlib.reload() would rebind every class in the module, so
        exceptions raised by the reloaded copy stop matching the ones other
        test modules imported earlier — the same duplicate-class-object
        contamination this file's old sys.modules scaffolding caused. A clean
        interpreter answers the question without touching this session.
        """
        import subprocess
        import textwrap

        script = textwrap.dedent("""
            import sys
            sys.path.insert(0, %r)
            import tradingview.webhooks.webhook_server as w
            print("BUILT" if w._app_instance is not None else "NOT_BUILT")
        """) % PROJECT_ROOT

        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        assert "NOT_BUILT" in out.stdout, (
            f"importing the module constructed the app\n{out.stdout}\n{out.stderr}"
        )

    def test_uvicorn_is_not_shadowed_by_a_mock(self):
        """test_webhook_server.py left a MagicMock uvicorn in sys.modules."""
        import sys

        import tests.test_webhook_server  # noqa: F401  (import for its side effects)

        uvicorn = sys.modules.get("uvicorn")
        if uvicorn is not None:
            assert "MagicMock" not in type(uvicorn).__name__

    def test_yaml_is_the_real_module(self):
        import sys

        import tests.test_webhook_server  # noqa: F401

        assert "MagicMock" not in type(sys.modules["yaml"]).__name__
