"""
Comprehensive tests for AlertDispatcher.

Covers: dispatch(), _send_console(), _send_discord(), _send_sms(),
_send_email(), _HAS_REQUESTS=False graceful degradation, all channel
types, message formatting, priority handling, and error handling.
"""

import sys
import os
import json
import logging

import pytest
from unittest.mock import MagicMock, patch, call
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.notifier.alert_dispatcher import (
    AlertDispatcher,
    PRIORITY_COLORS,
    PRIORITY_DISCORD_COLORS,
    RESET_COLOR,
    _HAS_REQUESTS,
)


# ── Helper ───────────────────────────────────────────────────────────────

def _make_dispatcher(channels=None, **extra_config):
    config = {"enabled_channels": channels or ["console"]}
    config.update(extra_config)
    return AlertDispatcher(config)


# ── __init__ Tests ───────────────────────────────────────────────────────

class TestInit:
    def test_default_channels(self):
        d = AlertDispatcher({})
        assert d.enabled_channels == ["console"]

    def test_custom_channels(self):
        d = AlertDispatcher({"enabled_channels": ["discord", "sms"]})
        assert d.enabled_channels == ["discord", "sms"]

    def test_config_stored(self):
        cfg = {"enabled_channels": ["console"], "extra": 42}
        d = AlertDispatcher(cfg)
        assert d.config["extra"] == 42


# ── dispatch() Tests ─────────────────────────────────────────────────────

class TestDispatch:
    def test_dispatch_calls_console(self):
        d = _make_dispatcher(["console"])
        with patch.object(d, "_send_console") as mock:
            d.dispatch("Title", "Body", "INFO")
            mock.assert_called_once_with("Title", "Body", "INFO")

    def test_dispatch_calls_multiple_channels(self):
        d = _make_dispatcher(["console", "discord"])
        with patch.object(d, "_send_console") as mc, \
             patch.object(d, "_send_discord") as md:
            d.dispatch("T", "B", "WARNING")
            mc.assert_called_once()
            md.assert_called_once()

    def test_dispatch_normalizes_priority_to_upper(self):
        d = _make_dispatcher(["console"])
        with patch.object(d, "_send_console") as mock:
            d.dispatch("T", "B", "warning")
            mock.assert_called_once_with("T", "B", "WARNING")

    def test_dispatch_invalid_priority_defaults_to_info(self):
        d = _make_dispatcher(["console"])
        with patch.object(d, "_send_console") as mock:
            d.dispatch("T", "B", "BOGUS")
            mock.assert_called_once_with("T", "B", "INFO")

    def test_dispatch_unknown_channel_logs_warning(self, caplog):
        d = _make_dispatcher(["nonexistent"])
        with caplog.at_level(logging.WARNING):
            d.dispatch("T", "B")
        assert any("Unknown notification channel" in r.message for r in caplog.records)

    def test_dispatch_exception_in_channel_does_not_propagate(self):
        d = _make_dispatcher(["console"])
        with patch.object(d, "_send_console", side_effect=RuntimeError("boom")):
            d.dispatch("T", "B")

    def test_dispatch_continues_after_one_channel_fails(self):
        d = _make_dispatcher(["discord", "console"])
        with patch.object(d, "_send_discord", side_effect=RuntimeError), \
             patch.object(d, "_send_console") as mock:
            d.dispatch("T", "B")
            mock.assert_called_once()


# ── _send_console() Tests ────────────────────────────────────────────────

class TestSendConsole:
    def test_console_prints_output(self, capsys):
        d = _make_dispatcher(["console"], console={"enabled": True})
        d._send_console("Alert Title", "Alert Body", "INFO")
        out = capsys.readouterr().out
        assert "Alert Title" in out
        assert "Alert Body" in out

    def test_console_includes_priority_tag(self, capsys):
        d = _make_dispatcher(["console"], console={"enabled": True})
        d._send_console("T", "B", "CRITICAL")
        out = capsys.readouterr().out
        assert "[CRITICAL]" in out

    def test_console_disabled_skips(self, capsys):
        d = _make_dispatcher(["console"], console={"enabled": False})
        d._send_console("T", "B", "INFO")
        out = capsys.readouterr().out
        assert out == ""

    def test_console_default_enabled(self, capsys):
        d = _make_dispatcher(["console"])
        d._send_console("T", "B", "WARNING")
        out = capsys.readouterr().out
        assert "[WARNING]" in out

    def test_console_color_codes(self):
        assert "INFO" in PRIORITY_COLORS
        assert "WARNING" in PRIORITY_COLORS
        assert "CRITICAL" in PRIORITY_COLORS


# ── _send_discord() Tests ────────────────────────────────────────────────

class TestSendDiscord:
    def test_discord_posts_embed(self):
        d = _make_dispatcher(
            ["discord"],
            discord={"webhook_url": "https://discord.com/api/webhooks/test"},
        )
        with patch("shared.notifier.alert_dispatcher.requests") as mock_req:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_req.post.return_value = mock_resp
            d._send_discord("Title", "Body", "INFO")
            mock_req.post.assert_called_once()
            call_kwargs = mock_req.post.call_args
            payload = json.loads(call_kwargs[1]["data"] if "data" in call_kwargs[1] else call_kwargs.kwargs["data"])
            assert payload["embeds"][0]["title"] == "[INFO] Title"

    def test_discord_embed_color_matches_priority(self):
        d = _make_dispatcher(
            ["discord"],
            discord={"webhook_url": "https://discord.com/api/webhooks/test"},
        )
        with patch("shared.notifier.alert_dispatcher.requests") as mock_req:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_req.post.return_value = mock_resp
            d._send_discord("T", "B", "CRITICAL")
            payload = json.loads(mock_req.post.call_args[1]["data"])
            assert payload["embeds"][0]["color"] == PRIORITY_DISCORD_COLORS["CRITICAL"]

    def test_discord_no_webhook_skips(self, caplog):
        d = _make_dispatcher(["discord"], discord={})
        with caplog.at_level(logging.WARNING):
            d._send_discord("T", "B", "INFO")
        assert any("webhook" in r.message.lower() for r in caplog.records)

    @patch.dict("shared.notifier.alert_dispatcher.__dict__", {"_HAS_REQUESTS": False})
    def test_discord_without_requests_skips(self, caplog):
        d = _make_dispatcher(["discord"], discord={"webhook_url": "https://x"})
        import shared.notifier.alert_dispatcher as mod
        original = mod._HAS_REQUESTS
        mod._HAS_REQUESTS = False
        try:
            with caplog.at_level(logging.WARNING):
                d._send_discord("T", "B", "INFO")
            assert any("requests" in r.message.lower() for r in caplog.records)
        finally:
            mod._HAS_REQUESTS = original


# ── _send_sms() Tests ────────────────────────────────────────────────────

class TestSendSms:
    def _sms_config(self):
        return {
            "sms": {
                "account_sid": "AC123",
                "auth_token": "token123",
                "from_number": "+1111",
                "to_number": "+2222",
            }
        }

    def test_sms_posts_to_twilio(self):
        d = _make_dispatcher(["sms"], **self._sms_config())
        with patch("shared.notifier.alert_dispatcher.requests") as mock_req:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_req.post.return_value = mock_resp
            d._send_sms("Alert", "Market crash", "CRITICAL")
            mock_req.post.assert_called_once()
            call_args = mock_req.post.call_args
            assert "twilio.com" in call_args[0][0]
            assert call_args[1]["data"]["Body"].startswith("[CRITICAL]")

    def test_sms_incomplete_config_skips(self, caplog):
        d = _make_dispatcher(["sms"], sms={"account_sid": "AC123"})
        with caplog.at_level(logging.WARNING):
            d._send_sms("T", "B", "INFO")
        assert any("incomplete" in r.message.lower() for r in caplog.records)

    def test_sms_without_requests_skips(self, caplog):
        d = _make_dispatcher(["sms"], **self._sms_config())
        import shared.notifier.alert_dispatcher as mod
        original = mod._HAS_REQUESTS
        mod._HAS_REQUESTS = False
        try:
            with caplog.at_level(logging.WARNING):
                d._send_sms("T", "B", "INFO")
            assert any("requests" in r.message.lower() for r in caplog.records)
        finally:
            mod._HAS_REQUESTS = original

    def test_sms_auth_uses_sid_and_token(self):
        d = _make_dispatcher(["sms"], **self._sms_config())
        with patch("shared.notifier.alert_dispatcher.requests") as mock_req:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_req.post.return_value = mock_resp
            d._send_sms("T", "B", "INFO")
            call_kwargs = mock_req.post.call_args[1]
            assert call_kwargs["auth"] == ("AC123", "token123")


# ── _send_email() Tests ──────────────────────────────────────────────────

class TestSendEmail:
    def _email_config(self):
        return {
            "email": {
                "smtp_host": "smtp.test.com",
                "smtp_port": 587,
                "smtp_user": "user",
                "smtp_password": "pass",
                "smtp_from": "from@test.com",
                "smtp_to": "to@test.com",
            }
        }

    def test_email_sends_via_smtp(self):
        d = _make_dispatcher(["email"], **self._email_config())
        with patch("shared.notifier.alert_dispatcher.smtplib.SMTP") as mock_smtp:
            server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
            d._send_email("Alert", "Body", "WARNING")
            server.sendmail.assert_called_once()

    def test_email_subject_has_priority(self):
        d = _make_dispatcher(["email"], **self._email_config())
        with patch("shared.notifier.alert_dispatcher.smtplib.SMTP") as mock_smtp:
            server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
            d._send_email("Test", "Body", "CRITICAL")
            sent_msg = server.sendmail.call_args[0][2]
            assert "[CRITICAL]" in sent_msg

    def test_email_incomplete_config_skips(self, caplog):
        d = _make_dispatcher(["email"], email={"smtp_host": "x"})
        with caplog.at_level(logging.WARNING):
            d._send_email("T", "B", "INFO")
        assert any("incomplete" in r.message.lower() for r in caplog.records)


# ── _HAS_REQUESTS integration ───────────────────────────────────────────

class TestHasRequestsFlag:
    def test_flag_is_boolean(self):
        assert isinstance(_HAS_REQUESTS, bool)

    def test_discord_and_sms_degrade_without_requests(self):
        """Without `requests`, both channels skip instead of raising.

        This asserted `"_HAS_REQUESTS" in inspect.getsource(...)`, which a
        comment mentioning the name would satisfy and which breaks on any
        equivalent rewrite. What matters is that a missing optional dependency
        does not take down alerting — so that is what is exercised.
        """
        from unittest.mock import patch

        dispatcher = AlertDispatcher({
            "enabled_channels": ["discord", "sms"],
            "discord": {"webhook_url": "https://example.test/hook"},
            "sms": {"account_sid": "x", "auth_token": "y",
                    "from_number": "+1", "to_number": "+2"},
        })

        with patch("shared.notifier.alert_dispatcher._HAS_REQUESTS", False):
            # Must not raise, and must not attempt any HTTP call.
            with patch("shared.notifier.alert_dispatcher.requests") as req:
                dispatcher.dispatch(title="t", message="m", priority="INFO")
                req.post.assert_not_called()


# ── Priority color constants ────────────────────────────────────────────

class TestConstants:
    def test_priority_colors_complete(self):
        for p in ("INFO", "WARNING", "CRITICAL"):
            assert p in PRIORITY_COLORS
            assert p in PRIORITY_DISCORD_COLORS

    def test_reset_color_defined(self):
        assert RESET_COLOR == "\033[0m"

    def test_discord_colors_are_ints(self):
        for v in PRIORITY_DISCORD_COLORS.values():
            assert isinstance(v, int)
