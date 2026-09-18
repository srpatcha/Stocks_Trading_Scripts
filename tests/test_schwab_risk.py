"""Tests for Schwab/thinkorswim client risk controls.

Covers: RiskManager integration blocks trades, daily P&L tracking,
daily P&L blocks when exceeded, daily reset works, record_trade_pnl
updates correctly.

12+ tests total.
"""

import os
import sys
import time
import pytest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from thinkorswim.api.schwab_client import SchwabClient, SchwabAPIError


# ─── Helpers ───

def _mock_auth():
    """Patch _authenticate to prevent real API calls."""
    return patch.object(SchwabClient, "_authenticate", return_value=None)


def _make_client(risk_manager=None, max_daily_loss=5000.0):
    """Create a SchwabClient with mocked auth."""
    config = {
        "client_id": "test_id",
        "client_secret": "test_secret",
        "refresh_token": "test_refresh",
        "account_id": "ACCT12345678",
    }
    with _mock_auth():
        client = SchwabClient(
            config,
            risk_manager=risk_manager,
            max_daily_loss=max_daily_loss,
        )
        client._access_token = "test_token"
        client._token_expiry = time.time() + 3600
    return client


# ═══════════════════════════════════════════════════════════════════════
#  1. RiskManager integration blocks trades
# ═══════════════════════════════════════════════════════════════════════


class TestRiskManagerIntegration:

    def test_risk_manager_stored(self):
        rm = MagicMock()
        client = _make_client(risk_manager=rm)
        assert client._risk_manager is rm

    def test_no_risk_manager_by_default(self):
        client = _make_client()
        assert client._risk_manager is None

    def test_risk_manager_blocks_market_order(self):
        rm = MagicMock()
        rm.can_trade.return_value = False
        client = _make_client(risk_manager=rm)
        with pytest.raises(SchwabAPIError, match="RiskManager blocked"):
            client.place_market_order("AAPL", "BUY", 100)

    def test_risk_manager_blocks_limit_order(self):
        rm = MagicMock()
        rm.can_trade.return_value = False
        client = _make_client(risk_manager=rm)
        with pytest.raises(SchwabAPIError, match="RiskManager blocked"):
            client.place_limit_order("AAPL", "BUY", 100, 150.0)

    def test_risk_manager_allows_order(self):
        rm = MagicMock()
        rm.can_trade.return_value = True
        client = _make_client(risk_manager=rm)

        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.headers = {"Location": "/orders/ORD123"}
        mock_resp.text = ""
        client._session.post = MagicMock(return_value=mock_resp)

        order_id = client.place_market_order("AAPL", "BUY", 10)
        assert order_id == "ORD123"


# ═══════════════════════════════════════════════════════════════════════
#  2. Daily P&L tracking
# ═══════════════════════════════════════════════════════════════════════


class TestDailyPnLTracking:

    def test_initial_pnl_zero(self):
        client = _make_client()
        assert client.daily_pnl == 0.0

    def test_record_positive_pnl(self):
        client = _make_client()
        client.record_trade_pnl("AAPL", 500.0)
        assert client.daily_pnl == 500.0

    def test_record_negative_pnl(self):
        client = _make_client()
        client.record_trade_pnl("AAPL", -300.0)
        assert client.daily_pnl == -300.0

    def test_pnl_accumulates(self):
        client = _make_client()
        client.record_trade_pnl("AAPL", -200.0)
        client.record_trade_pnl("MSFT", -300.0)
        client.record_trade_pnl("GOOG", 100.0)
        assert client.daily_pnl == -400.0


# ═══════════════════════════════════════════════════════════════════════
#  3. Daily P&L blocks when exceeded
# ═══════════════════════════════════════════════════════════════════════


class TestDailyPnLBlocks:

    def test_not_halted_under_limit(self):
        client = _make_client(max_daily_loss=1000.0)
        client.record_trade_pnl("AAPL", -500.0)
        assert client.is_trading_halted is False

    def test_halted_at_limit(self):
        client = _make_client(max_daily_loss=1000.0)
        client.record_trade_pnl("AAPL", -1000.0)
        assert client.is_trading_halted is True

    def test_halted_over_limit(self):
        client = _make_client(max_daily_loss=1000.0)
        client.record_trade_pnl("AAPL", -1500.0)
        assert client.is_trading_halted is True

    def test_market_order_blocked_when_halted(self):
        client = _make_client(max_daily_loss=1000.0)
        client.record_trade_pnl("AAPL", -1200.0)
        with pytest.raises(SchwabAPIError, match="Daily loss limit"):
            client.place_market_order("AAPL", "BUY", 100)

    def test_limit_order_blocked_when_halted(self):
        client = _make_client(max_daily_loss=1000.0)
        client.record_trade_pnl("AAPL", -1200.0)
        with pytest.raises(SchwabAPIError, match="Daily loss limit"):
            client.place_limit_order("AAPL", "BUY", 100, 150.0)


# ═══════════════════════════════════════════════════════════════════════
#  4. Daily reset works
# ═══════════════════════════════════════════════════════════════════════


class TestDailyReset:

    def test_reset_clears_pnl(self):
        client = _make_client(max_daily_loss=1000.0)
        client.record_trade_pnl("AAPL", -900.0)
        assert client.daily_pnl == -900.0
        client.reset_daily_pnl()
        assert client.daily_pnl == 0.0

    def test_reset_unblocks_trading(self):
        client = _make_client(max_daily_loss=1000.0)
        client.record_trade_pnl("AAPL", -1200.0)
        assert client.is_trading_halted is True
        client.reset_daily_pnl()
        assert client.is_trading_halted is False


# ═══════════════════════════════════════════════════════════════════════
#  5. record_trade_pnl updates correctly
# ═══════════════════════════════════════════════════════════════════════


class TestRecordTradePnl:

    def test_forwards_to_risk_manager(self):
        # spec= binds the mock to the real RiskManager signature. An unspecced
        # MagicMock accepts any keyword, which is how a call passing
        # quantity= — rejected by the real record_trade(symbol, pnl) — passed
        # this test while silently failing in production.
        from shared.risk_manager import RiskManager

        rm = MagicMock(spec=RiskManager)
        client = _make_client(risk_manager=rm)
        client.record_trade_pnl("AAPL", -200.0, quantity=50)
        rm.record_trade.assert_called_once_with(symbol="AAPL", pnl=-200.0)

    def test_forwarded_call_matches_real_risk_manager(self):
        """The forwarded call must be accepted by the real RiskManager."""
        from shared.risk_manager import RiskManager, RiskManagerConfig

        rm = RiskManager(RiskManagerConfig(total_capital=100_000.0))
        client = _make_client(risk_manager=rm)
        client.record_trade_pnl("AAPL", -200.0, quantity=50)

        # record_trade swallows nothing here: if the signature drifts again the
        # daily P&L simply will not move.
        assert rm.get_status()["daily_pnl"] == pytest.approx(-200.0)

    def test_risk_manager_error_handled_gracefully(self):
        rm = MagicMock()
        rm.record_trade.side_effect = Exception("RM error")
        client = _make_client(risk_manager=rm)
        # Should not raise
        client.record_trade_pnl("AAPL", -200.0)
        assert client.daily_pnl == -200.0

    def test_no_risk_manager_still_tracks(self):
        client = _make_client()
        client.record_trade_pnl("AAPL", -300.0)
        client.record_trade_pnl("MSFT", 100.0)
        assert client.daily_pnl == -200.0

    def test_multiple_symbols_tracked(self):
        client = _make_client()
        client.record_trade_pnl("AAPL", -100.0)
        client.record_trade_pnl("MSFT", -200.0)
        client.record_trade_pnl("TSLA", 50.0)
        assert client.daily_pnl == -250.0
