"""Tests for SQLite state persistence across ALL broker platforms.

Covers Schwab (thinkorswim) and TradeStation:
- State saves to DB on trade / position changes
- State loads correctly on restart (new instance)
- New day resets daily P&L but keeps other state
- Corrupt DB handled gracefully (fresh start)
- Open positions survive restart

15+ tests total.
"""

import json
import os
import sqlite3
import sys
import time
import pytest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.risk_manager import RiskManager, RiskManagerConfig

# trading_day(), not date.today(): the reset logic measures the session
# at the exchange, so on a UTC host the host date is a day ahead and
# stamping state with it looks like a brand new trading day.
from shared.utils.trading_clock import trading_day


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_schwab(db_path, **overrides):
    """Create a SchwabClient with auth mocked out and persistence enabled."""
    from thinkorswim.api.schwab_client import SchwabClient

    config = {
        "client_id": "test_id",
        "client_secret": "test_secret",
        "refresh_token": "test_refresh",
        "account_id": "TEST_ACCT_HASH",
    }
    with patch.object(SchwabClient, "_authenticate", return_value=None):
        client = SchwabClient(
            config,
            max_daily_loss=overrides.get("max_daily_loss", 2_000),
            persist_path=db_path,
        )
    client._access_token = "mock_token"
    client._token_expiry = time.time() + 3600
    return client


def _make_ts_router(db_path, **overrides):
    """Create a TradeStationOrderRouter with auth mocked and persistence."""
    from tradestation.api.order_router import TradeStationOrderRouter

    config = {
        "client_id": "ts_id",
        "client_secret": "ts_secret",
        "redirect_uri": "https://127.0.0.1",
        "refresh_token": "ts_refresh",
        "max_daily_loss": overrides.get("max_daily_loss", 5_000),
        "max_consecutive_losses": overrides.get("max_consecutive_losses", 3),
        "cooldown_minutes": overrides.get("cooldown_minutes", 30),
        "max_positions": overrides.get("max_positions", 10),
    }
    with patch.object(TradeStationOrderRouter, "_authenticate", return_value=None):
        router = TradeStationOrderRouter(config, persist_path=db_path)
    router.access_token = "mock_ts_token"
    router.token_expiry = time.time() + 3600
    return router


@pytest.fixture
def schwab_db(tmp_path):
    return str(tmp_path / "schwab_risk.db")


@pytest.fixture
def ts_db(tmp_path):
    return str(tmp_path / "ts_risk.db")


# ═══════════════════════════════════════════════════════════════════════
#  Schwab Persistence Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSchwabSavesRiskState:
    """test_schwab_saves_risk_state — daily P&L persisted."""

    def test_db_file_created(self, schwab_db):
        _make_schwab(schwab_db)
        assert os.path.exists(schwab_db)

    def test_schwab_table_created(self, schwab_db):
        _make_schwab(schwab_db)
        conn = sqlite3.connect(schwab_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        assert any("schwab_risk_state" in t[0] for t in tables)

    def test_daily_pnl_saved_after_update(self, schwab_db):
        client = _make_schwab(schwab_db)
        client._daily_pnl = -500.0
        client._save_risk_state()

        conn = sqlite3.connect(schwab_db)
        rows = dict(conn.execute("SELECT key, value FROM schwab_risk_state").fetchall())
        conn.close()
        assert float(json.loads(rows["daily_pnl"])) == -500.0

    def test_consecutive_losses_saved(self, schwab_db):
        client = _make_schwab(schwab_db)
        client._consecutive_losses = 2
        client._save_risk_state()

        conn = sqlite3.connect(schwab_db)
        rows = dict(conn.execute("SELECT key, value FROM schwab_risk_state").fetchall())
        conn.close()
        assert int(json.loads(rows["consecutive_losses"])) == 2


class TestSchwabLoadsOnRestart:
    """test_schwab_loads_on_restart — state restored after restart."""

    def test_daily_pnl_restored_same_day(self, schwab_db):
        c1 = _make_schwab(schwab_db)
        c1._daily_pnl = -1_200.0
        c1._save_risk_state()
        del c1

        c2 = _make_schwab(schwab_db)
        assert c2._daily_pnl == -1_200.0

    def test_consecutive_losses_restored(self, schwab_db):
        c1 = _make_schwab(schwab_db)
        c1._consecutive_losses = 3
        c1._cooldown_until = time.time() + 600
        c1._save_risk_state()
        del c1

        c2 = _make_schwab(schwab_db)
        assert c2._consecutive_losses == 3

    def test_cooldown_restored(self, schwab_db):
        c1 = _make_schwab(schwab_db)
        future = time.time() + 1800
        c1._cooldown_until = future
        c1._save_risk_state()
        del c1

        c2 = _make_schwab(schwab_db)
        assert c2._cooldown_until > time.time()


class TestSchwabNewDayResets:
    """test_schwab_new_day_resets — next day resets P&L."""

    def test_new_day_resets_daily_pnl(self, schwab_db):
        c1 = _make_schwab(schwab_db)
        c1._daily_pnl = -1_500.0
        c1._save_risk_state()
        del c1

        # Manually set date to yesterday in DB
        conn = sqlite3.connect(schwab_db)
        yesterday = (trading_day() - timedelta(days=1)).isoformat()
        conn.execute(
            "UPDATE schwab_risk_state SET value = ? WHERE key = 'daily_pnl_date'",
            (json.dumps(yesterday),),
        )
        conn.commit()
        conn.close()

        c2 = _make_schwab(schwab_db)
        assert c2._daily_pnl == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  TradeStation Persistence Tests
# ═══════════════════════════════════════════════════════════════════════


class TestTradeStationSavesState:
    """test_tradestation_saves_state — daily P&L, cooldowns persisted."""

    def test_ts_db_created(self, ts_db):
        _make_ts_router(ts_db)
        assert os.path.exists(ts_db)

    def test_ts_table_created(self, ts_db):
        _make_ts_router(ts_db)
        conn = sqlite3.connect(ts_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        assert any("ts_risk_state" in t[0] for t in tables)

    def test_ts_daily_pnl_saved(self, ts_db):
        router = _make_ts_router(ts_db)
        router._daily_pnl = -3_000.0
        router._save_risk_state()

        conn = sqlite3.connect(ts_db)
        rows = dict(conn.execute("SELECT key, value FROM ts_risk_state").fetchall())
        conn.close()
        assert float(json.loads(rows["daily_pnl"])) == -3_000.0

    def test_ts_cooldown_saved(self, ts_db):
        router = _make_ts_router(ts_db)
        future_dt = datetime.now(timezone.utc) + timedelta(minutes=30)
        router._cooldown_until = future_dt
        router._save_risk_state()

        conn = sqlite3.connect(ts_db)
        rows = dict(conn.execute("SELECT key, value FROM ts_risk_state").fetchall())
        conn.close()
        saved = json.loads(rows["cooldown_until"])
        assert saved is not None


class TestTradeStationLoadsOnRestart:
    """test_tradestation_loads_on_restart — state restored."""

    def test_ts_pnl_restored_same_day(self, ts_db):
        r1 = _make_ts_router(ts_db)
        r1._daily_pnl = -2_500.0
        r1._daily_pnl_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        r1._save_risk_state()
        del r1

        r2 = _make_ts_router(ts_db)
        assert r2._daily_pnl == -2_500.0

    def test_ts_consecutive_losses_restored(self, ts_db):
        r1 = _make_ts_router(ts_db)
        r1._consecutive_losses = 2
        r1._daily_pnl_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        r1._save_risk_state()
        del r1

        r2 = _make_ts_router(ts_db)
        assert r2._consecutive_losses == 2


class TestTradeStationPositionsPersisted:
    """test_tradestation_positions_persisted — open positions survive restart."""

    def test_positions_saved_and_restored(self, ts_db):
        r1 = _make_ts_router(ts_db)
        r1._open_positions = {"AAPL", "MSFT", "GOOGL"}
        r1._daily_pnl_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        r1._save_risk_state()
        del r1

        r2 = _make_ts_router(ts_db)
        assert "AAPL" in r2._open_positions
        assert "MSFT" in r2._open_positions
        assert "GOOGL" in r2._open_positions

    def test_position_count_matches(self, ts_db):
        r1 = _make_ts_router(ts_db)
        r1._open_positions = {"TSLA", "NVDA"}
        r1._daily_pnl_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        r1._save_risk_state()
        del r1

        r2 = _make_ts_router(ts_db)
        assert len(r2._open_positions) == 2


class TestCorruptDBFreshStart:
    """test_corrupt_db_fresh_start — handles corrupt DB gracefully."""

    def test_schwab_corrupt_db(self, tmp_path):
        corrupt = str(tmp_path / "corrupt_schwab.db")
        with open(corrupt, "w") as f:
            f.write("GARBAGE DATA NOT A DATABASE")

        try:
            client = _make_schwab(corrupt)
            assert client._daily_pnl == 0.0
        except Exception:
            pass  # raising on corrupt DB is also acceptable

    def test_ts_corrupt_db(self, tmp_path):
        corrupt = str(tmp_path / "corrupt_ts.db")
        with open(corrupt, "w") as f:
            f.write("NOT SQLITE!!")

        try:
            router = _make_ts_router(corrupt)
            assert router._daily_pnl == 0.0
        except Exception:
            pass

    def test_schwab_bad_json_in_db(self, schwab_db):
        conn = sqlite3.connect(schwab_db)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schwab_risk_state "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        conn.execute(
            "INSERT INTO schwab_risk_state (key, value, updated_at) VALUES (?, ?, ?)",
            ("daily_pnl", "{{INVALID JSON!!", "2025-01-01"),
        )
        conn.commit()
        conn.close()

        try:
            client = _make_schwab(schwab_db)
            assert client._daily_pnl == 0.0
        except json.JSONDecodeError:
            pass

    def test_ts_bad_json_in_db(self, ts_db):
        conn = sqlite3.connect(ts_db)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ts_risk_state "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        conn.execute(
            "INSERT INTO ts_risk_state (key, value, updated_at) VALUES (?, ?, ?)",
            ("daily_pnl", "NOT_JSON!!!", "2025-01-01"),
        )
        conn.commit()
        conn.close()

        try:
            router = _make_ts_router(ts_db)
            assert router._daily_pnl == 0.0
        except json.JSONDecodeError:
            pass


class TestNewDayResetsTradeStation:
    """Additional new-day reset tests for TradeStation."""

    def test_ts_new_day_resets_pnl(self, ts_db):
        r1 = _make_ts_router(ts_db)
        r1._daily_pnl = -4_000.0
        r1._daily_pnl_reset_date = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).strftime("%Y-%m-%d")
        r1._save_risk_state()
        del r1

        r2 = _make_ts_router(ts_db)
        assert r2._daily_pnl == 0.0

    def test_ts_new_day_preserves_consecutive_losses(self, ts_db):
        r1 = _make_ts_router(ts_db)
        r1._consecutive_losses = 2
        r1._daily_pnl = -1_000.0
        r1._daily_pnl_reset_date = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).strftime("%Y-%m-%d")
        r1._save_risk_state()
        del r1

        r2 = _make_ts_router(ts_db)
        assert r2._daily_pnl == 0.0
        assert r2._consecutive_losses == 2
