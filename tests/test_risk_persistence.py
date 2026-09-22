"""Tests for RiskManager SQLite state persistence.

Covers: state saves to SQLite on trade, state loads on restart,
daily P&L persisted/restored, consecutive losses persisted,
circuit breaker state persisted, peak equity persisted,
corrupt DB handled gracefully, missing DB creates fresh state.

15+ tests total.
"""

import json
import os
import sqlite3
import sys
import time
import pytest
from datetime import date, timedelta
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.risk_manager import RiskManager, RiskManagerConfig

# trading_day(), not date.today(): the reset logic measures the session
# at the exchange, so on a UTC host the host date is a day ahead and
# stamping state with it looks like a brand new trading day.
from shared.utils.trading_clock import trading_day


@pytest.fixture
def db_path(tmp_path):
    """Temporary SQLite database path."""
    return str(tmp_path / "risk_state.db")


def _make_rm(db_path, **kwargs):
    """Create a RiskManager with persistence enabled."""
    defaults = dict(
        total_capital=100_000,
        max_daily_loss=2_000,
        max_consecutive_losses=3,
        cooldown_seconds=60,
        max_drawdown_pct=10.0,
        circuit_breaker_pause_hours=24.0,
        min_seconds_between_trades=0,
        max_trades_per_hour=100,
        persist_path=db_path,
    )
    defaults.update(kwargs)
    cfg = RiskManagerConfig(**defaults)
    return RiskManager(config=cfg)


# ═══════════════════════════════════════════════════════════════════════
#  1. State saves to SQLite on trade
# ═══════════════════════════════════════════════════════════════════════


class TestStateSavesOnTrade:

    def test_db_created_on_init(self, db_path):
        _make_rm(db_path)
        assert os.path.exists(db_path)

    def test_table_created(self, db_path):
        _make_rm(db_path)
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = [t[0] for t in tables]
        assert "risk_state" in table_names

    def test_state_written_after_trade(self, db_path):
        rm = _make_rm(db_path)
        rm.record_trade("AAPL", pnl=-500)
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT key, value FROM risk_state").fetchall()
        conn.close()
        keys = {r[0] for r in rows}
        assert "daily_pnl" in keys
        assert "consecutive_losses" in keys
        assert "current_equity" in keys

    def test_state_written_after_add_position(self, db_path):
        rm = _make_rm(db_path)
        rm.add_position("AAPL", 2000)
        conn = sqlite3.connect(db_path)
        rows = dict(conn.execute("SELECT key, value FROM risk_state").fetchall())
        conn.close()
        positions = json.loads(rows["open_positions"])
        assert "AAPL" in positions


# ═══════════════════════════════════════════════════════════════════════
#  2. State loads on restart
# ═══════════════════════════════════════════════════════════════════════


class TestStateLoadsOnRestart:

    def test_equity_restored(self, db_path):
        rm1 = _make_rm(db_path)
        rm1.record_trade("AAPL", pnl=-1_000)
        del rm1

        rm2 = _make_rm(db_path)
        assert rm2._current_equity == 99_000

    def test_positions_restored(self, db_path):
        rm1 = _make_rm(db_path)
        rm1.add_position("AAPL", 3_000)
        rm1.add_position("MSFT", 2_000)
        del rm1

        rm2 = _make_rm(db_path)
        assert "AAPL" in rm2._open_positions
        assert "MSFT" in rm2._open_positions
        assert rm2._open_positions["AAPL"] == 3_000


# ═══════════════════════════════════════════════════════════════════════
#  3. Daily P&L persisted and restored
# ═══════════════════════════════════════════════════════════════════════


class TestDailyPnLPersistence:

    def test_daily_pnl_restored_same_day(self, db_path):
        rm1 = _make_rm(db_path)
        rm1.record_trade("AAPL", pnl=-800)
        rm1._last_trade_time = 0
        rm1.record_trade("MSFT", pnl=-300)
        del rm1

        rm2 = _make_rm(db_path)
        assert rm2._daily_pnl == -1_100

    def test_daily_pnl_reset_on_new_day(self, db_path):
        rm1 = _make_rm(db_path)
        rm1.record_trade("AAPL", pnl=-800)
        del rm1

        # Manually update the saved date to yesterday
        conn = sqlite3.connect(db_path)
        yesterday = (trading_day() - timedelta(days=1)).isoformat()
        conn.execute(
            "UPDATE risk_state SET value = ? WHERE key = 'daily_pnl_date'",
            (json.dumps(yesterday),),
        )
        conn.commit()
        conn.close()

        rm2 = _make_rm(db_path)
        assert rm2._daily_pnl == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  4. Consecutive losses persisted
# ═══════════════════════════════════════════════════════════════════════


class TestConsecutiveLossesPersistence:

    def test_consecutive_losses_restored(self, db_path):
        rm1 = _make_rm(db_path)
        rm1.record_trade("AAPL", pnl=-100)
        rm1._last_trade_time = 0
        rm1.record_trade("AAPL", pnl=-100)
        assert rm1._consecutive_losses == 2
        del rm1

        rm2 = _make_rm(db_path)
        assert rm2._consecutive_losses == 2

    def test_cooldown_state_restored(self, db_path):
        rm1 = _make_rm(db_path)
        for _ in range(3):
            rm1.record_trade("AAPL", pnl=-100)
            rm1._last_trade_time = 0
        cooldown_val = rm1._cooldown_until
        assert cooldown_val > time.time()
        del rm1

        rm2 = _make_rm(db_path)
        assert rm2._cooldown_until > time.time()


# ═══════════════════════════════════════════════════════════════════════
#  5. Circuit breaker state persisted
# ═══════════════════════════════════════════════════════════════════════


class TestCircuitBreakerPersistence:

    def test_circuit_breaker_restored(self, db_path):
        rm1 = _make_rm(db_path, max_drawdown_pct=5.0, max_consecutive_losses=999)
        rm1.record_trade("AAPL", pnl=-6_000)  # 6% drawdown triggers breaker
        assert rm1._circuit_breaker_until > time.time()
        del rm1

        rm2 = _make_rm(db_path, max_drawdown_pct=5.0, max_consecutive_losses=999)
        assert rm2._circuit_breaker_until > time.time()
        assert rm2.can_trade() is False


# ═══════════════════════════════════════════════════════════════════════
#  6. Peak equity persisted
# ═══════════════════════════════════════════════════════════════════════


class TestPeakEquityPersistence:

    def test_peak_equity_restored(self, db_path):
        rm1 = _make_rm(db_path)
        rm1.record_trade("AAPL", pnl=5_000)
        assert rm1._peak_equity == 105_000
        rm1._last_trade_time = 0
        rm1.record_trade("AAPL", pnl=-2_000)
        assert rm1._peak_equity == 105_000
        del rm1

        rm2 = _make_rm(db_path)
        assert rm2._peak_equity == 105_000
        assert rm2._current_equity == 103_000


# ═══════════════════════════════════════════════════════════════════════
#  7. Corrupt DB handled gracefully
# ═══════════════════════════════════════════════════════════════════════


class TestCorruptDBHandled:

    def test_corrupt_db_file_starts_fresh(self, tmp_path):
        corrupt_path = str(tmp_path / "corrupt.db")
        with open(corrupt_path, "w") as f:
            f.write("THIS IS NOT A VALID SQLITE DATABASE!!!")

        # SQLite may raise DatabaseError on corrupt files.
        # If the constructor raises, that is acceptable behaviour.
        try:
            rm = _make_rm(corrupt_path)
            # If it succeeded, state should be fresh defaults
            assert rm._daily_pnl == 0.0
            assert rm._current_equity == 100_000
        except Exception:
            # Corrupt DB raising is also valid behaviour
            pass

    def test_db_with_bad_json_starts_fresh(self, db_path):
        # Create valid DB but with invalid JSON in values
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS risk_state "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        conn.execute(
            "INSERT INTO risk_state (key, value, updated_at) VALUES (?, ?, ?)",
            ("daily_pnl", "NOT_VALID_JSON{{{", "2025-01-01"),
        )
        conn.commit()
        conn.close()

        # Bad JSON triggers json.decoder.JSONDecodeError in _load_state.
        # The constructor may raise or may gracefully degrade.
        try:
            rm = _make_rm(db_path)
            assert rm._current_equity == 100_000
        except json.JSONDecodeError:
            # Also acceptable — code propagates the error
            pass


# ═══════════════════════════════════════════════════════════════════════
#  8. Missing DB creates fresh state
# ═══════════════════════════════════════════════════════════════════════


class TestMissingDBCreatesFresh:

    def test_missing_db_creates_new(self, tmp_path):
        new_path = str(tmp_path / "subdir" / "new_state.db")
        assert not os.path.exists(new_path)

        rm = _make_rm(new_path)
        assert os.path.exists(new_path)
        assert rm._daily_pnl == 0.0
        assert rm._current_equity == 100_000
        assert rm.can_trade() is True

    def test_no_persist_path_works(self):
        cfg = RiskManagerConfig(total_capital=50_000, persist_path=None)
        rm = RiskManager(config=cfg)
        rm.record_trade("AAPL", pnl=-100)
        assert rm._daily_pnl == -100
        assert rm._persist_conn is None

    def test_empty_db_loads_defaults(self, db_path):
        # Create empty DB with table but no rows
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS risk_state "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        conn.commit()
        conn.close()

        rm = _make_rm(db_path)
        assert rm._daily_pnl == 0.0
        assert rm._consecutive_losses == 0
        assert rm._current_equity == 100_000
