"""
Tests for shared.ml.trade_memory — TradeMemory
=================================================

Covers:
- _create_tables(): schema creation
- record_trade(): insertion, return value, all fields persisted
- record_model_prediction(): correctness flag logic
- get_recent_trades(): ordering, limit
- get_trade_count(): count accuracy
- get_model_accuracy(): rolling window, regime filter, trend calc
- get_all_model_accuracies(): multi-model
- query_similar_regime(): regime filtering, insufficient data
- get_performance_summary(): overall stats, regime breakdown, drawdown
- Verify fix: thread-safety with check_same_thread=False and lock
- Concurrent writes via threading
- close() / cleanup
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from shared.ml.trade_memory import TradeDecisionRecord, TradeMemory


# ─── Fixtures ───


@pytest.fixture
def tmp_db(tmp_path):
    """Return path to a temporary SQLite database."""
    return str(tmp_path / "test_trades.db")


@pytest.fixture
def memory(tmp_db):
    """Fresh TradeMemory backed by temp DB."""
    mem = TradeMemory(db_path=tmp_db)
    yield mem
    mem.close()


def _make_trade(**overrides) -> TradeDecisionRecord:
    """Helper to create a TradeDecisionRecord with sensible defaults."""
    defaults = dict(
        timestamp=datetime.now().isoformat(),
        symbol="AAPL",
        action="BUY",
        entry_price=150.0,
        exit_price=155.0,
        pnl=5.0,
        pnl_pct=0.033,
        regime="TRENDING",
        regime_confidence=0.85,
        ensemble_signal=0.7,
        ensemble_confidence=0.8,
        decision_source="ensemble",
        is_winner=True,
        holding_period_bars=10,
    )
    defaults.update(overrides)
    return TradeDecisionRecord(**defaults)


# ─── _create_tables() ───


class TestCreateTables:
    def test_tables_exist(self, memory):
        cursor = memory._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "trades" in tables
        assert "model_performance" in tables

    def test_indexes_exist(self, memory):
        cursor = memory._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = {row["name"] for row in cursor.fetchall()}
        assert "idx_trades_symbol" in indexes
        assert "idx_trades_regime" in indexes
        assert "idx_trades_timestamp" in indexes
        assert "idx_model_perf_model" in indexes

    def test_create_tables_idempotent(self, memory):
        """Re-creating tables must preserve existing rows, not just not crash."""
        memory.record_trade(_make_trade())
        before = memory.get_trade_count()

        memory._create_tables()

        assert memory.get_trade_count() == before, "re-creating tables lost data"


# ─── record_trade() ───


class TestRecordTrade:
    def test_returns_row_id(self, memory):
        trade = _make_trade()
        row_id = memory.record_trade(trade)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_sequential_ids(self, memory):
        id1 = memory.record_trade(_make_trade(symbol="A"))
        id2 = memory.record_trade(_make_trade(symbol="B"))
        assert id2 == id1 + 1

    def test_all_fields_persisted(self, memory):
        trade = _make_trade(
            symbol="TSLA", action="SELL", entry_price=200.0, exit_price=190.0,
            pnl=-10.0, pnl_pct=-0.05, regime="VOLATILE", is_winner=False,
            lstm_prediction=0.5, transformer_prediction=-0.3, rl_action=-1,
        )
        row_id = memory.record_trade(trade)
        row = memory._conn.execute("SELECT * FROM trades WHERE id = ?", (row_id,)).fetchone()
        assert row["symbol"] == "TSLA"
        assert row["action"] == "SELL"
        assert row["entry_price"] == 200.0
        assert row["pnl"] == -10.0
        assert row["regime"] == "VOLATILE"
        assert row["is_winner"] == 0  # stored as int
        assert row["lstm_prediction"] == 0.5
        assert row["rl_action"] == -1

    def test_record_trade_with_features_snapshot(self, memory):
        trade = _make_trade(features_snapshot='{"rsi": 65, "adx": 30}')
        row_id = memory.record_trade(trade)
        row = memory._conn.execute("SELECT features_snapshot FROM trades WHERE id = ?", (row_id,)).fetchone()
        assert '"rsi"' in row["features_snapshot"]


# ─── record_model_prediction() ───


class TestRecordModelPrediction:
    def test_correct_prediction_flagged(self, memory):
        memory.record_model_prediction("lstm", prediction=0.5, actual_outcome=0.3)
        row = memory._conn.execute(
            "SELECT is_correct FROM model_performance ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["is_correct"] == 1  # same sign

    def test_incorrect_prediction_flagged(self, memory):
        memory.record_model_prediction("lstm", prediction=0.5, actual_outcome=-0.3)
        row = memory._conn.execute(
            "SELECT is_correct FROM model_performance ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["is_correct"] == 0  # different sign

    def test_regime_stored(self, memory):
        memory.record_model_prediction("rl", 1.0, 0.5, regime="TRENDING", symbol="AAPL")
        row = memory._conn.execute(
            "SELECT regime, symbol FROM model_performance ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["regime"] == "TRENDING"
        assert row["symbol"] == "AAPL"


# ─── get_recent_trades() ───


class TestGetRecentTrades:
    def test_empty_db_returns_empty(self, memory):
        assert memory.get_recent_trades() == []

    def test_returns_correct_count(self, memory):
        for i in range(5):
            memory.record_trade(_make_trade(symbol=f"SYM{i}"))
        result = memory.get_recent_trades(n=3)
        assert len(result) == 3

    def test_ordered_by_timestamp_desc(self, memory):
        for i in range(3):
            ts = (datetime.now() + timedelta(minutes=i)).isoformat()
            memory.record_trade(_make_trade(timestamp=ts, symbol=f"SYM{i}"))
        trades = memory.get_recent_trades(n=3)
        # First trade should be most recent
        assert trades[0]["symbol"] == "SYM2"

    def test_returns_dicts(self, memory):
        memory.record_trade(_make_trade())
        result = memory.get_recent_trades(n=1)
        assert isinstance(result[0], dict)


# ─── get_trade_count() ───


class TestGetTradeCount:
    def test_empty_db_returns_zero(self, memory):
        assert memory.get_trade_count() == 0

    def test_count_after_inserts(self, memory):
        for _ in range(7):
            memory.record_trade(_make_trade())
        assert memory.get_trade_count() == 7


# ─── get_model_accuracy() ───


class TestGetModelAccuracy:
    def test_no_data_returns_zero_accuracy(self, memory):
        result = memory.get_model_accuracy("lstm")
        assert result["accuracy"] == 0.0
        assert result["total_predictions"] == 0
        assert result["needs_retrain"] is True

    def test_perfect_accuracy(self, memory):
        for _ in range(10):
            memory.record_model_prediction("lstm", 0.5, 0.3)
        result = memory.get_model_accuracy("lstm", window=10)
        assert result["accuracy"] == 1.0

    def test_half_accuracy(self, memory):
        for i in range(10):
            outcome = 0.3 if i % 2 == 0 else -0.3
            memory.record_model_prediction("lstm", 0.5, outcome)
        result = memory.get_model_accuracy("lstm", window=10)
        assert result["accuracy"] == pytest.approx(0.5)

    def test_regime_filter(self, memory):
        for _ in range(5):
            memory.record_model_prediction("lstm", 0.5, 0.3, regime="TRENDING")
        for _ in range(5):
            memory.record_model_prediction("lstm", 0.5, -0.3, regime="VOLATILE")
        trending = memory.get_model_accuracy("lstm", regime="TRENDING")
        assert trending["accuracy"] == 1.0
        volatile = memory.get_model_accuracy("lstm", regime="VOLATILE")
        assert volatile["accuracy"] == 0.0

    def test_needs_retrain_low_accuracy(self, memory):
        for _ in range(10):
            memory.record_model_prediction("lstm", 0.5, -0.3)
        result = memory.get_model_accuracy("lstm")
        assert result["needs_retrain"] is True

    def test_trend_calculation(self, memory):
        # First half correct, second half wrong → negative trend
        for i in range(20):
            outcome = 0.3 if i >= 10 else -0.3  # older=wrong, recent=correct
            memory.record_model_prediction("lstm", 0.5, outcome)
        result = memory.get_model_accuracy("lstm", window=20)
        # Trend should be positive (recent is more accurate)
        assert result["trend"] > 0


# ─── get_all_model_accuracies() ───


class TestGetAllModelAccuracies:
    def test_returns_all_models(self, memory):
        memory.record_model_prediction("lstm", 0.5, 0.3)
        memory.record_model_prediction("transformer", 0.2, 0.1)
        result = memory.get_all_model_accuracies()
        assert "lstm" in result
        assert "transformer" in result


# ─── query_similar_regime() ───


class TestQuerySimilarRegime:
    def test_insufficient_data(self, memory):
        memory.record_trade(_make_trade(regime="TRENDING"))
        result = memory.query_similar_regime("TRENDING", min_trades=5)
        assert result["sufficient_data"] is False
        assert result["recommendation"] == "insufficient_history"

    def test_sufficient_data_returns_stats(self, memory):
        for i in range(10):
            memory.record_trade(_make_trade(
                regime="TRENDING",
                pnl=10.0 if i < 7 else -5.0,
                pnl_pct=0.02 if i < 7 else -0.01,
                is_winner=i < 7,
            ))
        result = memory.query_similar_regime("TRENDING", min_trades=5)
        assert result["sufficient_data"] is True
        assert result["win_rate"] == pytest.approx(0.7)
        assert result["trade_count"] == 10

    def test_best_source_identified(self, memory):
        for _ in range(5):
            memory.record_trade(_make_trade(regime="RANGING", decision_source="lstm", is_winner=True))
        for _ in range(5):
            memory.record_trade(_make_trade(regime="RANGING", decision_source="rl", is_winner=False))
        result = memory.query_similar_regime("RANGING", min_trades=5)
        assert result["best_source"] == "lstm"


# ─── get_performance_summary() ───


class TestGetPerformanceSummary:
    def test_empty_db(self, memory):
        result = memory.get_performance_summary()
        assert result["total_trades"] == 0

    def test_full_summary(self, memory):
        for i in range(5):
            memory.record_trade(_make_trade(
                pnl=10.0, is_winner=True, regime="TRENDING",
            ))
        for i in range(3):
            memory.record_trade(_make_trade(
                pnl=-5.0, is_winner=False, regime="VOLATILE",
            ))
        result = memory.get_performance_summary(lookback_days=1)
        assert result["total_trades"] == 8
        assert result["total_pnl"] == pytest.approx(35.0)
        assert "regime_breakdown" in result
        assert "TRENDING" in result["regime_breakdown"]
        assert "VOLATILE" in result["regime_breakdown"]


# ─── Thread-safety verification (bug fix) ───


class TestThreadSafety:
    def test_check_same_thread_false(self, tmp_db):
        """Verify fix: connection created with check_same_thread=False."""
        mem = TradeMemory(db_path=tmp_db)
        # The connection should allow access from different threads
        errors = []

        def insert_from_thread():
            try:
                mem.record_trade(_make_trade(symbol="THREAD"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=insert_from_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert mem.get_trade_count() == 5
        mem.close()

    def test_concurrent_writes_with_lock(self, tmp_db):
        """Verify concurrent writes don't corrupt data thanks to the lock."""
        mem = TradeMemory(db_path=tmp_db)
        num_threads = 10
        trades_per_thread = 5
        errors = []

        def worker():
            try:
                for _ in range(trades_per_thread):
                    mem.record_trade(_make_trade())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0
        assert mem.get_trade_count() == num_threads * trades_per_thread
        mem.close()

    def test_concurrent_reads_and_writes(self, tmp_db):
        """Mixed concurrent reads and writes should not crash."""
        mem = TradeMemory(db_path=tmp_db)
        # Seed some data
        for _ in range(10):
            mem.record_trade(_make_trade())

        errors = []

        def reader():
            try:
                for _ in range(5):
                    mem.get_recent_trades(n=5)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for _ in range(5):
                    mem.record_trade(_make_trade())
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=reader) for _ in range(2)]
            + [threading.Thread(target=writer) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # No errors at all. This used to filter out "SQLite API misuse" and
        # call it acceptable — but that error means a read was executing on
        # the shared connection while a write was mid-statement, because the
        # readers did not take the lock. Serialising them removed the cause,
        # so the test no longer has to excuse the symptom.
        assert errors == [], f"concurrent access raised: {errors}"
        # All writer trades should have been recorded
        assert mem.get_trade_count() >= 10  # at least the seeded ones
        mem.close()


# ─── close / repr ───


class TestCloseAndRepr:
    def test_repr_contains_db_path(self, memory):
        r = repr(memory)
        assert "TradeMemory" in r
        assert "trades=" in r

    def test_close_no_error(self, tmp_db):
        mem = TradeMemory(db_path=tmp_db)
        mem.record_trade(_make_trade())

        mem.close()

        # Closing twice must stay safe — __del__ calls it again.
        mem.close()

    def test_db_file_created(self, tmp_db):
        mem = TradeMemory(db_path=tmp_db)
        assert Path(tmp_db).exists()
        mem.close()

    def test_parent_directory_created(self, tmp_path):
        nested = str(tmp_path / "subdir" / "deep" / "test.db")
        mem = TradeMemory(db_path=nested)
        assert Path(nested).parent.exists()
        mem.close()
