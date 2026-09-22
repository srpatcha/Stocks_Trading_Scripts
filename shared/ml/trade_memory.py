"""
Trade Memory — Persistent Trade Journal with Full Context
============================================================

SQLite-backed trade memory that stores every decision with:
- Market features at entry/exit
- Regime classification at time of trade
- Model predictions (LSTM, Transformer, RL, Regime)
- Actual P&L outcome
- Strategy that generated the signal

Enables the self-learning agent to:
1. Query "what worked in similar conditions?"
2. Track per-model accuracy over rolling windows
3. Detect model degradation and trigger retraining
4. Build adaptive confidence based on historical hit rate

Usage:
    memory = TradeMemory("trades.db")
    memory.record_trade(trade_record)
    similar = memory.query_similar_regime("TRENDING", lookback_days=90)
    accuracy = memory.get_model_accuracy("lstm", window=50)
    memory.get_performance_summary()
"""

from __future__ import annotations

import json
import functools
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _synchronized(method):
    """Serialise access to the shared SQLite connection.

    One sqlite3 connection is shared across threads with
    check_same_thread=False. The writers took the lock but the readers did
    not, so a read could execute on the connection while a write was mid
    statement — surfacing as "SQLite API misuse" under load. A test excused
    that as "acceptable"; it is not, it is an unserialised shared cursor.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper



@dataclass
class TradeDecisionRecord:
    """Full context of a single trade decision."""

    timestamp: str
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    entry_price: float
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0

    # Market context at entry
    regime: str = "UNKNOWN"  # TRENDING, RANGING, VOLATILE
    regime_confidence: float = 0.0

    # Feature snapshot (JSON-serialized dict of top features at entry)
    features_snapshot: str = "{}"

    # Model predictions at time of decision
    lstm_prediction: float = 0.0
    transformer_prediction: float = 0.0
    rl_action: int = 0  # -1, 0, +1
    regime_prediction: str = "UNKNOWN"
    ensemble_signal: float = 0.0
    ensemble_confidence: float = 0.0

    # Which strategy/model drove the decision
    decision_source: str = "ensemble"

    # Outcome tracking
    holding_period_bars: int = 0
    max_favorable_excursion: float = 0.0  # best unrealized P&L during trade
    max_adverse_excursion: float = 0.0  # worst unrealized P&L during trade
    is_winner: bool = False

    # Risk context
    position_size: int = 0
    risk_amount: float = 0.0
    portfolio_value_at_entry: float = 0.0

    id: Optional[int] = None


class TradeMemory:
    """SQLite-backed persistent trade journal.

    Stores complete decision context for every trade, enabling
    the self-learning agent to improve over time.

    Args:
        db_path: Path to SQLite database file.
    """

    def __init__(self, db_path: str = "trade_memory.db") -> None:
        self._db_path = db_path
        # RLock so a synchronised method can call another, and so every
        # access to the shared connection is serialised.
        self._lock = threading.RLock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        # FIX 8: SQLite WAL mode for better concurrency
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("TradeMemory initialized: %s", db_path)

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        with self._lock:
            self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL DEFAULT 0,
                pnl REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0,

                regime TEXT DEFAULT 'UNKNOWN',
                regime_confidence REAL DEFAULT 0,
                features_snapshot TEXT DEFAULT '{}',

                lstm_prediction REAL DEFAULT 0,
                transformer_prediction REAL DEFAULT 0,
                rl_action INTEGER DEFAULT 0,
                regime_prediction TEXT DEFAULT 'UNKNOWN',
                ensemble_signal REAL DEFAULT 0,
                ensemble_confidence REAL DEFAULT 0,

                decision_source TEXT DEFAULT 'ensemble',

                holding_period_bars INTEGER DEFAULT 0,
                max_favorable_excursion REAL DEFAULT 0,
                max_adverse_excursion REAL DEFAULT 0,
                is_winner INTEGER DEFAULT 0,

                position_size INTEGER DEFAULT 0,
                risk_amount REAL DEFAULT 0,
                portfolio_value_at_entry REAL DEFAULT 0,

                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prediction REAL NOT NULL,
                actual_outcome REAL NOT NULL,
                is_correct INTEGER NOT NULL,
                regime TEXT DEFAULT 'UNKNOWN',
                symbol TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_regime ON trades(regime);
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_source ON trades(decision_source);
            CREATE INDEX IF NOT EXISTS idx_model_perf_model ON model_performance(model_name);
            CREATE INDEX IF NOT EXISTS idx_model_perf_regime ON model_performance(regime);
        """)
            self._conn.commit()

    # ─── Record Trades ───

    def record_trade(self, trade: TradeDecisionRecord) -> int:
        """Store a completed trade with full context.

        Args:
            trade: TradeDecisionRecord with all fields populated.

        Returns:
            Row ID of the inserted record.
        """
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO trades (
                    timestamp, symbol, action, entry_price, exit_price, pnl, pnl_pct,
                    regime, regime_confidence, features_snapshot,
                    lstm_prediction, transformer_prediction, rl_action,
                    regime_prediction, ensemble_signal, ensemble_confidence,
                    decision_source,
                    holding_period_bars, max_favorable_excursion, max_adverse_excursion,
                    is_winner, position_size, risk_amount, portfolio_value_at_entry
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade.timestamp, trade.symbol, trade.action,
                    trade.entry_price, trade.exit_price, trade.pnl, trade.pnl_pct,
                    trade.regime, trade.regime_confidence, trade.features_snapshot,
                    trade.lstm_prediction, trade.transformer_prediction, trade.rl_action,
                    trade.regime_prediction, trade.ensemble_signal, trade.ensemble_confidence,
                    trade.decision_source,
                    trade.holding_period_bars, trade.max_favorable_excursion,
                    trade.max_adverse_excursion, int(trade.is_winner),
                    trade.position_size, trade.risk_amount, trade.portfolio_value_at_entry,
                ),
            )
            self._conn.commit()
            row_id = cursor.lastrowid
        logger.info(
            "Trade recorded [#%d]: %s %s @ %.2f → %.2f | P&L: $%.2f (%.2f%%) | regime=%s",
            row_id, trade.action, trade.symbol,
            trade.entry_price, trade.exit_price,
            trade.pnl, trade.pnl_pct * 100, trade.regime,
        )
        return row_id

    def record_model_prediction(
        self,
        model_name: str,
        prediction: float,
        actual_outcome: float,
        regime: str = "UNKNOWN",
        symbol: str = "",
    ) -> None:
        """Record a model's prediction vs actual outcome for accuracy tracking.

        Args:
            model_name: Name of the model (e.g., "lstm", "transformer", "rl").
            prediction: What the model predicted (direction or return).
            actual_outcome: What actually happened.
            regime: Market regime at time of prediction.
            symbol: Symbol being predicted.
        """
        is_correct = int(np.sign(prediction) == np.sign(actual_outcome))
        with self._lock:
            self._conn.execute(
                """INSERT INTO model_performance
                   (timestamp, model_name, prediction, actual_outcome, is_correct, regime, symbol)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(), model_name, prediction,
                 actual_outcome, is_correct, regime, symbol),
            )
            self._conn.commit()

    # ─── Query History ───

    @_synchronized
    def query_similar_regime(
        self,
        regime: str,
        lookback_days: int = 90,
        min_trades: int = 5,
    ) -> Dict[str, Any]:
        """Find what worked in similar market regimes.

        Args:
            regime: Target regime ("TRENDING", "RANGING", "VOLATILE").
            lookback_days: How far back to look.
            min_trades: Minimum trades needed for statistical significance.

        Returns:
            Dict with win_rate, avg_pnl, best_source, trade_count, etc.
        """
        cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        rows = self._conn.execute(
            """SELECT action, pnl, pnl_pct, decision_source, is_winner,
                      ensemble_confidence, holding_period_bars
               FROM trades
               WHERE regime = ? AND timestamp >= ?
               ORDER BY timestamp DESC""",
            (regime, cutoff),
        ).fetchall()

        if len(rows) < min_trades:
            return {
                "regime": regime,
                "trade_count": len(rows),
                "sufficient_data": False,
                "recommendation": "insufficient_history",
            }

        wins = sum(1 for r in rows if r["is_winner"])
        total_pnl = sum(r["pnl"] for r in rows)
        avg_pnl_pct = np.mean([r["pnl_pct"] for r in rows])

        # Best decision source for this regime
        source_stats: Dict[str, Dict[str, float]] = {}
        for r in rows:
            src = r["decision_source"]
            if src not in source_stats:
                source_stats[src] = {"wins": 0, "total": 0, "pnl": 0.0}
            source_stats[src]["total"] += 1
            source_stats[src]["pnl"] += r["pnl"]
            if r["is_winner"]:
                source_stats[src]["wins"] += 1

        best_source = max(
            source_stats.items(),
            key=lambda x: x[1]["wins"] / max(x[1]["total"], 1),
        )

        # Optimal holding period
        winner_rows = [r for r in rows if r["is_winner"]]
        avg_winning_hold = (
            np.mean([r["holding_period_bars"] for r in winner_rows])
            if winner_rows else 0
        )

        return {
            "regime": regime,
            "trade_count": len(rows),
            "sufficient_data": True,
            "win_rate": wins / len(rows),
            "total_pnl": total_pnl,
            "avg_pnl_pct": float(avg_pnl_pct),
            "best_source": best_source[0],
            "best_source_win_rate": best_source[1]["wins"] / max(best_source[1]["total"], 1),
            "source_breakdown": {
                k: {"win_rate": v["wins"] / max(v["total"], 1), "total_pnl": v["pnl"], "trades": v["total"]}
                for k, v in source_stats.items()
            },
            "avg_winning_hold_period": float(avg_winning_hold),
            "recommendation": "trade" if wins / len(rows) > 0.45 else "caution",
        }

    @_synchronized
    def get_model_accuracy(
        self,
        model_name: str,
        window: int = 50,
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get rolling accuracy for a specific model.

        Args:
            model_name: Model to check ("lstm", "transformer", "rl", "ensemble").
            window: Number of recent predictions to evaluate.
            regime: Optional regime filter.

        Returns:
            Dict with accuracy, total_predictions, recent_trend, etc.
        """
        query = "SELECT is_correct, prediction, actual_outcome FROM model_performance WHERE model_name = ?"
        params: list = [model_name]

        if regime:
            query += " AND regime = ?"
            params.append(regime)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(window)

        rows = self._conn.execute(query, params).fetchall()

        if not rows:
            return {"model": model_name, "accuracy": 0.0, "total_predictions": 0, "needs_retrain": True}

        correct = sum(r["is_correct"] for r in rows)
        accuracy = correct / len(rows)

        # Trend: compare first half vs second half
        half = len(rows) // 2
        if half > 0:
            recent_acc = sum(r["is_correct"] for r in rows[:half]) / half
            older_acc = sum(r["is_correct"] for r in rows[half:]) / max(len(rows) - half, 1)
            trend = recent_acc - older_acc  # positive = improving, negative = degrading
        else:
            trend = 0.0

        return {
            "model": model_name,
            "accuracy": accuracy,
            "total_predictions": len(rows),
            "recent_accuracy": recent_acc if half > 0 else accuracy,
            "trend": trend,
            "needs_retrain": accuracy < 0.45 or trend < -0.1,
        }

    @_synchronized
    def get_all_model_accuracies(self, window: int = 50) -> Dict[str, Dict[str, float]]:
        """Get accuracy for all tracked models.

        Returns:
            Dict mapping model_name to accuracy stats.
        """
        models = self._conn.execute(
            "SELECT DISTINCT model_name FROM model_performance"
        ).fetchall()

        return {
            row["model_name"]: self.get_model_accuracy(row["model_name"], window)
            for row in models
        }

    @_synchronized
    def get_performance_summary(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Get comprehensive performance summary.

        Args:
            lookback_days: Number of days to summarize.

        Returns:
            Dict with overall stats, per-regime breakdown, model accuracies.
        """
        cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()

        trades = self._conn.execute(
            "SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC",
            (cutoff,),
        ).fetchall()

        if not trades:
            return {"period_days": lookback_days, "total_trades": 0, "message": "no trades in period"}

        total_pnl = sum(t["pnl"] for t in trades)
        wins = sum(1 for t in trades if t["is_winner"])

        # Per-regime breakdown
        regime_stats: Dict[str, Dict[str, Any]] = {}
        for t in trades:
            regime = t["regime"]
            if regime not in regime_stats:
                regime_stats[regime] = {"trades": 0, "wins": 0, "pnl": 0.0}
            regime_stats[regime]["trades"] += 1
            regime_stats[regime]["pnl"] += t["pnl"]
            if t["is_winner"]:
                regime_stats[regime]["wins"] += 1

        for v in regime_stats.values():
            v["win_rate"] = v["wins"] / max(v["trades"], 1)

        # Equity curve
        equity = []
        running = 0.0
        for t in reversed(trades):
            running += t["pnl"]
            equity.append(running)

        max_dd = 0.0
        peak = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd

        return {
            "period_days": lookback_days,
            "total_trades": len(trades),
            "win_rate": wins / len(trades),
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(trades),
            "max_drawdown": max_dd,
            "regime_breakdown": regime_stats,
            "model_accuracies": self.get_all_model_accuracies(),
        }

    @_synchronized
    def get_recent_trades(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get the N most recent trades.

        Args:
            n: Number of trades to return.

        Returns:
            List of trade dicts.
        """
        rows = self._conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    @_synchronized
    def get_trade_count(self) -> int:
        """Total number of trades in memory."""
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM trades").fetchone()
        return row["cnt"]

    # ─── Cleanup ───

    def cleanup_old_trades(self, max_age_days: int = 365, max_records: int = 10000) -> None:
        """Remove trades older than max_age_days or keep only most recent max_records.

        Args:
            max_age_days: Delete trades older than this many days.
            max_records: Cap total records to this number, keeping most recent.
        """
        with self._lock:
            cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
            self._conn.execute("DELETE FROM trades WHERE timestamp < ?", (cutoff,))
            # Also cap total records
            count = self._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            if count > max_records:
                self._conn.execute(
                    "DELETE FROM trades WHERE id NOT IN (SELECT id FROM trades ORDER BY id DESC LIMIT ?)",
                    (max_records,),
                )
            self._conn.commit()
        logger.info("Trade memory cleanup: max_age=%dd, max_records=%d", max_age_days, max_records)

    @_synchronized
    def close(self) -> None:
        """Close database connection."""
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        count = self.get_trade_count()
        return f"TradeMemory(db='{self._db_path}', trades={count})"
