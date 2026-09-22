"""
Market Data Cache with SQLite Backend
=======================================

Caches historical OHLCV data locally to avoid re-fetching from IB
on every run. Uses SQLite (zero-config, ships with Python).

DB location: ~/.stocks_plugin/cache/market_data.db

Usage:
    cache = MarketDataCache()
    cache.store_bars("AAPL", "1 day", df)
    cached = cache.get_bars("AAPL", "1 day", "2024-01-01", "2024-12-31")
    stats = cache.get_cache_stats()
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from shared.utils.paths import stocks_plugin_subdir

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = stocks_plugin_subdir("cache", create=False)
_DEFAULT_DB_FILE = "market_data.db"


class MarketDataCache:
    """SQLite-backed cache for historical market data.

    Thread-safe via threading.Lock. Auto-creates DB directory
    and tables on first use.

    Args:
        db_path: Path to SQLite database file. If None, uses
            ~/.stocks_plugin/cache/market_data.db
    """

    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS bars (
            symbol      TEXT    NOT NULL,
            bar_size    TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            open        REAL    NOT NULL,
            high        REAL    NOT NULL,
            low         REAL    NOT NULL,
            close       REAL    NOT NULL,
            volume      REAL    NOT NULL DEFAULT 0,
            fetched_at  TEXT    NOT NULL,
            PRIMARY KEY (symbol, bar_size, date)
        )
    """

    _CREATE_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_bars_lookup
        ON bars (symbol, bar_size, date)
    """

    _CREATE_META_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS fetch_metadata (
            symbol      TEXT NOT NULL,
            bar_size    TEXT NOT NULL,
            last_fetch  TEXT NOT NULL,
            row_count   INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (symbol, bar_size)
        )
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_dir = Path(_DEFAULT_DB_DIR)
            db_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = str(db_dir / _DEFAULT_DB_FILE)
        else:
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = db_path

        self._lock = threading.Lock()
        self._init_db()
        logger.info("MarketDataCache initialized: %s", self._db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new connection (SQLite connections are not thread-safe)."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        """Create tables and indices if they don't exist."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(self._CREATE_TABLE_SQL)
                conn.execute(self._CREATE_INDEX_SQL)
                conn.execute(self._CREATE_META_TABLE_SQL)
                conn.commit()
            finally:
                conn.close()

    def store_bars(self, symbol: str, bar_size: str, df: pd.DataFrame) -> int:
        """Store OHLCV bars into the cache (upsert).

        Args:
            symbol: Ticker symbol (e.g., "AAPL").
            bar_size: Bar size string (e.g., "1 day").
            df: DataFrame with columns: date/index, open, high, low, close, volume.

        Returns:
            Number of rows stored.
        """
        if df.empty:
            return 0

        df = df.copy()

        if "date" not in df.columns and df.index.name in ("date", "datetime"):
            df = df.reset_index()
        elif "date" not in df.columns and "datetime" in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)

        df.columns = [c.strip().lower() for c in df.columns]

        required = {"date", "open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        if "volume" not in df.columns:
            df["volume"] = 0

        now = datetime.utcnow().isoformat()
        rows = []
        for _, row in df.iterrows():
            rows.append((
                symbol,
                bar_size,
                str(row["date"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row.get("volume", 0)),
                now,
            ))

        with self._lock:
            conn = self._get_connection()
            try:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO bars
                        (symbol, bar_size, date, open, high, low, close, volume, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO fetch_metadata
                        (symbol, bar_size, last_fetch, row_count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (symbol, bar_size, now, len(rows)),
                )
                conn.commit()
                logger.info(
                    "Cached %d bars for %s [%s]", len(rows), symbol, bar_size
                )
                return len(rows)
            finally:
                conn.close()

    def get_bars(
        self,
        symbol: str,
        bar_size: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve cached bars.

        Args:
            symbol: Ticker symbol.
            bar_size: Bar size string.
            start_date: Optional start date filter (inclusive).
            end_date: Optional end date filter (inclusive).

        Returns:
            DataFrame with OHLCV data, or None if no cached data.
        """
        query = "SELECT date, open, high, low, close, volume FROM bars WHERE symbol = ? AND bar_size = ?"
        params: list = [symbol, bar_size]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        with self._lock:
            conn = self._get_connection()
            try:
                df = pd.read_sql_query(query, conn, params=params)
            finally:
                conn.close()

        if df.empty:
            logger.debug("Cache miss: %s [%s]", symbol, bar_size)
            return None

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        logger.info("Cache hit: %d bars for %s [%s]", len(df), symbol, bar_size)
        return df

    def is_stale(
        self,
        symbol: str,
        bar_size: str,
        max_age_hours: float = 24.0,
    ) -> bool:
        """Check if cached data is stale.

        Args:
            symbol: Ticker symbol.
            bar_size: Bar size string.
            max_age_hours: Maximum age in hours before data is considered stale.

        Returns:
            True if data is stale or missing, False if fresh.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT last_fetch FROM fetch_metadata WHERE symbol = ? AND bar_size = ?",
                    (symbol, bar_size),
                )
                row = cursor.fetchone()
            finally:
                conn.close()

        if row is None:
            return True

        last_fetch = datetime.fromisoformat(row[0])
        age = datetime.utcnow() - last_fetch
        is_old = age > timedelta(hours=max_age_hours)

        if is_old:
            logger.debug(
                "Cache stale for %s [%s]: age=%.1f hours (max=%.1f)",
                symbol, bar_size, age.total_seconds() / 3600, max_age_hours,
            )
        return is_old

    def clear(self, symbol: Optional[str] = None) -> int:
        """Clear cached data.

        Args:
            symbol: If provided, clear only this symbol. Otherwise clear all.

        Returns:
            Number of rows deleted.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                if symbol:
                    cursor = conn.execute(
                        "DELETE FROM bars WHERE symbol = ?", (symbol,)
                    )
                    conn.execute(
                        "DELETE FROM fetch_metadata WHERE symbol = ?", (symbol,)
                    )
                else:
                    cursor = conn.execute("DELETE FROM bars")
                    conn.execute("DELETE FROM fetch_metadata")
                conn.commit()
                deleted = cursor.rowcount
                logger.info(
                    "Cache cleared: %d rows%s",
                    deleted,
                    f" for {symbol}" if symbol else " (all)",
                )
                return deleted
            finally:
                conn.close()

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with row_count, symbols, db_size_mb, and per-symbol details.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                total_rows = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
                symbols = conn.execute(
                    "SELECT DISTINCT symbol FROM bars ORDER BY symbol"
                ).fetchall()
                symbol_list = [s[0] for s in symbols]

                per_symbol = {}
                for sym in symbol_list:
                    row = conn.execute(
                        "SELECT COUNT(*), MIN(date), MAX(date) FROM bars WHERE symbol = ?",
                        (sym,),
                    ).fetchone()
                    per_symbol[sym] = {
                        "rows": row[0],
                        "earliest": row[1],
                        "latest": row[2],
                    }
            finally:
                conn.close()

        db_size = os.path.getsize(self._db_path) / (1024 * 1024)

        stats = {
            "total_rows": total_rows,
            "symbols_cached": len(symbol_list),
            "symbols": symbol_list,
            "db_size_mb": round(db_size, 2),
            "db_path": self._db_path,
            "per_symbol": per_symbol,
        }

        logger.info(
            "Cache stats: %d rows, %d symbols, %.2f MB",
            total_rows, len(symbol_list), db_size,
        )
        return stats
