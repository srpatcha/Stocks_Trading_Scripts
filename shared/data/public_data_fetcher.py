"""
Public Data Fetcher
===================

Fetches market data and news from free public sources (Yahoo Finance, RSS feeds)
with optional in-memory caching. No brokerage account required.

Used by:
- LiveRunner (OHLCV + market-hours detection)
- NewsSentimentAnalyzer (news headlines)
- AI-webhook endpoint (OHLCV for agent decisions)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from shared.utils.paths import stocks_plugin_subdir

logger = logging.getLogger(__name__)

_CACHE_DIR = stocks_plugin_subdir("cache", create=False)

# ─── NYSE market hours (Eastern) ─────────────────────────────────────────────
_MARKET_OPEN_HOUR_ET = 9
_MARKET_OPEN_MINUTE_ET = 30
_MARKET_CLOSE_HOUR_ET = 16
_MARKET_CLOSE_MINUTE_ET = 0


class PublicDataFetcher:
    """Fetches OHLCV data and news headlines from public sources.

    Args:
        cache_enabled: If True, caches fetched data for ``cache_ttl_seconds``.
        cache_ttl_seconds: How long to keep cached data (default 5 min).
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 300,
        cache_persist_path: Optional[str] = None,
    ) -> None:
        self._cache_enabled = cache_enabled
        self._cache_ttl = cache_ttl_seconds
        self._ohlcv_cache: Dict[str, Dict[str, Any]] = {}
        self._news_cache: Dict[str, Dict[str, Any]] = {}

        # Fix 9: yfinance rate limiting — minimum 0.5s between calls
        self._last_fetch_time: float = 0.0
        self._min_fetch_interval: float = 0.5

        # FIX 10: Data fetcher circuit breaker
        self._consecutive_failures: int = 0
        self._fundamentals_failures: int = 0
        self._max_failures: int = 5

        # Fundamentals cache (separate from OHLCV)
        self._fundamentals_cache: Dict[str, Dict[str, Any]] = {}
        self._fundamentals_ttl: int = 3600  # 1 hour for fundamentals (changes rarely)

        # Fix 18: cache persistence path
        if cache_persist_path is None:
            self._cache_persist_path = os.path.join(_CACHE_DIR, "data_cache.json")
        else:
            self._cache_persist_path = cache_persist_path
        self._load_cache_from_disk()

    # ─── OHLCV ───────────────────────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g., "AAPL").
            period: History window — yfinance period string ("1d","5d","1mo",
                    "3mo","6mo","1y","2y","5y","10y","ytd","max").
            interval: Bar size — yfinance interval string ("1m","2m","5m",
                      "15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo").

        Returns:
            DataFrame with columns [open, high, low, close, volume] and a
            DatetimeIndex named "date", or None on failure.
        """
        cache_key = f"{symbol}:{period}:{interval}"
        if self._cache_enabled:
            cached = self._ohlcv_cache.get(cache_key)
            if cached and time.time() - cached["ts"] < self._cache_ttl:
                logger.debug("OHLCV cache hit: %s", cache_key)
                return cached["df"]

        try:
            import yfinance as yf

            # FIX 10: Circuit breaker — if too many consecutive failures, use cached data
            if self._consecutive_failures >= self._max_failures:
                logger.critical(
                    "Data fetcher circuit breaker OPEN: %d consecutive failures for %s. "
                    "Using cached data if available.",
                    self._consecutive_failures, symbol,
                )
                cached_fallback = self._ohlcv_cache.get(cache_key)
                if cached_fallback:
                    return cached_fallback["df"]
                # Exponential backoff before retry
                backoff = min(2 ** self._consecutive_failures, 60)
                time.sleep(backoff)

            # Fix 9: enforce minimum gap between yfinance API calls
            self._rate_limit_fetch()

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)

            if df is None or df.empty:
                logger.warning("No OHLCV data for %s (period=%s, interval=%s)", symbol, period, interval)
                self._consecutive_failures += 1
                return None

            # Normalise column names to lowercase
            df.columns = [c.lower() for c in df.columns]

            # Keep only OHLCV columns
            keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            df = df[keep].copy()

            # Ensure DatetimeIndex named "date"
            df.index.name = "date"
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)

            df = df.dropna(subset=["close"])

            # FIX 11: Validate data before returning
            df = self._validate_data(df, symbol)

            if self._cache_enabled:
                self._ohlcv_cache[cache_key] = {"df": df, "ts": time.time()}

            # FIX 10: Reset failure counter on success
            self._consecutive_failures = 0

            logger.debug("Fetched %d bars for %s", len(df), symbol)
            return df

        except Exception as e:
            logger.error("Failed to fetch OHLCV for %s: %s", symbol, e)
            # FIX 10: Increment failure counter, use cached data on failure
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_failures:
                logger.critical(
                    "Data fetcher: %d consecutive failures. Check data source connectivity.",
                    self._consecutive_failures,
                )
            # Return stale cached data if available
            cached_fallback = self._ohlcv_cache.get(cache_key)
            if cached_fallback:
                logger.warning("Returning stale cached data for %s", symbol)
                return cached_fallback["df"]
            return None

    def fetch_latest_price(self, symbol: str) -> float:
        """Return the most recent close price for a symbol.

        Returns 0.0 on failure so callers can check ``price > 0``
        before using the value in calculations.

        .. warning::
            A return value of 0.0 means the price could not be fetched.
            Callers MUST guard against 0.0 to avoid division-by-zero
            or placing $0 orders::

                price = fetcher.fetch_latest_price("AAPL")
                if price <= 0:
                    logger.warning("Skipping — could not fetch price for AAPL")
                    return
        """
        try:
            import yfinance as yf

            # Fix 9: enforce minimum gap between yfinance API calls
            self._rate_limit_fetch()

            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
            if price and price > 0:
                return float(price)
            # Fallback: last bar from 1-day history
            df = self.fetch_ohlcv(symbol, period="5d", interval="1d")
            if df is not None and not df.empty:
                return float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning("fetch_latest_price(%s) failed: %s", symbol, e)
        return 0.0

    # ─── FIX 11: Data Validation ─────────────────────────────────────────────

    def _validate_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate OHLCV data and drop invalid rows.

        Checks:
        - Prices must be > 0
        - Volume must be >= 0
        - Dates must be sorted
        - No duplicate timestamps

        Args:
            df: Raw OHLCV DataFrame.
            symbol: Symbol for logging.

        Returns:
            Cleaned DataFrame.
        """
        initial_len = len(df)

        # Drop rows with non-positive prices
        price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        for col in price_cols:
            bad_mask = df[col] <= 0
            if bad_mask.any():
                logger.warning(
                    "Data validation [%s]: dropping %d rows with %s <= 0",
                    symbol, bad_mask.sum(), col,
                )
                df = df[~bad_mask]

        # Drop rows with negative volume
        if "volume" in df.columns:
            bad_vol = df["volume"] < 0
            if bad_vol.any():
                logger.warning(
                    "Data validation [%s]: dropping %d rows with negative volume",
                    symbol, bad_vol.sum(),
                )
                df = df[~bad_vol]

        # Remove duplicate timestamps
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            logger.warning(
                "Data validation [%s]: removing %d duplicate timestamps",
                symbol, dup_count,
            )
            df = df[~df.index.duplicated(keep="last")]

        # Ensure dates are sorted
        if not df.index.is_monotonic_increasing:
            logger.warning("Data validation [%s]: sorting unsorted dates", symbol)
            df = df.sort_index()

        if len(df) < initial_len:
            logger.info(
                "Data validation [%s]: %d → %d rows after cleanup",
                symbol, initial_len, len(df),
            )

        return df

    # ─── Rate Limiting (Fix 9) ─────────────────────────────────────────────

    def _rate_limit_fetch(self) -> None:
        """Enforce minimum 0.5s gap between yfinance API calls."""
        now = time.time()
        elapsed = now - self._last_fetch_time
        if elapsed < self._min_fetch_interval:
            time.sleep(self._min_fetch_interval - elapsed)
        self._last_fetch_time = time.time()

    # ─── Cache Persistence (Fix 18) ────────────────────────────────────────

    def save_cache_to_disk(self) -> None:
        """Persist in-memory cache metadata to disk for session recovery.

        Only saves cache keys and timestamps — actual DataFrames are
        not serialised because pickle is fragile across versions.  On
        reload the cache will be populated lazily.
        """
        if not self._cache_enabled:
            return
        try:
            os.makedirs(os.path.dirname(self._cache_persist_path), exist_ok=True)
            meta = {
                "ohlcv_keys": {
                    k: {"ts": v["ts"]} for k, v in self._ohlcv_cache.items()
                },
                "news_keys": {
                    k: {"ts": v["ts"]} for k, v in self._news_cache.items()
                },
                "saved_at": time.time(),
            }
            with open(self._cache_persist_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            logger.info("Data cache metadata saved to %s", self._cache_persist_path)
        except Exception as e:
            logger.warning("Failed to save cache metadata: %s", e)

    def _load_cache_from_disk(self) -> None:
        """Reload cache metadata from disk on startup.

        Fresh data will be fetched lazily on the next access.
        """
        if not self._cache_enabled or not os.path.exists(self._cache_persist_path):
            return
        try:
            with open(self._cache_persist_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            saved_at = meta.get("saved_at", 0)
            age = time.time() - saved_at
            logger.info(
                "Cache metadata loaded from %s (age=%.0fs, ohlcv_keys=%d, news_keys=%d)",
                self._cache_persist_path,
                age,
                len(meta.get("ohlcv_keys", {})),
                len(meta.get("news_keys", {})),
            )
        except Exception as e:
            logger.debug("Failed to load cache metadata: %s", e)

    # ─── Fundamental Data ─────────────────────────────────────────────────────

    def fetch_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch fundamental data from Yahoo Finance.

        Returns dict with: pe_ratio, forward_pe, peg_ratio, price_to_book,
        dividend_yield, market_cap, revenue, earnings_growth, profit_margin,
        debt_to_equity, current_ratio, book_value, sector, industry.

        Uses a separate failure counter from OHLCV to prevent fundamental
        failures from blocking price data fetching.

        Args:
            symbol: Ticker symbol (e.g., "AAPL").

        Returns:
            Dict of fundamental data, or None on failure.
        """
        cache_key = f"fundamentals:{symbol}"
        if self._cache_enabled:
            cached = self._fundamentals_cache.get(cache_key)
            if cached and time.time() - cached["ts"] < self._fundamentals_ttl:
                logger.debug("Fundamentals cache hit: %s", symbol)
                return cached["data"]

        try:
            import yfinance as yf

            if self._fundamentals_failures >= self._max_failures:
                logger.critical(
                    "Fundamentals circuit breaker OPEN: %s", symbol
                )
                cached_fallback = self._fundamentals_cache.get(cache_key)
                if cached_fallback:
                    return cached_fallback["data"]
                return None

            self._rate_limit_fetch()

            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            fundamentals: Dict[str, Any] = {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "market_cap": info.get("marketCap"),
                "revenue": info.get("totalRevenue"),
                "earnings_growth": info.get("earningsGrowth"),
                "profit_margin": info.get("profitMargins"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "book_value": info.get("bookValue"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }

            # Try to get institutional ownership percentage
            try:
                inst_holders = ticker.institutional_holders
                if inst_holders is not None and not inst_holders.empty:
                    fundamentals["institutional_pct"] = float(
                        inst_holders["pctHeld"].sum() * 100
                    ) if "pctHeld" in inst_holders.columns else None
                else:
                    fundamentals["institutional_pct"] = None
            except Exception:
                fundamentals["institutional_pct"] = None

            if self._cache_enabled:
                self._fundamentals_cache[cache_key] = {"data": fundamentals, "ts": time.time()}

            self._fundamentals_failures = 0
            logger.debug("Fetched fundamentals for %s", symbol)
            return fundamentals

        except Exception as e:
            logger.error("Failed to fetch fundamentals for %s: %s", symbol, e)
            self._fundamentals_failures += 1
            cached_fallback = self._fundamentals_cache.get(cache_key)
            if cached_fallback:
                logger.warning("Returning stale cached fundamentals for %s", symbol)
                return cached_fallback["data"]
            return None

    def fetch_earnings_dates(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch earnings dates and surprise data from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g., "AAPL").

        Returns:
            List of dicts with keys: date, eps_estimate, eps_actual, surprise_pct.
            Returns empty list on failure.
        """
        cache_key = f"earnings:{symbol}"
        if self._cache_enabled:
            cached = self._fundamentals_cache.get(cache_key)
            if cached and time.time() - cached["ts"] < self._cache_ttl:
                logger.debug("Earnings cache hit: %s", symbol)
                return cached["data"]

        try:
            import yfinance as yf

            if self._fundamentals_failures >= self._max_failures:
                logger.critical(
                    "Fundamentals circuit breaker OPEN for earnings: %s", symbol
                )
                cached_fallback = self._fundamentals_cache.get(cache_key)
                if cached_fallback:
                    return cached_fallback["data"]
                return []

            self._rate_limit_fetch()

            ticker = yf.Ticker(symbol)
            earnings_dates = ticker.earnings_dates

            if earnings_dates is None or earnings_dates.empty:
                return []

            results: List[Dict[str, Any]] = []
            for idx, row in earnings_dates.iterrows():
                entry: Dict[str, Any] = {
                    "date": str(idx),
                    "eps_estimate": row.get("EPS Estimate"),
                    "eps_actual": row.get("Reported EPS"),
                    "surprise_pct": row.get("Surprise(%)"),
                }
                results.append(entry)

            if self._cache_enabled:
                self._fundamentals_cache[cache_key] = {"data": results, "ts": time.time()}

            self._fundamentals_failures = 0
            logger.debug("Fetched %d earnings dates for %s", len(results), symbol)
            return results

        except Exception as e:
            logger.error("Failed to fetch earnings dates for %s: %s", symbol, e)
            self._fundamentals_failures += 1
            return []

    # ─── News Headlines ───────────────────────────────────────────────────────

    def fetch_news_headlines(
        self,
        symbol: str,
        max_items: int = 20,
    ) -> List[Dict[str, str]]:
        """Fetch news headlines for a symbol.

        Tries Yahoo Finance news first, falls back to Google News RSS,
        then Reuters RSS.

        Args:
            symbol: Ticker symbol (e.g., "AAPL").
            max_items: Maximum number of headlines to return.

        Returns:
            List of dicts with keys: "title", "source", "published".
        """
        cache_key = f"news:{symbol}"
        if self._cache_enabled:
            cached = self._news_cache.get(cache_key)
            if cached and time.time() - cached["ts"] < self._cache_ttl:
                logger.debug("News cache hit: %s", symbol)
                return cached["items"][:max_items]

        headlines: List[Dict[str, str]] = []

        # 1. Yahoo Finance news via yfinance
        headlines.extend(self._fetch_yfinance_news(symbol, max_items))

        # 2. Google News RSS fallback
        if len(headlines) < max_items:
            headlines.extend(self._fetch_google_news_rss(symbol, max_items - len(headlines)))

        # Deduplicate by title
        seen: set = set()
        unique: List[Dict[str, str]] = []
        for h in headlines:
            title = h.get("title", "")
            if title and title not in seen:
                seen.add(title)
                unique.append(h)

        if self._cache_enabled:
            self._news_cache[cache_key] = {"items": unique, "ts": time.time()}

        logger.debug("Fetched %d headlines for %s", len(unique), symbol)
        return unique[:max_items]

    def _fetch_yfinance_news(self, symbol: str, limit: int) -> List[Dict[str, str]]:
        """Fetch news from Yahoo Finance via yfinance."""
        try:
            import yfinance as yf
            self._rate_limit_fetch()
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            results = []
            for item in news[:limit]:
                title = item.get("title") or item.get("content", {}).get("title", "")
                source = (item.get("publisher") or
                          item.get("content", {}).get("provider", {}).get("displayName", "Yahoo Finance"))
                published = ""
                ptime = item.get("providerPublishTime") or item.get("content", {}).get("pubDate", "")
                if isinstance(ptime, (int, float)):
                    published = datetime.fromtimestamp(ptime, tz=timezone.utc).isoformat()
                elif isinstance(ptime, str):
                    published = ptime
                if title:
                    results.append({"title": title, "source": source, "published": published})
            return results
        except Exception as e:
            logger.debug("yfinance news fetch failed for %s: %s", symbol, e)
            return []

    def _fetch_google_news_rss(self, symbol: str, limit: int) -> List[Dict[str, str]]:
        """Fetch headlines from Google News RSS for the symbol."""
        try:
            import feedparser
            url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            results = []
            for entry in feed.entries[:limit]:
                title = entry.get("title", "")
                source = entry.get("source", {}).get("title", "Google News") if hasattr(entry.get("source", {}), "get") else "Google News"
                published = entry.get("published", "")
                if title:
                    results.append({"title": title, "source": source, "published": published})
            return results
        except Exception as e:
            logger.debug("Google News RSS fetch failed for %s: %s", symbol, e)
            return []

    # ─── Market Hours ─────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        """Return True if the US equity market (NYSE/NASDAQ) is currently open.

        Uses system local time converted to Eastern Time. Does not account for
        market holidays — use a library like ``trading_calendars`` for holiday
        awareness if needed.
        """
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            try:
                from backports.zoneinfo import ZoneInfo  # type: ignore[no-reattr]
            except ImportError:
                # Best-effort fallback using UTC offset for ET (UTC-5 / UTC-4)
                return self._is_market_open_utc_fallback()

        now_et = datetime.now(ZoneInfo("America/New_York"))
        return self._check_market_hours(now_et)

    def _check_market_hours(self, now_et: datetime) -> bool:
        """Check if ``now_et`` falls within regular NYSE trading hours."""
        # Weekends
        if now_et.weekday() >= 5:
            return False

        open_time = now_et.replace(
            hour=_MARKET_OPEN_HOUR_ET,
            minute=_MARKET_OPEN_MINUTE_ET,
            second=0,
            microsecond=0,
        )
        close_time = now_et.replace(
            hour=_MARKET_CLOSE_HOUR_ET,
            minute=_MARKET_CLOSE_MINUTE_ET,
            second=0,
            microsecond=0,
        )
        return open_time <= now_et < close_time

    def _is_market_open_utc_fallback(self) -> bool:
        """Fallback market-hours check using UTC offset (ET = UTC-5 during EST, UTC-4 during EDT)."""
        now_utc = datetime.now(timezone.utc)
        # Approximate: use UTC-5 (EST). Slightly wrong during EDT but acceptable fallback.
        et_hour = (now_utc.hour - 5) % 24
        et_weekday = now_utc.weekday()
        # Adjust weekday if crossing midnight
        if now_utc.hour < 5:
            et_weekday = (et_weekday - 1) % 7

        if et_weekday >= 5:
            return False

        et_minutes = et_hour * 60 + now_utc.minute
        market_open = _MARKET_OPEN_HOUR_ET * 60 + _MARKET_OPEN_MINUTE_ET
        market_close = _MARKET_CLOSE_HOUR_ET * 60 + _MARKET_CLOSE_MINUTE_ET
        return market_open <= et_minutes < market_close

    def clear_cache(self) -> None:
        """Clear all in-memory caches."""
        self._ohlcv_cache.clear()
        self._news_cache.clear()
        self._fundamentals_cache.clear()
        logger.debug("Data cache cleared")

    def get_data_health(self) -> dict:
        """Return data source health status for monitoring.

        Returns dict with:
        - ohlcv_failures: consecutive OHLCV fetch failures
        - fundamentals_failures: consecutive fundamentals fetch failures
        - circuit_breaker_open: whether circuit breaker has tripped
        - cache_entries: number of cached items
        - last_fetch_age_s: seconds since last successful fetch
        """
        now = time.time()
        return {
            "ohlcv_failures": self._consecutive_failures,
            "fundamentals_failures": self._fundamentals_failures,
            "circuit_breaker_open": self._consecutive_failures >= self._max_failures,
            "fundamentals_cb_open": self._fundamentals_failures >= self._max_failures,
            "ohlcv_cache_entries": len(self._ohlcv_cache),
            "fundamentals_cache_entries": len(self._fundamentals_cache),
            "last_fetch_age_s": round(now - self._last_fetch_time, 1) if self._last_fetch_time > 0 else -1,
            "rate_limit_interval_s": self._min_fetch_interval,
        }

    def __repr__(self) -> str:
        return (
            f"PublicDataFetcher(cache={'on' if self._cache_enabled else 'off'}, "
            f"ttl={self._cache_ttl}s)"
        )
