"""
Earnings Calendar Strategy
=============================

Trades around earnings announcements using:
- Historical earnings surprise data (beat/miss patterns)
- Pre-earnings momentum (drift into earnings)
- Post-earnings drift (PEAD — Post Earnings Announcement Drift)
- Fundamental quality scoring

Data sources used:
- Earnings dates + EPS estimates/actuals (Yahoo Finance)
- OHLCV price history
- Fundamental data (P/E, growth, margins)
- News sentiment (optional confirmation)

Patterns traded:
1. Pre-earnings drift: Buy 5 days before earnings if stock has history of beating
2. Post-earnings drift: Buy after a positive surprise, sell after a negative
3. Earnings avoidance: Exit positions before earnings to avoid gap risk

Usage:
    from strategies.examples.earnings_strategy import EarningsStrategy
    strategy = EarningsStrategy()
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.indicators.technical_indicators import TechnicalIndicators as TI
from strategies import register_strategy

if TYPE_CHECKING:  # pragma: no cover
    # Imported at runtime inside generate_signals to avoid a circular
    # import; declared here so the annotation resolves for linters and
    # type checkers instead of tripping F821.
    from shared.backtesting.backtest_engine_v2 import BacktestContext


logger = logging.getLogger(__name__)


@dataclass
class EarningsConfig:
    """Configuration for the Earnings Strategy."""

    # Pre-earnings drift
    pre_earnings_days: int = 5
    min_beat_rate: float = 0.60  # stock must beat 60% of the time historically

    # Post-earnings drift (PEAD)
    post_earnings_hold_days: int = 20
    min_surprise_pct: float = 5.0

    # Earnings avoidance (for other strategies)
    exit_before_earnings_days: int = 2

    # Fundamental quality filter
    min_market_cap: float = 2_000_000_000.0
    max_pe_ratio: float = 50.0

    # Technical confirmation
    trend_ema_length: int = 50
    require_uptrend: bool = True

    # Risk management
    stop_loss_pct: float = 5.0
    position_size_pct: float = 1.5  # smaller size for event-driven

    # Data
    min_bars: int = 60
    use_enricher: bool = True  # adds news, volume, regime


@register_strategy("earnings")
class EarningsStrategy:
    """Earnings-driven strategy trading around announcement dates.

    Uses historical beat rates and surprise magnitude to determine
    pre-earnings positioning and post-earnings drift trades.
    """

    def __init__(self, config: EarningsConfig | None = None) -> None:
        self.config = config or EarningsConfig()
        self._entry_prices: Dict[str, float] = {}
        self._entry_bars: Dict[str, int] = {}
        self._earnings_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._fundamentals_cache: Dict[str, Dict[str, Any]] = {}
        self._fetcher: Optional[Any] = None
        self._enricher = None
        if self.config.use_enricher:
            try:
                from shared.strategy_enricher import StrategyEnricher
                self._enricher = StrategyEnricher()
            except Exception as e:
                logger.debug("Enricher init: %s", e)

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "EarningsStrategy":
        config = EarningsConfig(**{
            k: v for k, v in params.items() if hasattr(EarningsConfig, k)
        })
        return cls(config)

    def _get_fetcher(self):
        if self._fetcher is None:
            from shared.data.public_data_fetcher import PublicDataFetcher
            self._fetcher = PublicDataFetcher()
        return self._fetcher

    def _get_earnings(self, symbol: str) -> List[Dict[str, Any]]:
        if symbol in self._earnings_cache:
            return self._earnings_cache[symbol]
        try:
            fetcher = self._get_fetcher()
            data = fetcher.fetch_earnings_dates(symbol)
            self._earnings_cache[symbol] = data or []
            return self._earnings_cache[symbol]
        except Exception as e:
            logger.debug("Earnings fetch failed for %s: %s", symbol, e)
            return []

    def _get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        if symbol in self._fundamentals_cache:
            return self._fundamentals_cache[symbol]
        try:
            fetcher = self._get_fetcher()
            data = fetcher.fetch_fundamentals(symbol)
            if data:
                self._fundamentals_cache[symbol] = data
                return data
        except Exception:
            pass
        return {}

    def _calculate_beat_rate(self, earnings: List[Dict[str, Any]]) -> float:
        """Calculate historical earnings beat rate."""
        beats = [e for e in earnings if e.get("surprise_pct") is not None and e["surprise_pct"] > 0]
        total = [e for e in earnings if e.get("surprise_pct") is not None]
        if not total:
            return 0.0
        return len(beats) / len(total)

    def _avg_surprise(self, earnings: List[Dict[str, Any]]) -> float:
        """Calculate average earnings surprise percentage."""
        surprises = [e["surprise_pct"] for e in earnings if e.get("surprise_pct") is not None]
        if not surprises:
            return 0.0
        return float(np.mean(surprises))

    def _days_to_next_earnings(self, earnings: List[Dict[str, Any]]) -> Optional[int]:
        """Calculate days until next earnings date."""
        now = datetime.now()
        future_dates = []
        for e in earnings:
            try:
                edate = pd.Timestamp(e["date"]).to_pydatetime()
                if hasattr(edate, 'tzinfo') and edate.tzinfo:
                    edate = edate.replace(tzinfo=None)
                if edate > now:
                    future_dates.append(edate)
            except (ValueError, TypeError):
                continue
        if not future_dates:
            return None
        nearest = min(future_dates)
        return (nearest - now).days

    def _last_surprise(self, earnings: List[Dict[str, Any]]) -> Optional[float]:
        """Get the most recent earnings surprise %."""
        now = datetime.now()
        past = []
        for e in earnings:
            try:
                edate = pd.Timestamp(e["date"]).to_pydatetime()
                if hasattr(edate, 'tzinfo') and edate.tzinfo:
                    edate = edate.replace(tzinfo=None)
                if edate <= now and e.get("surprise_pct") is not None:
                    past.append((edate, e["surprise_pct"]))
            except (ValueError, TypeError):
                continue
        if not past:
            return None
        past.sort(key=lambda x: x[0], reverse=True)
        return past[0][1]

    def generate_signals(self, ctx: "BacktestContext") -> Dict[str, int]:
        """Generate earnings-driven signals."""
        from shared.backtesting.backtest_engine_v2 import BacktestContext

        cfg = self.config
        signals: Dict[str, int] = {}

        for sym, df in ctx.bars.items():
            if len(df) < cfg.min_bars:
                signals[sym] = 0
                continue

            current_pos = ctx.positions.get(sym, 0)
            close = df["close"]
            current_price = float(close.iloc[-1])

            # Stop loss check
            if current_pos > 0 and sym in self._entry_prices:
                entry = self._entry_prices[sym]
                loss_pct = (entry - current_price) / entry * 100
                if loss_pct >= cfg.stop_loss_pct:
                    signals[sym] = 0
                    self._entry_prices.pop(sym, None)
                    self._entry_bars.pop(sym, None)
                    continue

            # Hold period check for PEAD trades
            if current_pos > 0 and sym in self._entry_bars:
                bars_held = ctx.bar_index - self._entry_bars[sym]
                if bars_held >= cfg.post_earnings_hold_days:
                    signals[sym] = 0
                    self._entry_prices.pop(sym, None)
                    self._entry_bars.pop(sym, None)
                    continue

            earnings = self._get_earnings(sym)
            if not earnings:
                signals[sym] = 1 if current_pos > 0 else 0
                continue

            beat_rate = self._calculate_beat_rate(earnings)
            days_to_next = self._days_to_next_earnings(earnings)
            last_surprise = self._last_surprise(earnings)

            # Fundamental quality filter
            fundamentals = self._get_fundamentals(sym)
            mcap = fundamentals.get("market_cap", 0) or 0
            pe = fundamentals.get("pe_ratio")

            quality_ok = mcap >= cfg.min_market_cap
            if pe is not None and pe > cfg.max_pe_ratio:
                quality_ok = False

            # Technical trend filter
            trend_ok = True
            if cfg.require_uptrend:
                ema = TI.ema(close, cfg.trend_ema_length)
                trend_ok = current_price > float(ema.iloc[-1])

            if current_pos <= 0:
                # Enricher gate: check sentiment + fundamentals + regime before entry
                enricher_ok = True
                if getattr(self, "_enricher", None):
                    enriched = self._enricher.enrich(sym, df)
                    blocked, reason = self._enricher.should_block_entry(enriched)
                    if blocked:
                        enricher_ok = False
                        logger.debug("%s: earnings entry blocked: %s", sym, reason)

                # Volume confirmation
                volume_ok = True
                if "volume" in df.columns and len(df) >= 20:
                    vol_ma = float(df["volume"].rolling(20).mean().iloc[-1])
                    if vol_ma > 0:
                        volume_ok = float(df["volume"].iloc[-1]) > vol_ma * 0.8

                # Pattern 1: Pre-earnings drift — buy before earnings if high beat rate
                if (
                    enricher_ok
                    and volume_ok
                    and days_to_next is not None
                    and 0 < days_to_next <= cfg.pre_earnings_days
                    and beat_rate >= cfg.min_beat_rate
                    and quality_ok
                    and trend_ok
                ):
                    signals[sym] = 1
                    self._entry_prices[sym] = current_price
                    self._entry_bars[sym] = ctx.bar_index
                    logger.info(
                        "%s: EARNINGS PRE-DRIFT BUY (days_to=%d, beat_rate=%.0f%%)",
                        sym, days_to_next, beat_rate * 100,
                    )
                    continue

                # Pattern 2: Post-earnings drift — buy after positive surprise
                if (
                    enricher_ok
                    and volume_ok
                    and last_surprise is not None
                    and last_surprise >= cfg.min_surprise_pct
                    and quality_ok
                    and trend_ok
                ):
                    signals[sym] = 1
                    self._entry_prices[sym] = current_price
                    self._entry_bars[sym] = ctx.bar_index
                    logger.info(
                        "%s: EARNINGS PEAD BUY (surprise=%.1f%%)",
                        sym, last_surprise,
                    )
                    continue

                signals[sym] = 0
            else:
                # Pattern 3: Earnings avoidance — exit before upcoming earnings
                if days_to_next is not None and days_to_next <= cfg.exit_before_earnings_days:
                    signals[sym] = 0
                    self._entry_prices.pop(sym, None)
                    self._entry_bars.pop(sym, None)
                    logger.info("%s: EXIT before earnings (%d days)", sym, days_to_next)
                else:
                    signals[sym] = 1

        return signals
