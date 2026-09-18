"""
News Sentiment Strategy
=========================

Trades based on news sentiment combined with technical confirmation.
Uses NewsSentimentAnalyzer (FinBERT → VADER → keyword fallback) to
score headlines, then confirms with price trend and volume.

Data sources used:
- News headlines (Yahoo Finance + Google News RSS)
- Sentiment scoring (FinBERT/VADER/keyword)
- OHLCV price history (trend + volume confirmation)
- Fundamental data (market cap filter)

Entry: Bullish sentiment (score > 0.3) + price above 20-EMA + volume above average
Exit:  Sentiment turns bearish OR trailing stop hit

Usage:
    from strategies.examples.sentiment_strategy import SentimentStrategy
    strategy = SentimentStrategy()
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

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
class SentimentConfig:
    """Configuration for the Sentiment Strategy."""

    # Sentiment thresholds
    bullish_threshold: float = 0.3
    bearish_threshold: float = -0.2
    min_confidence: float = 0.3
    min_headlines: int = 3

    # Technical confirmation
    trend_ema_length: int = 20
    volume_ma_length: int = 20
    require_trend_confirm: bool = True
    require_volume_confirm: bool = True

    # Risk management
    stop_loss_atr_mult: float = 2.0
    trailing_stop: bool = True
    atr_length: int = 14

    # Market cap filter (avoid penny stocks)
    min_market_cap: float = 1_000_000_000.0  # $1B

    # Data
    max_headlines: int = 20
    min_bars: int = 50
    use_enricher: bool = True  # adds fundamentals, earnings, regime


@register_strategy("sentiment")
class SentimentStrategy:
    """News sentiment + technical confirmation strategy.

    Reads news headlines, scores sentiment, and combines with
    price trend and volume to generate BUY/SELL signals.
    """

    def __init__(self, config: SentimentConfig | None = None) -> None:
        self.config = config or SentimentConfig()
        self._trailing_stops: Dict[str, float] = {}
        self._sentiment_cache: Dict[str, Dict[str, Any]] = {}
        self._analyzer: Optional[Any] = None
        self._fetcher: Optional[Any] = None
        self._enricher = None
        if self.config.use_enricher:
            try:
                from shared.strategy_enricher import StrategyEnricher
                self._enricher = StrategyEnricher()
            except Exception as e:
                logger.debug("Enricher init: %s", e)

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "SentimentStrategy":
        config = SentimentConfig(**{
            k: v for k, v in params.items() if hasattr(SentimentConfig, k)
        })
        return cls(config)

    def _get_analyzer(self):
        if self._analyzer is None:
            from shared.ml.news_sentiment import NewsSentimentAnalyzer
            self._analyzer = NewsSentimentAnalyzer()
        return self._analyzer

    def _get_fetcher(self):
        if self._fetcher is None:
            from shared.data.public_data_fetcher import PublicDataFetcher
            self._fetcher = PublicDataFetcher()
        return self._fetcher

    def _get_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment score for a symbol (cached per session)."""
        if symbol in self._sentiment_cache:
            return self._sentiment_cache[symbol]

        try:
            analyzer = self._get_analyzer()
            result = analyzer.analyze(symbol)
            if result:
                self._sentiment_cache[symbol] = result
                return result
        except Exception as e:
            logger.debug("Sentiment analysis failed for %s: %s", symbol, e)

        return {
            "sentiment_score": 0.0,
            "sentiment_label": "NEUTRAL",
            "confidence": 0.0,
            "headlines_analyzed": 0,
        }

    def generate_signals(self, ctx: "BacktestContext") -> Dict[str, int]:
        """Generate sentiment-driven signals for each symbol."""
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

            atr = TI.atr(df, cfg.atr_length)
            current_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else current_price * 0.02

            # Trailing stop
            if cfg.trailing_stop and current_pos > 0:
                new_stop = current_price - cfg.stop_loss_atr_mult * current_atr
                self._trailing_stops[sym] = max(self._trailing_stops.get(sym, 0), new_stop)
                if current_price < self._trailing_stops[sym]:
                    signals[sym] = 0
                    self._trailing_stops.pop(sym, None)
                    logger.info("%s: sentiment trailing stop hit", sym)
                    continue

            # Get sentiment
            sentiment = self._get_sentiment(sym)
            score = sentiment.get("sentiment_score", 0.0)
            confidence = sentiment.get("confidence", 0.0)
            n_headlines = sentiment.get("headlines_analyzed", 0)

            if n_headlines < cfg.min_headlines:
                signals[sym] = 1 if current_pos > 0 else 0
                continue

            # Technical confirmation
            trend_ok = True
            if cfg.require_trend_confirm:
                ema = TI.ema(close, cfg.trend_ema_length)
                trend_ok = current_price > float(ema.iloc[-1])

            volume_ok = True
            if cfg.require_volume_confirm and "volume" in df.columns:
                vol_ma = float(df["volume"].rolling(cfg.volume_ma_length).mean().iloc[-1])
                volume_ok = float(df["volume"].iloc[-1]) > vol_ma

            # Signal logic
            # Enricher gate: fundamentals + earnings + regime
            enricher_ok = True
            if getattr(self, "_enricher", None) and current_pos <= 0:
                enriched = self._enricher.enrich(sym, df)
                if enriched.fundamentals_available and not enriched.fundamental_ok:
                    enricher_ok = False
                if enriched.earnings_available and not enriched.earnings_safe:
                    enricher_ok = False

            if (
                current_pos <= 0
                and score >= cfg.bullish_threshold
                and confidence >= cfg.min_confidence
                and trend_ok
                and volume_ok
                and enricher_ok
            ):
                signals[sym] = 1
                self._trailing_stops[sym] = current_price - cfg.stop_loss_atr_mult * current_atr
                logger.info(
                    "%s: SENTIMENT BUY (score=%.2f, confidence=%.2f, headlines=%d)",
                    sym, score, confidence, n_headlines,
                )
            elif current_pos > 0 and score <= cfg.bearish_threshold:
                signals[sym] = 0
                self._trailing_stops.pop(sym, None)
                logger.info("%s: SENTIMENT EXIT (score=%.2f)", sym, score)
            else:
                signals[sym] = 1 if current_pos > 0 else 0

        return signals
