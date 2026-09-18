"""
Meta Ensemble Strategy — The Master Strategy
===============================================

Combines ALL available data sources and strategy signals into one
unified trading decision. This is the most comprehensive strategy
in the system.

Data sources used:
- OHLCV price history (trend, momentum, volatility)
- News headlines + sentiment scoring (FinBERT/VADER)
- Fundamental data (P/E, P/B, earnings growth, debt, dividends)
- Earnings calendar (beat rate, surprise %, days to next)
- Technical indicators (EMA, RSI, MACD, ADX, ATR, Bollinger, Force Index)
- Market regime classification (TRENDING/RANGING/VOLATILE)
- Multi-timeframe trend confirmation

Signal components (weighted vote):
1. Technical score (trend + momentum + volume)
2. Fundamental score (value + quality + growth)
3. Sentiment score (news headlines analysis)
4. Earnings score (beat rate + surprise + calendar proximity)
5. Regime-adaptive weighting

Entry: Weighted composite score > threshold
Exit:  Score drops below exit threshold OR trailing stop

Usage:
    from strategies.examples.meta_strategy import MetaEnsembleStrategy
    strategy = MetaEnsembleStrategy()
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.indicators.technical_indicators import TechnicalIndicators as TI
from shared.indicators.multi_timeframe import MultiTimeframeTrend
from strategies import register_strategy

if TYPE_CHECKING:  # pragma: no cover
    # Imported at runtime inside generate_signals to avoid a circular
    # import; declared here so the annotation resolves for linters and
    # type checkers instead of tripping F821.
    from shared.backtesting.backtest_engine_v2 import BacktestContext


logger = logging.getLogger(__name__)


@dataclass
class MetaEnsembleConfig:
    """Configuration for the Meta Ensemble strategy."""

    # Component weights (sum to ~1.0)
    technical_weight: float = 0.30
    fundamental_weight: float = 0.20
    sentiment_weight: float = 0.20
    earnings_weight: float = 0.15
    regime_weight: float = 0.15

    # Thresholds
    entry_threshold: float = 0.40  # composite score > 0.4 to BUY
    exit_threshold: float = 0.20  # composite score < 0.2 to SELL
    min_agreement: int = 3  # at least 3 of 5 components must agree

    # Technical parameters
    fast_ema: int = 9
    slow_ema: int = 21
    trend_ema: int = 200
    rsi_length: int = 14
    adx_length: int = 14
    adx_threshold: float = 20.0
    volume_ma_length: int = 20

    # Fundamental thresholds
    max_pe_ratio: float = 30.0
    min_market_cap: float = 2_000_000_000.0

    # Risk management
    stop_loss_atr_mult: float = 2.5
    trailing_stop: bool = True
    atr_length: int = 14

    # Multi-timeframe
    use_mtf_filter: bool = True
    htf_period: str = "W"

    # Data requirements
    min_bars: int = 200


@register_strategy("meta_ensemble")
class MetaEnsembleStrategy:
    """Master strategy combining ALL available data sources.

    Scores each stock across 5 dimensions (technical, fundamental,
    sentiment, earnings, regime), computes a weighted composite score,
    and trades when the score exceeds the threshold with sufficient
    agreement across components.
    """

    def __init__(self, config: MetaEnsembleConfig | None = None) -> None:
        self.config = config or MetaEnsembleConfig()
        self._trailing_stops: Dict[str, float] = {}
        self._mtf = MultiTimeframeTrend(htf_period=self.config.htf_period)
        self._fetcher: Optional[Any] = None
        self._sentiment_analyzer: Optional[Any] = None
        self._sentiment_cache: Dict[str, Dict[str, Any]] = {}
        self._fundamentals_cache: Dict[str, Dict[str, Any]] = {}
        self._earnings_cache: Dict[str, list] = {}

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "MetaEnsembleStrategy":
        config = MetaEnsembleConfig(**{
            k: v for k, v in params.items() if hasattr(MetaEnsembleConfig, k)
        })
        return cls(config)

    def _get_fetcher(self):
        if self._fetcher is None:
            from shared.data.public_data_fetcher import PublicDataFetcher
            self._fetcher = PublicDataFetcher()
        return self._fetcher

    def _get_sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            try:
                from shared.ml.news_sentiment import NewsSentimentAnalyzer
                self._sentiment_analyzer = NewsSentimentAnalyzer()
            except Exception:
                self._sentiment_analyzer = None
        return self._sentiment_analyzer

    # ─── Component 1: Technical Score ───

    def _score_technical(self, df: pd.DataFrame) -> float:
        """Score technical signals on a 0-1 scale.

        Checks: EMA crossover, price vs 200 EMA, RSI zone, ADX strength,
        volume confirmation, MACD histogram direction.
        """
        cfg = self.config
        score = 0.0
        checks = 0

        close = df["close"]
        current_price = float(close.iloc[-1])

        # EMA crossover
        fast = TI.ema(close, cfg.fast_ema)
        slow = TI.ema(close, cfg.slow_ema)
        if float(fast.iloc[-1]) > float(slow.iloc[-1]):
            score += 1
        checks += 1

        # Price above 200 EMA (trend)
        if len(df) >= cfg.trend_ema:
            trend = TI.ema(close, cfg.trend_ema)
            if current_price > float(trend.iloc[-1]):
                score += 1
            checks += 1

        # RSI in bullish zone (40-70)
        rsi = TI.rsi(close, cfg.rsi_length)
        rsi_val = float(rsi.iloc[-1])
        if not np.isnan(rsi_val):
            if 40 < rsi_val < 70:
                score += 1
            elif rsi_val > 70:
                score += 0.3  # overbought — partial credit
            checks += 1

        # ADX trend strength
        adx_val, plus_di, minus_di = TI.adx(df, cfg.adx_length)
        adx_current = float(adx_val.iloc[-1])
        if not np.isnan(adx_current) and adx_current > cfg.adx_threshold:
            if float(plus_di.iloc[-1]) > float(minus_di.iloc[-1]):
                score += 1
            checks += 1

        # Volume above average
        if "volume" in df.columns:
            vol_ma = float(df["volume"].rolling(cfg.volume_ma_length).mean().iloc[-1])
            if vol_ma > 0 and float(df["volume"].iloc[-1]) > vol_ma:
                score += 1
            checks += 1

        # MACD histogram rising
        _, _, hist = TI.macd(close)
        if len(hist) > 1 and float(hist.iloc[-1]) > float(hist.iloc[-2]):
            score += 1
        checks += 1

        return score / max(checks, 1)

    # ─── Component 2: Fundamental Score ───

    def _score_fundamental(self, symbol: str) -> float:
        """Score fundamental quality on a 0-1 scale."""
        if symbol in self._fundamentals_cache:
            f = self._fundamentals_cache[symbol]
        else:
            try:
                f = self._get_fetcher().fetch_fundamentals(symbol) or {}
                self._fundamentals_cache[symbol] = f
            except Exception:
                return 0.5  # neutral if unavailable

        if not f:
            return 0.5

        score = 0.0
        checks = 0

        # P/E reasonable
        pe = f.get("pe_ratio")
        if pe is not None:
            if 0 < pe < self.config.max_pe_ratio:
                score += 1
            checks += 1

        # Earnings growing
        eg = f.get("earnings_growth")
        if eg is not None:
            if eg > 0:
                score += 1
            checks += 1

        # Profit margin positive
        pm = f.get("profit_margin")
        if pm is not None:
            if pm > 0.05:
                score += 1
            checks += 1

        # Reasonable debt
        dte = f.get("debt_to_equity")
        if dte is not None:
            ratio = dte / 100.0 if dte > 10 else dte
            if ratio < 2.0:
                score += 1
            checks += 1

        # Market cap filter
        mcap = f.get("market_cap")
        if mcap is not None:
            if mcap > self.config.min_market_cap:
                score += 1
            checks += 1

        # Dividend (bonus)
        div = f.get("dividend_yield")
        if div is not None and div > 0:
            score += 0.5
            checks += 1

        return score / max(checks, 1)

    # ─── Component 3: Sentiment Score ───

    def _score_sentiment(self, symbol: str) -> float:
        """Score news sentiment on a 0-1 scale."""
        if symbol in self._sentiment_cache:
            s = self._sentiment_cache[symbol]
        else:
            try:
                analyzer = self._get_sentiment_analyzer()
                if analyzer is None:
                    return 0.5
                s = analyzer.analyze(symbol) or {}
                self._sentiment_cache[symbol] = s
            except Exception:
                return 0.5

        raw_score = s.get("sentiment_score", 0.0)
        confidence = s.get("confidence", 0.0)
        n_headlines = s.get("headlines_analyzed", 0)

        if n_headlines < 3 or confidence < 0.2:
            return 0.5  # not enough data

        # Map [-1, +1] sentiment to [0, 1] score
        return (raw_score + 1.0) / 2.0

    # ─── Component 4: Earnings Score ───

    def _score_earnings(self, symbol: str) -> float:
        """Score earnings quality on a 0-1 scale."""
        if symbol in self._earnings_cache:
            earnings = self._earnings_cache[symbol]
        else:
            try:
                earnings = self._get_fetcher().fetch_earnings_dates(symbol) or []
                self._earnings_cache[symbol] = earnings
            except Exception:
                return 0.5

        if not earnings:
            return 0.5

        score = 0.0
        checks = 0

        # Beat rate
        surprises = [e["surprise_pct"] for e in earnings if e.get("surprise_pct") is not None]
        if surprises:
            beats = sum(1 for s in surprises if s > 0)
            beat_rate = beats / len(surprises)
            score += beat_rate
            checks += 1

            # Average surprise magnitude
            avg_surprise = float(np.mean(surprises))
            if avg_surprise > 5:
                score += 1
            elif avg_surprise > 0:
                score += 0.5
            checks += 1

        return score / max(checks, 1)

    # ─── Component 5: Regime Score ───

    def _score_regime(self, df: pd.DataFrame) -> float:
        """Score regime favorability on a 0-1 scale.

        Bullish = trending up with manageable volatility.
        """
        if len(df) < 60:
            return 0.5

        close = df["close"]

        # 20-day vs 60-day return comparison
        ret_20 = float(close.iloc[-1] / close.iloc[-20] - 1) if len(df) >= 20 else 0
        ret_60 = float(close.iloc[-1] / close.iloc[-60] - 1) if len(df) >= 60 else 0

        # Volatility (ATR as % of price)
        atr = TI.atr(df, 14)
        atr_pct = float(atr.iloc[-1]) / float(close.iloc[-1]) * 100 if float(close.iloc[-1]) > 0 else 5

        score = 0.0

        # Positive momentum
        if ret_20 > 0:
            score += 0.3
        if ret_60 > 0:
            score += 0.3

        # Accelerating momentum (short > long)
        if ret_20 > ret_60:
            score += 0.2

        # Low-to-moderate volatility is favorable
        if atr_pct < 3.0:
            score += 0.2
        elif atr_pct < 5.0:
            score += 0.1

        # MTF confirmation
        if self.config.use_mtf_filter:
            trend = self._mtf.get_htf_trend(df)
            if trend == "BULLISH":
                score = min(score + 0.2, 1.0)
            elif trend == "BEARISH":
                score = max(score - 0.3, 0.0)

        return min(score, 1.0)

    # ─── Main Signal Generation ───

    def generate_signals(self, ctx: "BacktestContext") -> Dict[str, int]:
        """Generate meta ensemble signals combining all data sources."""
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
                    continue

            # Score all 5 components
            tech_score = self._score_technical(df)
            fund_score = self._score_fundamental(sym)
            sent_score = self._score_sentiment(sym)
            earn_score = self._score_earnings(sym)
            regime_score = self._score_regime(df)

            # Weighted composite
            composite = (
                tech_score * cfg.technical_weight
                + fund_score * cfg.fundamental_weight
                + sent_score * cfg.sentiment_weight
                + earn_score * cfg.earnings_weight
                + regime_score * cfg.regime_weight
            )

            # Count bullish components (score > 0.5)
            bullish_components = sum(1 for s in [
                tech_score, fund_score, sent_score, earn_score, regime_score
            ] if s > 0.5)

            # Entry
            if (
                current_pos <= 0
                and composite >= cfg.entry_threshold
                and bullish_components >= cfg.min_agreement
            ):
                signals[sym] = 1
                self._trailing_stops[sym] = current_price - cfg.stop_loss_atr_mult * current_atr
                logger.info(
                    "%s: META BUY (composite=%.2f, agree=%d/5) "
                    "[tech=%.2f fund=%.2f sent=%.2f earn=%.2f regime=%.2f]",
                    sym, composite, bullish_components,
                    tech_score, fund_score, sent_score, earn_score, regime_score,
                )
            # Exit
            elif current_pos > 0 and composite < cfg.exit_threshold:
                signals[sym] = 0
                self._trailing_stops.pop(sym, None)
                logger.info(
                    "%s: META EXIT (composite=%.2f, agree=%d/5)",
                    sym, composite, bullish_components,
                )
            else:
                signals[sym] = 1 if current_pos > 0 else 0

        return signals
