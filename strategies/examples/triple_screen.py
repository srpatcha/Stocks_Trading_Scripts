"""
Triple Screen Strategy (Alexander Elder)
==========================================

Implements Elder's Triple Screen Trading System:
    Screen 1 (Weekly): Trend direction via EMA slope + MACD histogram
    Screen 2 (Daily):  Oscillator pullback (Force Index or Elder-ray)
    Screen 3 (Intraday): Entry trigger (breakout of prior bar high/low)

Entry: Screen 1 bullish + Screen 2 pullback complete + Screen 3 trigger hit
Exit:  Trailing stop or Screen 1 reversal

Usage:
    from strategies.examples.triple_screen import TripleScreenStrategy
    strategy = TripleScreenStrategy()
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, TYPE_CHECKING

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
class TripleScreenConfig:
    """Configuration for the Triple Screen strategy."""

    # Screen 1: Weekly trend (higher timeframe)
    htf_period: str = "W"
    htf_ema_length: int = 13
    htf_macd_fast: int = 12
    htf_macd_slow: int = 26
    htf_macd_signal: int = 9

    # Screen 2: Daily oscillator
    force_index_period: int = 2
    elder_ray_period: int = 13

    # Screen 3: Entry trigger
    entry_breakout_bars: int = 1

    # Risk management
    stop_loss_atr_mult: float = 2.0
    trailing_stop: bool = True
    atr_length: int = 14

    # Minimum data requirement
    min_bars: int = 100
    use_enricher: bool = True


@register_strategy("triple_screen")
class TripleScreenStrategy:
    """Elder Triple Screen Trading System.

    Uses three timeframes to filter and time entries:
    Screen 1 (trend), Screen 2 (oscillator), Screen 3 (trigger).
    """

    def __init__(self, config: TripleScreenConfig | None = None) -> None:
        self.config = config or TripleScreenConfig()
        self._trailing_stops: Dict[str, float] = {}
        self._mtf = MultiTimeframeTrend(htf_period=self.config.htf_period)
        self._enricher = None
        if self.config.use_enricher:
            try:
                from shared.strategy_enricher import StrategyEnricher
                self._enricher = StrategyEnricher()
            except Exception as e:
                logger.debug("Enricher init: %s", e)

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "TripleScreenStrategy":
        config = TripleScreenConfig(**{
            k: v for k, v in params.items() if hasattr(TripleScreenConfig, k)
        })
        return cls(config)

    def _screen1_trend(self, df: pd.DataFrame) -> str:
        """Screen 1: Weekly trend direction.

        Uses higher-timeframe EMA slope and MACD histogram direction.
        Returns 'BULLISH', 'BEARISH', or 'NEUTRAL'.
        """
        htf = self._mtf.resample_to_htf(df, self.config.htf_period)
        if len(htf) < self.config.htf_ema_length + 5:
            return "NEUTRAL"

        ema = TI.ema(htf["close"], self.config.htf_ema_length)
        _, _, histogram = TI.macd(
            htf["close"],
            fast=self.config.htf_macd_fast,
            slow=self.config.htf_macd_slow,
            signal=self.config.htf_macd_signal,
        )

        ema_rising = float(ema.iloc[-1]) > float(ema.iloc[-2])
        hist_rising = float(histogram.iloc[-1]) > float(histogram.iloc[-2])

        if ema_rising and hist_rising:
            return "BULLISH"
        elif not ema_rising and not hist_rising:
            return "BEARISH"
        return "NEUTRAL"

    def _screen2_pullback(self, df: pd.DataFrame, trend: str) -> bool:
        """Screen 2: Daily oscillator pullback detection.

        In an uptrend: look for Force Index dip below zero (buying dip).
        In a downtrend: look for Force Index rise above zero (selling rally).
        """
        if len(df) < self.config.force_index_period + 5:
            return False

        fi = TI.force_index(df, self.config.force_index_period)
        fi_current = float(fi.iloc[-1])
        fi_prev = float(fi.iloc[-2]) if len(fi) > 1 else 0

        if trend == "BULLISH":
            return fi_prev < 0 and fi_current > 0
        elif trend == "BEARISH":
            return fi_prev > 0 and fi_current < 0
        return False

    def _screen3_trigger(self, df: pd.DataFrame, trend: str) -> bool:
        """Screen 3: Entry trigger — breakout of prior bar.

        In uptrend: current close > previous high.
        In downtrend: current close < previous low.
        """
        if len(df) < 2:
            return False

        current_close = float(df["close"].iloc[-1])
        prev_high = float(df["high"].iloc[-2])
        prev_low = float(df["low"].iloc[-2])

        if trend == "BULLISH":
            return current_close > prev_high
        elif trend == "BEARISH":
            return current_close < prev_low
        return False

    def generate_signals(self, ctx: "BacktestContext") -> Dict[str, int]:
        """Generate Triple Screen signals for each symbol."""
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
            if cfg.trailing_stop and current_pos != 0:
                if current_pos > 0:
                    new_stop = current_price - cfg.stop_loss_atr_mult * current_atr
                    self._trailing_stops[sym] = max(self._trailing_stops.get(sym, 0), new_stop)
                    if current_price < self._trailing_stops[sym]:
                        signals[sym] = 0
                        self._trailing_stops.pop(sym, None)
                        continue
                elif current_pos < 0:
                    new_stop = current_price + cfg.stop_loss_atr_mult * current_atr
                    self._trailing_stops[sym] = min(self._trailing_stops.get(sym, float("inf")), new_stop)
                    if current_price > self._trailing_stops[sym]:
                        signals[sym] = 0
                        self._trailing_stops.pop(sym, None)
                        continue

            # Screen 1: Weekly trend
            trend = self._screen1_trend(df)

            # Screen 2: Daily pullback
            pullback = self._screen2_pullback(df, trend)

            # Screen 3: Entry trigger
            trigger = self._screen3_trigger(df, trend)

            if current_pos == 0:
                # Enricher gate
                enricher_ok = True
                if getattr(self, "_enricher", None):
                    enriched = self._enricher.enrich(sym, df)
                    blocked, _ = self._enricher.should_block_entry(enriched)
                    if blocked:
                        enricher_ok = False

                if trend == "BULLISH" and pullback and trigger and enricher_ok:
                    signals[sym] = 1
                    self._trailing_stops[sym] = current_price - cfg.stop_loss_atr_mult * current_atr
                    logger.info("%s: TRIPLE SCREEN BUY (trend=%s)", sym, trend)
                elif trend == "BEARISH" and pullback and trigger and enricher_ok:
                    signals[sym] = -1
                    self._trailing_stops[sym] = current_price + cfg.stop_loss_atr_mult * current_atr
                    logger.info("%s: TRIPLE SCREEN SELL (trend=%s)", sym, trend)
                else:
                    signals[sym] = 0
            elif current_pos > 0:
                if trend == "BEARISH":
                    signals[sym] = 0
                    self._trailing_stops.pop(sym, None)
                    logger.info("%s: TRIPLE SCREEN EXIT (trend reversal)", sym)
                else:
                    signals[sym] = 1
            elif current_pos < 0:
                if trend == "BULLISH":
                    signals[sym] = 0
                    self._trailing_stops.pop(sym, None)
                    logger.info("%s: TRIPLE SCREEN COVER (trend reversal)", sym)
                else:
                    signals[sym] = -1

        return signals
