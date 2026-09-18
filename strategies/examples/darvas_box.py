"""
Darvas Box Strategy (Nicolas Darvas)
======================================

Implements the Darvas Box breakout method from
"How I Made $2,000,000 in the Stock Market".

The Darvas Box is defined by:
1. A new high is established (box ceiling).
2. Price consolidates without exceeding the ceiling for N bars.
3. The lowest low during consolidation becomes the box floor.
4. Entry: price closes above box ceiling on above-average volume.
5. Stop: below box floor.

Usage:
    from strategies.examples.darvas_box import DarvasBoxStrategy
    strategy = DarvasBoxStrategy()
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
from strategies import register_strategy

if TYPE_CHECKING:  # pragma: no cover
    # Imported at runtime inside generate_signals to avoid a circular
    # import; declared here so the annotation resolves for linters and
    # type checkers instead of tripping F821.
    from shared.backtesting.backtest_engine_v2 import BacktestContext


logger = logging.getLogger(__name__)


@dataclass
class DarvasBoxConfig:
    """Configuration for the Darvas Box strategy."""

    box_confirm_bars: int = 3
    volume_ma_length: int = 20
    volume_surge_mult: float = 1.5
    stop_loss_buffer_pct: float = 0.5
    use_trailing_box_stop: bool = True
    min_box_height_pct: float = 3.0
    max_box_height_pct: float = 30.0
    trend_filter_length: int = 50
    use_enricher: bool = True


@register_strategy("darvas_box")
class DarvasBoxStrategy:
    """Darvas Box breakout strategy.

    Defines consolidation boxes from new highs, enters on volume-confirmed
    breakouts above the box ceiling, and stops below the box floor.
    """

    def __init__(self, config: DarvasBoxConfig | None = None) -> None:
        self.config = config or DarvasBoxConfig()
        self._box_ceilings: Dict[str, float] = {}
        self._box_floors: Dict[str, float] = {}
        self._entry_prices: Dict[str, float] = {}
        self._stop_prices: Dict[str, float] = {}
        self._enricher = None
        if self.config.use_enricher:
            try:
                from shared.strategy_enricher import StrategyEnricher
                self._enricher = StrategyEnricher()
            except Exception as e:
                logger.debug("Enricher init: %s", e)

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "DarvasBoxStrategy":
        config = DarvasBoxConfig(**{
            k: v for k, v in params.items() if hasattr(DarvasBoxConfig, k)
        })
        return cls(config)

    @staticmethod
    def detect_boxes(
        df: pd.DataFrame, confirm_bars: int = 3,
        min_height_pct: float = 3.0, max_height_pct: float = 30.0,
    ) -> pd.DataFrame:
        """Detect Darvas Boxes in price data.

        Returns DataFrame with columns: box_ceiling, box_floor, in_box.
        """
        n = len(df)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        ceiling = np.full(n, np.nan)
        floor = np.full(n, np.nan)
        in_box = np.zeros(n, dtype=bool)

        current_ceiling = high[0]
        ceiling_confirmed = False
        bars_since_high = 0
        current_floor = low[0]

        for i in range(1, n):
            if high[i] > current_ceiling:
                current_ceiling = high[i]
                bars_since_high = 0
                ceiling_confirmed = False
                current_floor = low[i]
            else:
                bars_since_high += 1
                current_floor = min(current_floor, low[i])

            if bars_since_high >= confirm_bars and not ceiling_confirmed:
                ceiling_confirmed = True

            if ceiling_confirmed:
                box_height_pct = (current_ceiling - current_floor) / current_ceiling * 100 if current_ceiling > 0 else 0
                if min_height_pct <= box_height_pct <= max_height_pct:
                    ceiling[i] = current_ceiling
                    floor[i] = current_floor
                    in_box[i] = True

        result = pd.DataFrame(index=df.index)
        result["box_ceiling"] = ceiling
        result["box_floor"] = floor
        result["in_box"] = in_box
        return result

    def generate_signals(self, ctx: "BacktestContext") -> Dict[str, int]:
        """Generate Darvas Box signals for each symbol."""
        from shared.backtesting.backtest_engine_v2 import BacktestContext

        cfg = self.config
        signals: Dict[str, int] = {}

        for sym, df in ctx.bars.items():
            if len(df) < cfg.trend_filter_length:
                signals[sym] = 0
                continue

            current_pos = ctx.positions.get(sym, 0)
            close = df["close"]
            current_price = float(close.iloc[-1])

            # Stop loss check
            if current_pos > 0 and sym in self._stop_prices:
                if current_price < self._stop_prices[sym]:
                    signals[sym] = 0
                    self._stop_prices.pop(sym, None)
                    self._entry_prices.pop(sym, None)
                    self._box_ceilings.pop(sym, None)
                    self._box_floors.pop(sym, None)
                    logger.info("%s: Darvas stop loss triggered at %.2f", sym, current_price)
                    continue

            # Trend filter: only trade above 50-SMA
            sma = TI.sma(close, cfg.trend_filter_length)
            sma_val = float(sma.iloc[-1])
            if current_price < sma_val:
                if current_pos > 0:
                    signals[sym] = 1
                else:
                    signals[sym] = 0
                continue

            boxes = self.detect_boxes(
                df, confirm_bars=cfg.box_confirm_bars,
                min_height_pct=cfg.min_box_height_pct,
                max_height_pct=cfg.max_box_height_pct,
            )

            latest_ceiling = boxes["box_ceiling"].iloc[-1]
            latest_floor = boxes["box_floor"].iloc[-1]
            in_box = bool(boxes["in_box"].iloc[-1])

            if not in_box or np.isnan(latest_ceiling):
                signals[sym] = 1 if current_pos > 0 else 0
                continue

            # Volume check
            vol_ok = True
            if "volume" in df.columns and len(df) >= cfg.volume_ma_length:
                vol_ma = float(df["volume"].rolling(cfg.volume_ma_length).mean().iloc[-1])
                current_vol = float(df["volume"].iloc[-1])
                vol_ok = current_vol > cfg.volume_surge_mult * vol_ma

            # Breakout above box ceiling
            # Enricher gate
            enricher_ok = True
            if getattr(self, "_enricher", None) and current_pos <= 0:
                enriched = self._enricher.enrich(sym, df)
                blocked, _ = self._enricher.should_block_entry(enriched)
                if blocked:
                    enricher_ok = False

            if current_pos <= 0 and current_price > latest_ceiling and vol_ok and enricher_ok:
                signals[sym] = 1
                self._entry_prices[sym] = current_price
                self._box_ceilings[sym] = float(latest_ceiling)
                self._box_floors[sym] = float(latest_floor)
                buffer = float(latest_floor) * (1 - cfg.stop_loss_buffer_pct / 100)
                self._stop_prices[sym] = buffer
                logger.info(
                    "%s: DARVAS BREAKOUT ceiling=%.2f floor=%.2f stop=%.2f",
                    sym, latest_ceiling, latest_floor, buffer,
                )
            elif current_pos > 0:
                # Update trailing box stop
                if cfg.use_trailing_box_stop and not np.isnan(latest_floor):
                    new_stop = float(latest_floor) * (1 - cfg.stop_loss_buffer_pct / 100)
                    if sym in self._stop_prices:
                        self._stop_prices[sym] = max(self._stop_prices[sym], new_stop)
                signals[sym] = 1
            else:
                signals[sym] = 0

        return signals
