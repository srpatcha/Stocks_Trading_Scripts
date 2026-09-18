"""
Sector Rotation Strategy
==========================

Rotates capital between sector ETFs based on relative momentum,
investing in the strongest sectors and avoiding the weakest.

Data sources used:
- OHLCV price history for sector ETFs
- Momentum scoring (12-1 month returns)
- Relative strength ranking across sectors
- Volume confirmation

Sector ETFs tracked: XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLU, XLC, XLB, XLRE

Entry: Top N sectors by momentum, above 200-SMA
Exit:  Sector drops out of top N or below 200-SMA
Rebalance: Monthly

Usage:
    from strategies.examples.sector_rotation import SectorRotationStrategy
    strategy = SectorRotationStrategy()
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

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

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLU": "Utilities",
    "XLC": "Communication Services",
    "XLB": "Materials",
    "XLRE": "Real Estate",
}


@dataclass
class SectorRotationConfig:
    """Configuration for the Sector Rotation strategy."""

    # Momentum parameters
    momentum_lookback: int = 252  # ~12 months
    momentum_skip: int = 21  # skip most recent month (mean-reversion effect)

    # Portfolio construction
    top_n_sectors: int = 3  # invest in top N sectors
    equal_weight: bool = True

    # Trend filter
    trend_sma_length: int = 200
    require_above_sma: bool = True

    # Rebalance frequency
    rebalance_period_bars: int = 21  # monthly

    # Risk management
    stop_loss_pct: float = 8.0

    # Data
    min_bars: int = 260
    use_enricher: bool = True
    use_volume_confirm: bool = True
    volume_ma_length: int = 20


@register_strategy("sector_rotation")
class SectorRotationStrategy:
    """Sector rotation strategy based on relative momentum.

    Ranks sector ETFs by momentum, invests in the strongest N sectors
    that are above their 200-day SMA. Rebalances monthly.
    """

    def __init__(self, config: SectorRotationConfig | None = None) -> None:
        self.config = config or SectorRotationConfig()
        self._last_rebalance_bar: int = -999
        self._target_sectors: List[str] = []
        self._entry_prices: Dict[str, float] = {}
        self._enricher = None
        if self.config.use_enricher:
            try:
                from shared.strategy_enricher import StrategyEnricher
                self._enricher = StrategyEnricher()
            except Exception as e:
                logger.debug("Enricher init: %s", e)

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "SectorRotationStrategy":
        config = SectorRotationConfig(**{
            k: v for k, v in params.items() if hasattr(SectorRotationConfig, k)
        })
        return cls(config)

    def _rank_sectors(self, bars: Dict[str, pd.DataFrame]) -> List[str]:
        """Rank available sector ETFs by momentum (12-1 month return)."""
        cfg = self.config
        scores: Dict[str, float] = {}

        for sym in SECTOR_ETFS:
            if sym not in bars:
                continue
            df = bars[sym]
            if len(df) < cfg.momentum_lookback:
                continue

            close = df["close"]
            # 12-1 month momentum: return over lookback period skipping recent month
            start_idx = -(cfg.momentum_lookback)
            end_idx = -(cfg.momentum_skip) if cfg.momentum_skip > 0 else None

            start_price = float(close.iloc[start_idx])
            end_price = float(close.iloc[end_idx]) if end_idx else float(close.iloc[-1])

            if start_price > 0:
                momentum = (end_price - start_price) / start_price
                scores[sym] = momentum

        # Sort by momentum descending
        ranked = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)

        # Filter: must be above 200-SMA
        if cfg.require_above_sma:
            filtered = []
            for sym in ranked:
                df = bars[sym]
                if len(df) >= cfg.trend_sma_length:
                    sma = TI.sma(df["close"], cfg.trend_sma_length)
                    if float(df["close"].iloc[-1]) > float(sma.iloc[-1]):
                        filtered.append(sym)
            ranked = filtered

        return ranked[:cfg.top_n_sectors]

    def generate_signals(self, ctx: "BacktestContext") -> Dict[str, int]:
        """Generate sector rotation signals."""
        from shared.backtesting.backtest_engine_v2 import BacktestContext

        cfg = self.config
        signals: Dict[str, int] = {}

        # Check if rebalance is due
        bars_since_rebalance = ctx.bar_index - self._last_rebalance_bar
        if bars_since_rebalance >= cfg.rebalance_period_bars:
            self._target_sectors = self._rank_sectors(ctx.bars)
            self._last_rebalance_bar = ctx.bar_index
            logger.info(
                "Sector rotation rebalance: top sectors = %s",
                self._target_sectors,
            )

        for sym, df in ctx.bars.items():
            if sym not in SECTOR_ETFS:
                signals[sym] = ctx.positions.get(sym, 0)
                continue

            if len(df) < cfg.min_bars:
                signals[sym] = 0
                continue

            current_pos = ctx.positions.get(sym, 0)
            current_price = float(df["close"].iloc[-1])

            # Stop loss
            if current_pos > 0 and sym in self._entry_prices:
                entry = self._entry_prices[sym]
                loss_pct = (entry - current_price) / entry * 100
                if loss_pct >= cfg.stop_loss_pct:
                    signals[sym] = 0
                    self._entry_prices.pop(sym, None)
                    logger.info("%s: sector rotation stop loss (%.1f%%)", sym, loss_pct)
                    continue

            if sym in self._target_sectors:
                # Enricher gate for new entries
                enricher_ok = True
                if getattr(self, "_enricher", None) and current_pos <= 0:
                    enriched = self._enricher.enrich(sym, df)
                    blocked, reason = self._enricher.should_block_entry(enriched)
                    if blocked:
                        enricher_ok = False
                        logger.debug("%s: sector entry blocked: %s", sym, reason)

                if enricher_ok:
                    if current_pos <= 0:
                        self._entry_prices[sym] = current_price
                    signals[sym] = 1
                else:
                    signals[sym] = 0
            else:
                if current_pos > 0:
                    self._entry_prices.pop(sym, None)
                    logger.info(
                        "%s (%s): exiting — no longer top sector",
                        sym, SECTOR_ETFS.get(sym, ""),
                    )
                signals[sym] = 0

        return signals
