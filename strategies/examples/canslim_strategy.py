"""
CAN SLIM Strategy (O'Neil)
============================

Implements William O'Neil's CAN SLIM scoring system combining
fundamental and technical criteria for growth stock selection.

Criteria scored 0-7 (one point each):
    C - Current quarterly earnings growth > 25%
    A - Annual earnings growth (3-year revenue growth)
    N - New highs (price within 5% of 52-week high)
    S - Supply/demand (volume surge > 1.5x average)
    L - Leader/Laggard (relative strength vs SPY > 80th percentile)
    I - Institutional ownership % (from yfinance)
    M - Market direction (above 200 SMA)

Entry: score >= 5
Exit:  score < 3 or stop hit

Usage:
    from strategies.examples.canslim_strategy import CANSLIMStrategy
    strategy = CANSLIMStrategy()
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
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
class CANSLIMConfig:
    """Configuration thresholds for CAN SLIM scoring."""

    # C - Current quarterly earnings growth
    min_earnings_growth_pct: float = 25.0

    # A - Annual earnings (revenue growth threshold)
    min_annual_revenue_growth_pct: float = 15.0

    # N - New highs (within X% of 52-week high)
    new_high_pct_threshold: float = 5.0
    new_high_lookback: int = 252  # trading days in a year

    # S - Supply/demand volume surge
    volume_surge_mult: float = 1.5
    volume_ma_length: int = 50

    # L - Leader/Laggard RS threshold
    rs_percentile_threshold: float = 80.0

    # I - Institutional ownership minimum
    min_institutional_pct: float = 20.0

    # M - Market direction
    market_sma_length: int = 200

    # Trade management
    entry_score_threshold: int = 5
    exit_score_threshold: int = 3
    stop_loss_pct: float = 7.0  # O'Neil's classic 7-8% stop
    use_enricher: bool = True


@register_strategy("canslim")
class CANSLIMStrategy:
    """CAN SLIM growth stock scoring strategy.

    Scores each stock 0-7 based on O'Neil's CAN SLIM criteria.
    Enters when score >= 5, exits when score < 3 or stop is hit.
    """

    def __init__(self, config: CANSLIMConfig | None = None) -> None:
        self.config = config or CANSLIMConfig()
        self._entry_prices: Dict[str, float] = {}
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
    def from_params(cls, params: Dict[str, Any]) -> "CANSLIMStrategy":
        config = CANSLIMConfig(**{
            k: v for k, v in params.items() if hasattr(CANSLIMConfig, k)
        })
        return cls(config)

    def _fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch and cache fundamental data for a symbol."""
        if symbol in self._fundamentals_cache:
            return self._fundamentals_cache[symbol]

        try:
            if self._fetcher is None:
                from shared.data.public_data_fetcher import PublicDataFetcher
                self._fetcher = PublicDataFetcher()
            data = self._fetcher.fetch_fundamentals(symbol)
            if data:
                self._fundamentals_cache[symbol] = data
                return data
        except Exception as e:
            logger.debug("Could not fetch fundamentals for %s: %s", symbol, e)

        return {}

    def _score_canslim(
        self,
        symbol: str,
        df: pd.DataFrame,
        market_df: Optional[pd.DataFrame] = None,
    ) -> int:
        """Calculate CAN SLIM score (0-7) for a symbol.

        Args:
            symbol: Ticker symbol.
            df: OHLCV DataFrame for the symbol.
            market_df: Optional OHLCV DataFrame for SPY/market benchmark.

        Returns:
            Integer score from 0 to 7.
        """
        cfg = self.config
        score = 0
        close = df["close"]
        current_price = float(close.iloc[-1])

        fundamentals = self._fetch_fundamentals(symbol)

        # C - Current quarterly earnings growth > 25%
        eg = fundamentals.get("earnings_growth")
        if eg is not None and eg * 100 > cfg.min_earnings_growth_pct:
            score += 1

        # A - Annual earnings growth (revenue growth as proxy)
        revenue = fundamentals.get("revenue", 0)
        profit_margin = fundamentals.get("profit_margin")
        eg_annual = fundamentals.get("earnings_growth")
        if eg_annual is not None and eg_annual * 100 > cfg.min_annual_revenue_growth_pct:
            score += 1
        elif profit_margin is not None and profit_margin > 0.10 and revenue and revenue > 0:
            score += 1

        # N - Price within 5% of 52-week high
        lookback = min(cfg.new_high_lookback, len(df))
        high_52w = float(df["high"].iloc[-lookback:].max())
        if high_52w > 0:
            pct_from_high = (high_52w - current_price) / high_52w * 100
            if pct_from_high <= cfg.new_high_pct_threshold:
                score += 1

        # S - Volume surge (current volume > 1.5x average)
        if "volume" in df.columns and len(df) >= cfg.volume_ma_length:
            vol_ma = float(df["volume"].rolling(cfg.volume_ma_length).mean().iloc[-1])
            current_vol = float(df["volume"].iloc[-1])
            if vol_ma > 0 and current_vol > cfg.volume_surge_mult * vol_ma:
                score += 1

        # L - Relative Strength vs benchmark
        if market_df is not None and len(market_df) >= 60:
            sym_ret = float(close.iloc[-1] / close.iloc[-60] - 1) if len(df) >= 60 else 0
            mkt_ret = float(market_df["close"].iloc[-1] / market_df["close"].iloc[-60] - 1)
            if mkt_ret != 0 and sym_ret / max(abs(mkt_ret), 0.001) > 1.0:
                score += 1
        else:
            rsi_val = TI.rsi(close, 14)
            if not np.isnan(rsi_val.iloc[-1]) and float(rsi_val.iloc[-1]) > 60:
                score += 1

        # I - Institutional ownership
        inst_pct = fundamentals.get("institutional_pct")
        if inst_pct is not None and inst_pct > cfg.min_institutional_pct:
            score += 1

        # M - Market direction (price above 200 SMA)
        if len(df) >= cfg.market_sma_length:
            sma_200 = TI.sma(close, cfg.market_sma_length)
            if current_price > float(sma_200.iloc[-1]):
                score += 1

        return score

    def generate_signals(self, ctx: "BacktestContext") -> Dict[str, int]:
        """Generate CAN SLIM signals for each symbol."""
        from shared.backtesting.backtest_engine_v2 import BacktestContext

        cfg = self.config
        signals: Dict[str, int] = {}

        market_df = ctx.bars.get("SPY")

        for sym, df in ctx.bars.items():
            if sym == "SPY":
                continue
            if len(df) < cfg.market_sma_length:
                signals[sym] = 0
                continue

            current_pos = ctx.positions.get(sym, 0)
            current_price = float(df["close"].iloc[-1])

            # Stop loss check
            if current_pos > 0 and sym in self._entry_prices:
                entry = self._entry_prices[sym]
                loss_pct = (entry - current_price) / entry * 100
                if loss_pct >= cfg.stop_loss_pct:
                    signals[sym] = 0
                    self._entry_prices.pop(sym, None)
                    logger.info("%s: stop loss triggered (%.1f%% loss)", sym, loss_pct)
                    continue

            score = self._score_canslim(sym, df, market_df)

            if current_pos <= 0 and score >= cfg.entry_score_threshold:
                # Enricher gate: check sentiment + earnings blackout
                enricher_ok = True
                if getattr(self, "_enricher", None):
                    enriched = self._enricher.enrich(sym, df)
                    blocked, reason = self._enricher.should_block_entry(enriched)
                    if blocked:
                        enricher_ok = False
                        logger.debug("%s: CAN SLIM entry blocked: %s", sym, reason)
                if enricher_ok:
                    signals[sym] = 1
                    self._entry_prices[sym] = current_price
                    logger.info("%s: CAN SLIM BUY (score=%d/7)", sym, score)
                else:
                    signals[sym] = 0
            elif current_pos > 0 and score < cfg.exit_score_threshold:
                signals[sym] = 0
                self._entry_prices.pop(sym, None)
                logger.info("%s: CAN SLIM EXIT (score=%d/7)", sym, score)
            else:
                signals[sym] = 1 if current_pos > 0 else 0

        return signals
