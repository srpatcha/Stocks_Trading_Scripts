"""
Graham Value Strategy
=======================

Implements Benjamin Graham's Intelligent Investor principles for
value-based stock selection using fundamental data.

Criteria scored 0-6:
    1. Margin of Safety: price / book_value < 1.5
    2. P/E Filter: pe_ratio < 15 (or below sector median)
    3. Debt Check: debt_to_equity < 1.0
    4. Dividend: dividend_yield > 0 (income-producing)
    5. Earnings Stability: positive earnings
    6. Market Cap Filter: > $2B (Graham preferred large-caps)

Entry: score >= 4 and technical support nearby
Exit:  score < 2 or P/E > 20

Usage:
    from strategies.examples.value_strategy import ValueStrategy
    strategy = ValueStrategy()
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
class ValueStrategyConfig:
    """Configuration thresholds for the Graham value strategy."""

    # Margin of Safety: price-to-book ratio
    max_price_to_book: float = 1.5

    # P/E thresholds
    max_pe_ratio: float = 15.0
    exit_pe_ratio: float = 20.0

    # Debt check
    max_debt_to_equity: float = 1.0

    # Dividend requirement
    min_dividend_yield: float = 0.0  # > 0 means must pay dividends

    # Market cap minimum ($2B)
    min_market_cap: float = 2_000_000_000.0

    # Score thresholds
    entry_score_threshold: int = 4
    exit_score_threshold: int = 2

    # Technical support: buy within X% of SMA support
    support_sma_length: int = 50
    support_pct_threshold: float = 5.0

    # Stop loss
    stop_loss_pct: float = 10.0
    use_enricher: bool = True


@register_strategy("value")
class ValueStrategy:
    """Graham-inspired value strategy using fundamental data.

    Scores stocks 0-6 based on fundamental criteria and enters
    when score >= 4 near technical support. Exits when score < 2
    or P/E exceeds 20.
    """

    def __init__(self, config: ValueStrategyConfig | None = None) -> None:
        self.config = config or ValueStrategyConfig()
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
    def from_params(cls, params: Dict[str, Any]) -> "ValueStrategy":
        config = ValueStrategyConfig(**{
            k: v for k, v in params.items() if hasattr(ValueStrategyConfig, k)
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

    def _score_value(self, symbol: str, df: pd.DataFrame) -> int:
        """Calculate Graham value score (0-6) for a symbol.

        Args:
            symbol: Ticker symbol.
            df: OHLCV DataFrame.

        Returns:
            Integer score from 0 to 6.
        """
        cfg = self.config
        score = 0

        fundamentals = self._fetch_fundamentals(symbol)
        if not fundamentals:
            return 0

        # 1. Margin of Safety: price/book < 1.5
        ptb = fundamentals.get("price_to_book")
        if ptb is not None and ptb < cfg.max_price_to_book and ptb > 0:
            score += 1

        # 2. P/E Filter: < 15
        pe = fundamentals.get("pe_ratio")
        if pe is not None and 0 < pe < cfg.max_pe_ratio:
            score += 1

        # 3. Debt Check: debt/equity < 1.0
        # yfinance debtToEquity is in percentage form (e.g., 150 = 1.5x)
        dte = fundamentals.get("debt_to_equity")
        if dte is not None:
            # Normalise: if dte > 10, assume percentage form; else raw ratio
            dte_ratio = dte / 100.0 if dte > 10 else dte
            if dte_ratio < cfg.max_debt_to_equity:
                score += 1

        # 4. Dividend: yield > 0
        div_yield = fundamentals.get("dividend_yield")
        if div_yield is not None and div_yield > cfg.min_dividend_yield:
            score += 1

        # 5. Earnings Stability: positive earnings growth
        eg = fundamentals.get("earnings_growth")
        pm = fundamentals.get("profit_margin")
        if (eg is not None and eg > 0) or (pm is not None and pm > 0):
            score += 1

        # 6. Market Cap Filter: > $2B
        mcap = fundamentals.get("market_cap")
        if mcap is not None and mcap > cfg.min_market_cap:
            score += 1

        return score

    def _near_support(self, df: pd.DataFrame) -> bool:
        """Check if price is near SMA support level."""
        cfg = self.config
        if len(df) < cfg.support_sma_length:
            return True  # not enough data, allow entry

        close = df["close"]
        sma = TI.sma(close, cfg.support_sma_length)
        current = float(close.iloc[-1])
        sma_val = float(sma.iloc[-1])

        if sma_val <= 0:
            return True

        pct_above = (current - sma_val) / sma_val * 100
        return pct_above <= cfg.support_pct_threshold

    def generate_signals(self, ctx: "BacktestContext") -> Dict[str, int]:
        """Generate Graham value signals for each symbol."""
        from shared.backtesting.backtest_engine_v2 import BacktestContext

        cfg = self.config
        signals: Dict[str, int] = {}

        for sym, df in ctx.bars.items():
            if len(df) < cfg.support_sma_length:
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
                    logger.info("%s: value stop loss triggered (%.1f%%)", sym, loss_pct)
                    continue

            score = self._score_value(sym, df)

            # P/E exit check
            fundamentals = self._fundamentals_cache.get(sym, {})
            pe = fundamentals.get("pe_ratio")
            pe_exit = pe is not None and pe > cfg.exit_pe_ratio

            if current_pos <= 0 and score >= cfg.entry_score_threshold and self._near_support(df):
                enricher_ok = True
                if getattr(self, "_enricher", None):
                    enriched = self._enricher.enrich(sym, df)
                    blocked, reason = self._enricher.should_block_entry(enriched)
                    if blocked:
                        enricher_ok = False
                        logger.debug("%s: VALUE entry blocked: %s", sym, reason)
                if enricher_ok:
                    signals[sym] = 1
                    self._entry_prices[sym] = current_price
                    logger.info("%s: VALUE BUY (score=%d/6)", sym, score)
                else:
                    signals[sym] = 0
            elif current_pos > 0 and (score < cfg.exit_score_threshold or pe_exit):
                signals[sym] = 0
                self._entry_prices.pop(sym, None)
                reason = f"score={score}/6" if not pe_exit else f"P/E={pe:.1f}"
                logger.info("%s: VALUE EXIT (%s)", sym, reason)
            else:
                signals[sym] = 1 if current_pos > 0 else 0

        return signals
