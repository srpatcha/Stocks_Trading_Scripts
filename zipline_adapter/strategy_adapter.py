"""
Zipline / QuantConnect Strategy Adapter
==========================================

Bridges stocks_plugin strategy functions into zipline-reloaded's
initialize/handle_data pattern. Also exports strategy skeletons
to QuantConnect Lean C# format.

Requires: pip install zipline-reloaded

Usage:
    adapter = ZiplineStrategyAdapter()
    result = adapter.run_backtest(my_strategy, ["AAPL", "MSFT"], "2020-01-01", "2024-01-01")
    adapter.export_to_lean(my_strategy, "lean_output/MyStrategy.cs")
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import zipline  # type: ignore[import-untyped]
    from zipline.api import (  # type: ignore[import-untyped]
        order_target_percent,
        set_commission,
        set_slippage,
        schedule_function,
        symbol as zipline_symbol,
    )
    from zipline import run_algorithm  # type: ignore[import-untyped]
    from zipline.finance.commission import PerShare  # type: ignore[import-untyped]
    from zipline.finance.slippage import FixedSlippage  # type: ignore[import-untyped]
    _HAS_ZIPLINE = True
except ImportError:
    _HAS_ZIPLINE = False
    logger.debug("zipline-reloaded not installed — zipline adapter unavailable")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.indicators.technical_indicators import TechnicalIndicators as TI


def _require_zipline() -> None:
    if not _HAS_ZIPLINE:
        raise ImportError(
            "zipline-reloaded is required for this adapter. "
            "Install with: pip install zipline-reloaded>=3.0"
        )


class ZiplineStrategyAdapter:
    """Adapter to run stocks_plugin strategies in zipline-reloaded.

    Wraps a strategy function into zipline's initialize/handle_data
    pattern and converts results to BacktestResultV2.

    Args:
        commission_per_share: Commission in dollars per share.
        slippage_spread: Fixed slippage spread in dollars.
    """

    def __init__(
        self,
        commission_per_share: float = 0.005,
        slippage_spread: float = 0.01,
    ) -> None:
        self._commission = commission_per_share
        self._slippage = slippage_spread

    def create_algorithm(
        self,
        strategy_fn: Callable,
        symbols: List[str],
        capital: float = 100_000.0,
        rebalance_freq: str = "daily",
    ) -> Dict[str, Callable]:
        """Create zipline initialize/handle_data functions from a strategy.

        Args:
            strategy_fn: A function that takes a dict of {symbol: DataFrame}
                and returns a dict of {symbol: signal} where signal is -1/0/+1.
            symbols: List of ticker symbols.
            capital: Starting capital.
            rebalance_freq: "daily" or "weekly".

        Returns:
            Dict with 'initialize' and 'handle_data' callables.
        """
        _require_zipline()
        commission = self._commission
        slippage = self._slippage

        def initialize(context: Any) -> None:
            context.symbols = [zipline_symbol(s) for s in symbols]
            context.strategy_fn = strategy_fn
            context.capital = capital
            context.bar_count = 0

            set_commission(PerShare(cost=commission))
            set_slippage(FixedSlippage(spread=slippage))

        def handle_data(context: Any, data: Any) -> None:
            context.bar_count += 1

            # Build OHLCV dict for each symbol
            bars: Dict[str, pd.DataFrame] = {}
            for sym in context.symbols:
                try:
                    hist = data.history(sym, ["open", "high", "low", "close", "volume"], 200, "1d")
                    if not hist.empty:
                        bars[str(sym)] = hist
                except Exception:
                    continue

            if not bars:
                return

            # Get signals from strategy
            try:
                signals = context.strategy_fn(bars)
            except Exception as e:
                logger.warning("Strategy function error: %s", e)
                return

            # Execute signals
            for sym_obj in context.symbols:
                sym_str = str(sym_obj)
                signal = signals.get(sym_str, 0)

                if signal == 1:
                    target_pct = 1.0 / len(context.symbols)
                elif signal == -1:
                    target_pct = -1.0 / len(context.symbols)
                else:
                    target_pct = 0.0

                try:
                    order_target_percent(sym_obj, target_pct)
                except Exception as e:
                    logger.warning("Order failed for %s: %s", sym_str, e)

        return {
            "initialize": initialize,
            "handle_data": handle_data,
        }

    def run_backtest(
        self,
        strategy_fn: Callable,
        symbols: List[str],
        start: str,
        end: str,
        capital: float = 100_000.0,
        data_bundle: str = "quandl",
    ) -> Any:
        """Run a full backtest using zipline.

        Args:
            strategy_fn: Strategy function returning signals.
            symbols: List of ticker symbols.
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).
            capital: Starting capital.
            data_bundle: Zipline data bundle name.

        Returns:
            BacktestResultV2 converted from zipline results.
        """
        _require_zipline()

        algo = self.create_algorithm(strategy_fn, symbols, capital)

        start_dt = pd.Timestamp(start, tz="utc")
        end_dt = pd.Timestamp(end, tz="utc")

        logger.info(
            "Running zipline backtest: %s to %s, symbols=%s, capital=$%s",
            start, end, symbols, capital,
        )

        perf = run_algorithm(
            start=start_dt,
            end=end_dt,
            initialize=algo["initialize"],
            handle_data=algo["handle_data"],
            capital_base=capital,
            bundle=data_bundle,
        )

        return self._convert_to_backtest_result(perf, capital)

    def run_backtest_from_dataframes(
        self,
        strategy_fn: Callable,
        data: Dict[str, pd.DataFrame],
        capital: float = 100_000.0,
    ) -> Any:
        """Run backtest directly from DataFrames (no zipline needed).

        Falls back to BacktestEngineV2 when zipline is not available.
        This provides a universal interface regardless of whether
        zipline is installed.

        Args:
            strategy_fn: Strategy function (context) -> {symbol: signal}.
            data: Dict of {symbol: OHLCV DataFrame}.
            capital: Starting capital.

        Returns:
            BacktestResultV2.
        """
        from shared.backtesting.backtest_engine_v2 import BacktestEngineV2

        engine = BacktestEngineV2(initial_capital=capital)
        engine.load_data(data)
        return engine.run(strategy_fn)

    @staticmethod
    def _convert_to_backtest_result(perf: pd.DataFrame, capital: float) -> Any:
        """Convert zipline performance DataFrame to BacktestResultV2.

        Args:
            perf: Zipline performance DataFrame.
            capital: Initial capital.

        Returns:
            BacktestResultV2 with computed metrics.
        """
        from shared.backtesting.backtest_engine_v2 import BacktestResultV2

        equity = perf["portfolio_value"].tolist()
        returns = perf["returns"].tolist()

        total_return = (equity[-1] - capital) / capital if equity else 0

        returns_arr = np.array(returns, dtype=float)
        mean_ret = float(np.mean(returns_arr))
        std_ret = float(np.std(returns_arr, ddof=1)) if len(returns_arr) > 1 else 0
        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0

        downside = returns_arr[returns_arr < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0
        sortino = mean_ret / downside_std * np.sqrt(252) if downside_std > 0 else 0

        peak = np.maximum.accumulate(equity)
        dd = (np.array(equity) - peak) / peak
        # Signed, not absolute. np.min(dd) is already negative, and every other
        # adapter reports it that way — backtrader_adapter/analyzers.py:126 and
        # portfolio_backtester/portfolio_engine.py:166 both use float(np.min(dd)).
        # Wrapping it in np.abs() made this adapter the only one returning a
        # positive drawdown, so the same BacktestResultV2 field carried opposite
        # signs depending on which engine produced it, and any downstream
        # threshold comparison or chart label was inverted for zipline runs.
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0

        n_trades = 0
        if "transactions" in perf.columns:
            for txns in perf["transactions"]:
                if isinstance(txns, list):
                    n_trades += len(txns)

        return BacktestResultV2(
            total_return=round(total_return, 6),
            sharpe_ratio=round(float(sharpe), 4),
            sortino_ratio=round(float(sortino), 4),
            max_drawdown=round(max_dd, 6),
            total_trades=n_trades,
            equity_curve=equity,
        )

    @staticmethod
    def export_to_lean(
        strategy_fn: Callable,
        output_path: str,
        strategy_name: str = "StocksPluginStrategy",
        symbols: Optional[List[str]] = None,
    ) -> None:
        """Export strategy as a QuantConnect Lean C# template.

        Generates a structured C# skeleton with indicator setup,
        Initialize(), and OnData() methods. User fills in specific logic.

        Args:
            strategy_fn: The strategy function (used for docstring extraction).
            output_path: File path for the C# output.
            strategy_name: Name for the C# class.
            symbols: List of symbols to include in universe.
        """
        symbols = symbols or ["AAPL", "MSFT", "GOOGL"]
        symbols_str = ", ".join(f'"{s}"' for s in symbols)

        doc = strategy_fn.__doc__ or "Ported from stocks_plugin Python strategy"
        doc_lines = doc.strip().split("\n")
        doc_comment = "\n".join(f"    /// {line.strip()}" for line in doc_lines[:5])

        template = f"""/*
 * {strategy_name} — QuantConnect Lean Algorithm
 * Auto-generated from stocks_plugin Python strategy.
 * Fill in specific entry/exit logic where indicated.
 *
 * Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 */

using QuantConnect;
using QuantConnect.Algorithm;
using QuantConnect.Data;
using QuantConnect.Indicators;
using QuantConnect.Orders;
using System;
using System.Collections.Generic;

namespace QuantConnect.Algorithm.CSharp
{{
{doc_comment}
    public class {strategy_name} : QCAlgorithm
    {{
        // Indicators
        private Dictionary<string, RelativeStrengthIndex> _rsi = new();
        private Dictionary<string, ExponentialMovingAverage> _emaFast = new();
        private Dictionary<string, ExponentialMovingAverage> _emaSlow = new();
        private Dictionary<string, BollingerBands> _bb = new();
        private Dictionary<string, AverageTrueRange> _atr = new();
        private Dictionary<string, AverageDirectionalIndex> _adx = new();

        private readonly string[] _symbols = new[] {{ {symbols_str} }};

        public override void Initialize()
        {{
            SetStartDate(2020, 1, 1);
            SetEndDate(DateTime.Now);
            SetCash(100000);

            foreach (var ticker in _symbols)
            {{
                var equity = AddEquity(ticker, Resolution.Daily);
                var symbol = equity.Symbol;

                _rsi[ticker] = RSI(symbol, 14, MovingAverageType.Wilders, Resolution.Daily);
                _emaFast[ticker] = EMA(symbol, 9, Resolution.Daily);
                _emaSlow[ticker] = EMA(symbol, 21, Resolution.Daily);
                _bb[ticker] = BB(symbol, 20, 2, MovingAverageType.Simple, Resolution.Daily);
                _atr[ticker] = ATR(symbol, 14, MovingAverageType.Wilders, Resolution.Daily);
                _adx[ticker] = ADX(symbol, 14, Resolution.Daily);
            }}

            // Rebalance daily at market open
            Schedule.On(DateRules.EveryDay(), TimeRules.AfterMarketOpen(_symbols[0], 30),
                () => Rebalance());
        }}

        public override void OnData(Slice data)
        {{
            // Data-driven events handled here if needed
            // Main logic is in Rebalance() scheduled function
        }}

        private void Rebalance()
        {{
            foreach (var ticker in _symbols)
            {{
                if (!_rsi[ticker].IsReady || !_adx[ticker].IsReady)
                    continue;

                var adxValue = _adx[ticker].Current.Value;
                var rsiValue = _rsi[ticker].Current.Value;
                var price = Securities[ticker].Price;

                // ═══ REGIME DETECTION ═══
                bool isTrending = adxValue > 25;
                bool isRanging = adxValue < 20;

                // ═══ ENTRY LOGIC ═══
                if (isTrending)
                {{
                    // Trend-following: EMA crossover
                    if (_emaFast[ticker].Current.Value > _emaSlow[ticker].Current.Value
                        && !Portfolio[ticker].Invested)
                    {{
                        var targetPct = 1.0m / _symbols.Length;
                        SetHoldings(ticker, targetPct);
                        Debug($"TREND LONG: {{ticker}} @ {{price}}");
                    }}
                }}
                else if (isRanging)
                {{
                    // Mean reversion: RSI + BB
                    if (rsiValue < 30 && price <= _bb[ticker].LowerBand.Current.Value
                        && !Portfolio[ticker].Invested)
                    {{
                        var targetPct = 1.0m / _symbols.Length;
                        SetHoldings(ticker, targetPct);
                        Debug($"MR LONG: {{ticker}} RSI={{rsiValue:F1}} @ {{price}}");
                    }}
                }}

                // ═══ EXIT LOGIC ═══
                if (Portfolio[ticker].Invested)
                {{
                    if (isRanging && price >= _bb[ticker].MiddleBand.Current.Value)
                    {{
                        Liquidate(ticker);
                        Debug($"MR EXIT: {{ticker}} at BB mid @ {{price}}");
                    }}
                    else if (isTrending && rsiValue > 70)
                    {{
                        Liquidate(ticker);
                        Debug($"TREND EXIT: {{ticker}} RSI={{rsiValue:F1}} @ {{price}}");
                    }}
                }}
            }}
        }}

        public override void OnOrderEvent(OrderEvent orderEvent)
        {{
            if (orderEvent.Status == OrderStatus.Filled)
            {{
                Debug($"Order filled: {{orderEvent.Symbol}} {{orderEvent.Direction}} "
                    + $"{{orderEvent.FillQuantity}} @ {{orderEvent.FillPrice}}");
            }}
        }}
    }}
}}
"""
        filepath = Path(output_path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(template, encoding="utf-8")

        logger.info(
            "Lean C# template exported to %s (%d symbols)",
            output_path,
            len(symbols),
        )
