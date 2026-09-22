"""
Backtrader Analyzers and Result Converter
==========================================

Custom analyzers for Backtrader and a converter to BacktestResultV2.

Usage:
    cerebro.addanalyzer(BacktestResultAnalyzer)
    results = cerebro.run()
    bt_result = to_backtest_result_v2(results[0])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import backtrader as bt  # type: ignore[import-untyped]
    _HAS_BT = True
except ImportError:
    _HAS_BT = False

try:
    from shared.backtesting.backtest_engine_v2 import BacktestResultV2, TradeRecord
    _HAS_ENGINE = True
except ImportError:
    _HAS_ENGINE = False


if _HAS_BT:

    class BacktestResultAnalyzer(bt.Analyzer):
        """Collects trades, equity curve, and performance metrics from a Backtrader run."""

        def __init__(self):
            self.equity_curve = []
            self.dates = []
            self.trades: List[Dict[str, Any]] = []

        def next(self):
            dt = self.strategy.datas[0].datetime.datetime(0)
            self.dates.append(dt)
            self.equity_curve.append(self.strategy.broker.getvalue())

        def notify_trade(self, trade):
            if trade.isclosed:
                self.trades.append({
                    "pnl": trade.pnl,
                    "pnlcomm": trade.pnlcomm,
                    "size": trade.size,
                    "price": trade.price,
                    "barlen": trade.barlen,
                    "dtopen": bt.num2date(trade.dtopen),
                    "dtclose": bt.num2date(trade.dtclose),
                })

        def get_analysis(self):
            return {
                "equity_curve": self.equity_curve,
                "dates": self.dates,
                "trades": self.trades,
            }

else:
    BacktestResultAnalyzer = None  # type: ignore[assignment,misc]


def _find_result_analyzer(analyzers):
    """Locate the BacktestResultAnalyzer on a strategy's analyzer collection.

    backtrader's ItemCollection.getitems() yields ``(name, analyzer)`` PAIRS,
    not bare analyzers, and the collection itself is iterable via
    __getitem__/__len__. Both shapes are handled so this does not depend on
    which backtrader version is installed.

    Returns the analyzer, or None when it was never registered.
    """
    if BacktestResultAnalyzer is None:
        return None

    candidates = []
    getitems = getattr(analyzers, "getitems", None)
    if callable(getitems):
        for entry in getitems():
            # (name, analyzer) in real backtrader; a bare analyzer elsewhere.
            if isinstance(entry, tuple) and len(entry) == 2:
                candidates.append(entry[1])
            else:
                candidates.append(entry)
    else:
        try:
            candidates = list(analyzers)
        except TypeError:
            candidates = []

    for candidate in candidates:
        if isinstance(candidate, BacktestResultAnalyzer):
            return candidate
    return None


def to_backtest_result_v2(
    strategy_result,
    initial_capital: float = 100_000.0,
) -> "BacktestResultV2":
    """Convert Backtrader strategy result to BacktestResultV2.

    Args:
        strategy_result: Result from cerebro.run()[0]
        initial_capital: Starting capital

    Returns:
        BacktestResultV2 with unified metrics
    """
    if not _HAS_ENGINE:
        raise ImportError("BacktestResultV2 not available")
    if not _HAS_BT:
        raise ImportError("backtrader is required for to_backtest_result_v2")

    # backtrader's ItemCollection exposes getbyname/getitems/getnames — there
    # is no getbytype, so this raised AttributeError on every call and
    # run_backtest() could never complete. The failure was invisible because
    # backtrader is an optional dependency and the test skipped without it.
    #
    # Searching by type rather than by name keeps this working whether the
    # analyzer was registered with an explicit _name or with backtrader's
    # auto-generated one.
    analyzer = _find_result_analyzer(strategy_result.analyzers)
    if analyzer is None:
        raise ValueError(
            "BacktestResultAnalyzer not found. Add it via "
            "cerebro.addanalyzer(BacktestResultAnalyzer) before cerebro.run()."
        )

    analysis = analyzer.get_analysis()

    equity = np.array(analysis["equity_curve"])
    dates = analysis["dates"]

    if len(equity) == 0:
        raise ValueError("Empty equity curve - no bars processed")

    # Core metrics
    total_return = (equity[-1] - initial_capital) / initial_capital
    n_days = max((dates[-1] - dates[0]).days, 1) if len(dates) > 1 else 1
    n_years = n_days / 365.25
    cagr = (equity[-1] / initial_capital) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Returns
    returns = np.diff(equity) / equity[:-1]
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0

    neg_returns = returns[returns < 0]
    if len(neg_returns) <= 1:
        downside_std = float(abs(neg_returns.mean())) if len(neg_returns) == 1 else 1e-10
    else:
        downside_std = float(np.std(neg_returns))
    if downside_std == 0:
        downside_std = 1e-10
    sortino = float(np.mean(returns) / downside_std * np.sqrt(252))

    # Drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Trades
    trades = analysis["trades"]
    trade_records = []
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for t in trades:
        pnl = t["pnlcomm"]
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            losses += 1
            gross_loss += abs(pnl)

        trade_records.append(TradeRecord(
            symbol="",
            direction="LONG" if t["size"] > 0 else "SHORT",
            entry_date=str(t.get("dtopen", "")),
            entry_price=t["price"],
            exit_date=str(t.get("dtclose", "")),
            exit_price=t["price"] + pnl / max(abs(t["size"]), 1),
            shares=abs(t["size"]),
            pnl=pnl,
            pnl_pct=pnl / initial_capital,
            commission=t["pnl"] - t["pnlcomm"],
            slippage=0.0,
            hold_bars=t.get("barlen", 0),
        ))

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_win = gross_profit / wins if wins > 0 else 0.0
    avg_loss = gross_loss / losses if losses > 0 else 0.0
    expectancy = (win_rate * avg_win - (1 - win_rate) * avg_loss) if total_trades > 0 else 0.0

    equity_series = pd.Series(equity, index=pd.DatetimeIndex(dates))

    return BacktestResultV2(
        total_return=total_return,
        cagr=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=total_trades,
        expectancy=expectancy,
        avg_trade_duration=sum(t.get("barlen", 0) for t in trades) / max(total_trades, 1),
        max_consecutive_wins=0,
        max_consecutive_losses=0,
        monthly_returns=pd.Series(dtype=float),
        alpha=0.0,
        beta=0.0,
        information_ratio=0.0,
        tracking_error=0.0,
        equity_curve=equity_series,
        trade_log=trade_records,
        trades=trade_records,
        long_trades=sum(1 for t in trades if t["size"] > 0),
        short_trades=sum(1 for t in trades if t["size"] < 0),
        avg_win=avg_win,
        avg_loss=avg_loss,
    )
