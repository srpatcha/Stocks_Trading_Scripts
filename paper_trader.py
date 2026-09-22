#!/usr/bin/env python3
"""
Paper Trading Simulator
=========================

Standalone paper trader that uses REAL market data from Yahoo Finance
but executes trades in a simulated account. No broker needed.

Features:
- Real-time OHLCV data from Yahoo Finance (free, 15-min delay)
- All 15 strategies available
- Full risk management active
- Trade journal logging
- Performance tracking with P&L, win rate, Sharpe
- Runs continuously or one-shot scan

Usage:
    # Scan once and show signals
    python paper_trader.py --symbols AAPL,MSFT,GOOGL --strategy trend_following

    # Run all strategies on a stock
    python paper_trader.py --symbols AAPL --strategy all

    # Run meta_ensemble (uses ALL data sources)
    python paper_trader.py --symbols AAPL,MSFT,TSLA,NVDA --strategy meta_ensemble

    # Full portfolio scan with top picks
    python paper_trader.py --scan-universe
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from shared.data.public_data_fetcher import PublicDataFetcher
from shared.risk_manager import RiskManager, RiskManagerConfig
from shared.backtesting.backtest_engine_v2 import BacktestContext
from shared.utils.paths import stocks_plugin_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_trader")

PAPER_STATE_FILE = os.path.join(
    stocks_plugin_root(), "paper_trader_state.json"
)

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "UNH", "JNJ", "XOM", "PG", "HD", "MA",
]


@dataclass
class PaperPosition:
    symbol: str
    shares: int
    entry_price: float
    entry_date: str
    strategy: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class PaperTradeRecord:
    symbol: str
    strategy: str
    direction: str
    shares: int
    entry_price: float
    exit_price: float
    pnl: float
    entry_date: str
    exit_date: str


@dataclass
class PaperAccount:
    initial_capital: float = 100_000.0
    cash: float = 100_000.0
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0


class PaperTrader:
    """Standalone paper trading simulator using real market data."""

    def __init__(
        self,
        capital: float = 100_000.0,
        risk_pct: float = 1.0,
    ) -> None:
        self.fetcher = PublicDataFetcher()
        self.risk_manager = RiskManager(config=RiskManagerConfig(
            total_capital=capital,
            risk_per_trade_pct=risk_pct,
            max_daily_loss=capital * 0.02,
            max_position_pct_equity=15.0,
            max_shares_per_order=5000,
            min_seconds_between_trades=0,
            max_trades_per_hour=100,
        ))
        self.account = PaperAccount(initial_capital=capital, cash=capital)
        self._strategies: Dict[str, Any] = {}
        self._load_state()

    def _load_strategies(self):
        """Load all strategy classes."""
        import strategies.examples.trend_following
        import strategies.examples.breakout
        import strategies.examples.mean_reversion
        import strategies.examples.canslim_strategy
        import strategies.examples.value_strategy
        import strategies.examples.darvas_box
        import strategies.examples.triple_screen
        import strategies.examples.sentiment_strategy
        import strategies.examples.earnings_strategy
        import strategies.examples.sector_rotation
        import strategies.examples.meta_strategy
        import strategies.examples.factor_portfolio
        import strategies.examples.ml_rl_strategy
        import strategies.examples.self_learning_strategy
        from strategies import STRATEGY_REGISTRY
        return STRATEGY_REGISTRY

    def get_strategy(self, name: str):
        """Get or create a strategy instance."""
        if name not in self._strategies:
            registry = self._load_strategies()
            if name not in registry:
                raise ValueError(f"Unknown strategy: {name}. Available: {list(registry.keys())}")
            self._strategies[name] = registry[name]()
        return self._strategies[name]

    def fetch_data(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for all symbols."""
        data = {}
        for sym in symbols:
            df = self.fetcher.fetch_ohlcv(sym, period=period, interval="1d")
            if df is not None and not df.empty:
                data[sym] = df
                logger.info("Fetched %d bars for %s (latest: $%.2f)",
                           len(df), sym, float(df["close"].iloc[-1]))
            else:
                logger.warning("No data for %s", sym)
        return data

    def run_strategy(
        self, strategy_name: str, symbols: List[str], period: str = "1y"
    ) -> Dict[str, Dict[str, Any]]:
        """Run a single strategy and return signals with context."""
        data = self.fetch_data(symbols, period)
        if not data:
            logger.error("No data available")
            return {}

        strategy = self.get_strategy(strategy_name)

        # Build context
        positions = {
            sym: pos.shares for sym, pos in self.account.positions.items()
        }
        portfolio_val = self.account.cash + sum(
            pos.shares * pos.current_price for pos in self.account.positions.values()
        )

        ctx = BacktestContext(
            bar_index=0,
            bars=data,
            positions=positions,
            capital=self.account.cash,
            portfolio_value=portfolio_val,
        )

        signals = strategy.generate_signals(ctx)

        results = {}
        for sym, signal in signals.items():
            if sym not in data:
                continue
            price = float(data[sym]["close"].iloc[-1])
            self.account.positions.get(sym, PaperPosition(sym, 0, 0, "", ""))

            # Validate order before recording
            if signal != 0:
                direction = "LONG" if signal > 0 else "SHORT"
                shares = self.risk_manager.calculate_position_size(sym, price)
                ok, reason = self.risk_manager.validate_order(
                    sym, shares, price, price, direction=direction,
                    avg_daily_volume=float(data[sym]["volume"].iloc[-20:].mean()) if len(data[sym]) >= 20 else None,
                )
            else:
                ok, reason = True, "FLAT"
                shares = 0
                direction = "FLAT"

            results[sym] = {
                "signal": signal,
                "signal_text": {1: "BUY", -1: "SELL", 0: "HOLD", 2: "ADD"}.get(signal, "?"),
                "price": price,
                "shares": shares,
                "direction": direction,
                "order_valid": ok,
                "order_reason": reason,
                "strategy": strategy_name,
            }

            # Execute paper trade
            if ok and signal == 1 and sym not in self.account.positions:
                self._paper_buy(sym, shares, price, strategy_name)
            elif signal == 0 and sym in self.account.positions:
                self._paper_sell(sym, price)

        return results

    def _paper_buy(self, symbol: str, shares: int, price: float, strategy: str):
        """Execute a paper buy."""
        cost = shares * price
        if cost > self.account.cash:
            shares = int(self.account.cash / price)
            cost = shares * price
        if shares <= 0:
            return

        self.account.cash -= cost
        self.account.positions[symbol] = PaperPosition(
            symbol=symbol,
            shares=shares,
            entry_price=price,
            entry_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            strategy=strategy,
            current_price=price,
        )
        logger.info("📈 PAPER BUY: %d shares of %s @ $%.2f ($%.0f)",
                    shares, symbol, price, cost)

    def _paper_sell(self, symbol: str, price: float):
        """Execute a paper sell (close position)."""
        if symbol not in self.account.positions:
            return

        pos = self.account.positions[symbol]
        proceeds = pos.shares * price
        pnl = (price - pos.entry_price) * pos.shares

        self.account.cash += proceeds
        self.account.total_pnl += pnl
        self.account.total_trades += 1
        if pnl > 0:
            self.account.winning_trades += 1

        self.account.trade_history.append({
            "symbol": symbol,
            "strategy": pos.strategy,
            "shares": pos.shares,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "pnl": round(pnl, 2),
            "entry_date": pos.entry_date,
            "exit_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

        emoji = "✅" if pnl > 0 else "❌"
        logger.info("%s PAPER SELL: %s %d shares @ $%.2f → P&L: $%.2f",
                    emoji, symbol, pos.shares, price, pnl)

        del self.account.positions[symbol]
        self.risk_manager.record_trade(symbol, pnl)

    def scan_all_strategies(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """Run all strategies and aggregate signals."""
        all_signals: Dict[str, List[Dict]] = {sym: [] for sym in symbols}

        strategy_names = [
            "trend_following", "breakout", "mean_reversion",
            "canslim", "value", "darvas_box", "triple_screen",
            "sentiment", "earnings", "meta_ensemble",
        ]

        for strat_name in strategy_names:
            try:
                results = self.run_strategy(strat_name, symbols)
                for sym, result in results.items():
                    if result["signal"] != 0:
                        all_signals[sym].append(result)
            except Exception as e:
                logger.warning("Strategy %s failed: %s", strat_name, e)

        return all_signals

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        # Update current prices
        for sym, pos in self.account.positions.items():
            price = self.fetcher.fetch_latest_price(sym)
            if price > 0:
                pos.current_price = price
                pos.unrealized_pnl = (price - pos.entry_price) * pos.shares

        positions_value = sum(
            pos.shares * pos.current_price for pos in self.account.positions.values()
        )
        total_equity = self.account.cash + positions_value
        total_return = (total_equity - self.account.initial_capital) / self.account.initial_capital * 100
        win_rate = (
            self.account.winning_trades / self.account.total_trades * 100
            if self.account.total_trades > 0 else 0
        )

        return {
            "equity": round(total_equity, 2),
            "cash": round(self.account.cash, 2),
            "positions_value": round(positions_value, 2),
            "total_return_pct": round(total_return, 2),
            "total_pnl": round(self.account.total_pnl, 2),
            "unrealized_pnl": round(sum(p.unrealized_pnl for p in self.account.positions.values()), 2),
            "total_trades": self.account.total_trades,
            "win_rate": round(win_rate, 1),
            "open_positions": len(self.account.positions),
            "positions": {
                sym: {
                    "shares": pos.shares,
                    "entry": pos.entry_price,
                    "current": pos.current_price,
                    "pnl": round(pos.unrealized_pnl, 2),
                    "strategy": pos.strategy,
                }
                for sym, pos in self.account.positions.items()
            },
            "risk_status": self.risk_manager.get_status(),
        }

    def _save_state(self):
        """Save paper trading state to disk."""
        state = {
            "cash": self.account.cash,
            "positions": {
                sym: {
                    "shares": pos.shares,
                    "entry_price": pos.entry_price,
                    "entry_date": pos.entry_date,
                    "strategy": pos.strategy,
                }
                for sym, pos in self.account.positions.items()
            },
            "trade_history": self.account.trade_history[-100:],
            "total_pnl": self.account.total_pnl,
            "total_trades": self.account.total_trades,
            "winning_trades": self.account.winning_trades,
            "saved_at": datetime.now().isoformat(),
        }
        Path(PAPER_STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(PAPER_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load paper trading state from disk."""
        if not os.path.exists(PAPER_STATE_FILE):
            return
        try:
            with open(PAPER_STATE_FILE) as f:
                state = json.load(f)
            self.account.cash = state.get("cash", self.account.initial_capital)
            self.account.total_pnl = state.get("total_pnl", 0)
            self.account.total_trades = state.get("total_trades", 0)
            self.account.winning_trades = state.get("winning_trades", 0)
            self.account.trade_history = state.get("trade_history", [])
            for sym, pos_data in state.get("positions", {}).items():
                self.account.positions[sym] = PaperPosition(
                    symbol=sym,
                    shares=pos_data["shares"],
                    entry_price=pos_data["entry_price"],
                    entry_date=pos_data.get("entry_date", ""),
                    strategy=pos_data.get("strategy", "unknown"),
                )
            if self.account.positions:
                logger.info("Resumed %d paper positions", len(self.account.positions))
        except Exception as e:
            logger.warning("Failed to load paper state: %s", e)


def print_signals(results: Dict[str, Dict[str, Any]]):
    """Pretty-print trading signals."""
    print("\n" + "=" * 70)
    print(f"{'Symbol':<8} {'Signal':<8} {'Price':>10} {'Shares':>8} {'Valid':>6} {'Reason'}")
    print("─" * 70)
    for sym, r in sorted(results.items()):
        signal_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪", "ADD": "🟡"}.get(r["signal_text"], "?")
        valid = "✅" if r["order_valid"] else "❌"
        print(f"{sym:<8} {signal_color} {r['signal_text']:<5} ${r['price']:>9.2f} {r['shares']:>8} {valid:>6}  {r['order_reason']}")
    print("=" * 70)


def print_portfolio(summary: Dict[str, Any]):
    """Pretty-print portfolio summary."""
    print("\n" + "=" * 70)
    print("  PAPER TRADING PORTFOLIO")
    print("=" * 70)
    print(f"  Equity:          ${summary['equity']:>12,.2f}")
    print(f"  Cash:            ${summary['cash']:>12,.2f}")
    print(f"  Positions Value: ${summary['positions_value']:>12,.2f}")
    print(f"  Total Return:    {summary['total_return_pct']:>12.2f}%")
    print(f"  Realized P&L:    ${summary['total_pnl']:>12,.2f}")
    print(f"  Unrealized P&L:  ${summary['unrealized_pnl']:>12,.2f}")
    print(f"  Total Trades:    {summary['total_trades']:>12}")
    print(f"  Win Rate:        {summary['win_rate']:>12.1f}%")
    print(f"  Open Positions:  {summary['open_positions']:>12}")

    if summary["positions"]:
        print("\n  Open Positions:")
        print(f"  {'Symbol':<8} {'Shares':>8} {'Entry':>10} {'Current':>10} {'P&L':>10} {'Strategy'}")
        print("  " + "─" * 60)
        for sym, pos in summary["positions"].items():
            pnl_emoji = "✅" if pos["pnl"] >= 0 else "❌"
            print(f"  {sym:<8} {pos['shares']:>8} ${pos['entry']:>9.2f} ${pos['current']:>9.2f} "
                  f"{pnl_emoji}${pos['pnl']:>8.2f}  {pos['strategy']}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Simulator")
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL",
                       help="Comma-separated stock symbols")
    parser.add_argument("--strategy", type=str, default="meta_ensemble",
                       help="Strategy name (or 'all' to scan all strategies)")
    parser.add_argument("--capital", type=float, default=100_000,
                       help="Starting capital (default: $100,000)")
    parser.add_argument("--risk-pct", type=float, default=1.0,
                       help="Risk per trade %% (default: 1.0)")
    parser.add_argument("--scan-universe", action="store_true",
                       help="Scan default universe of 15 stocks")
    parser.add_argument("--portfolio", action="store_true",
                       help="Show current portfolio status only")
    args = parser.parse_args()

    trader = PaperTrader(capital=args.capital, risk_pct=args.risk_pct)

    if args.portfolio:
        summary = trader.get_portfolio_summary()
        print_portfolio(summary)
        return

    symbols = DEFAULT_UNIVERSE if args.scan_universe else args.symbols.split(",")
    symbols = [s.strip().upper() for s in symbols]

    print(f"\n🔍 Scanning {len(symbols)} symbols with strategy: {args.strategy}")
    print(f"   Capital: ${args.capital:,.0f} | Risk: {args.risk_pct}% per trade")

    if args.strategy == "all" or args.scan_universe:
        all_signals = trader.scan_all_strategies(symbols)
        print("\n" + "=" * 70)
        print("  MULTI-STRATEGY SCAN RESULTS")
        print("=" * 70)
        for sym in sorted(all_signals.keys()):
            sigs = all_signals[sym]
            if sigs:
                buy_count = sum(1 for s in sigs if s["signal"] > 0)
                sell_count = sum(1 for s in sigs if s["signal"] < 0)
                strategies = [s["strategy"] for s in sigs if s["signal"] > 0]
                if buy_count > 0:
                    print(f"  🟢 {sym}: {buy_count} BUY signals from: {', '.join(strategies)}")
                if sell_count > 0:
                    print(f"  🔴 {sym}: {sell_count} SELL signals")
    else:
        results = trader.run_strategy(args.strategy, symbols)
        print_signals(results)

    # Show portfolio
    summary = trader.get_portfolio_summary()
    print_portfolio(summary)

    # Save state
    trader._save_state()
    print(f"\n💾 Paper trading state saved to {PAPER_STATE_FILE}")


if __name__ == "__main__":
    main()
