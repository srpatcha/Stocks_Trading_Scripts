"""
Strategy Runner CLI
=====================

Command-line interface for running, backtesting, and optimizing strategies.

Usage:
    python -m strategies.runner list
    python -m strategies.runner backtest --strategy trend_following --data synthetic
    python -m strategies.runner backtest --strategy breakout --data SPY.csv --chart
    python -m strategies.runner backtest --strategy mean_reversion --data BTC.csv --params rsi_oversold=25,rsi_overbought=75
    python -m strategies.runner optimize --strategy trend_following --data synthetic --n-trials 50
"""

from __future__ import annotations

import argparse
import json
import sys
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

# Import every example strategy so the @register_strategy decorators run.
#
# This was a hand-maintained list of six imports, which registered only 7 of
# the 15 strategies — canslim, darvas_box, earnings, meta_ensemble,
# sector_rotation, sentiment, triple_screen and value were simply unreachable
# from this CLI, while the README advertises all 15. Discovery keeps the list
# from drifting again when a strategy is added.
def _import_all_example_strategies() -> None:
    import importlib
    import pkgutil

    import strategies.examples as _examples

    for module in pkgutil.iter_modules(_examples.__path__):
        if module.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"strategies.examples.{module.name}")
        except Exception as exc:          # optional deps (torch, lightgbm, ...)
            logger.warning(
                "Strategy module %r could not be imported and will not be "
                "listed: %s", module.name, exc,
            )


_import_all_example_strategies()

from strategies import STRATEGY_REGISTRY, list_strategies
from shared.backtesting.backtest_engine_v2 import BacktestEngineV2, BacktestResultV2


def _generate_synthetic_data(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate default synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_bars)
    price = 100.0
    prices = []
    for i in range(n_bars):
        regime = np.sin(2 * np.pi * i / 200)
        drift = 0.0003 * regime
        ret = drift + rng.randn() * 0.015
        price *= 1 + ret
        high = price * (1 + abs(rng.randn()) * 0.006)
        low = price * (1 - abs(rng.randn()) * 0.006)
        prices.append({
            "date": dates[i],
            "open": price * (1 + rng.randn() * 0.002),
            "high": high,
            "low": low,
            "close": price,
            "volume": int(rng.uniform(500_000, 2_000_000)),
        })
    return pd.DataFrame(prices)


def _generate_synthetic_universe(
    n_stocks: int = 12, n_bars: int = 400, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic multi-stock universe for factor strategies."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_bars)
    tickers = [f"STOCK_{chr(65 + i)}" for i in range(n_stocks)]
    prices = {}
    for t in tickers:
        drift = rng.uniform(-0.0002, 0.001)
        vol = rng.uniform(0.01, 0.025)
        p = 50.0 + rng.uniform(0, 100)
        series = [p]
        for _ in range(n_bars - 1):
            p *= 1 + drift + rng.randn() * vol
            series.append(p)
        prices[t] = series
    return pd.DataFrame(prices, index=dates)


def _load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file or generate synthetic data."""
    if data_path.lower() == "synthetic":
        return _generate_synthetic_data()
    if not os.path.isfile(data_path):
        print(f"Error: File not found: {data_path}")
        sys.exit(1)
    df = pd.read_csv(data_path)
    return df


def _parse_params(params_str: str) -> Dict[str, Any]:
    """Parse key=value parameter string into dict."""
    params: Dict[str, Any] = {}
    if not params_str:
        return params
    for pair in params_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()
        # Auto-convert types
        if val.lower() in ("true", "false"):
            params[key] = val.lower() == "true"
        else:
            try:
                params[key] = int(val)
            except ValueError:
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val
    return params


def _print_result(result: BacktestResultV2) -> None:
    """Print backtest results summary."""
    print("\n" + "=" * 50)
    print("  BACKTEST RESULTS")
    print("=" * 50)
    print(f"  Total Return:       {result.total_return:>10.2%}")
    print(f"  CAGR:               {result.cagr:>10.2%}")
    print(f"  Sharpe Ratio:       {result.sharpe_ratio:>10.4f}")
    print(f"  Sortino Ratio:      {result.sortino_ratio:>10.4f}")
    print(f"  Max Drawdown:       {result.max_drawdown:>10.2%}")
    print(f"  Calmar Ratio:       {result.calmar_ratio:>10.4f}")
    print(f"  Win Rate:           {result.win_rate:>10.2%}")
    print(f"  Profit Factor:      {result.profit_factor:>10.4f}")
    print(f"  Total Trades:       {result.total_trades:>10d}")
    print(f"    Long Trades:      {result.long_trades:>10d}")
    print(f"    Short Trades:     {result.short_trades:>10d}")
    print(f"  Avg Win:            ${result.avg_win:>9.2f}")
    print(f"  Avg Loss:           ${result.avg_loss:>9.2f}")
    print(f"  Expectancy:         ${result.expectancy:>9.2f}")
    print(f"  Avg Trade Duration: {result.avg_trade_duration:>10.1f} bars")
    print(f"  Max Consec. Wins:   {result.max_consecutive_wins:>10d}")
    print(f"  Max Consec. Losses: {result.max_consecutive_losses:>10d}")
    if result.alpha != 0 or result.beta != 0:
        print(f"  Alpha:              {result.alpha:>10.4f}")
        print(f"  Beta:               {result.beta:>10.4f}")
    print("=" * 50)


def _result_to_dict(result: BacktestResultV2) -> Dict[str, Any]:
    """Convert BacktestResultV2 to a serializable dict."""
    return {
        "total_return": result.total_return,
        "cagr": result.cagr,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "max_drawdown": result.max_drawdown,
        "calmar_ratio": result.calmar_ratio,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "total_trades": result.total_trades,
        "long_trades": result.long_trades,
        "short_trades": result.short_trades,
        "avg_win": result.avg_win,
        "avg_loss": result.avg_loss,
        "expectancy": result.expectancy,
        "avg_trade_duration": result.avg_trade_duration,
        "max_consecutive_wins": result.max_consecutive_wins,
        "max_consecutive_losses": result.max_consecutive_losses,
    }


def cmd_list(args: argparse.Namespace) -> None:
    """List all registered strategies."""
    strategies = list_strategies()
    print("\nAvailable Strategies:")
    print("-" * 60)
    for name, desc in strategies.items():
        print(f"  {name:<20s} {desc}")
    print(f"\nTotal: {len(strategies)} strategies")


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run a backtest."""
    strategy_name = args.strategy
    if strategy_name not in STRATEGY_REGISTRY:
        print(f"Error: Unknown strategy '{strategy_name}'")
        print(f"Available: {', '.join(STRATEGY_REGISTRY.keys())}")
        sys.exit(1)

    # Handle factor strategy (needs universe data)
    if strategy_name == "factor":
        params = _parse_params(args.params) if args.params else {}
        strategy_cls = STRATEGY_REGISTRY[strategy_name]
        strategy = strategy_cls.from_params(params) if params else strategy_cls()

        if args.data.lower() == "synthetic":
            universe = _generate_synthetic_universe()
        else:
            universe = pd.read_csv(args.data, index_col=0, parse_dates=True)

        print(f"Running {strategy_name} on {len(universe.columns)} stocks, {len(universe)} bars...")
        result = strategy.run_backtest(universe, initial_capital=args.capital)
        _print_result(result)

        if args.chart:
            _render_chart(result)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(_result_to_dict(result), f, indent=2)
            print(f"Results saved to {args.output}")
        return

    # Handle ML/RL strategies (need training)
    if strategy_name in ("ml", "rl", "self_learning"):
        params = _parse_params(args.params) if args.params else {}
        strategy_cls = STRATEGY_REGISTRY[strategy_name]
        strategy = strategy_cls.from_params(params) if params else strategy_cls()

        df = _load_data(args.data)
        print(f"Training {strategy_name} strategy...")
        strategy.train(df)
        print(f"Running backtest on {len(df)} bars...")

        engine = BacktestEngineV2(initial_capital=args.capital)
        engine.load_data(df)
        result = engine.run(strategy.generate_signals)
        _print_result(result)

        if args.chart:
            _render_chart(result)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(_result_to_dict(result), f, indent=2)
            print(f"Results saved to {args.output}")
        return

    # Standard strategies
    params = _parse_params(args.params) if args.params else {}
    strategy_cls = STRATEGY_REGISTRY[strategy_name]
    strategy = strategy_cls.from_params(params) if params else strategy_cls()

    df = _load_data(args.data)
    print(f"Running {strategy_name} on {len(df)} bars...")

    engine = BacktestEngineV2(initial_capital=args.capital)
    engine.load_data(df)
    result = engine.run(strategy.generate_signals)
    _print_result(result)

    if args.chart:
        _render_chart(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(_result_to_dict(result), f, indent=2)
        print(f"Results saved to {args.output}")


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run strategy optimization."""
    strategy_name = args.strategy
    if strategy_name not in STRATEGY_REGISTRY:
        print(f"Error: Unknown strategy '{strategy_name}'")
        sys.exit(1)

    strategy_cls = STRATEGY_REGISTRY[strategy_name]

    df = _load_data(args.data)

    engine = BacktestEngineV2(initial_capital=args.capital)
    engine.load_data(df)

    # Define parameter spaces per strategy
    param_spaces = {
        "trend_following": {
            "fast_ma_length": {"type": "int", "low": 5, "high": 20},
            "slow_ma_length": {"type": "int", "low": 15, "high": 50},
            "adx_threshold": {"type": "int", "low": 15, "high": 35},
            "stop_loss_atr_mult": {"type": "float", "low": 1.0, "high": 4.0},
        },
        "mean_reversion": {
            "rsi_length": {"type": "int", "low": 7, "high": 21},
            "rsi_oversold": {"type": "int", "low": 20, "high": 40},
            "rsi_overbought": {"type": "int", "low": 60, "high": 80},
            "bb_length": {"type": "int", "low": 10, "high": 30},
            "bb_std": {"type": "float", "low": 1.5, "high": 3.0},
        },
        "breakout": {
            "channel_length": {"type": "int", "low": 10, "high": 40},
            "volume_mult": {"type": "float", "low": 1.0, "high": 3.0},
            "atr_stop_mult": {"type": "float", "low": 1.0, "high": 4.0},
            "confirm_bars": {"type": "int", "low": 1, "high": 3},
        },
    }

    space = param_spaces.get(strategy_name)
    if space is None:
        print(f"Error: No optimization parameter space defined for '{strategy_name}'")
        sys.exit(1)

    def strategy_factory(params: Dict[str, Any]):
        s = strategy_cls.from_params(params)
        return s.generate_signals

    try:
        from shared.backtesting.optimizer import StrategyOptimizer

        optimizer = StrategyOptimizer(engine)

        method = args.method
        n_trials = args.n_trials
        metric = args.metric

        print(f"Optimizing {strategy_name} via {method} ({n_trials} trials, metric={metric})...")

        if method == "bayesian":
            best = optimizer.bayesian_optimize(
                strategy_factory, space, n_trials=n_trials, metric=metric
            )
        else:
            # Convert space to grid for grid search
            grid = {}
            for k, v in space.items():
                if v["type"] == "int":
                    step = max(1, (v["high"] - v["low"]) // 3)
                    grid[k] = list(range(v["low"], v["high"] + 1, step))
                elif v["type"] == "float":
                    grid[k] = [
                        round(v["low"] + i * (v["high"] - v["low"]) / 3, 2)
                        for i in range(4)
                    ]
            results = optimizer.grid_search(strategy_factory, grid, metric=metric)
            best = results[0] if results else None

        if best:
            print(f"\nBest {metric}: {best.metric_value:.4f}")
            print(f"Best params: {best.params}")
            if best.metrics:
                print("Full metrics:")
                for k, v in best.metrics.items():
                    print(f"  {k}: {v}")

            if args.output:
                with open(args.output, "w") as f:
                    json.dump({"best_params": best.params, "metrics": best.metrics}, f, indent=2)
                print(f"Results saved to {args.output}")

    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _render_chart(result: BacktestResultV2) -> None:
    """Render a dashboard chart from backtest results."""
    try:
        from shared.visualization.chart_renderer import ChartRenderer
        print("Rendering dashboard...")
        ChartRenderer.dashboard(result, save_path="backtest_dashboard.png")
        print("Dashboard saved to backtest_dashboard.png")
    except ImportError:
        print("Warning: matplotlib not installed - chart rendering unavailable")
    except Exception as e:
        print(f"Warning: Chart rendering failed: {e}")


def main() -> None:
    """Main entry point for the CLI runner."""
    parser = argparse.ArgumentParser(
        prog="strategies.runner",
        description="Strategy Runner - backtest and optimize trading strategies",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list command
    list_parser = subparsers.add_parser("list", help="List available strategies")
    list_parser.set_defaults(func=cmd_list)

    # backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run a strategy backtest")
    bt_parser.add_argument(
        "--strategy", "-s", required=True, help="Strategy name (e.g. trend_following)"
    )
    bt_parser.add_argument(
        "--data", "-d", default="synthetic",
        help='Data source: CSV file path or "synthetic" (default: synthetic)',
    )
    bt_parser.add_argument(
        "--params", "-p", default="",
        help="Strategy params as key=value pairs (e.g. rsi_oversold=25,rsi_overbought=75)",
    )
    bt_parser.add_argument(
        "--capital", "-c", type=float, default=100_000,
        help="Initial capital (default: 100000)",
    )
    bt_parser.add_argument(
        "--chart", action="store_true",
        help="Generate dashboard chart",
    )
    bt_parser.add_argument(
        "--output", "-o", default="",
        help="Save results to JSON file",
    )
    bt_parser.set_defaults(func=cmd_backtest)

    # optimize command
    opt_parser = subparsers.add_parser("optimize", help="Optimize strategy parameters")
    opt_parser.add_argument(
        "--strategy", "-s", required=True, help="Strategy name"
    )
    opt_parser.add_argument(
        "--data", "-d", default="synthetic",
        help='Data source: CSV file path or "synthetic"',
    )
    opt_parser.add_argument(
        "--method", "-m", default="bayesian", choices=["bayesian", "grid"],
        help="Optimization method (default: bayesian)",
    )
    opt_parser.add_argument(
        "--n-trials", "-n", type=int, default=50,
        help="Number of optimization trials (default: 50)",
    )
    opt_parser.add_argument(
        "--metric", default="sharpe_ratio",
        help="Metric to optimize (default: sharpe_ratio)",
    )
    opt_parser.add_argument(
        "--capital", "-c", type=float, default=100_000,
        help="Initial capital (default: 100000)",
    )
    opt_parser.add_argument(
        "--output", "-o", default="",
        help="Save optimization results to JSON file",
    )
    opt_parser.set_defaults(func=cmd_optimize)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
