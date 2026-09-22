"""
Launch: Monitor Mode (Watch Only — No Trades)
================================================

The agent analyzes all symbols, computes decisions, logs everything,
but does NOT execute any trades. Safe for observation and learning.

All decisions logged to:
  ~/.stocks_plugin/logs/decisions.jsonl
  ~/.stocks_plugin/data/trade_memory.db
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from shared.daemon.live_runner import LiveRunner

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

def _parse_args(argv=None):
    """Parse CLI arguments.

    Without this, `--help` on a trading launcher was an unrecognised argument
    that argparse never saw — the script ignored it and STARTED THE DAEMON.
    Typing --help to find out what something does should never begin trading.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols", default=",".join(SYMBOLS),
        help="Comma-separated symbols to monitor (default: %(default)s)",
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Seconds between decision cycles (default: %(default)s)",
    )
    parser.add_argument(
        "--no-news", action="store_true", help="Disable news sentiment analysis",
    )
    parser.add_argument(
        "--no-train", action="store_true", help="Skip initial model training",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the configuration and exit without starting the daemon",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    SYMBOLS = [x.strip() for x in args.symbols.split(",") if x.strip()]
    print("=" * 60)
    print("  MONITOR MODE — WATCH ONLY")
    print("  Analyzing markets + news, logging decisions, NO trades")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print("=" * 60)

    if args.dry_run:
        print("\nDRY RUN — the daemon will NOT start.")
        print(f"  mode:     monitor (no orders are ever placed)")
        print(f"  symbols:  {', '.join(SYMBOLS)}")
        print(f"  interval: {args.interval}s")
        print(f"  news:     {'OFF' if args.no_news else 'ON'}")
        print(f"  training: {'skipped' if args.no_train else 'on start'}")
        raise SystemExit(0)

    runner = LiveRunner(
        symbols=SYMBOLS,
        mode="monitor",
        interval_seconds=args.interval,
        use_news=not args.no_news,
        models=["regime"],
    )
    runner.start(train_first=not args.no_train)
