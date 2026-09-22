"""
Launch: IB Paper Trading (Requires IB Gateway/TWS running)
============================================================

Connects to Interactive Brokers paper trading account via IB Gateway.
The agent makes real decisions and places real paper orders.

Prerequisites:
  1. IB Gateway/TWS running on port 7497 (paper)
  2. API enabled in IB Gateway settings
  3. See docs/ib_setup_guide.txt for setup instructions
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from shared.daemon.live_runner import LiveRunner

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")
IB_PORT = int(os.environ.get("IB_PORT", "7497"))  # 7497=paper, 7496=live

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
    parser.add_argument(
        "--host", default=IB_HOST, help="IB Gateway host (default: %(default)s)",
    )
    parser.add_argument(
        "--port", type=int, default=IB_PORT,
        help="IB Gateway port; 7497=paper, 7496=live (default: %(default)s)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    SYMBOLS = [x.strip() for x in args.symbols.split(",") if x.strip()]
    print("=" * 60)
    print("  IB PAPER TRADING MODE")
    print(f"  Connecting to IB Gateway at {IB_HOST}:{IB_PORT}")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print("  Orders will be placed on your IB PAPER account")
    print("=" * 60)

    # 7496/4001 are the LIVE ports. Say so before connecting rather than
    # after — this script is named "paper" and will happily drive a live
    # account if pointed at one.
    if args.port not in (7497, 4002):
        print(
            f"\n  WARNING: port {args.port} is NOT an IB paper port "
            f"(7497/4002). Orders may reach a LIVE account.\n"
        )

    if args.dry_run:
        print("\nDRY RUN — the daemon will NOT start and no connection is made.")
        print(f"  mode:     paper via IB Gateway")
        print(f"  gateway:  {args.host}:{args.port}"
              f"  ({'PAPER' if args.port in (7497, 4002) else 'LIVE'})")
        print(f"  symbols:  {', '.join(SYMBOLS)}")
        print(f"  interval: {args.interval}s")
        raise SystemExit(0)

    runner = LiveRunner(
        symbols=SYMBOLS,
        mode="paper",
        interval_seconds=args.interval,
        use_news=not args.no_news,
        models=["regime"],
        broker="ib",
        broker_config={"host": args.host, "port": args.port, "client_id": 1},
    )
    runner.start(train_first=not args.no_train)
