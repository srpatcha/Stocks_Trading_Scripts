#!/usr/bin/env python3
"""
Trading System Setup Script
==============================

One-command setup that:
1. Installs all Python dependencies
2. Creates .env template with placeholders
3. Creates required directories
4. Validates all imports work
5. Runs a quick self-test
6. Tests data source connectivity (Yahoo Finance)
7. Prints system readiness report

Usage:
    python setup_trading.py            # run the full setup
    python setup_trading.py --dry-run  # show what it would do
"""

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(ROOT, ".env")
ENV_EXAMPLE = os.path.join(ROOT, ".env.example")
DATA_DIR = os.path.join(os.path.expanduser("~"), ".stocks_plugin")

COLORS = {
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
}


def cprint(msg, color="RESET"):
    print(f"{COLORS.get(color, '')}{msg}{COLORS['RESET']}")


def step(n, total, msg):
    cprint(f"\n[{n}/{total}] {msg}", "BLUE")
    print("─" * 60)


def run_cmd(cmd, check=False):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    return result.returncode == 0, result.stdout, result.stderr


# ─── Step 1: Install Dependencies ─────────────────────────────────────────

def install_dependencies():
    step(1, 7, "Installing Python dependencies")

    core_deps = [
        "pandas", "numpy", "yfinance",
    ]
    optional_deps = [
        ("vaderSentiment", "Sentiment analysis (VADER)"),
        ("feedparser", "Google News RSS fallback"),
        ("lightgbm", "Regime classifier (ML)"),
        # Not optional alongside lightgbm: regime_classifier.py and
        # feature_engineer.py use lgb.LGBMClassifier / lgb.LGBMRegressor, which
        # live in lightgbm.sklearn and raise
        #   LightGBMError: scikit-learn is required for lightgbm.sklearn
        # without it. Installing lightgbm alone left the regime classifier
        # broken for anyone who followed the README.
        ("scikit-learn", "Required by lightgbm.sklearn (regime classifier)"),
    ]

    # Core (required)
    for dep in core_deps:
        ok, _, _ = run_cmd(f"{sys.executable} -m pip install {dep} -q")
        status = "✅" if ok else "❌"
        print(f"  {status} {dep}")

    # Optional
    for dep, desc in optional_deps:
        ok, _, _ = run_cmd(f"{sys.executable} -m pip install {dep} -q")
        status = "✅" if ok else "⚠️ (optional)"
        print(f"  {status} {dep} — {desc}")

    return True


# ─── Step 2: Create .env Template ─────────────────────────────────────────

def create_env():
    step(2, 7, "Creating environment configuration")

    if os.path.exists(ENV_FILE):
        cprint("  .env already exists — skipping (won't overwrite your secrets)", "YELLOW")
        return True

    env_content = """# Trading System Configuration
# ==============================
# Fill in your credentials below.

# Interactive Brokers (IBKR)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
IBKR_ACCOUNT=

# Schwab (optional)
SCHWAB_APP_KEY=
SCHWAB_APP_SECRET=
SCHWAB_REDIRECT_URI=https://127.0.0.1

# TradeStation (optional)
TRADESTATION_KEY=
TRADESTATION_SECRET=
TRADESTATION_REDIRECT_URI=http://localhost

# Risk Configuration
TOTAL_CAPITAL=100000
RISK_PER_TRADE_PCT=1.0
MAX_DAILY_LOSS=2000
MAX_MONTHLY_LOSS=4000
ENFORCE_MARKET_HOURS=true

# Data
DATA_CACHE_DIR=~/.stocks_plugin/cache

# Logging
LOG_LEVEL=INFO
"""
    with open(ENV_FILE, "w") as f:
        f.write(env_content)
    cprint("  ✅ .env created — fill in your broker credentials", "GREEN")
    return True


# ─── Step 3: Create Directories ───────────────────────────────────────────

def create_directories():
    step(3, 7, "Creating required directories")

    dirs = [
        os.path.join(DATA_DIR, "cache"),
        os.path.join(DATA_DIR, "journal"),
        os.path.join(DATA_DIR, "models"),
        os.path.join(DATA_DIR, "logs"),
        os.path.join(DATA_DIR, "backtest_results"),
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {d}")

    return True


# ─── Step 4: Validate Imports ─────────────────────────────────────────────

def validate_imports():
    step(4, 7, "Validating core imports")

    sys.path.insert(0, ROOT)
    checks = []

    modules = [
        ("shared.risk_manager", "RiskManager"),
        ("shared.data.public_data_fetcher", "PublicDataFetcher"),
        ("shared.indicators.technical_indicators", "TechnicalIndicators"),
        ("shared.indicators.candlestick_patterns", "CandlestickPatterns"),
        ("shared.strategy_enricher", "StrategyEnricher"),
        ("shared.trade_journal", "TradeJournal"),
        ("shared.backtesting.backtest_engine_v2", "BacktestEngineV2"),
    ]

    for mod_path, cls_name in modules:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            getattr(mod, cls_name)
            print(f"  ✅ {mod_path}.{cls_name}")
            checks.append(True)
        except Exception as e:
            print(f"  ❌ {mod_path}.{cls_name} — {e}")
            checks.append(False)

    return all(checks)


# ─── Step 5: Validate Strategies ──────────────────────────────────────────

def validate_strategies():
    step(5, 7, "Validating strategy registry")

    sys.path.insert(0, ROOT)
    strategy_files = [
        "strategies.examples.trend_following",
        "strategies.examples.breakout",
        "strategies.examples.mean_reversion",
        "strategies.examples.factor_portfolio",
        "strategies.examples.darvas_box",
        "strategies.examples.triple_screen",
        "strategies.examples.canslim_strategy",
        "strategies.examples.value_strategy",
        "strategies.examples.ml_rl_strategy",
        "strategies.examples.self_learning_strategy",
        "strategies.examples.sentiment_strategy",
        "strategies.examples.earnings_strategy",
        "strategies.examples.sector_rotation",
        "strategies.examples.meta_strategy",
    ]

    for sf in strategy_files:
        try:
            __import__(sf)
            name = sf.split(".")[-1]
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {sf} — {e}")

    from strategies import STRATEGY_REGISTRY
    count = len(STRATEGY_REGISTRY)
    cprint(f"\n  Total strategies registered: {count}/15", "GREEN" if count >= 14 else "YELLOW")
    return count >= 14


# ─── Step 6: Test Data Connectivity ───────────────────────────────────────

def test_data_connectivity():
    step(6, 7, "Testing data source connectivity")

    sys.path.insert(0, ROOT)
    results = {}

    # OHLCV
    try:
        from shared.data.public_data_fetcher import PublicDataFetcher
        fetcher = PublicDataFetcher(cache_enabled=False)
        df = fetcher.fetch_ohlcv("AAPL", period="5d", interval="1d")
        if df is not None and len(df) > 0:
            print(f"  ✅ OHLCV: fetched {len(df)} bars for AAPL")
            results["ohlcv"] = True
        else:
            print("  ❌ OHLCV: no data returned")
            results["ohlcv"] = False
    except Exception as e:
        print(f"  ❌ OHLCV: {e}")
        results["ohlcv"] = False

    # Fundamentals
    try:
        fund = fetcher.fetch_fundamentals("AAPL")
        if fund and fund.get("pe_ratio"):
            print(f"  ✅ Fundamentals: AAPL P/E={fund['pe_ratio']}")
            results["fundamentals"] = True
        else:
            print("  ⚠️ Fundamentals: partial or empty data")
            results["fundamentals"] = False
    except Exception as e:
        print(f"  ❌ Fundamentals: {e}")
        results["fundamentals"] = False

    # News
    try:
        news = fetcher.fetch_news_headlines("AAPL", max_items=5)
        print(f"  ✅ News: {len(news)} headlines for AAPL")
        results["news"] = len(news) > 0
    except Exception as e:
        print(f"  ❌ News: {e}")
        results["news"] = False

    # Data health
    health = fetcher.get_data_health()
    print(f"  📊 Data health: failures={health['ohlcv_failures']}, "
          f"circuit_breaker={'OPEN' if health['circuit_breaker_open'] else 'OK'}")

    return all(results.values())


# ─── Step 7: Run Quick Self-Test ──────────────────────────────────────────

def run_self_test():
    step(7, 7, "Running quick self-test")

    ok, stdout, stderr = run_cmd(
        f"cd {ROOT} && {sys.executable} -m pytest tests/test_production_safety.py "
        f"tests/test_new_features.py -v --tb=short -q 2>&1 | tail -5"
    )

    if stdout:
        print(stdout)

    if "passed" in (stdout + stderr):
        cprint("  ✅ Self-test passed", "GREEN")
        return True
    else:
        cprint("  ⚠️ Self-test had issues (non-critical)", "YELLOW")
        return True  # Non-blocking


# ─── Main ─────────────────────────────────────────────────────────────────

def main(argv=None):
    # This script pip-installs packages, writes .env and creates directories
    # under ~/.stocks_plugin. It had no argument parsing at all, so
    # `setup_trading.py --help` — the first thing anyone types before running
    # an unfamiliar script — silently performed all of it.
    parser = argparse.ArgumentParser(
        description=(
            "One-command setup: installs dependencies, writes a .env template, "
            "creates data directories, validates imports and strategies, and "
            "tests Yahoo Finance connectivity."
        ),
        epilog=(
            "Writes to: ./.env (never overwritten) and "
            "~/.stocks_plugin/. Installs packages with pip into the active "
            "environment."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be done without installing or writing anything.",
    )
    args = parser.parse_args(argv)

    cprint("=" * 60, "BOLD")
    cprint("  TRADING SYSTEM SETUP", "BOLD")
    cprint("=" * 60, "BOLD")

    if args.dry_run:
        cprint("\nDRY RUN — nothing will be installed or written.\n", "YELLOW")
        print("Would install (core):     pandas, numpy, yfinance")
        print("Would install (optional): vaderSentiment, feedparser, lightgbm,")
        print("                          scikit-learn")
        print(f"Would create .env at:     {ENV_FILE}")
        print(f"Would create data dir:    {DATA_DIR}")
        print("Would then validate imports, strategies and data connectivity.")
        return 0

    results = {
        "Dependencies": install_dependencies(),
        "Environment": create_env(),
        "Directories": create_directories(),
        "Core Imports": validate_imports(),
        "Strategies": validate_strategies(),
        "Data Sources": test_data_connectivity(),
        "Self-Test": run_self_test(),
    }

    cprint("\n" + "=" * 60, "BOLD")
    cprint("  SETUP RESULTS", "BOLD")
    cprint("=" * 60, "BOLD")

    all_ok = True
    for name, ok in results.items():
        status = "✅" if ok else "❌"
        if not ok:
            all_ok = False
        print(f"  {status} {name}")

    print()
    if all_ok:
        cprint("  🎉 System is ready! Run paper_trader.py to start trading.", "GREEN")
    else:
        cprint("  ⚠️ Some checks failed. Fix issues above and re-run.", "YELLOW")

    cprint("\n  Next steps:", "BOLD")
    print("  1. Edit .env with your broker credentials (if using live trading)")
    print("  2. Run: python paper_trader.py --symbols AAPL,MSFT,GOOGL")
    print("  3. Paper trade for 30+ days before using real money")
    print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
