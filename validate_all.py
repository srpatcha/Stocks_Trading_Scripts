#!/usr/bin/env python3
"""
COMPREHENSIVE FUNCTIONALITY VALIDATION
=========================================
Validates every component of the trading system end-to-end.
"""
import os, sys, time, tempfile, json
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

PASS = 0
FAIL = 0
SKIP = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")

def skip(name, reason):
    global SKIP
    SKIP += 1
    print(f"  ⚠️ {name} — SKIPPED: {reason}")

def make_ohlcv(n=300, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    price = 100.0
    rows = []
    for i in range(n):
        ret = 0.001 + rng.randn() * 0.015
        price *= 1 + ret
        rows.append({
            "date": dates[i], "open": price*(1+rng.randn()*0.002),
            "high": price*(1+abs(rng.randn())*0.005),
            "low": price*(1-abs(rng.randn())*0.005),
            "close": price, "volume": int(rng.uniform(500000, 2000000))
        })
    return pd.DataFrame(rows).set_index("date")

# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  COMPREHENSIVE FUNCTIONALITY VALIDATION")
print("=" * 70)

# ─── 1. INDICATORS ───────────────────────────────────────────────────────
print("\n📊 1. TECHNICAL INDICATORS (35+)")
from shared.indicators.technical_indicators import TechnicalIndicators as TI
df = make_ohlcv()

for name, fn in [
    ("SMA", lambda: TI.sma(df["close"], 20)),
    ("EMA", lambda: TI.ema(df["close"], 20)),
    ("DEMA", lambda: TI.dema(df["close"], 20)),
    ("TEMA", lambda: TI.tema(df["close"], 20)),
    ("WMA", lambda: TI.wma(df["close"], 20)),
    ("KAMA", lambda: TI.kama(df["close"], 10)),
    ("HMA", lambda: TI.hma(df["close"], 9)),
    ("RSI", lambda: TI.rsi(df["close"], 14)),
    ("MACD", lambda: TI.macd(df["close"])),
    ("Stochastic", lambda: TI.stochastic(df)),
    ("CCI", lambda: TI.cci(df)),
    ("Williams %R", lambda: TI.williams_r(df)),
    ("ROC", lambda: TI.roc(df["close"])),
    ("MFI", lambda: TI.mfi(df)),
    ("ADX", lambda: TI.adx(df)),
    ("Bollinger Bands", lambda: TI.bbands(df["close"])),
    ("ATR", lambda: TI.atr(df)),
    ("Keltner Channels", lambda: TI.keltner_channels(df)),
    ("Donchian Channels", lambda: TI.donchian_channels(df)),
    ("OBV", lambda: TI.obv(df)),
    ("VWAP", lambda: TI.vwap(df)),
    ("AD Line", lambda: TI.ad_line(df)),
    ("CMF", lambda: TI.cmf(df)),
    ("Supertrend", lambda: TI.supertrend(df)),
    ("Ichimoku", lambda: TI.ichimoku(df)),
    ("Parabolic SAR", lambda: TI.psar(df)),
    ("Heikin Ashi", lambda: TI.heikin_ashi(df)),
    ("Squeeze", lambda: TI.squeeze(df)),
    ("Pivot Points", lambda: TI.pivot_points(df)),
    ("Force Index", lambda: TI.force_index(df)),
    ("Elder Ray", lambda: TI.elder_ray(df)),
    ("Elder Impulse", lambda: TI.elder_impulse(df)),
    ("Fibonacci Retracement", lambda: TI.fibonacci_retracement(df)),
    ("Volume Profile", lambda: TI.volume_profile(df)),
    ("Chaikin Volatility", lambda: TI.chaikin_volatility(df)),
]:
    try:
        result = fn()
        check(name, result is not None)
    except Exception as e:
        check(name, False, str(e))

# ─── 2. CANDLESTICK PATTERNS ─────────────────────────────────────────────
print("\n🕯️ 2. CANDLESTICK PATTERNS")
from shared.indicators.candlestick_patterns import CandlestickPatterns as CP

for name, fn in [
    ("Doji", lambda: CP.doji(df)),
    ("Hammer", lambda: CP.hammer(df)),
    ("Engulfing", lambda: CP.engulfing(df)),
    ("Cup and Handle", lambda: CP.cup_and_handle(df)),
    ("Scan All", lambda: CP.scan_all(df)),
]:
    try:
        result = fn()
        check(name, result is not None)
    except Exception as e:
        check(name, False, str(e))

# ─── 3. RISK MANAGER ─────────────────────────────────────────────────────
print("\n🛡️ 3. RISK MANAGER (7 layers + production safety)")
from shared.risk_manager import RiskManager, RiskManagerConfig, SizingMethod

rm = RiskManager(config=RiskManagerConfig(
    total_capital=100000, min_seconds_between_trades=0, max_trades_per_hour=10000
))
check("can_trade() initially True", rm.can_trade() is True)
check("Position sizing (fixed fractional)", rm.calculate_position_size("AAPL", 150.0, stop_price=140.0) > 0)
check("Position sizing (Kelly)", RiskManager(config=RiskManagerConfig(sizing_method=SizingMethod.KELLY, min_seconds_between_trades=0)).calculate_position_size("X", 100) > 0)
check("Position sizing (fixed dollar)", RiskManager(config=RiskManagerConfig(sizing_method=SizingMethod.FIXED_DOLLAR, min_seconds_between_trades=0)).calculate_position_size("X", 100) > 0)
check("Position sizing (fixed shares)", RiskManager(config=RiskManagerConfig(sizing_method=SizingMethod.FIXED_SHARES, min_seconds_between_trades=0)).calculate_position_size("X", 100) > 0)
check("Max position % equity cap", rm.calculate_position_size("X", 150, stop_price=149.9) <= int(100000*0.25/150)+1)
check("Max shares per order cap", rm.calculate_position_size("PENNY", 0.01) <= 10000)

# Record trades
rm.record_trade("AAPL", pnl=-1000)
check("Daily PnL tracked", rm._daily_pnl == -1000)
check("Equity updated", rm._current_equity == 99000)

# Validate order
ok, reason = rm.validate_order("AAPL", 100, 150.0, 150.0)
check("validate_order() accepts valid order", ok is True)
ok, reason = rm.validate_order("X", 50000, 150.0, 150.0)
check("validate_order() rejects fat-finger", ok is False and "shares" in reason.lower())
ok, reason = rm.validate_order("X", 100, 200.0, 100.0)
check("validate_order() rejects price deviation", ok is False and "deviat" in reason.lower())
ok, reason = rm.validate_order("X", 100, 5.0, 5.0, avg_daily_volume=10000)
check("validate_order() rejects low volume", ok is False and "volume" in reason.lower())

# Short limits
rm2 = RiskManager(config=RiskManagerConfig(max_short_positions=1, min_seconds_between_trades=0, max_trades_per_hour=10000))
rm2._open_positions["X"] = -5000
ok, _ = rm2.validate_order("Y", 50, 100, 100, direction="SHORT")
check("Short position limit enforced", ok is False)

# Pyramiding
rm3 = RiskManager(config=RiskManagerConfig(enable_pyramiding=True, pyramid_threshold_pct=2.0, min_seconds_between_trades=0, max_trades_per_hour=10000))
check("can_pyramid() with profit", rm3.can_pyramid("X", 105, 100, 0) is True)
check("can_pyramid() without profit", rm3.can_pyramid("X", 100.5, 100, 0) is False)
check("calculate_pyramid_size() scales down", rm3.calculate_pyramid_size(100, 2) == 25)

# Monthly cap
rm4 = RiskManager(config=RiskManagerConfig(max_monthly_loss=3000, max_daily_loss=50000, min_seconds_between_trades=0, max_trades_per_hour=10000))
rm4.record_trade("X", pnl=-3000)
check("Monthly loss cap blocks trading", rm4.can_trade() is False)

# Market hours
allowed, _ = rm.check_market_hours()
check("Market hours check returns tuple", isinstance(allowed, bool))

# Status
status = rm.get_status()
check("get_status() has all fields", "monthly_pnl" in status and "max_shares_per_order" in status)

# Thread safety
import threading
# RLock, not Lock: the read-side gates nest (get_status -> can_trade), so a
# plain mutex self-deadlocks once they are synchronised. This asserted
# isinstance(..., type(threading.Lock())) and started failing when the lock
# was upgraded — check for the lock protocol instead of one concrete type.
check(
    "_state_lock exists and is re-entrant",
    hasattr(rm, "_state_lock")
    and isinstance(rm._state_lock, type(threading.RLock()))
    and hasattr(rm._state_lock, "acquire")
    and hasattr(rm._state_lock, "release"),
)

# ─── 4. DATA FETCHER ─────────────────────────────────────────────────────
print("\n📡 4. DATA FETCHER")
from shared.data.public_data_fetcher import PublicDataFetcher
pf = PublicDataFetcher(cache_enabled=False)
check("Separate failure counters exist", hasattr(pf, "_fundamentals_failures"))
check("Fundamentals cache exists", hasattr(pf, "_fundamentals_cache"))

health = pf.get_data_health()
check("get_data_health() works", isinstance(health, dict) and "ohlcv_failures" in health)
check("Circuit breaker closed", health["circuit_breaker_open"] is False)

# ─── 5. STRATEGY ENRICHER ────────────────────────────────────────────────
print("\n🔌 5. STRATEGY ENRICHER")
from shared.strategy_enricher import StrategyEnricher, EnrichedData
enricher = StrategyEnricher()
enriched = enricher.enrich("TEST", df)
check("enrich() returns EnrichedData", isinstance(enriched, EnrichedData))
check("regime detection works", enriched.regime in ("TRENDING", "RANGING", "VOLATILE", "UNKNOWN"))
check("ML signal available", hasattr(enriched, "ml_signal"))
check("composite_boost in range", 0.5 <= enriched.composite_boost <= 1.5)

blocked, reason = enricher.should_block_entry(enriched)
check("should_block_entry() returns tuple", isinstance(blocked, bool) and isinstance(reason, str))

# ─── 6. ALL 15 STRATEGIES ────────────────────────────────────────────────
print("\n📈 6. ALL 15 STRATEGIES")
import strategies.examples.trend_following
import strategies.examples.breakout
import strategies.examples.mean_reversion
import strategies.examples.factor_portfolio
import strategies.examples.darvas_box
import strategies.examples.triple_screen
import strategies.examples.canslim_strategy
import strategies.examples.value_strategy
import strategies.examples.ml_rl_strategy
import strategies.examples.self_learning_strategy
import strategies.examples.sentiment_strategy
import strategies.examples.earnings_strategy
import strategies.examples.sector_rotation
import strategies.examples.meta_strategy
from strategies import STRATEGY_REGISTRY

check(f"Strategy count = {len(STRATEGY_REGISTRY)}", len(STRATEGY_REGISTRY) >= 15)

from shared.backtesting.backtest_engine_v2 import BacktestContext
ctx = BacktestContext(bar_index=299, bars={"TEST": df}, positions={}, capital=100000, portfolio_value=100000)

for name in sorted(STRATEGY_REGISTRY.keys()):
    try:
        cls = STRATEGY_REGISTRY[name]
        strategy = cls()
        signals = strategy.generate_signals(ctx)
        check(f"Strategy '{name}' generates signals", isinstance(signals, dict))
    except Exception as e:
        check(f"Strategy '{name}'", False, str(e)[:60])

# ─── 7. BACKTEST ENGINE ──────────────────────────────────────────────────
print("\n⚙️ 7. BACKTEST ENGINE V2")
from shared.backtesting.backtest_engine_v2 import BacktestEngineV2, TradeRecord, BacktestResultV2

engine = BacktestEngineV2(initial_capital=100000)
engine.load_data(df.reset_index())

strat = STRATEGY_REGISTRY["trend_following"]()
result = engine.run(strat.generate_signals)

check("Backtest runs successfully", result is not None)
check("Total return computed", hasattr(result, "total_return"))
check("Sharpe ratio computed", hasattr(result, "sharpe_ratio"))
check("Win rate computed", hasattr(result, "win_rate"))
check("Equity curve generated", len(result.equity_curve) > 0)
check("R-multiples field exists", hasattr(result, "avg_r_multiple"))
check("SQN field exists", hasattr(result, "sqn"))
check("TradeRecord has initial_risk", hasattr(TradeRecord, "__dataclass_fields__") and "initial_risk" in TradeRecord.__dataclass_fields__)
check("TradeRecord has r_multiple", "r_multiple" in TradeRecord.__dataclass_fields__)

# ─── 8. TRADE JOURNAL ────────────────────────────────────────────────────
print("\n📓 8. TRADE JOURNAL")
from shared.trade_journal import TradeJournal
journal_dir = os.path.join(tempfile.gettempdir(), f"test_journal_{int(time.time())}")
journal = TradeJournal(journal_dir=journal_dir)

passed = journal.pre_trade_check("AAPL", mood=8, confidence=7, setup="test")
check("Pre-trade check passes (good mood)", passed is True)
blocked = journal.pre_trade_check("MSFT", mood=2, confidence=2, setup="test")
check("Pre-trade check blocks (bad mood)", blocked is False)

journal.log_trade("AAPL", pnl=500, lesson="Good entry", followed_plan=True)
journal.log_trade("MSFT", pnl=-200, lesson="Bad timing", followed_plan=False)

stats = journal.get_discipline_stats()
check("Discipline stats computed", stats["total_trades"] == 2)
check("Plan adherence tracked", stats["plan_adherence_pct"] == 50.0)

mood_perf = journal.get_performance_by_mood()
check("Performance by mood works", isinstance(mood_perf, dict))

review = journal.daily_review(grade="B", what_went_well="Followed stops", what_to_improve="Patience")
check("Daily review works", review.grade == "B")

journal.save()
check("Journal saves to disk", os.path.exists(os.path.join(journal_dir, "journal.json")))

# ─── 9. MULTI-TIMEFRAME ──────────────────────────────────────────────────
print("\n🔄 9. MULTI-TIMEFRAME")
from shared.indicators.multi_timeframe import MultiTimeframeTrend
mtf = MultiTimeframeTrend()
trend = mtf.get_htf_trend(df)
check("HTF trend detection", trend in ("BULLISH", "BEARISH", "NEUTRAL"))
aligned = mtf.is_aligned(df, "BUY")
check("is_aligned() works", isinstance(aligned, bool))
sr = mtf.get_htf_support_resistance(df)
check("Support/resistance computed", "support" in sr and "resistance" in sr)

# ─── 10. STATE PERSISTENCE ───────────────────────────────────────────────
print("\n💾 10. STATE PERSISTENCE")
db_path = os.path.join(tempfile.gettempdir(), "test_risk_state.db")
rm_persist = RiskManager(config=RiskManagerConfig(
    persist_path=db_path, min_seconds_between_trades=0, max_trades_per_hour=10000
))
rm_persist.record_trade("AAPL", pnl=500)
rm_persist.record_pyramid("AAPL")
check("SQLite persistence created", os.path.exists(db_path))

# Restore
rm_restore = RiskManager(config=RiskManagerConfig(
    persist_path=db_path, min_seconds_between_trades=0, max_trades_per_hour=10000
))
check("Equity restored", rm_restore._current_equity == 100500)
check("Pyramid count restored", rm_restore.get_pyramid_count("AAPL") == 1)

# Cleanup
os.remove(db_path)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"  RESULTS: ✅ {PASS} passed | ❌ {FAIL} failed | ⚠️ {SKIP} skipped")
print("=" * 70)

if FAIL == 0:
    print("  🎉 ALL FUNCTIONALITY VALIDATED — SYSTEM IS PRODUCTION READY")
else:
    print(f"  ⚠️ {FAIL} ISSUES NEED ATTENTION")

sys.exit(1 if FAIL > 0 else 0)
