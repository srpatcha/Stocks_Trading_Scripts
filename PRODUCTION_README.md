# Stocks Plugin — Automated Trading System

Production-ready algorithmic trading framework with 15 strategies, 7 data sources,
7-layer risk management, and comprehensive backtesting.

---

## ⚠️ IMPORTANT DISCLAIMER

**No trading system can guarantee zero losses.** Every trade carries risk. Even with the
tightest controls, individual trades WILL lose money — that is normal. The system limits
HOW MUCH you can lose per trade, per day, per month, and total. Past backtest performance
does NOT guarantee future results. **Paper trade for 30+ days before using real money.**

---

## How to Use — Step by Step

### Step 1: Install

```bash
cd stocks_plugin
python setup_trading.py    # installs everything, validates system
```

### Step 2: Paper Trade (No Real Money)

```bash
# Scan 3 stocks with the best strategy (uses all data sources)
python paper_trader.py --symbols AAPL,MSFT,GOOGL --strategy meta_ensemble

# Scan 15 stocks with ALL strategies and show consensus
python paper_trader.py --scan-universe

# Use a specific strategy
python paper_trader.py --symbols TSLA --strategy trend_following

# Check your paper portfolio
python paper_trader.py --portfolio
```

### Step 3: Understand the Signals

When you run the paper trader, you'll see:
```
Symbol   Signal   Price       Shares   Valid  Reason
AAPL     🟢 BUY   $195.00       128   ✅     OK
MSFT     ⚪ HOLD  $420.00         0   ✅     FLAT
TSLA     🔴 SELL  $175.00        50   ✅     OK
GOOGL    🟢 BUY   $175.00         0   ❌     Bearish sentiment (-0.45)
```

- **🟢 BUY** — Strategy says buy. `Valid ✅` means risk checks pass.
- **⚪ HOLD** — No action. Keep current position (or stay out).
- **🔴 SELL** — Strategy says sell/exit.
- **❌ Rejected** — Risk manager blocked the trade (bad sentiment, earnings coming, etc.)

### Step 4: Backtest Before Live Trading

```bash
# Run a strategy on historical data to see how it would have performed
python -m strategies.examples.trend_following
python -m strategies.examples.meta_strategy
```

Output shows: Total Return, Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor.

---

## How to Set Your Risk Limits

### The Reality About Losses

You said you don't want to lose a single dollar. Here's the truth:

| What You Want | What's Possible | How to Get Close |
|---------------|----------------|-----------------|
| Zero losses ever | ❌ Impossible | Don't trade — keep money in savings |
| Lose < $50/day | ✅ Possible | Set `max_daily_loss=50` |
| Lose < $100/month | ✅ Possible | Set `max_monthly_loss=100` |
| Never lose > 1% on any trade | ✅ Possible | Set `risk_per_trade_pct=1.0` |
| Stop all trading if down $500 total | ✅ Possible | Set `max_drawdown_pct=0.5` (on $100K) |

### Risk Configuration — How to Set Limits

Edit the configuration in your strategy runner or paper_trader.py:

```python
from shared.risk_manager import RiskManager, RiskManagerConfig

# ═══════════════════════════════════════════════════════════
# ULTRA-CONSERVATIVE CONFIG — Minimum possible risk
# ═══════════════════════════════════════════════════════════
config = RiskManagerConfig(

    # ─── How much money you're trading with ───
    total_capital=10000.0,          # Start with $10K (small!)

    # ─── Per-Trade Risk ───
    risk_per_trade_pct=0.5,         # Risk only 0.5% per trade = $50 max loss per trade
    max_position_pct_equity=10.0,   # Never put more than 10% ($1,000) in one stock
    max_shares_per_order=100,       # Never buy more than 100 shares at once

    # ─── Daily Limit ───
    max_daily_loss=100.0,           # Stop ALL trading if you lose $100 in a day
    auto_flatten_on_daily_loss=True, # Auto-close all positions when daily limit hit

    # ─── Monthly Limit ───
    max_monthly_loss=300.0,         # Stop ALL trading if you lose $300 in a month (3%)

    # ─── Total Loss Circuit Breaker ───
    max_drawdown_pct=5.0,           # Stop for 24 hours if account drops 5% ($500)
    circuit_breaker_pause_hours=24, # Pause for 24 hours after circuit breaker

    # ─── Emotional Protection ───
    max_consecutive_losses=2,       # Pause after just 2 losses in a row
    cooldown_seconds=3600,          # Pause for 1 HOUR after consecutive losses
    max_trades_per_hour=3,          # Max 3 trades per hour (prevent overtrading)
    min_seconds_between_trades=60,  # Wait at least 1 minute between trades

    # ─── Position Limits ───
    max_open_positions=3,           # Max 3 stocks at a time
    max_portfolio_heat_pct=10.0,    # Max 10% of portfolio at risk at any time

    # ─── Safety Checks ───
    max_price_deviation_pct=5.0,    # Block trades if price moved >5% from last known
    min_avg_volume=100000,          # Only trade stocks with 100K+ daily volume
    max_position_pct_adv=2.0,       # Position can't be >2% of daily volume

    # ─── Short Selling (disable for safety) ───
    max_short_positions=0,          # NO short selling at all
    max_short_exposure_pct=0.0,     # Zero short exposure

    # ─── Market Hours ───
    enforce_market_hours=True,      # Only trade during NYSE hours

    # ─── Pyramiding (disable) ───
    enable_pyramiding=False,        # Don't add to positions

    # ─── Save State (crash recovery) ───
    persist_path="~/.stocks_plugin/risk_state.db",
)

rm = RiskManager(config=config)
```

### What This Config Does

With the ultra-conservative config above ($10K account):

| Scenario | Maximum Loss | What Happens |
|----------|-------------|-------------|
| Worst single trade | **$50** (0.5%) | Fixed fractional sizing |
| 2 losses in a row | **$100** then **1 hour pause** | Cooldown triggers |
| Worst single day | **$100** then **trading stops** | Daily limit |
| Worst single month | **$300** then **trading stops** | Monthly limit |
| Worst total drawdown | **$500** then **24 hour pause** | Circuit breaker |
| Fat-finger mistake | **Blocked** | 100 share max + price check |
| Short selling gone wrong | **Impossible** | Shorts disabled (max=0) |
| Illiquid stock | **Blocked** | 100K min volume |
| Weekend gap | **$1,000 max** | 10% position cap |

### How Each Limit Works

```
YOU WANT TO BUY AAPL AT $195

Step 1: Position Sizing
   Account: $10,000
   Risk per trade: 0.5% = $50
   Stop loss: $195 - ATR*2 = ~$190 (risk $5/share)
   Shares = $50 / $5 = 10 shares ($1,950 total)
   ✅ Under 10% cap ($1,000 limit → actually capped to 5 shares = $975)

Step 2: Pre-Order Validation
   ✅ 5 shares < 100 share max (fat-finger OK)
   ✅ $195 is within 5% of last price (price sanity OK)
   ✅ AAPL volume 50M > 100K min (liquidity OK)
   ✅ Not a short position (shorts disabled)
   ✅ Daily loss $0 < $100 limit (daily OK)
   ✅ Monthly loss $0 < $300 limit (monthly OK)

Step 3: Enricher Check
   ✅ News sentiment > -0.4 (not strongly bearish)
   ✅ Fundamentals quality > 0.4 (P/E reasonable)
   ✅ No earnings in next 2 days (no gap risk)
   ✅ Market is open (enforce_market_hours=True)

→ APPROVED: Buy 5 shares of AAPL at $195 ($975 total)
→ Stop loss at $190 → Max loss on this trade = $25
```

---

## All 15 Strategies Explained Simply

| Strategy | What It Does | When It Buys | When It Sells | Risk Level |
|----------|-------------|-------------|--------------|------------|
| `trend_following` | Follows the trend | Price trending up + strong momentum | Trend reverses | Low |
| `breakout` | Catches new highs | Price breaks above N-day high with volume | Trailing stop | Medium |
| `mean_reversion` | Buys dips | RSI oversold + at bottom of Bollinger Band | Price returns to middle | Medium |
| `canslim` | O'Neil growth picks | 5+ of 7 criteria (earnings, growth, new highs) | Score drops below 3 | Medium |
| `value` | Graham value picks | Cheap P/E + low debt + dividends | Score drops or P/E > 20 | Low |
| `darvas_box` | Box breakouts | Price breaks above consolidation box | Below box floor | Medium |
| `triple_screen` | 3 timeframe filter | Weekly + Daily + Intraday all agree | Trend reversal | Low |
| `sentiment` | News-driven | Bullish news + technical confirmation | Bearish news | Medium |
| `earnings` | Around earnings | Before earnings (stocks that usually beat) | After event | High |
| `sector_rotation` | Best sectors | Top 3 sectors by 12-month momentum | Sector drops out | Low |
| `factor` | Momentum ranking | Long strongest, short weakest | Monthly rebalance | Medium |
| `ml` | LSTM prediction | Neural network predicts up | Predicts down | Medium |
| `rl` | RL agent | Reinforcement learning agent acts | Agent says sell | Medium |
| `self_learning` | Adaptive AI | Learns from own trades, adapts | AI decides | Medium |
| `meta_ensemble` | Combines ALL | 5-component score > 0.4 + 3/5 agree | Score < 0.2 | **Lowest** |

**Recommended for beginners:** `meta_ensemble` (safest — requires multiple signals to agree)

---

## Configuration Reference

### All Risk Parameters

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `total_capital` | 100,000 | Your account size in dollars |
| `risk_per_trade_pct` | 2.0 | Max % of account to risk per trade |
| `max_daily_loss` | 5,000 | Dollar amount — stop trading for the day |
| `max_monthly_loss` | 0 (off) | Dollar amount — stop trading for the month |
| `max_drawdown_pct` | 10.0 | % drop from peak → 24h circuit breaker |
| `max_consecutive_losses` | 3 | Losses in a row → cooldown |
| `cooldown_seconds` | 1800 | Seconds to pause after consecutive losses |
| `max_trades_per_hour` | 10 | Prevent overtrading |
| `min_seconds_between_trades` | 30 | Minimum gap between trades |
| `max_open_positions` | 10 | Max stocks held at once |
| `max_portfolio_heat_pct` | 20.0 | Max % of equity at risk across all positions |
| `max_position_pct_equity` | 25.0 | Max % of equity in one stock |
| `max_position_notional` | 0 (off) | Hard dollar cap per position |
| `max_shares_per_order` | 10,000 | Fat-finger protection |
| `max_price_deviation_pct` | 10.0 | Block if price moved too much |
| `max_short_positions` | 5 | Max number of short positions |
| `max_short_exposure_pct` | 30.0 | Max % of equity in shorts |
| `enforce_market_hours` | False | Only trade during NYSE hours |
| `min_avg_volume` | 50,000 | Skip illiquid stocks |
| `enable_pyramiding` | False | Add to winning positions |

### Choosing Your Settings

| Your Situation | `risk_per_trade_pct` | `max_daily_loss` | `max_monthly_loss` |
|---------------|---------------------|-------------------|---------------------|
| **Very cautious** ($10K account) | 0.5% ($50) | $100 | $300 |
| **Conservative** ($50K account) | 1.0% ($500) | $1,000 | $2,500 |
| **Standard** ($100K account) | 2.0% ($2,000) | $5,000 | $6,000 |
| **Aggressive** ($100K account) | 3.0% ($3,000) | $10,000 | $10,000 |

---

## Testing

```bash
# Quick validation (< 2 seconds)
python validate_all.py                # 108 functional checks

# Full test suite (< 2 seconds)
python -m pytest tests/test_production_safety.py tests/test_new_features.py \
    tests/test_shared_risk_manager.py tests/test_strategy_examples.py -v

# End-to-end pipeline test
python e2e_test.py
```

---

## FAQ

**Q: Can I guarantee I won't lose money?**
A: No. Every trade can lose. The system limits HOW MUCH you lose, not IF you lose.

**Q: What's the safest strategy?**
A: `meta_ensemble` — requires 3 of 5 data sources to agree before buying.

**Q: How do I start with minimum risk?**
A: Use the ultra-conservative config above with $10K, 0.5% risk, $100 daily limit.

**Q: Do I need a broker account?**
A: No — `paper_trader.py` works with free Yahoo Finance data, no broker needed.

**Q: What if Yahoo Finance goes down?**
A: The system has circuit breakers and stale cache fallback. It stops trading, never crashes.

**Q: Is my money safe from bugs?**
A: 7 independent safety layers protect you, and the full suite passes. Note that the layers in
`shared/risk_manager.py` apply only where a RiskManager is attached — pass one to
`BrokerBridge(risk_manager=...)` or `LiveRunner(risk_manager=...)`. Always start with paper trading.
