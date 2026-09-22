# Stocks Trading — Scripts & Plugins

[![CI](https://github.com/srpatcha/Stocks_Trading_Scripts/actions/workflows/ci.yml/badge.svg)](https://github.com/srpatcha/Stocks_Trading_Scripts/actions/workflows/ci.yml)
[![CodeQL](https://github.com/srpatcha/Stocks_Trading_Scripts/actions/workflows/codeql.yml/badge.svg)](https://github.com/srpatcha/Stocks_Trading_Scripts/actions/workflows/codeql.yml)
[![Scorecard](https://github.com/srpatcha/Stocks_Trading_Scripts/actions/workflows/scorecard.yml/badge.svg)](https://github.com/srpatcha/Stocks_Trading_Scripts/actions/workflows/scorecard.yml)
[![Book](https://github.com/srpatcha/Stocks_Trading_Scripts/actions/workflows/book-build.yml/badge.svg)](https://github.com/srpatcha/Stocks_Trading_Scripts/actions/workflows/book-build.yml)

A comprehensive algorithmic trading system with **15 strategies**, **7 data sources**, **7-layer risk management**, and production safety controls. 2,190+ tests passing on Python 3.12.

## Quick Start

```bash
# Setup (installs deps, validates system, tests connectivity)
python setup_trading.py

# Paper trade with real Yahoo Finance data (no broker needed)
python paper_trader.py --symbols AAPL,MSFT,GOOGL --strategy meta_ensemble

# Scan 15 stocks with all strategies
python paper_trader.py --scan-universe

# Run tests
python -m pytest tests/test_production_safety.py tests/test_new_features.py -v
```

## 15 Trading Strategies

| # | Strategy | Data Sources | Based On |
|---|----------|-------------|----------|
| 1 | `trend_following` | 📈📊💰📰📅🧠🌊 | EMA crossover + ADX + trailing stop |
| 2 | `breakout` | 📈📊💰📰📅🧠🌊 | Donchian channel breakout |
| 3 | `mean_reversion` | 📈📊💰📰📅🧠🌊 | RSI + Bollinger Bands |
| 4 | `factor` | 📈📊💰📰📅🧠🌊 | 12-1 month momentum long/short |
| 5 | `darvas_box` | 📈📊💰📰📅🧠🌊 | Darvas box breakout |
| 6 | `triple_screen` | 📈📊💰📰📅🧠🌊 | Elder triple screen system |
| 7 | `canslim` | 📈📊💰📰📅🧠🌊 | O'Neil CAN SLIM 7-criteria |
| 8 | `value` | 📈📊💰📰📅🧠🌊 | Graham fundamental value |
| 9 | `ml` | 📈📊💰📰📅🧠🌊 | LSTM deep learning |
| 10 | `rl` | 📈📊💰📰📅🧠🌊 | PPO reinforcement learning |
| 11 | `self_learning` | 📈📊💰📰📅🧠🌊 | Adaptive ML ensemble |
| 12 | `sentiment` | 📈📊💰📰📅🧠🌊 | News sentiment + technicals |
| 13 | `earnings` | 📈📊💰📰📅🧠🌊 | Earnings calendar trading |
| 14 | `sector_rotation` | 📈📊💰📰📅🧠🌊 | Sector ETF momentum |
| 15 | `meta_ensemble` | 📈📊💰📰📅🧠🌊 | All sources combined |

📈Price 📊Volume 💰Fundamentals 📰News 📅Earnings 🧠ML 🌊Regime — **all 15 strategies use all 7 data sources**

## 7-Layer Risk Management

```
Layer 7 ─ PORTFOLIO HEAT ── Max 20% equity at risk
Layer 6 ─ POSITION CAP ──── Max 25% equity / 10K shares per position
Layer 5 ─ CIRCUIT BREAKER ─ 10% drawdown → 24h pause
Layer 4 ─ MONTHLY CAP ───── Elder 6% monthly loss limit
Layer 3 ─ DAILY LIMIT ───── $5,000/day hard stop
Layer 2 ─ COOLDOWN ──────── 30-min pause after 3 consecutive losses
Layer 1 ─ PER-TRADE RISK ── 2% of equity per trade
```

**Production safety**: fat-finger protection (10K shares), price deviation (±10%), short limits (5 positions / 30%), liquidity filter (50K min volume), market hours enforcement, thread-safe state persistence (SQLite WAL).

## Architecture

```
stocks_plugin/
├── shared/
│   ├── risk_manager.py           ← 7-layer risk engine (thread-safe)
│   ├── strategy_enricher.py      ← Multi-source data enrichment for all strategies
│   ├── trade_journal.py          ← Human psychology/discipline journal
│   ├── data/public_data_fetcher.py ← OHLCV, fundamentals, news, earnings
│   ├── indicators/               ← 35+ indicators, 13 candlestick patterns
│   ├── backtesting/              ← Multi-asset backtester with R-multiples/SQN
│   └── ml/                       ← Sentiment, regime, ensemble, LSTM, RL
├── strategies/examples/          ← 15 registered strategies
├── tests/                        ← 2,190+ tests (production safety + features)
├── setup_trading.py              ← One-command setup
├── paper_trader.py               ← Paper trading simulator (no broker needed)
└── .github/workflows/ci.yml      ← CI/CD with security scan + release automation
```

## CI/CD

- **Tests**: Python 3.10/3.11/3.12 matrix, production safety + feature + strategy tests
- **Security**: Bandit scan, hardcoded secrets check, eval/exec scan
- **Lint**: Syntax validation on all files, flake8 error detection
- **Strategy validation**: Verifies all 15 strategies register correctly
- **Release**: Tag `v*` → automated GitHub release with changelog

## Release

```bash
git tag v1.0.0
git push origin v1.0.0
# → CI runs all checks → creates GitHub Release automatically
```

See [PRODUCTION_README.md](PRODUCTION_README.md) for full production documentation.

---

## Platform Support Matrix

| Platform | Native Language | API / Integration | Broker(s) | Automation | Indicators | Data/Analytics |
|---|---|---|---|---|---|---|
| **TradingView** | Pine Script (v5+) | Webhooks, Broker API | Interactive Brokers, TradeStation | ✅ Alerts → Orders | ✅ Custom indicators & scanners | ✅ Strategy backtesting |
| **thinkorswim** | thinkScript | TDA/Schwab API (OAuth) | Charles Schwab | ✅ Conditional orders | ✅ Custom studies & scans | ✅ ThinkBack, watchlists |
| **Interactive Brokers** | Python (TWS API) | IB Gateway / TWS API | Interactive Brokers | ✅ Full algo trading | ✅ Real-time data feeds | ✅ Portfolio analytics |
| **TradeStation** | EasyLanguage | TradeStation API (REST) | TradeStation | ✅ Strategy automation | ✅ RadarScreen, scanners | ✅ Historical data export |

---

## Architecture & Project Structure

```
stocks_plugin/
├── README.md
├── docs/                          # Documentation & design notes
│   ├── architecture.md
│   └── platform-guides/
│       ├── tradingview.md
│       ├── thinkorswim.md
│       ├── interactive-brokers.md
│       └── tradestation.md
│
├── tradingview/                   # TradingView — Pine Script
│   ├── strategies/                # Automated trading strategies
│   │   ├── mean_reversion.pine
│   │   ├── momentum_breakout.pine
│   │   └── multi_timeframe.pine
│   ├── indicators/                # Custom technical indicators
│   │   ├── volume_profile.pine
│   │   ├── custom_rsi.pine
│   │   └── market_structure.pine
│   ├── scanners/                  # Market scanners / screeners
│   │   └── gap_scanner.pine
│   └── webhooks/                  # Webhook receivers for alert → order routing
│       ├── webhook_server.py
│       └── config.yaml
│
├── thinkorswim/                   # thinkorswim — thinkScript
│   ├── strategies/                # Automated strategies & conditional orders
│   │   └── earnings_play.ts
│   ├── studies/                   # Custom studies (indicators)
│   │   ├── custom_macd.ts
│   │   └── relative_strength.ts
│   ├── scans/                     # Custom stock scans
│   │   └── unusual_volume.ts
│   └── watchlists/                # Watchlist configurations
│       └── sector_rotation.ts
│
├── interactive_brokers/           # Interactive Brokers — Python / TWS API
│   ├── strategies/                # Algo trading bots
│   │   ├── pairs_trading.py
│   │   └── options_wheel.py
│   ├── data/                      # Market data collection & storage
│   │   ├── historical_fetcher.py
│   │   └── realtime_stream.py
│   ├── analytics/                 # Portfolio analytics & reporting
│   │   ├── portfolio_tracker.py
│   │   └── risk_analyzer.py
│   ├── utils/                     # Shared utilities (connection, logging)
│   │   ├── ib_connection.py
│   │   └── order_manager.py
│   └── requirements.txt
│
├── tradestation/                  # TradeStation — EasyLanguage + API
│   ├── strategies/                # EasyLanguage strategies
│   │   └── trend_following.el
│   ├── indicators/                # EasyLanguage indicators
│   │   └── adaptive_moving_avg.el
│   ├── scanners/                  # RadarScreen indicators / scanners
│   │   └── sector_momentum.el
│   ├── api/                       # TradeStation REST API scripts (Python)
│   │   ├── order_router.py
│   │   └── account_monitor.py
│   └── requirements.txt
│
├── shared/                        # Cross-platform shared utilities
│   ├── config/                    # API keys, broker configs (gitignored)
│   │   └── config.example.yaml
│   ├── notifier/                  # Trade notification system (email, SMS, Discord)
│   │   └── alert_dispatcher.py
│   └── backtesting/               # Platform-agnostic backtesting framework
│       └── backtest_engine.py
│
├── tests/                         # Test suites
│   ├── test_ib_connection.py
│   ├── test_webhook_server.py
│   └── test_backtest_engine.py
│
├── .gitignore
├── .env.example
└── requirements.txt               # Top-level Python dependencies
```

---

## Development Roadmap

### Phase 1: TradingView (Pine Script) — ✅ Implemented

TradingView is the starting point due to its rapid prototyping capabilities, built-in backtesting, and broad broker integration.

**Deliverables:**
- [x] Pine Script v5 strategy templates (momentum, mean reversion, multi-timeframe)
- [x] Custom indicators (volume profile, market structure, enhanced RSI)
- [x] Market scanners (gap scanner, unusual volume, relative strength)
- [x] Webhook server for routing TradingView alerts to Interactive Brokers and TradeStation
- [x] Alert-to-order automation pipeline (TradingView → webhook → broker API)
- [x] Backtesting documentation and performance benchmarks

**Why first?** Pine Script allows fast iteration. Strategies proven here can be ported to other platforms. Webhook integration provides a bridge to IB and TradeStation for live execution.

---

### Phase 2: thinkorswim (thinkScript) — ✅ Implemented

Expand to thinkorswim for its powerful charting, ThinkBack historical analysis, and native scanning capabilities.

**Deliverables:**
- [x] Custom thinkScript studies (MACD variants, relative strength, breadth indicators)
- [x] Stock scans (unusual volume, earnings setups, sector rotation)
- [x] Conditional order templates for semi-automated strategies
- [x] Schwab API integration (OAuth2 authentication, order placement)
- [x] Watchlist generators and sector rotation tools

**Why second?** thinkScript studies complement TradingView indicators. The Schwab API (successor to TDA API) enables programmatic access for data collection and order routing.

---

### Phase 3: Interactive Brokers (TWS API — Python) — ✅ Implemented

Build full Python-based algorithmic trading infrastructure using the TWS API.

**Deliverables:**
- [x] IB Gateway / TWS connection manager with auto-reconnect
- [x] Real-time and historical market data collection pipeline
- [x] Algo trading bots (pairs trading, options wheel, delta-neutral strategies)
- [x] Portfolio analytics dashboard (P&L tracking, risk metrics, Greeks exposure)
- [x] Order management system with position sizing and risk controls
- [x] Paper trading validation framework

**Why third?** IB provides the most comprehensive API for full automation. Python scripts can implement complex strategies that are impossible in Pine Script or thinkScript.

---

### Phase 4: TradeStation (EasyLanguage + API) — ✅ Implemented

Complete platform coverage with TradeStation's EasyLanguage and REST API.

**Deliverables:**
- [x] EasyLanguage strategy templates (trend following, breakout, reversal)
- [x] RadarScreen scanner indicators for real-time market monitoring
- [x] Custom EasyLanguage indicators (adaptive moving averages, volatility measures)
- [x] TradeStation API integration (Python — account data, order placement, market data)
- [x] Cross-platform strategy performance comparison tools

**Why fourth?** EasyLanguage is powerful but platform-locked. The REST API provides modern programmatic access. This phase rounds out coverage across all four major platforms.

---

## Platform Details

### TradingView

| Attribute | Detail |
|---|---|
| **Language** | Pine Script v5+ |
| **Editor** | TradingView in-browser Pine Editor |
| **Backtesting** | Built-in Strategy Tester |
| **Alerts** | Webhook-based (HTTP POST to custom server) |
| **Broker Integration** | Native broker panel (IB, TradeStation, others) |
| **Data** | TradingView data feed (stocks, futures, crypto, forex) |

**Key capabilities:**
- Write strategies that generate `strategy.entry()` / `strategy.exit()` signals
- Create custom indicators with `plot()`, `plotshape()`, `bgcolor()`
- Build scanners using `request.security()` for multi-symbol analysis
- Route alerts via webhooks to external systems for live order execution

---

### thinkorswim (Charles Schwab)

| Attribute | Detail |
|---|---|
| **Language** | thinkScript |
| **Editor** | thinkorswim desktop platform editor |
| **Backtesting** | ThinkBack (historical replay) |
| **Automation** | Conditional orders, alerts |
| **API** | Schwab API (OAuth2, REST) — successor to TDA API |
| **Data** | Real-time + historical via platform; API for programmatic access |

**Key capabilities:**
- Custom studies rendered on charts (lower/upper studies)
- Stock Hacker scans with custom thinkScript filters
- Conditional orders triggered by study conditions
- Schwab API for external automation and data retrieval

---

### Interactive Brokers (TWS API)

| Attribute | Detail |
|---|---|
| **Language** | Python (via `ibapi` or `ib-async`) |
| **Connection** | TWS desktop or IB Gateway (port 7497/7496) |
| **Backtesting** | Custom (use `backtrader`, `zipline`, or custom engine) |
| **Automation** | Full programmatic order placement and management |
| **Data** | Real-time streaming + historical bars via API |
| **Asset Classes** | Stocks, options, futures, forex, bonds, funds |

**Key capabilities:**
- Full order lifecycle management (place, modify, cancel, monitor)
- Real-time market data streaming with tick-level granularity
- Historical data download for backtesting and analysis
- Portfolio and account data access (positions, P&L, margin)
- Complex order types (bracket, adaptive, algorithmic)

---

### TradeStation

| Attribute | Detail |
|---|---|
| **Language** | EasyLanguage / EasyLanguage Objects (ELO) |
| **Editor** | TradeStation desktop development environment |
| **Backtesting** | Built-in Strategy Tester, Walk-Forward Optimizer |
| **Automation** | Strategy automation in TradeStation + API |
| **API** | TradeStation REST API (OAuth2) |
| **Data** | Real-time + historical via platform and API |

**Key capabilities:**
- EasyLanguage strategies with built-in optimization and walk-forward testing
- RadarScreen for real-time multi-symbol indicator monitoring
- Scanner / market screener with custom EasyLanguage criteria
- REST API for external Python-based automation and data access

---

## Plugin Categories

### Automated Strategies
Scripts and bots that generate and execute trade signals automatically. Includes entry/exit logic, position sizing, and risk management.

- **TradingView**: Pine Script strategies with alert-based execution
- **thinkorswim**: Conditional order chains triggered by study conditions
- **Interactive Brokers**: Python algo bots with full order management
- **TradeStation**: EasyLanguage strategies with built-in automation

### Technical Indicators & Scanners
Custom indicators for chart analysis and scanners for market-wide screening.

- **TradingView**: Pine Script indicators and multi-symbol scanners
- **thinkorswim**: thinkScript studies and Stock Hacker scans
- **Interactive Brokers**: Python-based real-time data analysis
- **TradeStation**: EasyLanguage indicators and RadarScreen columns

### Data Collection & Portfolio Analytics
Tools for gathering market data, tracking portfolio performance, and generating analytical reports.

- **TradingView**: Strategy Tester reports, webhook-logged trade data
- **thinkorswim**: ThinkBack analysis, API-based data collection
- **Interactive Brokers**: Historical data pipelines, portfolio dashboards
- **TradeStation**: API data export, strategy performance reports

---

## Getting Started

### Prerequisites

| Platform | Requirements |
|---|---|
| **TradingView** | TradingView account (Pro+ recommended for webhook alerts), Python 3.9+ for webhook server |
| **thinkorswim** | Charles Schwab brokerage account, thinkorswim desktop platform |
| **Interactive Brokers** | IB account, TWS or IB Gateway installed, Python 3.9+, `ibapi` or `ib-async` package |
| **TradeStation** | TradeStation account, TradeStation desktop platform, Python 3.9+ for API scripts |

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url> stocks_plugin
   cd stocks_plugin
   ```

2. **Install Python dependencies** (for IB, TradeStation API, and webhook scripts):
   ```bash
   pip install -r requirements-core.txt   # required, version-bounded
   pip install -r requirements.txt        # core + all optional extras
   ```

3. **Configure API credentials:**
   ```bash
   cp .env.example .env
   # Edit .env with your broker API keys and configuration
   ```

4. **Start with TradingView (Phase 1):**
   - Open the `tradingview/` directory
   - Copy Pine Script code into TradingView's Pine Editor
   - Backtest strategies using TradingView's Strategy Tester
   - Set up webhook alerts for live execution

### Development Workflow

1. **Prototype** strategies in TradingView (Pine Script) for fast iteration
2. **Validate** with TradingView's built-in backtester
3. **Port** proven strategies to other platforms as needed
4. **Automate** using webhook → broker API pipelines or direct API integration
5. **Monitor** with portfolio analytics and alerting tools

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-strategy`)
3. Follow the folder structure conventions above
4. Include comments explaining strategy logic and parameters
5. Add backtesting results or performance notes where applicable
6. Submit a pull request

---

## License

Licensed under the **GNU Affero General Public License v3.0** — see
[LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

AGPL-3.0 applies because this repository vendors AGPL-3.0 (Ghostfolio) and
GPL-3.0 (alpaca-trend-alert) code in-tree. Note that **network use counts as
distribution**: if you run the webhook server or dashboard where others can
reach it, those users may request the source. Running it privately for
yourself triggers no obligation.

Trading involves risk. Nothing here is financial advice — use at your own
discretion with proper risk management.

---

## Advanced Strategies

### Regime-Aware Trading
| Strategy | Platform | File | Description |
|---|---|---|---|
| **Chameleon Regime Switcher** | TradingView | `tradingview/strategies/chameleon_regime_switcher.pine` | ADX-based regime detection switching between EMA crossover (trending) and BB+RSI mean reversion (ranging). Dashboard, webhook alerts. |
| **Advanced Multi-Regime** | TradingView | `tradingview/strategies/advanced_multi_regime.pine` | Multi-timeframe regime confirmation, 3:1 R:R targets, cooldown after losses, volatility squeeze detection. Enhanced dashboard. |
| **Regime Detector** | thinkorswim | `thinkorswim/studies/regime_detector.ts` | Scanner-compatible ADX regime study. Background coloring, labels, Stock Hacker filter output. |
| **Regime Switcher** | TradeStation | `tradestation/strategies/regime_switcher.el` | EasyLanguage regime-switching strategy with Chandelier Exit (trend) and BB exits (range). |
| **Regime Trader** | IBKR (Python) | `interactive_brokers/strategies/regime_trader.py` | Python bot: ADX/ATR/EMA regime detection → trend pullback or RSI mean-reversion entries. Bracket orders. |

### VWAP Strategies
| Strategy | Platform | File | Description |
|---|---|---|---|
| **VWAP Bands** | TradingView | `tradingview/indicators/vwap_bands.pine` | VWAP indicator with ±1σ, ±2σ, ±3σ deviation bands. Anchored VWAP (session/week/month). Distance label. |
| **VWAP Mean Reversion** | TradingView | `tradingview/strategies/vwap_mean_reversion.pine` | Mean reversion to VWAP at ±2σ bands. RSI confirmation. ADX regime filter (trade only when ranging). |
| **VWAP Deviation** | thinkorswim | `thinkorswim/studies/vwap_deviation.ts` | VWAP with deviation bands, distance labels, and ±2σ/±3σ alerts. |

### Intraday & Breakout
| Strategy | Platform | File | Description |
|---|---|---|---|
| **Opening Range Breakout** | TradingView | `tradingview/strategies/opening_range_breakout.pine` | ORB with configurable period (15/30/60 min). Split targets, volume confirmation, EOD exit. |
| **Volatility Squeeze Breakout** | TradingView | `tradingview/strategies/volatility_breakout.pine` | BB inside KC squeeze detection. Momentum direction entry. ATR trailing stops. Squeeze dot indicator. |
| **Trend Pullback Engine** | TradingView | `tradingview/strategies/trend_pullback.pine` | 200 EMA trend filter, entry on pullback to 20 EMA with RSI confirmation. 2:1 R:R targeting. |

### Portfolio Automation (Python / IBKR)
| Strategy | File | Description |
|---|---|---|
| **Momentum Rebalancer** | `interactive_brokers/strategies/momentum_rebalancer.py` | Monthly/weekly rebalancer. Ranks universe by composite momentum (1M/3M/6M ROC). 200-SMA filter. Equal-weight allocation. |
| **DCA Bot** | `interactive_brokers/strategies/dca_bot.py` | Dollar Cost Averaging with regime-aware pausing. Skips buys when RSI > 75 or death cross. Tracks cost basis. |
| **Risk Manager** | `shared/risk_manager.py` | Standalone risk module: position sizing (fixed fractional / Kelly), daily loss limits, cooldown after losses, drawdown circuit breaker, portfolio heat. |

### 🧠 Self-Learning AI Agent (Python)
| Component | File | Description |
|---|---|---|
| **Self-Learning Agent** | `shared/ml/self_learning_agent.py` | Autonomous orchestrator: classifies regime → queries trade memory → runs ensemble prediction → applies risk management → decides BUY/SELL/HOLD → records outcome → detects model degradation → triggers retraining. |
| **Ensemble Predictor** | `shared/ml/ensemble_predictor.py` | Combines LSTM, Transformer, RL, and momentum signals with adaptive weights. Models that perform better get more influence. Regime-conditional weighting. |
| **Trade Memory** | `shared/ml/trade_memory.py` | SQLite-backed persistent journal storing every trade with full context: features, regime, model predictions, P&L. Enables "what worked in similar conditions?" queries. |
| **Regime Classifier** | `shared/ml/regime_classifier.py` | LightGBM classifier (30+ features) auto-labeling TRENDING/RANGING/VOLATILE regimes. |
| **LSTM Predictor** | `shared/ml/deep_learning/lstm_predictor.py` | PyTorch LSTM/GRU for next-day return prediction. Walk-forward backtesting. |
| **Transformer Predictor** | `shared/ml/deep_learning/transformer_predictor.py` | Attention-based time series predictor. Positional encoding + self-attention. |
| **RL Trading Agent** | `shared/ml/rl_agent.py` | PPO/A2C/SAC via Stable-Baselines3. Custom Gymnasium environment with configurable reward (PnL/Sharpe/Sortino). |

---

### Strategy Quick-Reference Guide

<details>
<summary><strong>Chameleon Regime Switcher</strong> — How does it work?</summary>

**Regime Detection:** Uses ADX to classify market into TRENDING (ADX > 25), RANGING (ADX < 20), or VOLATILE (ATR spike).

**When TRENDING:** Enters on EMA 9/21 crossover with ATR trailing stop.
**When RANGING:** Enters at Bollinger Band extremes + RSI confirmation, exits at BB midline.
**When VOLATILE:** No new entries — protects capital.

**Key Inputs:** All inputs have `?` tooltip descriptions in the TradingView settings panel.
**Webhook:** Alert messages are JSON-formatted for direct webhook integration.
</details>

<details>
<summary><strong>Advanced Multi-Regime</strong> — What's different from Chameleon?</summary>

- **HTF Confirmation:** Confirms regime on daily timeframe via `request.security()` to reduce false signals
- **200 EMA Filter:** Only long above 200 EMA, only short below — prevents counter-trend entries
- **Cooldown:** Pauses trading for N bars after a losing trade
- **Max Daily Trades:** Limits trade count per day to prevent overtrading
- **3:1 R:R Targets:** Fixed stop and take-profit based on ATR, not trailing stops
- **Squeeze Detection:** Shows when BB compresses inside KC (pre-breakout conditions)
</details>

<details>
<summary><strong>VWAP Strategies</strong> — When to use which?</summary>

- **VWAP Bands** (indicator): Add to any chart for visual reference. Use ±2σ bands as support/resistance. ±3σ = extreme levels.
- **VWAP Mean Reversion** (strategy): Automated trading at ±2σ bands with RSI confirmation. Only trades in RANGING markets (ADX < 25). Stop is fixed at entry (does not trail).
- **VWAP Deviation** (thinkorswim): Same concept for thinkorswim platform with scanner-compatible output and alerts.
</details>

<details>
<summary><strong>Opening Range Breakout</strong> — Intraday setup</summary>

- Defines the high/low of the first 15/30/60 minutes after market open
- Enters on breakout above ORB high (long) or below ORB low (short)
- Stop at opposite end of the opening range
- Dual targets: 1.5× and 2× the range width (split position)
- Auto-exits 15 minutes before market close to avoid overnight risk
- One trade per day to avoid whipsaws
</details>

<details>
<summary><strong>Python Bots</strong> — Setup and usage</summary>

All Python bots share the same architecture:
```python
# 1. Create connection and dependencies
from interactive_brokers.utils.ib_connection import IBInsyncConnection
from interactive_brokers.utils.order_manager import OrderManager
from interactive_brokers.data.historical_fetcher import HistoricalDataFetcher
from shared.risk_manager import RiskManager
from shared.notifier.alert_dispatcher import AlertDispatcher

connection = IBInsyncConnection()
order_mgr = OrderManager(connection)
fetcher = HistoricalDataFetcher(connection)
risk_mgr = RiskManager()

# 2. Create and run bot
from interactive_brokers.strategies.regime_trader import RegimeTrader
bot = RegimeTrader(connection, order_mgr, fetcher, risk_manager=risk_mgr)
bot.run("AAPL", interval_seconds=300)
```

**Risk Manager** is shared across all bots and provides:
- Position sizing (fixed fractional or Kelly criterion)
- Daily loss limit with auto-flatten
- Cooldown after 3 consecutive losses (30 min pause)
- Drawdown circuit breaker (24h pause at 10% drawdown)
- Portfolio heat limit (max 20% of capital at risk)
- Trade frequency throttle (max 10/hour)
</details>

<details>
<summary><strong>🧠 Self-Learning AI Agent</strong> — How it works</summary>

The self-learning agent is the most advanced component in stocks_plugin. It combines all ML models into an autonomous decision-making loop that improves over time.

**Decision Loop (every bar):**
1. **Classify Regime** — LightGBM predicts TRENDING/RANGING/VOLATILE from 30+ features
2. **Query Memory** — "In similar TRENDING conditions, what worked? Which model was most accurate?"
3. **Gather Predictions** — LSTM, Transformer, RL, and momentum each produce a signal
4. **Ensemble Vote** — Weighted combination with adaptive weights (accurate models get more influence)
5. **Risk Gate** — Daily loss limit, drawdown circuit breaker, cooldown checks
6. **Execute** — BUY, SELL, or HOLD
7. **Record** — Full context (features, predictions, regime, P&L) saved to SQLite
8. **Self-Evaluate** — Detect model degradation, recommend retraining

**Self-Improvement Mechanisms:**
- Adaptive ensemble weights update every 20 decisions based on per-model accuracy
- Regime-conditional weighting: e.g., momentum gets amplified in trends, suppressed in volatility
- Confidence calibration: historical win rate adjusts confidence thresholds per regime
- Model degradation detection: accuracy below 45% or declining trend triggers retrain alert

**Quick Start:**
```python
from shared.ml.self_learning_agent import SelfLearningAgent

agent = SelfLearningAgent()
agent.train(df_historical)                    # train all models
decision = agent.decide(df_current, "AAPL")   # autonomous decision
print(decision["action"], decision["confidence"], decision["reasoning"])

# After trade completes:
agent.record_outcome(exit_price=155, pnl=500)

# Check what the agent has learned:
print(agent.get_performance(lookback_days=30))
print(agent.get_weight_summary())
```

**Via CLI:**
```bash
python -m strategies.runner backtest --strategy self_learning --data synthetic
python -m strategies.runner backtest --strategy self_learning --data SPY.csv --params models=regime,lstm
```
</details>

## Security

Please see [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).


---
Part of the [EmbeddedOS Organization](https://embeddedos-org.github.io).
