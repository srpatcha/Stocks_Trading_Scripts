"""
Live Trading Runner — 24/7 Autonomous Daemon
================================================

Runs the SelfLearningAgent continuously, handling:
- Market hours detection (trade during open, learn after close)
- Public data fetching (Yahoo Finance, no brokerage needed)
- News sentiment analysis before every decision
- Trade memory accumulation across sessions
- Model health monitoring and retrain alerts
- Logging everything for audit trail
- Graceful shutdown (Ctrl+C)

Platform Integration:
- Standalone: runs independently, fetches its own data
- TradingView: receives webhook alerts and routes through the agent
- IB Gateway: (future) can route orders to Interactive Brokers

Modes:
- "monitor": Watch and learn only — no trades, just log decisions
- "paper": Simulated trading — track P&L without real orders
- "live": Real trading via IB Gateway (requires IB connection)

Usage:
    # Monitor mode (safe — just watches and learns):
    python -m shared.daemon.live_runner --mode monitor --symbols AAPL,MSFT,GOOGL

    # Paper trading:
    python -m shared.daemon.live_runner --mode paper --symbols SPY,QQQ --interval 300

    # With news sentiment:
    python -m shared.daemon.live_runner --mode paper --symbols AAPL --news
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.utils.market_hours import MarketHours

# Configure logging
LOG_DIR = os.path.join(os.path.expanduser("~"), ".stocks_plugin", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "live_runner.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("live_runner")


class LiveRunner:
    """24/7 autonomous trading daemon.

    Continuously monitors markets, fetches data, analyzes news,
    and makes trading decisions using the SelfLearningAgent.

    Args:
        symbols: List of symbols to monitor.
        mode: "monitor" (watch only), "paper" (simulated), or "live" (real orders).
        interval_seconds: Seconds between decision cycles.
        use_news: Whether to analyze news sentiment.
        db_path: Path to trade memory database.
        models: ML models to train ("regime", "lstm", "transformer", "rl").
        risk_manager: Optional shared RiskManager. Passed through to the
            BrokerBridge so per-trade risk sizing and its gates apply to real
            orders, not just to the paper simulator.
    """

    def __init__(
        self,
        symbols: List[str],
        mode: str = "monitor",
        interval_seconds: int = 300,
        use_news: bool = True,
        db_path: Optional[str] = None,
        models: Optional[List[str]] = None,
        broker: Optional[str] = None,
        broker_config: Optional[Dict[str, Any]] = None,
        risk_manager: Optional[Any] = None,
    ) -> None:
        self._symbols = symbols
        self._mode = mode
        self._interval = interval_seconds
        self._use_news = use_news
        self._running = False
        self._cycle_count = 0
        self._broker_name = broker
        self._broker_config: Dict[str, Any] = broker_config or {}

        # Default paths
        if db_path is None:
            db_dir = os.path.join(os.path.expanduser("~"), ".stocks_plugin", "data")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "trade_memory.db")

        self._models_to_train = models or ["regime"]

        # Initialize components
        from shared.data.public_data_fetcher import PublicDataFetcher
        self._data_fetcher = PublicDataFetcher()

        self._sentiment_analyzer = None
        if use_news:
            try:
                from shared.ml.news_sentiment import NewsSentimentAnalyzer
                self._sentiment_analyzer = NewsSentimentAnalyzer()
            except Exception as e:
                logger.warning("News sentiment unavailable: %s", e)

        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig
        self._agent = SelfLearningAgent(AgentConfig(
            db_path=db_path,
            adaptive_thresholds=True,
        ))

        # Broker Bridge (for live/paper trading with real brokers)
        self._broker_bridge = None
        if broker and mode in ("paper", "live"):
            try:
                from shared.daemon.broker_bridge import BrokerBridge
                self._broker_bridge = BrokerBridge(
                    broker=broker,
                    config=broker_config or {},
                    mode=mode,
                    risk_manager=risk_manager,
                )
                logger.info("BrokerBridge initialized: %s (%s mode)", broker, mode)
            except Exception as e:
                logger.warning("BrokerBridge init failed: %s. Falling back to paper simulation.", e)

        # Paper trading state (fallback when no broker bridge)
        self._paper_positions: Dict[str, Dict[str, Any]] = {}
        self._paper_capital = 100_000.0
        self._paper_pnl = 0.0

        # Holding period tracking (Fix 5)
        self._tick_count = 0
        self._entry_bars: Dict[str, int] = {}

        # FIX 5: Commission rate for paper trading P&L
        self._commission_per_share = 0.005

        # FIX 8: Max holding period for paper positions (bars)
        self._max_holding_bars = 240

        # Market hours enforcement
        self._market_hours = MarketHours()

        # Risk manager (optional). Also handed to the BrokerBridge above so
        # per-trade risk sizing actually applies on the live order path.
        self._risk_manager = risk_manager

        # Signal handler for graceful shutdown (Fix 17: guard for non-main thread)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, OSError):
            logger.warning("Could not install signal handlers (not in main thread)")

        # Fix 2: consecutive error escalation
        self._consecutive_errors: int = 0

        # Optional LLM reasoning layer
        self._llm_reasoner = None

        logger.info(
            "LiveRunner initialized: symbols=%s, mode=%s, interval=%ds, news=%s",
            symbols, mode, interval_seconds, use_news,
        )

    def start(self, train_first: bool = True) -> None:
        """Start the 24/7 daemon loop.

        Args:
            train_first: If True, train models on 1 year of history before starting.
        """
        self._running = True

        self._print_banner()

        if train_first:
            self._initial_training()

        # Connect to broker if configured
        if self._broker_bridge and self._mode in ("paper", "live"):
            logger.info("Connecting to broker: %s", self._broker_name)
            if self._broker_bridge.connect():
                logger.info("Broker connected: %s", self._broker_bridge)
                acct = self._broker_bridge.get_account_info()
                logger.info("Account info: %s", acct)
            else:
                logger.warning("Broker connection failed. Running in paper-simulation mode.")
                self._broker_bridge = None

        logger.info("=" * 60)
        logger.info("DAEMON STARTED — Mode: %s", self._mode.upper())
        logger.info("Monitoring: %s", ", ".join(self._symbols))
        logger.info("Interval: %d seconds", self._interval)
        logger.info("Press Ctrl+C to stop gracefully")
        logger.info("=" * 60)

        while self._running:
            try:
                self._run_cycle()
                self._consecutive_errors = 0  # Fix 2: reset on success
                self._cycle_count += 1

                if self._running:
                    self._smart_sleep()

            except KeyboardInterrupt:
                break
            except Exception as e:
                # Fix 2: error escalation
                self._consecutive_errors += 1
                logger.error(
                    "Cycle error (#%d consecutive): %s",
                    self._consecutive_errors, e, exc_info=True,
                )
                if self._consecutive_errors > 10:
                    logger.critical(
                        "SHUTDOWN: %d consecutive errors exceeded threshold (10). "
                        "Stopping daemon.",
                        self._consecutive_errors,
                    )
                    self._running = False
                elif self._consecutive_errors > 5:
                    alert_msg = (
                        f"⚠️ LiveRunner: {self._consecutive_errors} consecutive errors. "
                        f"Latest: {e}"
                    )
                    logger.warning(alert_msg)
                    if hasattr(self, '_broker_bridge') and self._broker_bridge:
                        try:
                            from shared.notifier import AlertDispatcher
                            # AlertDispatcher requires a config dict, and
                            # dispatch() takes priority=, not level=. Both were
                            # wrong, so the TypeError went into the bare except
                            # below and this alert never actually fired.
                            dispatcher = AlertDispatcher(
                                self._broker_config.get("notifications", {})
                            )
                            dispatcher.dispatch(
                                title="LiveRunner Error Escalation",
                                message=alert_msg,
                                priority="WARNING",
                            )
                        except Exception as alert_exc:
                            logger.error("Error-escalation alert failed: %s", alert_exc)
                time.sleep(60)

        self._shutdown()

    def _run_cycle(self) -> None:
        """Execute one decision cycle for all symbols."""
        self._tick_count += 1

        # FIX 9: Sync position tracking to avoid dual-tracking divergence
        self._sync_positions()

        market_open = self._market_hours.is_market_open()
        trading_allowed = self._market_hours.is_trading_allowed()
        now = datetime.now(timezone.utc)

        logger.info(
            "─── Cycle #%d | %s | Market: %s | Trading: %s ───",
            self._cycle_count + 1,
            now.strftime("%Y-%m-%d %H:%M:%S"),
            "OPEN" if market_open else "CLOSED",
            "ALLOWED" if trading_allowed else "BLOCKED",
        )

        # GAP #7: Periodic retrain on a schedule
        RETRAIN_INTERVAL_CYCLES = 5000  # roughly every ~17 hours at 12 cycles/hr
        if self._tick_count > 0 and self._tick_count % RETRAIN_INTERVAL_CYCLES == 0:
            logger.info("Scheduled periodic retrain (cycle %d)", self._tick_count)
            try:
                for sym in self._symbols[:1]:  # Use first symbol as representative
                    retrain_df = self._data_fetcher.fetch_ohlcv(sym, period="2y", interval="1d")
                    if retrain_df is not None and len(retrain_df) > 200:
                        self._agent.train(retrain_df, verbose=False)
                        logger.info("Periodic retrain complete on %s (%d bars)", sym, len(retrain_df))
                        break
            except Exception as e:
                logger.error("Periodic retrain failed: %s", e)

        # Market hours enforcement: skip processing if trading is not allowed
        if not trading_allowed:
            # GAP #8: Use off-hours productively
            if self._tick_count % 360 == 0:  # Every ~30 min during off-hours
                self._off_hours_maintenance()
            logger.info("Trading not allowed outside configured market hours. Skipping cycle.")
            return

        # End-of-day flatten: close all positions before market close
        if self._market_hours.should_flatten_eod():
            remaining = self._market_hours.time_to_close()
            logger.warning(
                "EOD flatten triggered — %s remaining before close. Closing all positions.",
                remaining,
            )
            self._flatten_all_positions()
            return

        for symbol in self._symbols:
            try:
                # FIX 8: Check max holding period for paper positions
                if symbol in self._paper_positions:
                    bars_held = self._tick_count - self._entry_bars.get(symbol, self._tick_count)
                    if self._max_holding_bars > 0 and bars_held > self._max_holding_bars:
                        logger.warning(
                            "⚠️ PAPER MAX HOLDING: %s held %d bars (max=%d). Force closing.",
                            symbol, bars_held, self._max_holding_bars,
                        )
                        df = self._data_fetcher.fetch_ohlcv(symbol, period="5d", interval="1d")
                        price = float(df["close"].iloc[-1]) if df is not None and not df.empty else self._paper_positions[symbol]["entry_price"]
                        self._close_paper_position(symbol, price)
                        continue

                self._process_symbol(symbol, market_open)
            except Exception as e:
                logger.error("Error processing %s: %s", symbol, e)

        # Position reconciliation + force-close (from hyperliquid-trading-agent patterns)
        if self._broker_bridge and self._broker_bridge.is_connected():
            try:
                self._broker_bridge.reconcile_positions()
                force_closed = self._broker_bridge.check_and_force_close(agent=self._agent)
                if force_closed:
                    logger.warning("Force-closed %d positions exceeding max loss", len(force_closed))
            except Exception as e:
                logger.debug("Reconciliation/force-close check: %s", e)

        # Periodic status report
        if self._cycle_count % 12 == 0:  # every 12 cycles
            self._print_status_report()

    def _sync_positions(self) -> None:
        """FIX 9: Reconcile dual position tracking to prevent divergence.

        If broker_bridge is active and connected, paper positions are cleared
        (bridge is the source of truth). If the bridge fails mid-session,
        log CRITICAL instead of silently falling back to paper.
        """
        if self._broker_bridge is not None:
            if self._broker_bridge.is_connected():
                if self._paper_positions:
                    logger.info(
                        "Position sync: clearing %d paper positions (broker bridge is active)",
                        len(self._paper_positions),
                    )
                    self._paper_positions.clear()
                    self._entry_bars.clear()
            else:
                # Bridge exists but disconnected — do NOT silently fall back
                logger.critical(
                    "POSITION SYNC: broker bridge configured but DISCONNECTED. "
                    "NOT falling back to paper trading. Attempting reconnect..."
                )
                try:
                    if self._broker_bridge.connect():
                        logger.info("Broker reconnected successfully")
                    else:
                        logger.critical("Broker reconnect FAILED. Skipping trades this cycle.")
                except Exception as e:
                    logger.critical("Broker reconnect error: %s", e)

        # Position count reconciliation check
        if self._broker_bridge and self._broker_bridge.is_connected():
            bridge_count = len(self._broker_bridge.get_positions())
            paper_count = len(self._paper_positions)
            if paper_count > 0 and bridge_count > 0:
                logger.warning(
                    "POSITION DIVERGENCE: bridge has %d positions, paper has %d. "
                    "Clearing paper positions.",
                    bridge_count, paper_count,
                )
                self._paper_positions.clear()
                self._entry_bars.clear()

    def _process_symbol(self, symbol: str, market_open: bool) -> None:
        """Process a single symbol: fetch data → analyze → decide."""

        # Step 1: Fetch latest data
        df = self._data_fetcher.fetch_ohlcv(symbol, period="6mo", interval="1d")
        if df is None or df.empty or len(df) < 60:
            logger.warning("Insufficient data for %s (%d bars)", symbol, len(df) if df is not None else 0)
            return

        # Step 2: News sentiment (if enabled)
        sentiment_score = 0.0
        if self._sentiment_analyzer:
            try:
                sentiment = self._sentiment_analyzer.analyze(symbol, max_headlines=10)
                sentiment_score = sentiment.get("sentiment_score", 0.0)
                logger.info(
                    "  %s sentiment: %.3f (%s) | %d headlines | %d bull, %d bear",
                    symbol, sentiment_score, sentiment.get("sentiment_label", "?"),
                    sentiment.get("headlines_analyzed", 0),
                    sentiment.get("bullish_count", 0),
                    sentiment.get("bearish_count", 0),
                )
            except Exception as e:
                logger.debug("Sentiment analysis failed for %s: %s", symbol, e)

        # Step 3: Agent decision
        decision = self._agent.decide(df, symbol=symbol)

        action = decision["action"]
        confidence = decision["confidence"]
        regime = decision["regime"]
        price = decision["price"]

        # Incorporate sentiment into action
        if sentiment_score != 0:
            action = self._apply_sentiment_filter(action, sentiment_score, confidence)

        logger.info(
            "  %s: %s @ $%.2f | confidence=%.2f | regime=%s | sentiment=%.3f",
            symbol, action, price, confidence, regime, sentiment_score,
        )

        # Step 4b: Optional LLM reasoning layer (like hyperliquid-trading-agent)
        if hasattr(self, '_llm_reasoner') and self._llm_reasoner is not None:
            try:
                llm_result = self._llm_reasoner.reason(
                    symbol=symbol,
                    price=price,
                    regime=regime,
                    regime_confidence=decision.get("regime_probabilities", {}).get(regime, 0),
                    predictions=decision.get("predictions", {}),
                    ensemble_signal=decision.get("ensemble_signal", {}),
                    sentiment={"sentiment_score": sentiment_score} if sentiment_score else None,
                    risk_status=decision.get("risk_status"),
                )
                llm_action = llm_result.get("action", action)
                llm_confidence = llm_result.get("confidence", confidence)

                # Re-check risk manager before accepting LLM override
                if llm_action != "HOLD" and llm_action != action:
                    if self._risk_manager is not None and not self._risk_manager.can_trade():
                        logger.warning(
                            "LLM override %s->%s blocked by risk manager",
                            action, llm_action,
                        )
                        llm_action = "HOLD"
                        llm_confidence = 0.0

                action = llm_action
                confidence = llm_confidence
                # Pass TP/SL from LLM to broker bridge
                decision["tp_price"] = llm_result.get("tp_price")
                decision["sl_price"] = llm_result.get("sl_price")
                decision["exit_plan"] = llm_result.get("exit_plan", "")
                decision["reasoning"] = llm_result.get("reasoning", "")[:500]
                logger.info(
                    "  LLM override: %s (confidence=%.2f) | TP=%s SL=%s",
                    action, confidence, llm_result.get("tp_price"), llm_result.get("sl_price"),
                )
            except Exception as e:
                logger.debug("LLM reasoning skipped: %s", e)

        # Step 4: Execute based on mode
        if self._mode == "monitor":
            self._log_decision(symbol, action, price, confidence, regime)
        elif self._mode == "paper":
            if self._broker_bridge and self._broker_bridge.is_connected():
                self._execute_live_trade(symbol, action, price, confidence)
            else:
                self._execute_paper_trade(symbol, action, price, confidence)
        elif self._mode == "live" and market_open:
            self._execute_live_trade(symbol, action, price, confidence)

    def _apply_sentiment_filter(
        self, action: str, sentiment: float, confidence: float,
    ) -> str:
        """Adjust trading action based on news sentiment.

        Strong negative sentiment can block buys.
        Strong positive sentiment can block sells.
        """
        if action == "BUY" and sentiment < -0.3:
            logger.info("  Sentiment override: blocking BUY (sentiment=%.2f)", sentiment)
            return "HOLD"
        if action == "SELL" and sentiment > 0.3:
            logger.info("  Sentiment override: blocking SELL (sentiment=%.2f)", sentiment)
            return "HOLD"
        return action

    # ─── Paper Trading ───

    def _execute_paper_trade(
        self, symbol: str, action: str, price: float, confidence: float,
    ) -> None:
        """Simulate a trade in paper mode."""
        has_position = symbol in self._paper_positions
        current_pos = self._paper_positions.get(symbol)

        if action == "BUY" and not has_position:
            # Open long position
            shares = int(self._paper_capital * 0.1 / price)
            if shares > 0:
                cost = shares * price
                self._paper_positions[symbol] = {
                    "shares": shares,
                    "entry_price": price,
                    "entry_time": datetime.now().isoformat(),
                    "direction": "long",
                }
                self._entry_bars[symbol] = self._tick_count
                logger.info(
                    "  📈 PAPER BUY: %d shares of %s @ $%.2f ($%.2f)",
                    shares, symbol, price, cost,
                )

        elif action == "BUY" and has_position and current_pos["direction"] == "short":
            # Close short position
            self._close_paper_position(symbol, price)

        elif action == "SELL" and has_position and current_pos["direction"] == "long":
            # Close long position
            self._close_paper_position(symbol, price)

        elif action == "SELL" and not has_position:
            # Open short position (paper)
            shares = int(self._paper_capital * 0.1 / price)
            if shares > 0:
                self._paper_positions[symbol] = {
                    "shares": shares,
                    "entry_price": price,
                    "entry_time": datetime.now().isoformat(),
                    "direction": "short",
                }
                self._entry_bars[symbol] = self._tick_count
                logger.info(
                    "  📉 PAPER SHORT: %d shares of %s @ $%.2f",
                    shares, symbol, price,
                )

        elif action == "SELL" and has_position and current_pos["direction"] == "short":
            logger.debug("Already short %s, skipping SELL", symbol)

        elif action == "BUY" and has_position and current_pos["direction"] == "long":
            logger.debug("Already long %s, skipping BUY", symbol)

    def _close_paper_position(self, symbol: str, exit_price: float) -> None:
        """Close a paper position (long or short) and record outcome."""
        pos = self._paper_positions.pop(symbol, None)
        if not pos:
            return

        direction = pos["direction"]
        entry_price = pos["entry_price"]
        shares = pos["shares"]

        if direction == "long":
            raw_pnl = (exit_price - entry_price) * shares
        else:
            raw_pnl = (entry_price - exit_price) * shares

        # FIX 5: Deduct commissions (entry + exit)
        commission = self._commission_per_share * shares * 2
        pnl = raw_pnl - commission

        pnl_pct = pnl / (entry_price * shares) * 100
        self._paper_pnl += pnl

        holding_bars = self._tick_count - self._entry_bars.pop(symbol, self._tick_count)
        close_label = "PAPER SELL" if direction == "long" else "PAPER COVER"
        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            "  %s %s: %d shares of %s @ $%.2f | raw=$%.2f comm=$%.2f net=$%.2f (%.2f%%) | held %d bars",
            emoji, close_label, shares, symbol, exit_price, raw_pnl, commission, pnl, pnl_pct, holding_bars,
        )

        self._agent.record_outcome(
            exit_price=exit_price,
            pnl=pnl,
            holding_period_bars=holding_bars,
        )

    def _execute_live_trade(
        self, symbol: str, action: str, price: float, confidence: float,
    ) -> None:
        """Execute a live trade via the connected BrokerBridge."""
        if not self._broker_bridge:
            logger.warning(
                "LIVE TRADING: no broker connected. Use --broker ib|tradestation|schwab"
            )
            return

        if not self._broker_bridge.is_connected():
            logger.info("Connecting to broker...")
            if not self._broker_bridge.connect():
                logger.error("Broker connection failed. Skipping trade.")
                return

        # Build a decision dict for the bridge
        decision = {"action": action, "confidence": confidence, "price": price}
        result = self._broker_bridge.execute_decision(decision, symbol, agent=self._agent)

        if result and result.success:
            logger.info(
                "🔴 LIVE TRADE: %s %d %s @ $%.2f via %s [order=%s]",
                action, result.shares, symbol, price,
                result.broker, result.order_id,
            )
        elif result:
            logger.error("LIVE TRADE FAILED: %s | %s", symbol, result.message)

    def _log_decision(
        self, symbol: str, action: str, price: float, confidence: float, regime: str,
    ) -> None:
        """Log decision in monitor mode (no execution)."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "price": price,
            "confidence": confidence,
            "regime": regime,
            "mode": "monitor",
        }

        log_file = os.path.join(LOG_DIR, "decisions.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    # ─── Training ───

    def _initial_training(self) -> None:
        """Train models on historical data before starting the live loop."""
        logger.info("=" * 60)
        logger.info("INITIAL TRAINING PHASE")
        logger.info("=" * 60)

        # GAP #1: Wire data fetcher into agent for fresh auto-retrain data
        self._agent.set_data_fetcher(self._data_fetcher, self._symbols)

        # GAP #6: Try to load existing saved models first
        model_dir = os.path.join(os.path.expanduser("~"), ".stocks_plugin", "models")
        if os.path.isdir(model_dir) and os.listdir(model_dir):
            try:
                self._agent.load_models(model_dir)
                logger.info("Loaded pre-trained models from %s", model_dir)

                # Also load ensemble weights
                weights_path = os.path.join(model_dir, "ensemble_weights.json")
                if hasattr(self._agent, '_ensemble') and hasattr(self._agent._ensemble, 'load_weights'):
                    self._agent._ensemble.load_weights(weights_path)

                return  # Skip fresh training
            except Exception as e:
                logger.warning("Failed to load saved models: %s. Training fresh.", e)

        # No saved models — train from scratch
        primary_symbol = self._symbols[0]
        logger.info("No saved models found. Fetching 2 years of %s data for training...", primary_symbol)

        df = self._data_fetcher.fetch_ohlcv(primary_symbol, period="2y", interval="1d")
        if df is None or df.empty:
            logger.warning("Could not fetch training data for %s. Using untrained agent.", primary_symbol)
            return

        logger.info("Training on %d bars of %s...", len(df), primary_symbol)
        try:
            results = self._agent.train(
                df.reset_index() if "date" not in df.columns else df,
                models=self._models_to_train,
                verbose=True,
            )
            logger.info("Training complete: %s", list(results.keys()))
        except Exception as e:
            logger.error("Training failed: %s. Agent will use momentum fallback.", e)

    # ─── Sleep & Scheduling ───

    def _smart_sleep(self) -> None:
        """Sleep with market-aware intervals.

        During market hours: use configured interval
        After hours: check less frequently (15 min)
        Weekends: check very infrequently (1 hour)
        """
        now = datetime.now()
        is_weekend = now.weekday() >= 5

        if is_weekend:
            sleep_time = 3600  # 1 hour on weekends
        elif self._data_fetcher.is_market_open():
            sleep_time = self._interval  # configured interval during market
        else:
            sleep_time = 900  # 15 minutes after hours

        logger.debug("Sleeping %d seconds...", sleep_time)

        # Sleep in small chunks for responsive shutdown
        elapsed = 0
        while elapsed < sleep_time and self._running:
            time.sleep(min(10, sleep_time - elapsed))
            elapsed += 10

    # ─── Status Reporting ───

    def _print_status_report(self) -> None:
        """Print periodic status report."""
        logger.info("=" * 50)
        logger.info("STATUS REPORT — Cycle #%d", self._cycle_count)
        logger.info("=" * 50)

        # Paper positions
        if self._paper_positions:
            logger.info("Open positions:")
            for sym, pos in self._paper_positions.items():
                logger.info("  %s: %d shares @ $%.2f (%s)",
                           sym, pos["shares"], pos["entry_price"], pos["direction"])
        else:
            logger.info("No open positions")

        logger.info("Paper P&L: $%.2f", self._paper_pnl)

        # Agent stats
        try:
            perf = self._agent.get_performance(lookback_days=7)
            logger.info("Agent (7d): %d trades, win_rate=%.0f%%",
                       perf.get("total_trades", 0),
                       perf.get("win_rate", 0) * 100)
        except Exception:
            pass

        # Weights
        try:
            weights = self._agent.get_weight_summary()
            for model, w in weights.items():
                logger.info("  %s weight: %.3f", model, w.get("effective_weight", 0))
        except Exception:
            pass

        logger.info("=" * 50)

    def _print_banner(self) -> None:
        """Print startup banner."""
        print()
        print("╔══════════════════════════════════════════════════╗")
        print("║     🧠 SELF-LEARNING TRADING AGENT — LIVE       ║")
        print("║                                                  ║")
        print(f"║  Mode:    {self._mode.upper():<40s}║")
        print(f"║  Symbols: {', '.join(self._symbols):<40s}║")
        print(f"║  Interval: {self._interval}s{' ' * (38 - len(str(self._interval)))}║")
        print(f"║  News:    {'ON' if self._use_news else 'OFF':<40s}║")
        broker_str = self._broker_name.upper() if self._broker_name else "NONE"
        print(f"║  Broker:  {broker_str:<40s}║")
        print("║                                                  ║")
        print("║  Press Ctrl+C to stop                            ║")
        print("╚══════════════════════════════════════════════════╝")
        print()

    # ─── Shutdown ───

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received (signal=%d)", signum)
        self._running = False

    def _flatten_all_positions(self) -> None:
        """Close all open positions for end-of-day flatten."""
        # Close paper positions
        if self._paper_positions:
            symbols_to_close = list(self._paper_positions.keys())
            for symbol in symbols_to_close:
                try:
                    df = self._data_fetcher.fetch_ohlcv(symbol, period="5d", interval="1d")
                    if df is not None and not df.empty:
                        price = float(df["close"].iloc[-1])
                    else:
                        price = self._paper_positions[symbol]["entry_price"]
                    logger.info("  EOD flatten: closing %s @ $%.2f", symbol, price)
                    self._close_paper_position(symbol, price)
                except Exception as e:
                    logger.error("Failed to flatten %s: %s", symbol, e)

        # Close broker positions.
        #
        # This used to iterate get_positions() — a Dict[str, Position] — and
        # call pos.get("symbol") on each dict *key*, i.e. on a str. Every
        # iteration raised AttributeError, which the except below swallowed, so
        # broker positions were never flattened at EOD or on shutdown. It also
        # passed price=0, which would have booked a fabricated -100% P&L.
        # close_all_positions() already does this correctly: it fetches the
        # live price, falls back to the entry price, and routes through
        # _close_position so the portfolio gate and agent memory both update.
        if self._broker_bridge and self._broker_bridge.is_connected():
            try:
                results = self._broker_bridge.close_all_positions(agent=self._agent)
                for result in results:
                    if result.success:
                        logger.info("  EOD flatten: closed %s via broker", result.symbol)
                    else:
                        logger.error(
                            "  EOD flatten FAILED for %s: %s", result.symbol, result.message,
                        )
            except Exception as e:
                logger.error("Failed to flatten broker positions: %s", e)

    def _shutdown(self) -> None:
        """Graceful shutdown: flatten positions, save state, report."""
        logger.info("=" * 60)
        logger.info("SHUTTING DOWN")
        logger.info("=" * 60)

        # FIX 6: Flatten ALL positions (broker + paper) before saving
        logger.info("Shutdown: flattening all positions before exit")
        try:
            self._flatten_all_positions()
        except Exception as e:
            logger.error("Flatten during shutdown failed (continuing): %s", e)

        # Fix 5: log final paper P&L summary
        logger.info("═" * 40)
        logger.info("FINAL PAPER P&L SUMMARY")
        logger.info("  Total paper P&L: $%.2f", self._paper_pnl)
        logger.info("  Total cycles: %d", self._cycle_count)
        logger.info("═" * 40)

        # Save models
        try:
            model_dir = os.path.join(os.path.expanduser("~"), ".stocks_plugin", "models")
            self._agent.save_models(model_dir)
            logger.info("Models saved to %s", model_dir)

            # GAP #6: Also save ensemble weights on shutdown
            weights_path = os.path.join(model_dir, "ensemble_weights.json")
            if hasattr(self._agent, '_ensemble') and hasattr(self._agent._ensemble, 'save_weights'):
                self._agent._ensemble.save_weights(weights_path)
        except Exception as e:
            logger.warning("Failed to save models: %s", e)

        # Fix 18: save data cache to disk
        try:
            self._data_fetcher.save_cache_to_disk()
        except Exception as e:
            logger.debug("Cache save failed: %s", e)

        self._agent.close()
        logger.info("Daemon stopped.")

    def _off_hours_maintenance(self) -> None:
        """GAP #8: Use market-closed time for model maintenance and analysis."""
        try:
            # 1. Clean up old trade memory (GAP #2)
            if hasattr(self._agent, '_memory') and self._agent._memory:
                self._agent._memory.cleanup_old_trades(max_age_days=365, max_records=10000)

            # 2. Update ensemble weights from accumulated memory (GAP #3)
            if hasattr(self._agent, '_ensemble') and hasattr(self._agent, '_memory'):
                self._agent._ensemble.update_regime_multipliers_from_memory(self._agent._memory)

            # 3. Run a quick model health check
            if hasattr(self._agent, '_check_retrain_needed'):
                self._agent._check_retrain_needed()

            # 4. Save current state
            model_dir = os.path.join(os.path.expanduser("~"), ".stocks_plugin", "models")
            os.makedirs(model_dir, exist_ok=True)
            self._agent.save_models(model_dir)

            # Also save ensemble weights
            weights_path = os.path.join(model_dir, "ensemble_weights.json")
            if hasattr(self._agent, '_ensemble') and hasattr(self._agent._ensemble, 'save_weights'):
                self._agent._ensemble.save_weights(weights_path)

            logger.info("Off-hours maintenance complete")
        except Exception as e:
            logger.debug("Off-hours maintenance error: %s", e)


# ─── CLI Entry Point ───

def main() -> None:
    """CLI entry point for the live runner daemon."""
    parser = argparse.ArgumentParser(
        prog="live_runner",
        description="24/7 Self-Learning Trading Agent Daemon",
    )
    parser.add_argument(
        "--symbols", "-s", required=True,
        help="Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--mode", "-m", default="monitor",
        choices=["monitor", "paper", "live"],
        help="Trading mode: monitor (watch only), paper (simulated), live (real). Default: monitor",
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=300,
        help="Seconds between decision cycles (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--news", action="store_true",
        help="Enable news sentiment analysis",
    )
    parser.add_argument(
        "--no-train", action="store_true",
        help="Skip initial model training",
    )
    parser.add_argument(
        "--models", default="regime",
        help="Comma-separated models to train (default: regime). Options: regime,lstm,transformer,rl",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to trade memory database",
    )
    parser.add_argument(
        "--broker", "-b", default=None,
        choices=["ib", "tradestation", "schwab"],
        help="Broker for live/paper trading: ib, tradestation, or schwab",
    )
    parser.add_argument(
        "--broker-port", type=int, default=None,
        help="Broker connection port (IB: 7497=paper, 7496=live; default: auto)",
    )
    parser.add_argument(
        "--broker-host", default="127.0.0.1",
        help="Broker host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--llm", default=None,
        choices=["anthropic", "openai"],
        help="Enable LLM reasoning layer: anthropic (Claude) or openai (GPT)",
    )
    parser.add_argument(
        "--llm-key", default=None,
        help="API key for the LLM provider (or set ANTHROPIC_API_KEY / OPENAI_API_KEY env var)",
    )

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    models = [m.strip() for m in args.models.split(",")]

    broker_config = {}
    if args.broker:
        broker_config["host"] = args.broker_host
        if args.broker_port:
            broker_config["port"] = args.broker_port

    runner = LiveRunner(
        symbols=symbols,
        mode=args.mode,
        interval_seconds=args.interval,
        use_news=args.news,
        db_path=args.db,
        models=models,
        broker=args.broker,
        broker_config=broker_config if broker_config else None,
    )

    # Initialize LLM reasoning if requested (must be before runner.start() which blocks)
    if args.llm:
        try:
            from shared.ml.llm_reasoning import LLMReasoner
            api_key = args.llm_key or os.environ.get(
                "ANTHROPIC_API_KEY" if args.llm == "anthropic" else "OPENAI_API_KEY"
            )
            if api_key:
                runner._llm_reasoner = LLMReasoner(provider=args.llm, api_key=api_key)
                logger.info("LLM reasoning enabled: %s", args.llm)
            else:
                logger.warning("No API key for %s. Set --llm-key or env var.", args.llm)
        except ImportError as e:
            logger.warning("LLM provider not installed: %s", e)

    runner.start(train_first=not args.no_train)


if __name__ == "__main__":
    main()
