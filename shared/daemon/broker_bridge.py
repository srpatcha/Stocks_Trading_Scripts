"""
Broker Bridge — Connects AI Agent Decisions to Real Broker APIs
================================================================

The critical last-mile wiring that turns AI decisions into real trades.
Maps SelfLearningAgent.decide() output → broker-specific order execution.

Supported Brokers:
- Interactive Brokers (via ib_async / ibapi)
- TradeStation (via REST API)
- Schwab/thinkorswim (via Schwab API)

Features:
- Unified interface across all brokers
- Position tracking with entry prices
- Fill callbacks that auto-record outcomes to TradeMemory
- Configurable safety: paper-only mode, max order size, confirmation
- Translates agent signals to broker-native order types

Usage:
    bridge = BrokerBridge(broker="ib", mode="paper")
    bridge.connect()
    bridge.execute_decision(agent_decision, symbol="AAPL")
    bridge.get_positions()
    bridge.disconnect()
"""

from __future__ import annotations

import glob
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from shared.risk_manager_unified import UnifiedPortfolioRiskGate, UnifiedRiskConfig
from shared.utils.paths import stocks_plugin_data_dir

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Tracked position state."""
    symbol: str
    direction: str  # "long" or "short"
    shares: int
    entry_price: float
    entry_time: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    order_id: Optional[str] = None
    max_holding_bars: int = 0
    bars_held: int = 0


@dataclass
class TrailingStop:
    """Trailing stop configuration for a position."""
    symbol: str
    direction: str  # "long" or "short"
    activation_pct: float  # e.g., 0.02 = activate after 2% profit
    trail_pct: float  # e.g., 0.015 = trail by 1.5%
    highest_price: float = 0.0
    lowest_price: float = float('inf')
    activated: bool = False
    stop_price: float = 0.0


@dataclass
class ExecutionResult:
    """Result of a trade execution."""
    success: bool
    broker: str
    symbol: str
    action: str
    shares: int
    price: float
    order_id: str = ""
    message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    fill_price: Optional[float] = None
    filled_quantity: Optional[int] = None


# ─── Commission Model
#
# The CommissionModel provides a unified way to estimate trading commissions
# across all supported brokers.  The default values reflect typical US equity
# commission schedules (similar to Interactive Brokers tiered pricing):
#
#   per_share   — charge per share traded (default $0.005)
#   min_per_order — minimum commission per order (default $1.00)
#   max_pct     — cap as a fraction of trade value (default 0.5%)
#
# Formula:  commission = max(min_per_order,
#                            min(shares * per_share,
#                                shares * price * max_pct))
#
# Brokers with different fee structures can override per_share / min /
# max_pct when constructing BrokerBridge.  The commission is subtracted
# from realized P&L on every position close.

@dataclass
class CommissionModel:
    """Configurable per-trade commission estimator."""
    per_share: float = 0.005
    min_per_order: float = 1.0
    max_pct: float = 0.005

    def calculate_commission(self, shares: int, price: float) -> float:
        """Estimate the commission for a trade.

        Returns:
            Commission in dollars (always >= 0).
        """
        if shares <= 0 or price <= 0:
            return 0.0
        raw = shares * self.per_share
        cap = shares * price * self.max_pct
        return max(self.min_per_order, min(raw, cap))


class BaseBrokerAdapter(ABC):
    """Abstract broker adapter interface."""

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def place_market_order(self, symbol: str, action: str, quantity: int) -> ExecutionResult: ...

    @abstractmethod
    def place_limit_order(self, symbol: str, action: str, quantity: int, price: float) -> ExecutionResult: ...

    def place_stop_order(self, symbol: str, action: str, quantity: int, stop_price: float) -> ExecutionResult:
        """Place a stop order. Fails closed when the adapter has no stop support.

        This used to fall back to ``place_limit_order(..., stop_price)``. That is
        not a degraded stop — a SELL limit below the market (or a BUY limit
        above it) is immediately marketable, so the "protective stop" filled at
        once and closed the very position it was created to guard. Reporting
        failure lets the caller decide, instead of silently flattening.
        """
        logger.error(
            "place_stop_order() not implemented for %s — refusing to substitute a "
            "marketable limit order. The position is UNPROTECTED.",
            getattr(self, '_name', 'unknown'),
        )
        return ExecutionResult(
            False, getattr(self, '_name', 'unknown'), symbol, action, quantity, stop_price,
            message="stop orders unsupported by this adapter; no protective order placed",
        )

    def get_fill(self, order_id: str) -> Optional[tuple]:
        """Return ``(avg_fill_price, filled_quantity)``, or None if unknown.

        No adapter used to implement this, so ExecutionResult.fill_price and
        .filled_quantity were always None and every P&L in the diary, in agent
        memory and in the portfolio gate was computed from the *decision*
        price. Slippage and partial fills were invisible.
        """
        return None

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]: ...

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# ─── Interactive Brokers Adapter ───

class IBAdapter(BaseBrokerAdapter):
    """Real Interactive Brokers adapter using ib_async OrderManager."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._connection = None
        self._order_manager = None
        self._data_fetcher = None
        self._connected = False
        self._name = "interactive_brokers"
        self._event_loop = None
        self._loop_thread = None

    @property
    def name(self) -> str:
        return self._name

    def _ensure_event_loop(self) -> None:
        """Start a dedicated asyncio event loop in a background daemon thread.

        ib_async requires a running event loop for all operations. BrokerBridge
        is synchronous, so we spin up a dedicated loop thread and submit
        coroutines to it via run_coroutine_threadsafe().
        """
        import asyncio
        import threading

        if self._event_loop is not None and self._event_loop.is_running():
            return

        self._event_loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(self._event_loop)
            self._event_loop.run_forever()

        self._loop_thread = threading.Thread(target=_run, daemon=True, name="ib-async-loop")
        self._loop_thread.start()

    def _run_async(self, coro, timeout: float = 30.0):
        """Run an async coroutine on the IB event loop from synchronous code."""
        import asyncio
        self._ensure_event_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
        return future.result(timeout=timeout)

    def connect(self) -> bool:
        try:
            import inspect
            from interactive_brokers.utils.ib_connection import IBConnection
            from interactive_brokers.utils.order_manager import OrderManager

            self._connection = IBConnection.create(
                backend="ib_async",
                host=self._host,
                port=self._port,
                client_id=self._client_id,
            )

            # IBAsyncConnection.connect() is a coroutine — bridge to sync via
            # a dedicated background event loop so the IB message loop keeps
            # running after the initial handshake.
            if inspect.iscoroutinefunction(self._connection.connect):
                self._run_async(self._connection.connect(), timeout=30.0)
            else:
                self._connection.connect()

            self._order_manager = OrderManager(self._connection)

            try:
                from interactive_brokers.data.historical_fetcher import HistoricalDataFetcher
                self._data_fetcher = HistoricalDataFetcher(self._connection)
            except Exception:
                pass

            self._connected = True
            mode = "PAPER" if self._port in (7497, 4002) else "LIVE"
            logger.info("IB connected [%s:%d, mode=%s]", self._host, self._port, mode)
            return True
        except Exception as e:
            logger.error("IB connection failed: %s", e)
            return False

    def disconnect(self) -> None:
        if self._connection:
            self._connection.disconnect()
            self._connected = False
            logger.info("IB disconnected")
        if self._event_loop and self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)

    def is_connected(self) -> bool:
        if self._connection:
            return self._connection.is_connected()
        return False

    def place_market_order(self, symbol: str, action: str, quantity: int) -> ExecutionResult:
        if not self._order_manager:
            return ExecutionResult(False, self.name, symbol, action, quantity, 0, message="Not connected")

        try:
            trade = self._order_manager.market_order(symbol, action.upper(), float(quantity))
            order_id = str(trade.order.orderId) if hasattr(trade, 'order') else str(id(trade))
            logger.info("IB market order: %s %d %s → %s", action, quantity, symbol, order_id)
            return ExecutionResult(True, self.name, symbol, action, quantity, 0, order_id=order_id,
                                   message=f"Market order placed: {action} {quantity} {symbol}")
        except Exception as e:
            logger.error("IB market order failed: %s", e)
            return ExecutionResult(False, self.name, symbol, action, quantity, 0, message=str(e))

    def get_fill(self, order_id: str) -> Optional[tuple]:
        if not self._order_manager:
            return None
        try:
            return self._order_manager.get_fill(int(order_id))
        except Exception as e:
            logger.debug("IB fill lookup failed for %s: %s", order_id, e)
            return None

    def place_limit_order(self, symbol: str, action: str, quantity: int, price: float) -> ExecutionResult:
        if not self._order_manager:
            return ExecutionResult(False, self.name, symbol, action, quantity, price, message="Not connected")

        try:
            trade = self._order_manager.limit_order(symbol, action.upper(), float(quantity), price)
            order_id = str(trade.order.orderId) if hasattr(trade, 'order') else str(id(trade))
            return ExecutionResult(True, self.name, symbol, action, quantity, price, order_id=order_id,
                                   message=f"Limit order: {action} {quantity} {symbol} @ ${price:.2f}")
        except Exception as e:
            return ExecutionResult(False, self.name, symbol, action, quantity, price, message=str(e))

    def place_stop_order(self, symbol: str, action: str, quantity: int, stop_price: float) -> ExecutionResult:
        """Place a native IB stop order via ib_async."""
        if not self._order_manager:
            return ExecutionResult(False, self.name, symbol, action, quantity, stop_price, message="Not connected")
        try:
            trade = self._order_manager.stop_order(symbol, action.upper(), float(quantity), stop_price)
            order_id = str(trade.order.orderId) if hasattr(trade, 'order') else str(id(trade))
            logger.info("IB stop order: %s %d %s @ $%.2f → %s", action, quantity, symbol, stop_price, order_id)
            return ExecutionResult(True, self.name, symbol, action, quantity, stop_price, order_id=order_id,
                                   message=f"Stop order: {action} {quantity} {symbol} @ ${stop_price:.2f}")
        except Exception as e:
            # Do NOT fall back to a limit order at stop_price — it is marketable
            # and would immediately close the position this stop protects.
            logger.error("IB native stop order failed — position UNPROTECTED: %s", e)
            return ExecutionResult(False, self.name, symbol, action, quantity, stop_price,
                                   message=f"stop order failed, no protective order placed: {e}")

    def cancel_order(self, order_id: str) -> bool:
        if self._order_manager:
            return self._order_manager.cancel_order(int(order_id))
        return False

    def get_positions(self) -> List[Dict[str, Any]]:
        if not self._connection or not self.is_connected():
            return []
        try:
            positions = self._connection.positions()
            return [
                {
                    "symbol": p.contract.symbol,
                    "quantity": p.position,
                    "avg_cost": p.avgCost,
                    "market_value": p.marketValue if hasattr(p, 'marketValue') else 0,
                }
                for p in positions
            ]
        except Exception as e:
            logger.error("Failed to get IB positions: %s", e)
            return []

    def get_account_info(self) -> Dict[str, Any]:
        if not self._connection or not self.is_connected():
            return {"broker": self.name, "connected": False}
        try:
            summary = self._connection.accountSummary()
            info = {"broker": self.name, "connected": True}
            for item in summary:
                if item.tag in ("NetLiquidation", "BuyingPower", "TotalCashValue", "GrossPositionValue"):
                    info[item.tag] = float(item.value)
            return info
        except Exception:
            return {"broker": self.name, "connected": True}

    def get_latest_price(self, symbol: str) -> float:
        if self._data_fetcher:
            try:
                df = self._data_fetcher.fetch_bars(symbol, duration="1 D", bar_size="1 min")
                if df is not None and not df.empty:
                    return float(df["close"].iloc[-1])
            except Exception:
                pass
        return 0.0


# ─── TradeStation Adapter ───

class TradeStationAdapter(BaseBrokerAdapter):
    """Real TradeStation adapter using the REST API OrderRouter."""

    def __init__(self, config: Dict[str, str], account_id: str = ""):
        self._config = config
        self._account_id = account_id
        self._router = None
        self._connected = False
        self._name = "tradestation"

    @property
    def name(self) -> str:
        return self._name

    def connect(self) -> bool:
        try:
            from tradestation.api.order_router import TradeStationOrderRouter
            self._router = TradeStationOrderRouter(self._config)

            if not self._account_id:
                # Try to get first account
                try:
                    from tradestation.api.account_monitor import AccountMonitor
                    monitor = AccountMonitor(self._router, {})
                    balances = monitor.get_balances(self._config.get("account_id", ""))
                    self._account_id = self._config.get("account_id", "")
                except Exception:
                    self._account_id = self._config.get("account_id", "")

            self._connected = True
            logger.info("TradeStation connected (account=%s)", self._account_id)
            return True
        except Exception as e:
            logger.error("TradeStation connection failed: %s", e)
            return False

    def disconnect(self) -> None:
        self._connected = False
        logger.info("TradeStation disconnected")

    def is_connected(self) -> bool:
        return self._connected and self._router is not None

    def place_market_order(self, symbol: str, action: str, quantity: int) -> ExecutionResult:
        if not self._router:
            return ExecutionResult(False, self.name, symbol, action, quantity, 0, message="Not connected")
        try:
            order_id = self._router.place_market_order(self._account_id, symbol, action.upper(), quantity)
            return ExecutionResult(True, self.name, symbol, action, quantity, 0, order_id=order_id,
                                   message=f"TS market order: {action} {quantity} {symbol}")
        except Exception as e:
            return ExecutionResult(False, self.name, symbol, action, quantity, 0, message=str(e))

    def get_fill(self, order_id: str) -> Optional[tuple]:
        if not self._router:
            return None
        try:
            data = self._router.get_order_status(self._account_id, order_id) or {}
        except Exception as e:
            logger.debug("TS fill lookup failed for %s: %s", order_id, e)
            return None
        orders = data.get("Orders") or [data]
        for o in orders:
            filled = o.get("FilledQuantity") or o.get("filled_quantity")
            price = (o.get("FilledPrice") or o.get("AverageFilledPrice")
                     or o.get("ExecutionPrice"))
            if filled and price:
                try:
                    return float(price), int(float(filled))
                except (TypeError, ValueError):
                    continue
        return None

    def place_limit_order(self, symbol: str, action: str, quantity: int, price: float) -> ExecutionResult:
        if not self._router:
            return ExecutionResult(False, self.name, symbol, action, quantity, price, message="Not connected")
        try:
            order_id = self._router.place_limit_order(self._account_id, symbol, action.upper(), quantity, price)
            return ExecutionResult(True, self.name, symbol, action, quantity, price, order_id=order_id,
                                   message=f"TS limit order: {action} {quantity} {symbol} @ ${price:.2f}")
        except Exception as e:
            return ExecutionResult(False, self.name, symbol, action, quantity, price, message=str(e))

    def place_stop_order(self, symbol: str, action: str, quantity: int, stop_price: float) -> ExecutionResult:
        """Place a native TradeStation StopMarket order."""
        if not self._router:
            return ExecutionResult(False, self.name, symbol, action, quantity, stop_price, message="Not connected")
        try:
            order_id = self._router.place_stop_order(self._account_id, symbol, action.upper(), quantity, stop_price)
            return ExecutionResult(True, self.name, symbol, action, quantity, stop_price, order_id=order_id,
                                   message=f"TS stop order: {action} {quantity} {symbol} @ ${stop_price:.2f}")
        except Exception as e:
            # See BaseBrokerAdapter.place_stop_order — a limit at stop_price is
            # marketable and would flatten the position it is meant to protect.
            logger.error("TradeStation stop order failed — position UNPROTECTED: %s", e)
            return ExecutionResult(False, self.name, symbol, action, quantity, stop_price,
                                   message=f"stop order failed, no protective order placed: {e}")

    def cancel_order(self, order_id: str) -> bool:
        # FIX 14: Match order_router.cancel_order(order_id) signature (no account_id)
        if self._router:
            try:
                self._router.cancel_order(order_id)
                return True
            except Exception:
                return False
        return False

    def get_positions(self) -> List[Dict[str, Any]]:
        if not self._router:
            return []
        try:
            from tradestation.api.account_monitor import AccountMonitor
            monitor = AccountMonitor(self._router, {})
            return monitor.get_positions(self._account_id)
        except Exception:
            return []

    def get_account_info(self) -> Dict[str, Any]:
        if not self._router:
            return {"broker": self.name, "connected": False}
        try:
            from tradestation.api.account_monitor import AccountMonitor
            monitor = AccountMonitor(self._router, {})
            balances = monitor.get_balances(self._account_id)
            return {"broker": self.name, "connected": True, **balances}
        except Exception:
            return {"broker": self.name, "connected": True}

    def get_latest_price(self, symbol: str) -> float:
        if not self._router:
            return 0.0
        try:
            quote = self._router.get_quote(symbol)
            # TradeStation v3 quote fields: Last, Ask, Bid, Close
            for field_name in ("Last", "Ask", "Bid", "Close"):
                val = quote.get(field_name)
                if val is not None:
                    price = float(val)
                    if price > 0:
                        return price
        except Exception as e:
            logger.warning("TradeStation get_latest_price(%s) failed: %s", symbol, e)
        return 0.0


# ─── Schwab/thinkorswim Adapter ───

class SchwabAdapter(BaseBrokerAdapter):
    """Schwab API adapter for thinkorswim."""

    def __init__(self, config: Dict[str, str]):
        self._config = config
        self._client = None
        self._connected = False
        self._name = "schwab"

    @property
    def name(self) -> str:
        return self._name

    def connect(self) -> bool:
        try:
            from thinkorswim.api.schwab_client import SchwabClient
            self._client = SchwabClient(self._config)
            self._connected = True
            logger.info("Schwab connected")
            return True
        except Exception as e:
            logger.error("Schwab connection failed: %s", e)
            return False

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def place_market_order(self, symbol: str, action: str, quantity: int) -> ExecutionResult:
        if not self._client:
            return ExecutionResult(False, self.name, symbol, action, quantity, 0, message="Not connected")
        try:
            order_id = self._client.place_market_order(symbol, action.upper(), quantity)
            return ExecutionResult(True, self.name, symbol, action, quantity, 0, order_id=order_id)
        except Exception as e:
            return ExecutionResult(False, self.name, symbol, action, quantity, 0, message=str(e))

    def get_fill(self, order_id: str) -> Optional[tuple]:
        if not self._client:
            return None
        try:
            orders = self._client.get_orders(status="FILLED") or []
        except Exception as e:
            logger.debug("Schwab fill lookup failed for %s: %s", order_id, e)
            return None
        for o in orders:
            if str(o.get("orderId")) != str(order_id):
                continue
            filled = o.get("filledQuantity")
            legs = o.get("orderActivityCollection") or []
            prices = [
                float(leg["executionPrice"])
                for act in legs
                for leg in (act.get("executionLegs") or [])
                if leg.get("executionPrice")
            ]
            if filled and prices:
                return sum(prices) / len(prices), int(float(filled))
            if filled and o.get("price"):
                return float(o["price"]), int(float(filled))
        return None

    def place_limit_order(self, symbol: str, action: str, quantity: int, price: float) -> ExecutionResult:
        if not self._client:
            return ExecutionResult(False, self.name, symbol, action, quantity, price, message="Not connected")
        try:
            order_id = self._client.place_limit_order(symbol, action.upper(), quantity, price)
            return ExecutionResult(True, self.name, symbol, action, quantity, price, order_id=order_id)
        except Exception as e:
            return ExecutionResult(False, self.name, symbol, action, quantity, price, message=str(e))

    def place_stop_order(self, symbol: str, action: str, quantity: int, stop_price: float) -> ExecutionResult:
        """Place a native Schwab STOP order with stopPrice."""
        if not self._client:
            return ExecutionResult(False, self.name, symbol, action, quantity, stop_price, message="Not connected")
        try:
            order_id = self._client.place_stop_order(symbol, action.upper(), quantity, stop_price)
            return ExecutionResult(True, self.name, symbol, action, quantity, stop_price, order_id=order_id,
                                   message=f"Schwab stop order: {action} {quantity} {symbol} @ ${stop_price:.2f}")
        except Exception as e:
            # See BaseBrokerAdapter.place_stop_order — a limit at stop_price is
            # marketable and would flatten the position it is meant to protect.
            logger.error("Schwab stop order failed — position UNPROTECTED: %s", e)
            return ExecutionResult(False, self.name, symbol, action, quantity, stop_price,
                                   message=f"stop order failed, no protective order placed: {e}")

    def cancel_order(self, order_id: str) -> bool:
        if self._client:
            try:
                self._client.cancel_order(order_id)
                return True
            except Exception:
                return False
        return False

    def get_positions(self) -> List[Dict[str, Any]]:
        if self._client:
            try:
                return self._client.get_positions()
            except Exception:
                return []
        return []

    def get_account_info(self) -> Dict[str, Any]:
        if self._client:
            try:
                return {"broker": self.name, "connected": True, **self._client.get_account_info()}
            except Exception:
                pass
        return {"broker": self.name, "connected": False}

    def get_latest_price(self, symbol: str) -> float:
        if self._client:
            try:
                quote = self._client.get_quote(symbol)
                return quote.get("lastPrice", 0.0)
            except Exception:
                pass
        return 0.0


# ─── Broker Bridge (Main Entry Point) ───

class BrokerBridge:
    """Unified bridge connecting AI agent decisions to any broker.

    Handles:
    - Broker selection and connection
    - Translating BUY/SELL/HOLD → market/limit orders
    - Auto TP/SL placement after every trade
    - Position tracking with entry prices
    - Position reconciliation with broker state
    - Force-close on max loss threshold
    - P&L calculation on exit
    - JSONL trade diary for audit trail
    - Auto-recording outcomes to SelfLearningAgent

    Args:
        broker: Broker name — "ib", "tradestation", "schwab", "paper"
        config: Broker-specific config dict.
        mode: "paper" (IB port 7497) or "live" (IB port 7496).
        max_position_pct: Max % of capital per position.
        max_shares: Maximum shares per order.
        max_loss_pct: Force-close if position loses more than this %.
        default_tp_pct: Default take-profit distance as % of entry.
        default_sl_pct: Default stop-loss distance as % of entry.
        diary_path: Path to JSONL trade diary file.
        risk_manager: Optional shared RiskManager. When supplied, position
            sizing goes through its per-trade risk rules instead of the flat
            ``max_position_pct`` fallback, and a size of 0 blocks the order.
    """

    BROKERS = {
        "ib": IBAdapter,
        "interactive_brokers": IBAdapter,
        "tradestation": TradeStationAdapter,
        "ts": TradeStationAdapter,
        "schwab": SchwabAdapter,
        "thinkorswim": SchwabAdapter,
        "tos": SchwabAdapter,
    }

    def __init__(
        self,
        broker: str = "ib",
        config: Optional[Dict[str, Any]] = None,
        mode: str = "paper",
        max_position_pct: float = 0.10,
        max_shares: int = 500,
        capital: float = 100_000.0,
        max_loss_pct: float = 5.0,
        default_tp_pct: float = 3.0,
        default_sl_pct: float = 2.0,
        diary_path: Optional[str] = None,
        commission: Optional[CommissionModel] = None,
        max_holding_bars: int = 240,
        risk_manager: Optional[Any] = None,
    ) -> None:
        self._broker_name = broker.lower()
        self._config = config or {}
        self._mode = mode
        self._max_position_pct = max_position_pct
        self._max_shares = max_shares
        self._capital = capital
        self._max_loss_pct = max_loss_pct
        self._default_tp_pct = default_tp_pct
        self._default_sl_pct = default_sl_pct
        self._commission = commission or CommissionModel()
        self._max_holding_bars = max_holding_bars

        # Diary path
        if diary_path is None:
            # NB: no local `import os` here — os is imported at module scope,
            # and a function-local import makes the name local to the WHOLE
            # function, so any earlier or later use of os in __init__ raises
            # UnboundLocalError.
            diary_dir = os.path.join(os.path.expanduser("~"), ".stocks_plugin", "logs")
            os.makedirs(diary_dir, exist_ok=True)
            self._diary_path = os.path.join(diary_dir, "trade_diary.jsonl")
        else:
            self._diary_path = diary_path

        # Position tracking
        self._positions: Dict[str, Position] = {}
        # FIX 2: Threading lock for position state
        self._position_lock = threading.Lock()

        # OCO pair tracking: maps order_id -> paired order_id
        self._oco_pairs: Dict[str, str] = {}

        # Trailing stops
        self._trailing_stops: Dict[str, TrailingStop] = {}

        # Optional RiskManager. There was no way to supply one: the attribute
        # was set to None here and never assigned anywhere in the repository,
        # so _calculate_shares always fell through to a flat 10% of a static
        # $100k and the per-trade risk limit was never applied on a live path.
        self._risk_manager: Optional[Any] = risk_manager

        # Unified portfolio risk gate (cross-strategy coordination).
        #
        # This was get_instance() with no config at all, which meant the only
        # risk gate on this order path had: no persistence (a crash mid-day
        # zeroed the daily P&L and every tracked position), a hard-coded
        # $100,000 equity that ignored the capital= argument above, and a
        # $10,000 daily loss limit rather than the documented $5,000. Build a
        # real config from what the caller actually passed.
        risk_cfg = self._config.get("portfolio_risk", {})
        default_persist = os.path.join(
            stocks_plugin_data_dir(), "unified_risk.db",
        )
        self._portfolio_gate = UnifiedPortfolioRiskGate.get_instance(
            UnifiedRiskConfig(
                account_equity=capital,
                max_daily_loss=risk_cfg.get("max_daily_loss", 5_000.0),
                max_portfolio_exposure=risk_cfg.get("max_portfolio_exposure", 0.80),
                max_single_stock_pct=risk_cfg.get("max_single_stock_pct", 0.15),
                max_sector_pct=risk_cfg.get("max_sector_pct", 0.30),
                max_correlated_exposure=risk_cfg.get("max_correlated_exposure", 0.40),
                persist_path=risk_cfg.get("persist_path", default_persist),
            )
        )

        # Create adapter
        self._adapter = self._create_adapter()

        logger.info(
            "BrokerBridge created: broker=%s, mode=%s, max_shares=%d, sl=%.1f%%, tp=%.1f%%",
            self._broker_name, mode, max_shares, default_sl_pct, default_tp_pct,
        )

    def _create_adapter(self) -> BaseBrokerAdapter:
        """Create the appropriate broker adapter."""
        if self._broker_name in ("ib", "interactive_brokers"):
            port = 7497 if self._mode == "paper" else 7496
            return IBAdapter(
                host=self._config.get("host", "127.0.0.1"),
                port=self._config.get("port", port),
                client_id=self._config.get("client_id", 1),
            )
        elif self._broker_name in ("tradestation", "ts"):
            return TradeStationAdapter(self._config, self._config.get("account_id", ""))
        elif self._broker_name in ("schwab", "thinkorswim", "tos"):
            return SchwabAdapter(self._config)
        else:
            raise ValueError(f"Unknown broker: {self._broker_name}. Options: {list(self.BROKERS.keys())}")

    def connect(self) -> bool:
        """Connect to the broker."""
        return self._adapter.connect()

    def disconnect(self) -> None:
        """Disconnect from the broker."""
        self._adapter.disconnect()

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._adapter.is_connected()

    # ─── Execute AI Decisions ───

    def execute_decision(
        self,
        decision: Dict[str, Any],
        symbol: str,
        agent: Optional[Any] = None,
    ) -> Optional[ExecutionResult]:
        """Execute an AI agent decision via the connected broker.

        Args:
            decision: Output from SelfLearningAgent.decide() or LLMReasoner.reason()
            symbol: Ticker symbol.
            agent: Optional SelfLearningAgent for auto-recording outcomes.

        Returns:
            ExecutionResult or None if HOLD.
        """
        action = decision.get("action", "HOLD")
        confidence = decision.get("confidence", 0)
        price = decision.get("price", 0)
        tp_price = decision.get("tp_price")
        sl_price = decision.get("sl_price")
        exit_plan = decision.get("exit_plan", "")

        # Write every decision to diary
        self._write_diary({
            "action": action,
            "symbol": symbol,
            "price": price,
            "confidence": confidence,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "exit_plan": exit_plan,
            "regime": decision.get("regime", ""),
            "reasoning": decision.get("reasoning", "")[:500] if decision.get("reasoning") else "",
        })

        if action == "HOLD":
            logger.debug("Decision: HOLD %s (confidence=%.2f)", symbol, confidence)
            return None

        with self._position_lock:
            has_position = symbol in self._positions
            current_pos = self._positions.get(symbol)

        if action == "BUY":
            if has_position and current_pos.direction == "long":
                logger.debug("Already long %s, skipping BUY", symbol)
                return None

            if has_position and current_pos.direction == "short":
                self._close_position(symbol, price, agent)

            shares = self._calculate_shares(price)
            if shares <= 0:
                logger.warning("Sizing returned 0 shares for %s — order refused", symbol)
                return ExecutionResult(
                    False, self._adapter.name, symbol, action, 0, price,
                    message="position sizing returned 0 shares",
                )
            notional = shares * price if price > 0 else 0

            # reserve() checks the limits AND books the notional under one
            # lock. can_open_position() released the lock before the order went
            # out, so concurrent callers all saw the same headroom — 8
            # concurrent $14k orders were measured landing at 112% of equity
            # against an 80% cap.
            gate_ok, gate_reason = self._portfolio_gate.reserve(symbol, notional)
            if not gate_ok:
                logger.warning(
                    "Portfolio risk gate BLOCKED BUY %s ($%.0f): %s",
                    symbol, notional, gate_reason,
                )
                return ExecutionResult(
                    False, self._adapter.name, symbol, action, shares, price,
                    message=f"Portfolio risk gate blocked: {gate_reason}",
                )

            try:
                result = self._adapter.place_market_order(symbol, "BUY", shares)
            except Exception:
                self._portfolio_gate.release_reservation(symbol, notional)
                raise
            if not result.success:
                self._portfolio_gate.release_reservation(symbol, notional)

            if result.success:
                actual_price, filled = self._resolve_fill(result, price, shares)
                if filled == 0:
                    logger.error("Order filled 0 shares for %s — not opening position", symbol)
                    # Nothing was opened, so the held headroom must go back.
                    self._portfolio_gate.release_reservation(symbol, notional)
                    return result
                if filled < shares:
                    logger.warning("Partial fill: requested %d, filled %d for %s", shares, filled, symbol)

                with self._position_lock:
                    self._positions[symbol] = Position(
                        symbol=symbol, direction="long", shares=filled,
                        entry_price=actual_price, entry_time=datetime.now().isoformat(),
                        order_id=result.order_id,
                        max_holding_bars=self._max_holding_bars,
                    )
                logger.info("📈 OPENED LONG: %d shares %s @ $%.2f [%s]",
                           filled, symbol, actual_price, self._adapter.name)

                self._portfolio_gate.commit_reservation(
                    symbol, notional, filled, actual_price, "agent", self._adapter.name,
                )

                self._place_tp_sl(symbol, actual_price, "long", tp_price, sl_price, filled)

                self._write_diary({
                    "action": "OPEN_LONG",
                    "symbol": symbol,
                    "shares": filled,
                    "entry_price": actual_price,
                    "decision_price": price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "order_id": result.order_id,
                    "exit_plan": exit_plan,
                })
            return result

        elif action == "SELL":
            if has_position and current_pos.direction == "long":
                return self._close_position(symbol, price, agent)

            if has_position and current_pos.direction == "short":
                logger.debug("Already short %s, skipping SELL", symbol)
                return None

            shares = self._calculate_shares(price)
            if shares <= 0:
                logger.warning("Sizing returned 0 shares for %s — order refused", symbol)
                return ExecutionResult(
                    False, self._adapter.name, symbol, action, 0, price,
                    message="position sizing returned 0 shares",
                )
            notional = shares * price if price > 0 else 0

            # reserve() checks the limits AND books the notional under one
            # lock. can_open_position() released the lock before the order went
            # out, so concurrent callers all saw the same headroom — 8
            # concurrent $14k orders were measured landing at 112% of equity
            # against an 80% cap.
            gate_ok, gate_reason = self._portfolio_gate.reserve(symbol, notional)
            if not gate_ok:
                logger.warning(
                    "Portfolio risk gate BLOCKED SELL %s ($%.0f): %s",
                    symbol, notional, gate_reason,
                )
                return ExecutionResult(
                    False, self._adapter.name, symbol, action, shares, price,
                    message=f"Portfolio risk gate blocked: {gate_reason}",
                )

            try:
                result = self._adapter.place_market_order(symbol, "SELL", shares)
            except Exception:
                self._portfolio_gate.release_reservation(symbol, notional)
                raise
            if not result.success:
                self._portfolio_gate.release_reservation(symbol, notional)

            if result.success:
                actual_price, filled = self._resolve_fill(result, price, shares)
                if filled == 0:
                    logger.error("Order filled 0 shares for %s — not opening position", symbol)
                    # Nothing was opened, so the held headroom must go back.
                    self._portfolio_gate.release_reservation(symbol, notional)
                    return result
                if filled < shares:
                    logger.warning("Partial fill: requested %d, filled %d for %s", shares, filled, symbol)

                with self._position_lock:
                    self._positions[symbol] = Position(
                        symbol=symbol, direction="short", shares=filled,
                        entry_price=actual_price, entry_time=datetime.now().isoformat(),
                        order_id=result.order_id,
                        max_holding_bars=self._max_holding_bars,
                    )
                logger.info("📉 OPENED SHORT: %d shares %s @ $%.2f [%s]",
                           filled, symbol, actual_price, self._adapter.name)

                self._portfolio_gate.commit_reservation(
                    symbol, notional, filled, actual_price, "agent", self._adapter.name,
                )

                self._place_tp_sl(symbol, actual_price, "short", tp_price, sl_price, filled)

                self._write_diary({
                    "action": "OPEN_SHORT",
                    "symbol": symbol,
                    "shares": filled,
                    "entry_price": actual_price,
                    "decision_price": price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "order_id": result.order_id,
                    "exit_plan": exit_plan,
                })
            return result

        return None

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        agent: Optional[Any] = None,
    ) -> ExecutionResult:
        """Close an existing position and record outcome."""
        with self._position_lock:
            pos = self._positions.get(symbol)
        if not pos:
            return ExecutionResult(False, self._adapter.name, symbol, "CLOSE", 0, exit_price,
                                   message="No position to close")

        close_action = "SELL" if pos.direction == "long" else "BUY"
        result = self._adapter.place_market_order(symbol, close_action, pos.shares)

        if result.success:
            actual_exit, _ = self._resolve_fill(result, exit_price, pos.shares)

            # Calculate P&L
            if pos.direction == "long":
                raw_pnl = (actual_exit - pos.entry_price) * pos.shares
            else:
                raw_pnl = (pos.entry_price - actual_exit) * pos.shares

            # Fix 4: deduct commissions from both entry and exit
            entry_comm = self._commission.calculate_commission(pos.shares, pos.entry_price)
            exit_comm = self._commission.calculate_commission(pos.shares, actual_exit)
            pnl = raw_pnl - entry_comm - exit_comm

            emoji = "✅" if pnl > 0 else "❌"
            logger.info(
                "%s CLOSED %s: %d shares %s | entry=$%.2f exit=$%.2f | "
                "raw=$%.2f comm=$%.2f net=$%.2f",
                emoji, pos.direction.upper(), pos.shares, symbol,
                pos.entry_price, actual_exit, raw_pnl, entry_comm + exit_comm, pnl,
            )

            # Record outcome in agent memory
            if agent:
                try:
                    agent.record_outcome(
                        exit_price=actual_exit,
                        pnl=pnl,
                        holding_period_bars=pos.bars_held or 1,
                    )
                except Exception as e:
                    logger.warning("Failed to record outcome: %s", e)

            with self._position_lock:
                self._positions.pop(symbol, None)

            self._portfolio_gate.close_position(symbol, pnl)

        return result

    def _resolve_fill(self, result: ExecutionResult, fallback_price: float,
                      requested_shares: int) -> tuple:
        """Return the real (price, quantity) for a placed order.

        ``getattr(result, 'fill_price', None) or price`` always resolved to
        the decision price because no adapter ever populated fill_price, so
        every downstream P&L was computed against a price the trade never got.
        Ask the adapter; fall back only when the broker cannot tell us.
        """
        price = getattr(result, "fill_price", None)
        qty = getattr(result, "filled_quantity", None)

        if price is None or qty is None:
            try:
                fill = self._adapter.get_fill(result.order_id)
            except Exception as e:
                logger.debug("Fill lookup failed for %s: %s", result.order_id, e)
                fill = None
            # Defensive: this runs AFTER the order is live, so a malformed
            # adapter response must degrade to the fallback, never raise.
            fetched_price = fetched_qty = None
            try:
                if fill is not None:
                    fetched_price, fetched_qty = fill
                    fetched_price = float(fetched_price)
                    fetched_qty = int(fetched_qty)
            except (TypeError, ValueError) as e:
                logger.warning(
                    "Adapter returned an unusable fill for %s: %r (%s)",
                    result.order_id or "?", fill, e,
                )
                fetched_price = fetched_qty = None

            if fetched_price is not None and price is None:
                price = fetched_price
                result.fill_price = price
            if fetched_qty is not None and qty is None:
                qty = fetched_qty
                result.filled_quantity = qty

        if price is None or price <= 0:
            logger.warning(
                "No fill price available for %s order %s — using the decision "
                "price $%.2f; realized P&L will not include slippage",
                result.symbol, result.order_id or "?", fallback_price,
            )
            price = fallback_price
        try:
            qty = requested_shares if qty is None else int(qty)
        except (TypeError, ValueError):
            qty = requested_shares
        return float(price), qty

    def _calculate_shares(self, price: float, stop_price: Optional[float] = None) -> int:
        """Calculate position size based on capital and limits.

        If a RiskManager is attached, delegates to its position sizing logic.
        """
        if price <= 0:
            return 0

        if self._risk_manager is not None:
            try:
                size_result = self._risk_manager.calculate_position_size(
                    symbol="",
                    entry_price=price,
                    stop_price=stop_price,
                )
                rm_shares = size_result if isinstance(size_result, int) else getattr(size_result, 'shares', size_result)
                # A RiskManager returning 0 is a rejection — the position does
                # not fit inside its caps. Falling through to the default
                # sizing here would silently overrule the risk manager.
                return min(int(rm_shares), self._max_shares)
            except Exception as e:
                # Fail closed: a sizing error must not quietly become a
                # full-size default position.
                logger.error("RiskManager sizing failed, refusing to size the order: %s", e)
                return 0

        dollar_amount = self._capital * self._max_position_pct
        shares = int(dollar_amount / price)
        return min(shares, self._max_shares)

    # ─── Position & Account Info ───

    def get_positions(self) -> Dict[str, Position]:
        """Get locally tracked positions."""
        with self._position_lock:
            return dict(self._positions)

    def get_broker_positions(self) -> List[Dict[str, Any]]:
        """Get positions directly from the broker."""
        return self._adapter.get_positions()

    def get_account_info(self) -> Dict[str, Any]:
        """Get account info from the broker."""
        return self._adapter.get_account_info()

    def get_latest_price(self, symbol: str) -> float:
        """Get latest price from the broker.

        .. warning::
            Returns 0.0 when the price cannot be fetched.  Callers MUST
            check ``price > 0`` before using the value in order sizing,
            P&L calculations, or order placement to avoid $0 trades.
        """
        price = self._adapter.get_latest_price(symbol)
        if price <= 0:
            logger.warning("get_latest_price(%s) returned 0.0 — callers must guard", symbol)
        return price

    def close_all_positions(self, agent: Optional[Any] = None) -> List[ExecutionResult]:
        """Close all open positions (emergency flatten)."""
        results = []
        with self._position_lock:
            symbols = list(self._positions.keys())
        for symbol in symbols:
            price = self.get_latest_price(symbol)
            if price <= 0:
                with self._position_lock:
                    pos = self._positions.get(symbol)
                price = pos.entry_price if pos else 0
            result = self._close_position(symbol, price, agent)
            results.append(result)
        return results

    # ─── OCO (One-Cancels-Other) Management ───

    def _cancel_paired_order(self, filled_order_id: str) -> None:
        """Cancel the OCO partner when one side fills."""
        paired_id = self._oco_pairs.pop(filled_order_id, None)
        if paired_id:
            self._oco_pairs.pop(paired_id, None)
            try:
                cancelled = self._adapter.cancel_order(paired_id)
                if cancelled:
                    logger.info("  OCO: cancelled paired order %s (filled=%s)", paired_id, filled_order_id)
                else:
                    logger.warning("  OCO: failed to cancel paired order %s", paired_id)
            except Exception as e:
                logger.warning("  OCO cancel error for %s: %s", paired_id, e)

    def on_fill(self, order_id: str, symbol: str, fill_price: float) -> None:
        """Callback when an order fills. Cancels the OCO partner and closes the position."""
        logger.info("Fill received: order=%s, symbol=%s, price=%.2f", order_id, symbol, fill_price)
        self._cancel_paired_order(order_id)

        with self._position_lock:
            pos = self._positions.get(symbol)
        if pos:
            if pos.direction == "long":
                raw_pnl = (fill_price - pos.entry_price) * pos.shares
            else:
                raw_pnl = (pos.entry_price - fill_price) * pos.shares

            # Match _close_position: commissions on both legs, so the P&L that
            # reaches the portfolio gate is the same number either exit path
            # would produce.
            entry_comm = self._commission.calculate_commission(pos.shares, pos.entry_price)
            exit_comm = self._commission.calculate_commission(pos.shares, fill_price)
            pnl = raw_pnl - entry_comm - exit_comm

            emoji = "✅" if pnl > 0 else "❌"
            logger.info("%s TP/SL fill: %s %s | P&L=$%.2f", emoji, pos.direction.upper(), symbol, pnl)
            with self._position_lock:
                self._positions.pop(symbol, None)
            self._trailing_stops.pop(symbol, None)

            # A TP/SL fill is the normal way a position closes. Without this the
            # gate's exposure only ever grew (eventually blocking all trading)
            # and realized P&L never reached the daily-loss limit.
            self._portfolio_gate.close_position(symbol, pnl)

    # ─── TP/SL Auto-Management (from hyperliquid-trading-agent) ───

    def _place_tp_sl(
        self,
        symbol: str,
        entry_price: float,
        direction: str,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        shares: Optional[int] = None,
    ) -> None:
        """Place take-profit and stop-loss orders after opening a position.

        TP is placed as a limit order, SL is placed as a stop order.
        Both are tracked as an OCO pair so filling one cancels the other.
        Uses provided shares (for partial fills) or falls back to position shares.
        """
        if tp_price is None:
            if direction == "long":
                tp_price = round(entry_price * (1 + self._default_tp_pct / 100), 2)
            else:
                tp_price = round(entry_price * (1 - self._default_tp_pct / 100), 2)

        if sl_price is None:
            if direction == "long":
                sl_price = round(entry_price * (1 - self._default_sl_pct / 100), 2)
            else:
                sl_price = round(entry_price * (1 + self._default_sl_pct / 100), 2)

        with self._position_lock:
            pos = self._positions.get(symbol)
        if pos:
            order_shares = shares or pos.shares
            close_action = "SELL" if direction == "long" else "BUY"
            tp_order_id = None
            sl_order_id = None

            # Place TP as limit order
            try:
                tp_result = self._adapter.place_limit_order(symbol, close_action, order_shares, tp_price)
                if tp_result.success:
                    tp_order_id = tp_result.order_id
                    logger.info("  TP placed: %s %s @ $%.2f [order=%s]", symbol, close_action, tp_price, tp_order_id)
            except Exception as e:
                logger.warning("  TP order failed: %s", e)

            # Place SL as stop order (NOT limit order)
            try:
                sl_result = self._adapter.place_stop_order(symbol, close_action, order_shares, sl_price)
                if sl_result.success:
                    sl_order_id = sl_result.order_id
                    logger.info("  SL placed (STOP): %s %s @ $%.2f [order=%s]", symbol, close_action, sl_price, sl_order_id)
                else:
                    logger.error("  SL stop order rejected: %s", sl_result.message)
            except Exception as e:
                logger.error("  SL stop order failed: %s", e)

            # Track OCO pair
            if tp_order_id and sl_order_id:
                self._oco_pairs[tp_order_id] = sl_order_id
                self._oco_pairs[sl_order_id] = tp_order_id
                logger.info("  OCO pair linked: TP=%s <-> SL=%s", tp_order_id, sl_order_id)

            # Report what was actually placed, not what was intended. An open
            # position with no stop is the single most dangerous state here, so
            # it escalates rather than being buried in an info line.
            if sl_order_id:
                logger.info(
                    "  Auto TP/SL for %s: TP=$%.2f (+%.1f%%), SL=$%.2f (-%.1f%%)",
                    symbol, tp_price, self._default_tp_pct, sl_price, self._default_sl_pct,
                )
            else:
                logger.critical(
                    "  %s is OPEN WITHOUT A STOP LOSS (%d shares, entry $%.2f, intended SL $%.2f) — "
                    "manual intervention required",
                    symbol, order_shares, entry_price, sl_price,
                )
                self._notify_unprotected_position(symbol, order_shares, entry_price, sl_price)

    def _notify_unprotected_position(
        self, symbol: str, shares: int, entry_price: float, intended_sl: float,
    ) -> None:
        """Escalate an open position that has no stop-loss order behind it."""
        try:
            from shared.notifier import AlertDispatcher
            notify_config = self._config.get("notifications", {})
            AlertDispatcher(notify_config).dispatch(
                title=f"UNPROTECTED POSITION: {symbol}",
                message=(
                    f"{shares} shares of {symbol} are open at ${entry_price:.2f} with no "
                    f"stop-loss order. The intended stop was ${intended_sl:.2f}. "
                    f"Close the position or place the stop manually."
                ),
                priority="CRITICAL",
            )
        except Exception as e:
            # The log line above already recorded the condition at CRITICAL.
            logger.error("Could not dispatch unprotected-position alert: %s", e)

    # ─── Trailing Stops ───

    def set_trailing_stop(
        self,
        symbol: str,
        direction: str,
        activation_pct: float = 0.02,
        trail_pct: float = 0.015,
    ) -> None:
        """Configure a trailing stop for a position.

        Args:
            symbol: Ticker symbol.
            direction: "long" or "short".
            activation_pct: Activate after this % profit (e.g. 0.02 = 2%).
            trail_pct: Trail by this % from peak/trough (e.g. 0.015 = 1.5%).
        """
        pos = self._positions.get(symbol)
        if not pos:
            logger.warning("Cannot set trailing stop: no position for %s", symbol)
            return

        self._trailing_stops[symbol] = TrailingStop(
            symbol=symbol,
            direction=direction,
            activation_pct=activation_pct,
            trail_pct=trail_pct,
            highest_price=pos.entry_price,
            lowest_price=pos.entry_price,
        )
        logger.info(
            "Trailing stop set: %s %s | activate=%.1f%%, trail=%.1f%%",
            symbol, direction, activation_pct * 100, trail_pct * 100,
        )

    def _update_trailing_stops(self, symbol: str, current_price: float) -> bool:
        """Update trailing stop for a symbol. Returns True if stop triggered.

        For longs: tracks highest price, sets stop at highest*(1-trail_pct).
        For shorts: tracks lowest price, sets stop at lowest*(1+trail_pct).
        """
        ts = self._trailing_stops.get(symbol)
        if not ts:
            return False

        pos = self._positions.get(symbol)
        if not pos:
            self._trailing_stops.pop(symbol, None)
            return False

        entry = pos.entry_price

        if ts.direction == "long":
            ts.highest_price = max(ts.highest_price, current_price)

            if not ts.activated:
                if current_price >= entry * (1 + ts.activation_pct):
                    ts.activated = True
                    ts.stop_price = ts.highest_price * (1 - ts.trail_pct)
                    logger.info(
                        "Trailing stop ACTIVATED: %s long | stop=$%.2f",
                        symbol, ts.stop_price,
                    )
            else:
                ts.stop_price = ts.highest_price * (1 - ts.trail_pct)
                if current_price <= ts.stop_price:
                    logger.info(
                        "⚠️ TRAILING STOP TRIGGERED: %s long | price=$%.2f <= stop=$%.2f",
                        symbol, current_price, ts.stop_price,
                    )
                    return True

        elif ts.direction == "short":
            ts.lowest_price = min(ts.lowest_price, current_price)

            if not ts.activated:
                if current_price <= entry * (1 - ts.activation_pct):
                    ts.activated = True
                    ts.stop_price = ts.lowest_price * (1 + ts.trail_pct)
                    logger.info(
                        "Trailing stop ACTIVATED: %s short | stop=$%.2f",
                        symbol, ts.stop_price,
                    )
            else:
                ts.stop_price = ts.lowest_price * (1 + ts.trail_pct)
                if current_price >= ts.stop_price:
                    logger.info(
                        "⚠️ TRAILING STOP TRIGGERED: %s short | price=$%.2f >= stop=$%.2f",
                        symbol, current_price, ts.stop_price,
                    )
                    return True

        return False

    # ─── Force-Close on Max Loss (from hyperliquid-trading-agent) ───

    def check_and_force_close(self, agent: Optional[Any] = None) -> List[ExecutionResult]:
        """Check all positions for max loss / trailing stops and force-close.

        Returns:
            List of ExecutionResults for force-closed positions.
        """
        results = []
        with self._position_lock:
            positions_snapshot = list(self._positions.items())
        for symbol, pos in positions_snapshot:
            current_price = self.get_latest_price(symbol)
            if current_price <= 0:
                continue

            # FIX 8: Increment bars held and check max holding period
            pos.bars_held += 1
            if pos.max_holding_bars > 0 and pos.bars_held > pos.max_holding_bars:
                logger.warning(
                    "⚠️ MAX HOLDING PERIOD: %s %s held %d bars (max=%d). Force closing.",
                    pos.direction.upper(), symbol, pos.bars_held, pos.max_holding_bars,
                )
                result = self._close_position(symbol, current_price, agent)
                if result.success:
                    self._trailing_stops.pop(symbol, None)
                    self._write_diary({
                        "action": "MAX_HOLDING_CLOSE",
                        "symbol": symbol,
                        "direction": pos.direction,
                        "bars_held": pos.bars_held,
                        "max_holding_bars": pos.max_holding_bars,
                        "entry_price": pos.entry_price,
                        "exit_price": current_price,
                    })
                results.append(result)
                continue

            # Check trailing stops first
            if self._update_trailing_stops(symbol, current_price):
                logger.warning(
                    "⚠️ TRAILING STOP CLOSE: %s %s @ $%.2f",
                    pos.direction.upper(), symbol, current_price,
                )
                result = self._close_position(symbol, current_price, agent)
                if result.success:
                    self._trailing_stops.pop(symbol, None)
                    self._write_diary({
                        "action": "TRAILING_STOP_CLOSE",
                        "symbol": symbol,
                        "direction": pos.direction,
                        "entry_price": pos.entry_price,
                        "exit_price": current_price,
                    })
                results.append(result)
                continue

            # Calculate loss %
            if pos.direction == "long":
                loss_pct = (pos.entry_price - current_price) / pos.entry_price * 100
            else:
                loss_pct = (current_price - pos.entry_price) / pos.entry_price * 100

            if loss_pct >= self._max_loss_pct:
                logger.warning(
                    "⚠️ FORCE-CLOSE: %s %s at %.2f%% loss (max=%.1f%%)",
                    pos.direction.upper(), symbol, loss_pct, self._max_loss_pct,
                )
                result = self._close_position(symbol, current_price, agent)
                if result.success:
                    self._write_diary({
                        "action": "FORCE_CLOSE",
                        "symbol": symbol,
                        "loss_pct": round(loss_pct, 2),
                        "reason": f"Exceeded max loss threshold of {self._max_loss_pct}%",
                        "entry_price": pos.entry_price,
                        "exit_price": current_price,
                    })
                results.append(result)

        return results

    # ─── Position Reconciliation (from hyperliquid-trading-agent) ───

    def reconcile_positions(self) -> Dict[str, Any]:
        """Sync local position tracking with actual broker positions.

        Detects:
        - Stale local positions (no longer on broker)
        - Missing local positions (on broker but not tracked)

        Returns:
            Dict with reconciliation details.
        """
        broker_positions = self._adapter.get_positions()
        broker_symbols = set()
        reconciliation = {"removed": [], "added": [], "matched": 0}

        for bp in broker_positions:
            sym = bp.get("symbol", "")
            qty = float(bp.get("quantity", 0))
            if abs(qty) > 0:
                broker_symbols.add(sym)

        # Remove stale local positions
        with self._position_lock:
            for sym in list(self._positions.keys()):
                if sym not in broker_symbols:
                    logger.info("Reconcile: removing stale position %s (not on broker)", sym)
                    reconciliation["removed"].append(sym)
                    del self._positions[sym]

        # Release the gate's exposure for anything we just dropped. Without
        # this, reconciliation leaked notional into _total_exposure until the
        # gate refused every new position. P&L is unknown for a position that
        # vanished from the broker, so record 0 rather than inventing a number.
        for sym in reconciliation["removed"]:
            try:
                self._portfolio_gate.close_position(sym, 0.0)
            except Exception as e:
                logger.warning("Reconcile: could not release gate exposure for %s: %s", sym, e)

        # FIX 6: Adopt broker positions not tracked locally
        for bp in broker_positions:
            sym = bp.get("symbol", "")
            qty = float(bp.get("quantity", 0))
            avg_cost = float(bp.get("avg_cost", bp.get("averagePrice", bp.get("avg_price", 0))))
            if abs(qty) > 0 and sym not in self._positions:
                direction = "long" if qty > 0 else "short"
                with self._position_lock:
                    self._positions[sym] = Position(
                        symbol=sym,
                        direction=direction,
                        shares=int(abs(qty)),
                        entry_price=avg_cost,
                        entry_time=datetime.now().isoformat(),
                    )
                notional = abs(qty * avg_cost)
                self._portfolio_gate.register_position(
                    sym, int(abs(qty)), avg_cost, "adopted", self._adapter.name,
                )
                reconciliation["added"].append(sym)
                logger.info(
                    "Adopted broker position: %s %d shares @ $%.2f",
                    sym, int(abs(qty)), avg_cost,
                )
            elif abs(qty) > 0 and sym in self._positions:
                reconciliation["matched"] += 1

        if reconciliation["removed"] or reconciliation["added"]:
            logger.info("Reconciliation: %s", reconciliation)

        return reconciliation

    # ─── Trade Diary — JSONL Logging (from hyperliquid-trading-agent) ───

    # Fix 3: diary file rotation constants
    _DIARY_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    _DIARY_MAX_ROTATED = 5

    def _rotate_diary(self) -> None:
        """Rotate diary file when it exceeds 10 MB. Keeps max 5 rotated files."""
        # Remove oldest rotated files beyond the keep limit
        for i in range(self._DIARY_MAX_ROTATED, 0, -1):
            old = f"{self._diary_path}.{i}.jsonl"
            if i == self._DIARY_MAX_ROTATED and os.path.exists(old):
                os.remove(old)
            elif os.path.exists(old):
                os.rename(old, f"{self._diary_path}.{i + 1}.jsonl")

        # Rotate current file to .1.jsonl
        if os.path.exists(self._diary_path):
            os.rename(self._diary_path, f"{self._diary_path}.1.jsonl")
            logger.info("Diary rotated: %s (exceeded %d MB)",
                        self._diary_path, self._DIARY_MAX_SIZE // (1024 * 1024))

    def _write_diary(self, entry: Dict[str, Any]) -> None:
        """Write an entry to the JSONL trade diary with automatic rotation."""
        import json as _json
        entry["timestamp"] = datetime.now().isoformat()
        entry["broker"] = self._adapter.name
        entry["mode"] = self._mode

        try:
            # Fix 3: rotate if file exceeds max size
            if os.path.exists(self._diary_path):
                if os.path.getsize(self._diary_path) > self._DIARY_MAX_SIZE:
                    self._rotate_diary()

            with open(self._diary_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.warning("Diary write failed: %s", e)

    def get_diary(self, n: int = 50) -> List[Dict[str, Any]]:
        """Read the last N entries from the trade diary."""
        import json as _json
        entries = []
        try:
            with open(self._diary_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-n:]:
                try:
                    entries.append(_json.loads(line))
                except Exception:
                    pass
        except FileNotFoundError:
            pass
        return entries

    def __repr__(self) -> str:
        return (f"BrokerBridge(broker='{self._adapter.name}', mode='{self._mode}', "
                f"connected={self.is_connected()}, positions={len(self._positions)})")
