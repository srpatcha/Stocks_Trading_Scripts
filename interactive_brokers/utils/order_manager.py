"""
Order Management for Interactive Brokers
==========================================

Provides a comprehensive OrderManager class for placing, tracking,
modifying, and canceling orders with built-in risk controls.

Usage:
    manager = OrderManager(connection, config={"max_position_pct": 0.05})
    trade = manager.market_order("AAPL", "BUY", 100)
    status = manager.get_order_status(trade.order_id)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from shared.utils.trading_clock import trading_day


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderState(Enum):
    PENDING = "PendingSubmit"
    SUBMITTED = "Submitted"
    PRE_SUBMITTED = "PreSubmitted"
    FILLED = "Filled"
    PARTIALLY_FILLED = "PartiallyFilled"
    CANCELLED = "Cancelled"
    INACTIVE = "Inactive"
    API_PENDING = "ApiPending"
    API_CANCELLED = "ApiCancelled"


# ib_async reports order state as a plain string. Map it onto OrderState so
# sync_orders() can mark terminal orders and stop counting them as open.
_IB_STATUS_MAP = {
    "PendingSubmit": OrderState.PENDING,
    "ApiPending": OrderState.API_PENDING,
    "PreSubmitted": OrderState.PRE_SUBMITTED,
    "Submitted": OrderState.SUBMITTED,
    "Filled": OrderState.FILLED,
    "Cancelled": OrderState.CANCELLED,
    "ApiCancelled": OrderState.API_CANCELLED,
    "Inactive": OrderState.INACTIVE,
}


@dataclass
class OrderStatus:
    """Tracks the status and details of an order."""
    order_id: int
    symbol: str
    action: str
    order_type: str
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderState = OrderState.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    submitted_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    parent_id: Optional[int] = None
    oca_group: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.status in (
            OrderState.PENDING, OrderState.SUBMITTED,
            OrderState.PRE_SUBMITTED, OrderState.PARTIALLY_FILLED,
            OrderState.API_PENDING,
        )

    @property
    def is_filled(self) -> bool:
        return self.status == OrderState.FILLED


@dataclass
class RiskConfig:
    """Risk control configuration."""
    max_position_pct: float = 0.05
    max_order_value: float = 50000.0
    max_daily_loss: float = 5000.0
    max_open_orders: int = 20
    max_position_size: int = 1000
    require_confirmation_above: float = 25000.0
    cooldown_after_losses: int = 3
    cooldown_pause_seconds: int = 1800


class OrderManager:
    """Manages order placement, tracking, and risk controls for IB.

    Args:
        connection: An IBInsyncConnection or IBApiConnection instance.
        config: Risk configuration dict or RiskConfig instance.
        notifier: Optional AlertDispatcher for order notifications.
    """

    def __init__(
        self,
        connection: Any,
        config: Optional[dict | RiskConfig] = None,
        notifier: Any = None,
    ) -> None:
        self.connection = connection
        self.notifier = notifier
        self._orders: Dict[int, OrderStatus] = {}
        # order_id -> live ib_async Trade, used by sync_orders().
        self._trades: Dict[int, Any] = {}
        self._daily_pnl: float = 0.0
        self._daily_pnl_date: date = trading_day()
        self._loss_streak: int = 0
        self._cooldown_until: float = 0.0

        if isinstance(config, RiskConfig):
            self.risk_config = config
        elif isinstance(config, dict):
            self.risk_config = RiskConfig(**{
                k: v for k, v in config.items()
                if k in RiskConfig.__dataclass_fields__
            })
        else:
            self.risk_config = RiskConfig()

    def _create_contract(self, symbol: str, sec_type: str = "STK",
                         exchange: str = "SMART", currency: str = "USD",
                         **kwargs: Any) -> Any:
        """Create a Contract object using ib_async or ibapi.

        Args:
            symbol: Ticker symbol.
            sec_type: Security type (STK, OPT, FUT, etc.).
            exchange: Exchange name.
            currency: Currency code.

        Returns:
            A qualified Contract object.
        """
        try:
            from ib_async import Stock, Option, Future, Contract

            if sec_type == "STK":
                contract = Stock(symbol, exchange, currency)
            elif sec_type == "OPT":
                contract = Option(
                    symbol, kwargs.get("expiry", ""),
                    kwargs.get("strike", 0), kwargs.get("right", "C"),
                    exchange, currency=currency,
                )
            elif sec_type == "FUT":
                contract = Future(symbol, kwargs.get("expiry", ""), exchange, currency=currency)
            else:
                contract = Contract(
                    symbol=symbol, secType=sec_type,
                    exchange=exchange, currency=currency,
                )

            self.connection.qualifyContracts(contract)
            return contract
        except ImportError:
            try:
                from ibapi.contract import Contract as IBApiContract

                contract = IBApiContract()
                contract.symbol = symbol
                contract.secType = sec_type
                contract.exchange = exchange
                contract.currency = currency
                if sec_type == "OPT":
                    contract.lastTradeDateOrContractMonth = kwargs.get("expiry", "")
                    contract.strike = kwargs.get("strike", 0)
                    contract.right = kwargs.get("right", "C")
                elif sec_type == "FUT":
                    contract.lastTradeDateOrContractMonth = kwargs.get("expiry", "")
                return contract
            except ImportError:
                logger.error("Neither ib_async nor ibapi available for contract creation")
                raise

    def _create_order(self, action: str, quantity: float,
                      order_type: str = "MKT", **kwargs: Any) -> Any:
        """Create an ib_async Order object.

        Args:
            action: "BUY" or "SELL".
            quantity: Number of shares/contracts.
            order_type: "MKT", "LMT", "STP", "STP LMT", "TRAIL".

        Returns:
            An Order object.
        """
        from ib_async import Order

        order = Order(
            action=action,
            totalQuantity=quantity,
            orderType=order_type,
        )

        if order_type in ("LMT", "STP LMT") and "limit_price" in kwargs:
            order.lmtPrice = kwargs["limit_price"]
        if order_type in ("STP", "STP LMT") and "stop_price" in kwargs:
            order.auxPrice = kwargs["stop_price"]
        if order_type == "TRAIL":
            if "trail_amount" in kwargs:
                order.auxPrice = kwargs["trail_amount"]
            elif "trail_percent" in kwargs:
                order.trailingPercent = kwargs["trail_percent"]

        if "tif" in kwargs:
            order.tif = kwargs["tif"]
        if "oca_group" in kwargs:
            order.ocaGroup = kwargs["oca_group"]
            order.ocaType = kwargs.get("oca_type", 1)
        if "parent_id" in kwargs:
            order.parentId = kwargs["parent_id"]
        if "transmit" in kwargs:
            order.transmit = kwargs["transmit"]

        return order

    def _validate_order(self, symbol: str, action: str, quantity: float,
                        estimated_price: Optional[float] = None) -> None:
        """Run all risk checks before placing an order.

        Raises:
            ValueError: If any risk check fails.
        """
        self._check_position_size(quantity)
        self._check_daily_loss()
        self._check_cooldown()

        # The notional cap was gated behind `estimated_price is not None`,
        # and market_order()/trailing_stop_order() pass no price — so the cap
        # never applied to exactly the orders whose cost is unknown up front.
        # Fall back to a live quote; refuse only if we cannot price it at all.
        price = estimated_price
        if price is None:
            price = self._estimate_price(symbol)
        if price is not None:
            order_value = quantity * price
            if order_value > self.risk_config.max_order_value:
                raise ValueError(
                    f"Order value ${order_value:,.2f} exceeds max "
                    f"${self.risk_config.max_order_value:,.2f}"
                )
        else:
            logger.warning(
                "No price available for %s — max_order_value not enforced "
                "for this order", symbol,
            )

        # Reconcile first: without this, filled orders were still counted as
        # open and the account locked itself out after max_open_orders.
        self.sync_orders()
        open_count = len(self.get_open_orders())
        if open_count >= self.risk_config.max_open_orders:
            raise ValueError(
                f"Open order count ({open_count}) at maximum "
                f"({self.risk_config.max_open_orders})"
            )

    def _check_position_size(self, quantity: float) -> None:
        """Verify position size is within limits.

        Raises:
            ValueError: If quantity exceeds maximum position size.
        """
        if quantity > self.risk_config.max_position_size:
            raise ValueError(
                f"Quantity {quantity} exceeds max position size "
                f"{self.risk_config.max_position_size}"
            )
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

    def _check_daily_loss(self) -> None:
        """Check if daily loss limit has been hit.

        Raises:
            ValueError: If daily loss limit is exceeded.
        """
        if self._daily_pnl_date != trading_day():
            self._daily_pnl = 0.0
            self._daily_pnl_date = trading_day()

        if self._daily_pnl <= -self.risk_config.max_daily_loss:
            raise ValueError(
                f"Daily loss limit reached: ${self._daily_pnl:,.2f} "
                f"(max: ${self.risk_config.max_daily_loss:,.2f})"
            )

    def _track_order(self, trade: Any, symbol: str, action: str,
                     order_type: str, quantity: float,
                     limit_price: Optional[float] = None,
                     stop_price: Optional[float] = None,
                     parent_id: Optional[int] = None) -> OrderStatus:
        """Register an order in the tracking system."""
        order_id = trade.order.orderId if hasattr(trade, 'order') else int(uuid.uuid4().hex[:12], 16)

        status = OrderStatus(
            order_id=order_id,
            symbol=symbol,
            action=action,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            parent_id=parent_id,
        )
        self._orders[order_id] = status
        # Keep the live Trade object so sync_orders() can read the real fill
        # back off it. Nothing used to update OrderStatus after submission, so
        # every order stayed PENDING forever (see sync_orders).
        self._trades[order_id] = trade
        logger.info(
            "Order tracked: %s %s %.0f %s @ %s [id=%d]",
            action, symbol, quantity, order_type,
            limit_price if limit_price is not None else "MKT", order_id,
        )
        return status

    def _notify(self, message: str, level: str = "info") -> None:
        """Send notification via the notifier if available."""
        if self.notifier:
            try:
                if level == "warning":
                    self.notifier.warning(message)
                elif level == "error":
                    self.notifier.error(message)
                else:
                    self.notifier.info(message)
            except Exception as e:
                logger.warning("Notification failed: %s", e)

    def market_order(self, symbol: str, action: str, quantity: float,
                     sec_type: str = "STK", **kwargs: Any) -> Any:
        """Place a market order.

        Args:
            symbol: Ticker symbol.
            action: "BUY" or "SELL".
            quantity: Number of shares/contracts.
            sec_type: Security type.

        Returns:
            The Trade object from ib_async.
        """
        self._validate_order(symbol, action, quantity)

        contract = self._create_contract(symbol, sec_type, **kwargs)
        order = self._create_order(action, quantity, "MKT")
        trade = self.connection.placeOrder(contract, order)

        self._track_order(trade, symbol, action, "MKT", quantity)
        self._notify(f"Market order: {action} {quantity} {symbol}")
        return trade

    def limit_order(self, symbol: str, action: str, quantity: float,
                    limit_price: float, sec_type: str = "STK",
                    tif: str = "GTC", **kwargs: Any) -> Any:
        """Place a limit order.

        Args:
            symbol: Ticker symbol.
            action: "BUY" or "SELL".
            quantity: Number of shares/contracts.
            limit_price: Limit price (must be > 0).
            sec_type: Security type.
            tif: Time in force (GTC, DAY, IOC, etc.).

        Returns:
            The Trade object from ib_async.

        Raises:
            ValueError: If limit_price is <= 0.
        """
        if limit_price <= 0:
            raise ValueError(f"Limit price must be positive, got {limit_price}")
        self._validate_order(symbol, action, quantity, limit_price)

        contract = self._create_contract(symbol, sec_type, **kwargs)
        order = self._create_order(
            action, quantity, "LMT",
            limit_price=limit_price, tif=tif,
        )
        trade = self.connection.placeOrder(contract, order)

        self._track_order(trade, symbol, action, "LMT", quantity, limit_price=limit_price)
        self._notify(f"Limit order: {action} {quantity} {symbol} @ ${limit_price:.2f}")
        return trade

    def stop_order(self, symbol: str, action: str, quantity: float,
                   stop_price: float, sec_type: str = "STK",
                   **kwargs: Any) -> Any:
        """Place a stop order.

        Args:
            symbol: Ticker symbol.
            action: "BUY" or "SELL".
            quantity: Number of shares/contracts.
            stop_price: Stop trigger price.

        Returns:
            The Trade object from ib_async.
        """
        self._validate_order(symbol, action, quantity, stop_price)

        contract = self._create_contract(symbol, sec_type, **kwargs)
        order = self._create_order(
            action, quantity, "STP",
            stop_price=stop_price,
        )
        trade = self.connection.placeOrder(contract, order)

        self._track_order(trade, symbol, action, "STP", quantity, stop_price=stop_price)
        self._notify(f"Stop order: {action} {quantity} {symbol} @ ${stop_price:.2f}")
        return trade

    def bracket_order(self, symbol: str, action: str, quantity: float,
                      limit_price: float, take_profit: float,
                      stop_loss: float, sec_type: str = "STK",
                      **kwargs: Any) -> list:
        """Place a bracket order (entry + take profit + stop loss).

        Args:
            symbol: Ticker symbol.
            action: "BUY" or "SELL".
            quantity: Number of shares/contracts.
            limit_price: Entry limit price.
            take_profit: Take profit price.
            stop_loss: Stop loss price.

        Returns:
            List of Trade objects [parent, take_profit, stop_loss].
        """
        self._validate_order(symbol, action, quantity, limit_price)

        try:
            from ib_async import BracketOrder
            contract = self._create_contract(symbol, sec_type, **kwargs)

            bracket = BracketOrder(
                action=action,
                quantity=quantity,
                limitPrice=limit_price,
                takeProfitPrice=take_profit,
                stopLossPrice=stop_loss,
            )

            trades = []
            for o in bracket:
                trade = self.connection.placeOrder(contract, o)
                trades.append(trade)

            self._track_order(
                trades[0], symbol, action, "BRACKET",
                quantity, limit_price=limit_price,
            )
            self._notify(
                f"Bracket order: {action} {quantity} {symbol} "
                f"entry=${limit_price:.2f} TP=${take_profit:.2f} SL=${stop_loss:.2f}"
            )
            return trades
        except ImportError:
            logger.error("ib_async required for bracket orders")
            raise

    def trailing_stop_order(self, symbol: str, action: str, quantity: float,
                            trail_amount: Optional[float] = None,
                            trail_percent: Optional[float] = None,
                            sec_type: str = "STK", **kwargs: Any) -> Any:
        """Place a trailing stop order.

        Args:
            symbol: Ticker symbol.
            action: "BUY" or "SELL".
            quantity: Number of shares/contracts.
            trail_amount: Trailing amount in dollars.
            trail_percent: Trailing percentage.

        Returns:
            The Trade object from ib_async.
        """
        if trail_amount is None and trail_percent is None:
            raise ValueError("Must specify either trail_amount or trail_percent")

        self._validate_order(symbol, action, quantity)

        contract = self._create_contract(symbol, sec_type, **kwargs)
        order_kwargs: dict = {}
        if trail_amount is not None:
            order_kwargs["trail_amount"] = trail_amount
        if trail_percent is not None:
            order_kwargs["trail_percent"] = trail_percent

        order = self._create_order(action, quantity, "TRAIL", **order_kwargs)
        trade = self.connection.placeOrder(contract, order)

        trail_desc = f"${trail_amount}" if trail_amount else f"{trail_percent}%"
        self._track_order(trade, symbol, action, "TRAIL", quantity)
        self._notify(f"Trailing stop: {action} {quantity} {symbol} trail={trail_desc}")
        return trade

    def cancel_order(self, order_id: int) -> bool:
        """Cancel an open order.

        Args:
            order_id: The order ID to cancel.

        Returns:
            True if cancellation request was sent successfully.
        """
        if order_id in self._orders:
            status = self._orders[order_id]
            if not status.is_active:
                logger.warning("Order %d is not active (status=%s)", order_id, status.status)
                return False

        try:
            if not hasattr(self.connection, 'ib'):
                logger.warning("cancel_order requires IBAsyncConnection with .ib attribute")
                return False
            open_orders = self.connection.ib.openOrders()
            for order in open_orders:
                if order.orderId == order_id:
                    self.connection.cancelOrder(order)
                    if order_id in self._orders:
                        self._orders[order_id].status = OrderState.CANCELLED
                    self._notify(f"Order {order_id} cancelled")
                    logger.info("Order %d cancelled", order_id)
                    return True
            logger.warning("Order %d not found in open orders", order_id)
            return False
        except Exception as e:
            logger.error("Failed to cancel order %d: %s", order_id, e)
            return False

    def modify_order(self, order_id: int, new_limit_price: Optional[float] = None,
                     new_quantity: Optional[float] = None) -> bool:
        """Modify an existing order's price or quantity.

        Args:
            order_id: The order ID to modify.
            new_limit_price: New limit price (if applicable).
            new_quantity: New quantity.

        Returns:
            True if modification request was sent successfully.
        """
        if order_id not in self._orders:
            logger.warning("Order %d not tracked", order_id)
            return False

        status = self._orders[order_id]
        if not status.is_active:
            logger.warning("Cannot modify inactive order %d", order_id)
            return False

        try:
            if not hasattr(self.connection, 'ib'):
                logger.warning("modify_order requires IBAsyncConnection with .ib attribute")
                return False
            open_trades = self.connection.ib.openTrades()
            for trade in open_trades:
                if trade.order.orderId == order_id:
                    if new_limit_price is not None:
                        trade.order.lmtPrice = new_limit_price
                        status.limit_price = new_limit_price
                    if new_quantity is not None:
                        trade.order.totalQuantity = new_quantity
                        status.quantity = new_quantity

                    self.connection.placeOrder(trade.contract, trade.order)
                    logger.info("Order %d modified", order_id)
                    self._notify(f"Order {order_id} modified")
                    return True
            logger.warning("Order %d not found in open trades", order_id)
            return False
        except Exception as e:
            logger.error("Failed to modify order %d: %s", order_id, e)
            return False


    def _estimate_price(self, symbol: str) -> Optional[float]:
        """Best-effort last price, used to apply the notional cap.

        Returns None when no quote is reachable; the caller decides what to do
        with that rather than silently skipping the limit.
        """
        getter = getattr(self.connection, "get_latest_price", None)
        if getter is None:
            return None
        try:
            price = float(getter(symbol))
        except Exception as e:
            logger.debug("Price lookup failed for %s: %s", symbol, e)
            return None
        return price if price > 0 else None

    def sync_orders(self) -> int:
        """Refresh tracked orders from their live broker state.

        Nothing previously updated OrderStatus after submission: every order
        stayed PENDING, `is_active` stayed True, and once max_open_orders
        filled orders had accumulated the account locked itself out of placing
        anything further — including the market orders used to exit and the
        stops used to protect. Only cancel_order() ever changed a status.

        Returns:
            Number of orders whose status changed.
        """
        changed = 0
        for order_id, status in list(self._orders.items()):
            trade = self._trades.get(order_id)
            if trade is None:
                continue
            ib_status = getattr(trade, "orderStatus", None)
            if ib_status is None:
                continue

            filled = float(getattr(ib_status, "filled", 0.0) or 0.0)
            avg_price = float(getattr(ib_status, "avgFillPrice", 0.0) or 0.0)
            raw_state = str(getattr(ib_status, "status", "") or "")

            new_state = _IB_STATUS_MAP.get(raw_state)
            if new_state is None and filled >= status.quantity > 0:
                new_state = OrderState.FILLED

            updated = False
            if filled != status.filled_quantity:
                status.filled_quantity = filled
                updated = True
            if avg_price and avg_price != status.avg_fill_price:
                status.avg_fill_price = avg_price
                updated = True
            if new_state is not None and new_state != status.status:
                status.status = new_state
                if new_state == OrderState.FILLED and status.filled_at is None:
                    status.filled_at = datetime.now()
                updated = True

            if updated:
                changed += 1
            if not status.is_active:
                # Terminal: stop holding the broker object.
                self._trades.pop(order_id, None)

        if changed:
            logger.debug("sync_orders: %d order(s) updated", changed)
        return changed

    def get_fill(self, order_id: int) -> Optional[tuple]:
        """Return ``(avg_fill_price, filled_quantity)`` for a filled order.

        Returns None when the order is unknown or nothing has filled yet.
        """
        self.sync_orders()
        status = self._orders.get(int(order_id))
        if status is None or status.filled_quantity <= 0:
            return None
        return status.avg_fill_price, status.filled_quantity

    def get_open_orders(self) -> List[OrderStatus]:
        """Get all currently active/open orders.

        Returns:
            List of OrderStatus objects for active orders.
        """
        return [s for s in self._orders.values() if s.is_active]

    def get_order_status(self, order_id: int) -> Optional[OrderStatus]:
        """Get the status of a specific order.

        Args:
            order_id: The order ID to look up.

        Returns:
            OrderStatus if found, None otherwise.
        """
        return self._orders.get(order_id)

    def get_all_orders(self) -> Dict[int, OrderStatus]:
        """Get all tracked orders (active and inactive).

        Returns:
            Dictionary mapping order IDs to OrderStatus objects.
        """
        return dict(self._orders)

    def _check_cooldown(self) -> None:
        """Check if loss streak cooldown is active.

        Raises:
            ValueError: If currently in cooldown after consecutive losses.
        """
        if time.time() < self._cooldown_until:
            remaining = int(self._cooldown_until - time.time())
            raise ValueError(
                f"Cooldown active: {self._loss_streak} consecutive losses. "
                f"{remaining}s remaining."
            )

    def record_trade_result(self, pnl: float) -> None:
        """Record the result of a trade for loss streak tracking.

        If consecutive losses reach the configured threshold,
        a cooldown period is activated.

        Args:
            pnl: Realized P&L of the completed trade.
        """
        if pnl < 0:
            self._loss_streak += 1
            logger.info(
                "Loss streak: %d (threshold: %d)",
                self._loss_streak,
                self.risk_config.cooldown_after_losses,
            )
            if self._loss_streak >= self.risk_config.cooldown_after_losses:
                self._cooldown_until = (
                    time.time() + self.risk_config.cooldown_pause_seconds
                )
                logger.warning(
                    "Cooldown activated: %d consecutive losses → pausing %ds",
                    self._loss_streak,
                    self.risk_config.cooldown_pause_seconds,
                )
                self._notify(
                    f"⚠️ Cooldown: {self._loss_streak} consecutive losses. "
                    f"Pausing {self.risk_config.cooldown_pause_seconds}s.",
                    level="warning",
                )
        else:
            if self._loss_streak > 0:
                logger.info(
                    "Loss streak reset (was %d) after profitable trade",
                    self._loss_streak,
                )
            self._loss_streak = 0

    def update_daily_pnl(self, pnl: float) -> None:
        """Update the daily P&L tracker (call periodically).

        Args:
            pnl: Current daily realized P&L.
        """
        if self._daily_pnl_date != trading_day():
            self._daily_pnl = 0.0
            self._daily_pnl_date = trading_day()
        self._daily_pnl = pnl
