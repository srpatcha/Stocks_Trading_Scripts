"""TradeStation Order Router — OAuth2-authenticated order management via the v3 REST API."""

import json
import logging
import sqlite3
import threading
import time
from collections import deque
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Set

import requests

logger = logging.getLogger(__name__)

from shared.utils.secret_file import write_secret_file
from shared.utils.paths import stocks_plugin_root

try:
    from shared.risk_manager import RiskManager  # type: ignore
except ImportError:
    RiskManager = None


class TradeStationAPIError(Exception):
    """Custom exception for TradeStation API errors."""

    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class TradingBlockedError(Exception):
    """Raised when an order is rejected because trading is blocked."""


class TradeStationOrderRouter:
    """Handles order placement, cancellation, and status queries against TradeStation v3 API.

    Includes OAuth2 token management with automatic refresh, rate limiting
    (max 120 requests/minute), daily loss limits, consecutive-loss cooldown,
    max position count enforcement, and integration with AccountMonitor
    trading blocks and an optional shared RiskManager.
    """

    BASE_URL = "https://api.tradestation.com/v3"
    TOKEN_URL = "https://signin.tradestation.com/oauth/token"
    MAX_REQUESTS_PER_MINUTE = 120

    def __init__(
        self,
        config: dict,
        account_monitor=None,
        risk_manager=None,
        persist_path: Optional[str] = None,
    ):
        """Initialize the order router.

        Args:
            config: Dictionary containing:
                - client_id (str): OAuth2 client ID
                - client_secret (str): OAuth2 client secret
                - redirect_uri (str): OAuth2 redirect URI
                - refresh_token (str): OAuth2 refresh token for token exchange
                - max_daily_loss (float): Maximum daily P&L loss before blocking (default 5000)
                - max_consecutive_losses (int): Losses before cooldown kicks in (default 3)
                - cooldown_minutes (int): Cooldown duration after consecutive losses (default 30)
                - max_positions (int): Maximum simultaneous open positions (default 10)
            account_monitor: Optional AccountMonitor whose trading-block gate
                is checked before every order placement.
            risk_manager: Optional shared RiskManager for delegated
                can_trade / position_sizing decisions.
            persist_path: Optional path to a SQLite database for crash-recovery
                persistence of daily P&L, cooldown, and position state.
        """
        self.client_id = config["client_id"]
        self.client_secret = config["client_secret"]
        self.redirect_uri = config["redirect_uri"]
        self.refresh_token = config["refresh_token"]

        self.access_token: Optional[str] = None
        self.token_expiry: float = 0.0
        self._request_timestamps: deque = deque()
        self._session = requests.Session()

        # FIX 4: Token refresh thread safety
        self._token_lock = threading.Lock()

        # FIX 2: AccountMonitor integration
        self._monitor = account_monitor

        # FIX 6: Shared RiskManager integration
        self._risk_manager = risk_manager

        # FIX 3: Daily loss limit
        self._daily_pnl: float = 0.0
        self._max_daily_loss: float = config.get("max_daily_loss", 5000.0)
        self._daily_pnl_reset_date: Optional[str] = None

        # FIX 4: Consecutive loss cooldown
        self._consecutive_losses: int = 0
        self._max_consecutive_losses: int = config.get("max_consecutive_losses", 3)
        self._cooldown_minutes: int = config.get("cooldown_minutes", 30)
        self._cooldown_until: Optional[datetime] = None

        # FIX 5: Max position count
        self._open_positions: Set[str] = set()
        self._max_positions: int = config.get("max_positions", 10)

        # SQLite state persistence
        self._persist_path = persist_path
        self._persist_conn: Optional[sqlite3.Connection] = None
        self._persist_lock = threading.Lock()
        if self._persist_path:
            self._init_persistence(self._persist_path)
            self._load_risk_state()

        self._authenticate()

    # ── Risk State Persistence ───────────────────────────────────────────

    def _init_persistence(self, db_path: str) -> None:
        """Initialize SQLite database for risk state persistence."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._persist_conn = sqlite3.connect(db_path, check_same_thread=False)
        # FIX 8: SQLite WAL mode for better concurrency
        self._persist_conn.execute("PRAGMA journal_mode=WAL")
        self._persist_conn.execute("PRAGMA busy_timeout=5000")
        self._persist_conn.execute(
            "CREATE TABLE IF NOT EXISTS ts_risk_state "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        self._persist_conn.commit()
        logger.info("TradeStation persistence initialized: %s", db_path)

    def _save_risk_state(self) -> None:
        """Persist daily P&L, cooldown, and position state to SQLite."""
        if self._persist_conn is None:
            return

        now = datetime.now(timezone.utc).isoformat()
        state = {
            "daily_pnl": self._daily_pnl,
            "daily_pnl_reset_date": self._daily_pnl_reset_date,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            "open_positions": list(self._open_positions),
        }

        with self._persist_lock:
            try:
                for key, value in state.items():
                    self._persist_conn.execute(
                        "INSERT OR REPLACE INTO ts_risk_state (key, value, updated_at) "
                        "VALUES (?, ?, ?)",
                        (key, json.dumps(value), now),
                    )
                self._persist_conn.commit()
            except Exception as e:
                logger.error("Failed to save TradeStation risk state: %s", e)

    def _load_risk_state(self) -> None:
        """Restore risk state from SQLite on startup."""
        if self._persist_conn is None:
            return

        with self._persist_lock:
            try:
                rows = self._persist_conn.execute(
                    "SELECT key, value FROM ts_risk_state"
                ).fetchall()
            except Exception as e:
                logger.error("Failed to load TradeStation risk state: %s", e)
                return

        if not rows:
            logger.info("No persisted TradeStation risk state found — starting fresh")
            return

        state = {key: json.loads(value) for key, value in rows}

        saved_date = state.get("daily_pnl_reset_date")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if saved_date == today:
            self._daily_pnl = float(state.get("daily_pnl", 0.0))
            self._daily_pnl_reset_date = saved_date
        else:
            logger.info("Persisted TradeStation state is from %s — resetting daily P&L", saved_date)
            self._daily_pnl_reset_date = today

        self._consecutive_losses = int(state.get("consecutive_losses", 0))

        cooldown_str = state.get("cooldown_until")
        if cooldown_str:
            try:
                self._cooldown_until = datetime.fromisoformat(cooldown_str)
            except (ValueError, TypeError):
                self._cooldown_until = None

        positions = state.get("open_positions")
        if isinstance(positions, list):
            self._open_positions = set(positions)

        logger.info(
            "TradeStation risk state restored: daily_pnl=$%.2f, "
            "consecutive_losses=%d, open_positions=%d",
            self._daily_pnl, self._consecutive_losses, len(self._open_positions),
        )

    # ── Risk Gate ────────────────────────────────────────────────────────

    def _pre_order_checks(self, symbol: str) -> None:
        """Run all risk gates before placing an order.

        Raises:
            TradingBlockedError: When any gate blocks the order.
        """
        if self._monitor:
            blocked, reason = self._monitor.is_trading_blocked()
            if blocked:
                msg = f"Order rejected — AccountMonitor block: {reason}"
                logger.error(msg)
                raise TradingBlockedError(msg)

        if self._risk_manager is not None:
            try:
                if not self._risk_manager.can_trade():
                    msg = "Order rejected — RiskManager.can_trade() returned False"
                    logger.error(msg)
                    raise TradingBlockedError(msg)
            except Exception as exc:
                if isinstance(exc, TradingBlockedError):
                    raise
                # Fail closed. A risk gate that errors is a gate whose answer
                # is unknown, and "unknown" must not mean "yes" on an order
                # path. Signature drift between this caller and RiskManager has
                # already happened elsewhere in this codebase, so this branch
                # is reachable in practice, not theoretical.
                msg = f"Order rejected — RiskManager check errored: {exc}"
                logger.error(msg)
                raise TradingBlockedError(msg) from exc

        if not self.can_trade():
            msg = (
                f"Order rejected — daily loss limit reached "
                f"(P&L ${self._daily_pnl:,.2f}, limit -${self._max_daily_loss:,.2f})"
            )
            logger.error(msg)
            raise TradingBlockedError(msg)

        if self.is_in_cooldown():
            remaining = (self._cooldown_until - datetime.now(timezone.utc)).total_seconds()
            msg = (
                f"Order rejected — cooldown active after {self._max_consecutive_losses} "
                f"consecutive losses ({remaining:.0f}s remaining)"
            )
            logger.error(msg)
            raise TradingBlockedError(msg)

        if symbol not in self._open_positions and len(self._open_positions) >= self._max_positions:
            msg = (
                f"Order rejected — max open positions reached "
                f"({len(self._open_positions)}/{self._max_positions})"
            )
            logger.error(msg)
            raise TradingBlockedError(msg)

    # ── Daily Loss Limit ────────────────────────────────────────────────

    def record_trade_pnl(self, pnl: float) -> None:
        """Record a realized trade P&L and update the daily total."""
        self._auto_reset_daily()
        self._daily_pnl += pnl
        logger.info("Trade P&L recorded: %+.2f — daily total: %+.2f", pnl, self._daily_pnl)
        self._save_risk_state()

    def can_trade(self) -> bool:
        """Return False when cumulative daily losses exceed the limit.

        Both limits apply. Returning the shared RiskManager's verdict directly
        skipped the local check entirely, and since ``record_trade_pnl`` only
        updates the local ``_daily_pnl``, attaching a RiskManager disabled this
        router's own daily stop: a $6,000 local loss against a $5,000 limit
        still returned True. Attaching the better risk manager made the system
        strictly less safe.
        """
        self._auto_reset_daily()
        local_ok = self._daily_pnl > -self._max_daily_loss
        if not local_ok:
            return False

        if self._risk_manager is not None:
            try:
                return bool(self._risk_manager.can_trade())
            except Exception as exc:
                # Fail closed — see _pre_order_checks.
                logger.error("RiskManager.can_trade() failed, blocking trade: %s", exc)
                return False
        return True

    def reset_daily(self) -> None:
        """Reset the daily P&L counter (call at start of trading day)."""
        self._daily_pnl = 0.0
        self._daily_pnl_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info("Daily P&L reset to 0.00")

    def _auto_reset_daily(self) -> None:
        """Auto-reset daily P&L when the UTC date rolls over."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_pnl_reset_date != today:
            self._daily_pnl = 0.0
            self._daily_pnl_reset_date = today

    # ── Consecutive Loss Cooldown ───────────────────────────────────────

    def record_trade_result(self, won: bool) -> None:
        """Record whether the last trade was a win or loss.

        After ``max_consecutive_losses`` losses in a row a cooldown period is
        activated during which no new orders may be placed.
        """
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            logger.warning("Consecutive losses: %d", self._consecutive_losses)
            if self._consecutive_losses >= self._max_consecutive_losses:
                self._cooldown_until = datetime.now(timezone.utc) + timedelta(
                    minutes=self._cooldown_minutes
                )
                logger.warning(
                    "Cooldown activated until %s after %d consecutive losses",
                    self._cooldown_until.isoformat(),
                    self._consecutive_losses,
                )
        self._save_risk_state()

    def is_in_cooldown(self) -> bool:
        """Return True if the router is in a post-loss cooldown period."""
        if self._cooldown_until is None:
            return False
        if datetime.now(timezone.utc) >= self._cooldown_until:
            self._cooldown_until = None
            self._consecutive_losses = 0
            return False
        return True

    # ── Position Count Tracking ─────────────────────────────────────────

    def add_position(self, symbol: str) -> None:
        """Register a newly opened position."""
        self._open_positions.add(symbol)
        logger.info(
            "Position added: %s — open positions: %d/%d",
            symbol, len(self._open_positions), self._max_positions,
        )
        self._save_risk_state()

    def remove_position(self, symbol: str) -> None:
        """Remove a closed position."""
        self._open_positions.discard(symbol)
        logger.info(
            "Position removed: %s — open positions: %d/%d",
            symbol, len(self._open_positions), self._max_positions,
        )
        self._save_risk_state()

    # ── OAuth2 ──────────────────────────────────────────────────────────

    def _authenticate(self):
        """Exchange the refresh token for an access token (thread-safe)."""
        with self._token_lock:
            payload = {
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "redirect_uri": self.redirect_uri,
            }
            resp = self._session.post(self.TOKEN_URL, data=payload, timeout=30)
            if resp.status_code != 200:
                raise TradeStationAPIError(
                    f"Authentication failed: {resp.text}",
                    status_code=resp.status_code,
                    response_body=resp.text,
                )

            data = resp.json()
            token = data.get("access_token")
            if not token:
                raise TradeStationAPIError(f"Auth failed: {data}")
            self.access_token = token
            self.token_expiry = time.time() + data.get("expires_in", 1200) - 60
            if "refresh_token" in data:
                self.refresh_token = data["refresh_token"]
                # FIX 4: Persist refresh token to disk
                self._persist_refresh_token()
            logger.info("TradeStation authentication successful")

    def _persist_refresh_token(self) -> None:
        """Persist the current refresh token to disk."""
        try:
            token_path = Path(stocks_plugin_root()) / "ts_refresh_token.txt"
            write_secret_file(token_path, self.refresh_token)
            logger.debug("TradeStation refresh token persisted to %s", token_path)
        except Exception as e:
            logger.warning("Failed to persist TradeStation refresh token: %s", e)

    def _refresh_access_token(self):
        """Refresh the access token if it is expired or about to expire."""
        if time.time() >= self.token_expiry:
            logger.info("Access token expired, refreshing...")
            self._authenticate()

    def _get_headers(self) -> dict:
        """Return authorization headers, refreshing the token if needed."""
        self._refresh_access_token()
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    # ── Rate Limiting ───────────────────────────────────────────────────

    def _enforce_rate_limit(self):
        """Block until a request slot is available (120 req/min sliding window)."""
        now = time.time()
        while self._request_timestamps and self._request_timestamps[0] < now - 60:
            self._request_timestamps.popleft()

        if len(self._request_timestamps) >= self.MAX_REQUESTS_PER_MINUTE:
            sleep_for = 60 - (now - self._request_timestamps[0])
            if sleep_for > 0:
                logger.warning("Rate limit reached, sleeping %.2fs", sleep_for)
                time.sleep(sleep_for)

        self._request_timestamps.append(time.time())

    # ── HTTP helpers ────────────────────────────────────────────────────

    def _request(self, method: str, path: str, json_body: dict = None) -> dict:
        """Execute an API request with rate limiting and error handling."""
        self._enforce_rate_limit()
        url = f"{self.BASE_URL}{path}"
        headers = self._get_headers()

        try:
            resp = self._session.request(
                method, url, headers=headers, json=json_body, timeout=30
            )
        except requests.RequestException as exc:
            raise TradeStationAPIError(f"Network error: {exc}") from exc

        if resp.status_code == 401:
            logger.info("Received 401, attempting token refresh and retry")
            self._authenticate()
            self._enforce_rate_limit()
            headers = self._get_headers()
            resp = self._session.request(
                method, url, headers=headers, json=json_body, timeout=30
            )

        if resp.status_code not in (200, 201, 204):
            raise TradeStationAPIError(
                f"API error {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
                response_body=resp.text,
            )

        if resp.text:
            try:
                return resp.json()
            except ValueError:
                return {}
        return {}

    # ── Order Methods ───────────────────────────────────────────────────

    def place_market_order(
        self, account_id: str, symbol: str, action: str, quantity: int
    ) -> str:
        """Place a market order.

        Args:
            account_id: TradeStation account ID.
            symbol: Ticker symbol (e.g. "AAPL").
            action: "BUY" or "SELL".
            quantity: Number of shares.

        Returns:
            The order ID assigned by TradeStation.

        Raises:
            TradingBlockedError: If any risk gate blocks the order.
        """
        self._pre_order_checks(symbol)

        body = {
            "AccountID": account_id,
            "Symbol": symbol,
            "Quantity": str(quantity),
            "OrderType": "Market",
            "TradeAction": action,
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent",
        }
        logger.info("Placing market %s order: %s x%d", action, symbol, quantity)
        result = self._request("POST", "/orderexecution/orders", json_body=body)
        orders = result.get("Orders", [])
        order_id = orders[0].get("OrderID") if orders else None
        logger.info("Market order placed — OrderID: %s", order_id)
        return order_id

    def place_limit_order(
        self,
        account_id: str,
        symbol: str,
        action: str,
        quantity: int,
        limit_price: float,
    ) -> str:
        """Place a limit order.

        Returns:
            The order ID assigned by TradeStation.

        Raises:
            TradingBlockedError: If any risk gate blocks the order.
        """
        self._pre_order_checks(symbol)

        body = {
            "AccountID": account_id,
            "Symbol": symbol,
            "Quantity": str(quantity),
            "OrderType": "Limit",
            "LimitPrice": str(limit_price),
            "TradeAction": action,
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent",
        }
        logger.info(
            "Placing limit %s order: %s x%d @ %.2f",
            action,
            symbol,
            quantity,
            limit_price,
        )
        result = self._request("POST", "/orderexecution/orders", json_body=body)
        orders = result.get("Orders", [])
        order_id = orders[0].get("OrderID") if orders else None
        logger.info("Limit order placed — OrderID: %s", order_id)
        return order_id

    def place_bracket_order(
        self,
        account_id: str,
        symbol: str,
        action: str,
        quantity: int,
        limit_price: float,
        profit_target: float,
        stop_loss: float,
    ) -> dict:
        """Place a bracket (OCO) order with profit target and stop loss.

        Returns:
            Dict with OrderIDs for the entry, target, and stop orders.

        Raises:
            TradingBlockedError: If any risk gate blocks the order.
        """
        self._pre_order_checks(symbol)

        body = {
            "AccountID": account_id,
            "Symbol": symbol,
            "Quantity": str(quantity),
            "OrderType": "Limit",
            "LimitPrice": str(limit_price),
            "TradeAction": action,
            "TimeInForce": {"Duration": "GTC"},
            "Route": "Intelligent",
            "OSOs": [
                {
                    "Type": "BRK",
                    "Orders": [
                        {
                            "AccountID": account_id,
                            "Symbol": symbol,
                            "Quantity": str(quantity),
                            "OrderType": "Limit",
                            "LimitPrice": str(profit_target),
                            "TradeAction": {"BUY": "SELL", "SELL": "BUY", "BUYTOCOVER": "SELLSHORT", "SELLSHORT": "BUYTOCOVER"}.get(action, "SELL"),
                            "TimeInForce": {"Duration": "GTC"},
                            "Route": "Intelligent",
                        },
                        {
                            "AccountID": account_id,
                            "Symbol": symbol,
                            "Quantity": str(quantity),
                            "OrderType": "StopMarket",
                            "StopPrice": str(stop_loss),
                            "TradeAction": {"BUY": "SELL", "SELL": "BUY", "BUYTOCOVER": "SELLSHORT", "SELLSHORT": "BUYTOCOVER"}.get(action, "SELL"),
                            "TimeInForce": {"Duration": "GTC"},
                            "Route": "Intelligent",
                        },
                    ],
                }
            ],
        }
        logger.info(
            "Placing bracket %s order: %s x%d @ %.2f  TP=%.2f  SL=%.2f",
            action,
            symbol,
            quantity,
            limit_price,
            profit_target,
            stop_loss,
        )
        result = self._request("POST", "/orderexecution/orders", json_body=body)
        orders = result.get("Orders", [])
        return {
            "entry_order_id": orders[0]["OrderID"] if len(orders) > 0 else None,
            "bracket_orders": [o["OrderID"] for o in orders[1:]] if len(orders) > 1 else [],
        }

    def place_stop_order(self, account_id: str, symbol: str, action: str, quantity: int, stop_price: float) -> str:
        """Place a stop-market order via TradeStation API."""
        body = {
            "AccountID": account_id,
            "Symbol": symbol,
            "Quantity": str(quantity),
            "OrderType": "StopMarket",
            "StopPrice": str(stop_price),
            "TradeAction": action.upper(),
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent"
        }
        # BASE_URL already ends in /v3, and _request's payload parameter is
        # json_body — "/v3/..." plus json= produced a doubled path and a
        # TypeError on every stop-order placement.
        result = self._request("POST", "/orderexecution/orders", json_body=body)
        orders = result.get("Orders", [])
        return orders[0]["OrderID"] if orders else None

    def cancel_order(self, order_id: str):
        """Cancel an open order.

        Args:
            order_id: The order to cancel.
        """
        logger.info("Cancelling order %s", order_id)
        self._request("DELETE", f"/orderexecution/orders/{order_id}")
        logger.info("Order %s cancelled", order_id)

    def get_order_status(self, account_id: str, order_id: str) -> dict:
        """Get the status of a specific order.

        Returns:
            Dict with order status details.
        """
        result = self._request(
            "GET", f"/brokerage/accounts/{account_id}/orders/{order_id}"
        )
        return result

    def get_orders(self, account_id: str) -> list:
        """Get all orders for an account.

        Returns:
            List of order dicts.
        """
        result = self._request(
            "GET", f"/brokerage/accounts/{account_id}/orders"
        )
        return result.get("Orders", [])

    def get_quote(self, symbol: str) -> dict:
        """Get a real-time quote for a symbol using the MarketData v3 API.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").

        Returns:
            Dict with at least a "Last" key (last trade price as float),
            or an empty dict on failure.
        """
        try:
            result = self._request("GET", f"/marketdata/quotes/{symbol}")
            quotes = result.get("Quotes", [])
            if quotes:
                return quotes[0]
        except TradeStationAPIError as e:
            logger.warning("get_quote(%s) failed: %s", symbol, e)
        return {}

    def get_quotes(self, symbols: list) -> list:
        """Get real-time quotes for multiple symbols.

        Args:
            symbols: List of ticker symbols.

        Returns:
            List of quote dicts.
        """
        if not symbols:
            return []
        joined = ",".join(symbols)
        try:
            result = self._request("GET", f"/marketdata/quotes/{joined}")
            return result.get("Quotes", [])
        except TradeStationAPIError as e:
            logger.warning("get_quotes(%s) failed: %s", joined, e)
        return []
