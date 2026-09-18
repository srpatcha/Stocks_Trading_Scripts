"""
Schwab API Client (for thinkorswim)
=====================================

OAuth2-authenticated REST client for the Charles Schwab Trader API.
Replaces the old TDA API. Provides market data, order placement,
account info, and position management.

API Docs: https://developer.schwab.com

Setup:
    1. Register at https://developer.schwab.com
    2. Create an app to get client_id and client_secret
    3. Complete OAuth2 flow to get initial refresh_token
    4. Pass credentials to SchwabClient

Usage:
    client = SchwabClient({
        "client_id": "your_app_key",
        "client_secret": "your_secret",
        "refresh_token": "your_refresh_token",
        "account_id": "your_account_hash",
    })
    quote = client.get_quote("AAPL")
    order_id = client.place_market_order("AAPL", "BUY", 100)
    positions = client.get_positions()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from collections import deque
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

from shared.utils.secret_file import write_secret_file

# --- RiskManager integration (optional, from shared module) ---
try:
    from shared.risk_manager import RiskManager, RiskManagerConfig
    _HAS_RISK_MANAGER = True
except ImportError:
    _HAS_RISK_MANAGER = False
    RiskManager = None  # type: ignore[assignment,misc]
    RiskManagerConfig = None  # type: ignore[assignment,misc]


class SchwabAPIError(Exception):
    """Custom exception for Schwab API errors."""

    def __init__(self, message: str, status_code: int = 0, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class SchwabClient:
    """Charles Schwab Trader API client with OAuth2.

    Handles token management, rate limiting, and provides methods
    for market data, order placement, account queries, and positions.

    Args:
        config: Dict with:
            - client_id: OAuth2 app key
            - client_secret: OAuth2 app secret
            - refresh_token: OAuth2 refresh token
            - account_id: Account hash (encrypted account number)
            - redirect_uri: OAuth2 redirect URI (default: https://127.0.0.1)
    """

    AUTH_URL = "https://api.schwabapi.com/v1/oauth/token"
    BASE_URL = "https://api.schwabapi.com"
    TRADER_URL = "https://api.schwabapi.com/trader/v1"
    MARKETDATA_URL = "https://api.schwabapi.com/marketdata/v1"

    MAX_REQUESTS_PER_SECOND = 2

    def __init__(
        self,
        config: Dict[str, str],
        risk_manager: Optional[Any] = None,
        max_daily_loss: float = 5000.0,
        persist_path: Optional[str] = None,
    ) -> None:
        self._client_id = config["client_id"]
        self._client_secret = config["client_secret"]
        self._refresh_token = config["refresh_token"]
        self._account_id = config.get("account_id", "")
        self._redirect_uri = config.get("redirect_uri", "https://127.0.0.1")

        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._request_times: deque = deque()
        self._session = requests.Session()

        # FIX 4: Token refresh thread safety
        self._token_lock = threading.Lock()

        # RiskManager integration
        self._risk_manager = risk_manager

        # Daily P&L tracking
        self._daily_pnl: float = 0.0
        self._max_daily_loss: float = max_daily_loss
        self._daily_pnl_date: date = date.today()
        self._consecutive_losses: int = 0
        self._cooldown_until: float = 0.0

        # SQLite state persistence
        self._persist_path = persist_path
        self._persist_conn: Optional[sqlite3.Connection] = None
        self._persist_lock = threading.Lock()
        if self._persist_path:
            self._init_persistence(self._persist_path)
            self._load_risk_state()

        self._authenticate()
        logger.info("SchwabClient initialized (account=%s)", self._account_id[:8] + "..." if self._account_id else "?")

    # ─── Risk State Persistence ───

    def _init_persistence(self, db_path: str) -> None:
        """Initialize SQLite database for risk state persistence."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._persist_conn = sqlite3.connect(db_path, check_same_thread=False)
        # FIX 8: SQLite WAL mode for better concurrency
        self._persist_conn.execute("PRAGMA journal_mode=WAL")
        self._persist_conn.execute("PRAGMA busy_timeout=5000")
        self._persist_conn.execute(
            "CREATE TABLE IF NOT EXISTS schwab_risk_state "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        self._persist_conn.commit()
        logger.info("SchwabClient persistence initialized: %s", db_path)

    def _save_risk_state(self) -> None:
        """Persist daily P&L and cooldown state to SQLite."""
        if self._persist_conn is None:
            return

        from datetime import datetime as _dt
        now = _dt.now().isoformat()
        state = {
            "daily_pnl": self._daily_pnl,
            "daily_pnl_date": self._daily_pnl_date.isoformat(),
            "consecutive_losses": self._consecutive_losses,
            "cooldown_until": self._cooldown_until,
        }

        with self._persist_lock:
            try:
                for key, value in state.items():
                    self._persist_conn.execute(
                        "INSERT OR REPLACE INTO schwab_risk_state (key, value, updated_at) "
                        "VALUES (?, ?, ?)",
                        (key, json.dumps(value), now),
                    )
                self._persist_conn.commit()
            except Exception as e:
                logger.error("Failed to save Schwab risk state: %s", e)

    def _load_risk_state(self) -> None:
        """Restore risk state from SQLite on startup."""
        if self._persist_conn is None:
            return

        with self._persist_lock:
            try:
                rows = self._persist_conn.execute(
                    "SELECT key, value FROM schwab_risk_state"
                ).fetchall()
            except Exception as e:
                logger.error("Failed to load Schwab risk state: %s", e)
                return

        if not rows:
            logger.info("No persisted Schwab risk state found — starting fresh")
            return

        state = {key: json.loads(value) for key, value in rows}

        saved_date_str = state.get("daily_pnl_date")
        if saved_date_str:
            saved_date = date.fromisoformat(saved_date_str)
            if saved_date == date.today():
                self._daily_pnl = float(state.get("daily_pnl", 0.0))
                self._daily_pnl_date = saved_date
            else:
                logger.info("Persisted Schwab state is from %s — resetting daily P&L", saved_date_str)

        self._consecutive_losses = int(state.get("consecutive_losses", 0))
        self._cooldown_until = float(state.get("cooldown_until", 0.0))

        logger.info(
            "Schwab risk state restored: daily_pnl=$%.2f, consecutive_losses=%d",
            self._daily_pnl, self._consecutive_losses,
        )

    # ─── OAuth2 Authentication ───

    def _authenticate(self) -> None:
        """Exchange refresh_token for access_token (thread-safe)."""
        import base64
        with self._token_lock:
            credentials = base64.b64encode(
                f"{self._client_id}:{self._client_secret}".encode()
            ).decode()

            headers = {
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
            }

            resp = self._session.post(self.AUTH_URL, headers=headers, data=data, timeout=30)
            if resp.status_code != 200:
                raise SchwabAPIError(
                    f"Authentication failed: {resp.text}",
                    status_code=resp.status_code, body=resp.text,
                )

            token_data = resp.json()
            token = token_data.get("access_token")
            if not token:
                raise SchwabAPIError(f"Auth failed: {token_data}")
            self._access_token = token
            self._token_expiry = time.time() + token_data.get("expires_in", 1800) - 60

            if "refresh_token" in token_data:
                self._refresh_token = token_data["refresh_token"]
                # FIX 4: Persist refresh token to disk
                self._persist_refresh_token()

            logger.info("Schwab authentication successful")

    def _persist_refresh_token(self) -> None:
        """Persist the current refresh token to disk."""
        try:
            token_path = Path.home() / ".stocks_plugin" / "schwab_refresh_token.txt"
            write_secret_file(token_path, self._refresh_token)
            logger.debug("Schwab refresh token persisted to %s", token_path)
        except Exception as e:
            logger.warning("Failed to persist Schwab refresh token: %s", e)

    def _ensure_token(self) -> None:
        """Refresh token if expired (thread-safe)."""
        if time.time() >= self._token_expiry:
            logger.info("Schwab token expired, refreshing...")
            self._authenticate()

    def _headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        self._ensure_token()
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ─── Rate Limiting ───

    def _rate_limit(self) -> None:
        """Enforce rate limit (2 requests/second)."""
        now = time.time()
        while self._request_times and self._request_times[0] < now - 1:
            self._request_times.popleft()

        if len(self._request_times) >= self.MAX_REQUESTS_PER_SECOND:
            sleep_for = 1.0 - (now - self._request_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)

        self._request_times.append(time.time())

    # ─── HTTP ───

    def _request(self, method: str, url: str, json_body: Any = None, params: Any = None) -> Any:
        """Make an authenticated API request."""
        self._rate_limit()
        headers = self._headers()

        try:
            resp = self._session.request(
                method, url, headers=headers, json=json_body, params=params, timeout=30,
            )
        except requests.RequestException as e:
            raise SchwabAPIError(f"Network error: {e}") from e

        if resp.status_code == 401:
            self._authenticate()
            headers = self._headers()
            resp = self._session.request(
                method, url, headers=headers, json=json_body, params=params, timeout=30,
            )

        if resp.status_code not in (200, 201, 204):
            raise SchwabAPIError(
                f"API error {resp.status_code}: {resp.text}",
                status_code=resp.status_code, body=resp.text,
            )

        if resp.text:
            try:
                return resp.json()
            except ValueError:
                return {}
        return {}

    # ─── Market Data ───

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get a real-time quote for a symbol.

        Returns:
            Dict with: lastPrice, bidPrice, askPrice, totalVolume, etc.
        """
        resp = self._request("GET", f"{self.MARKETDATA_URL}/{symbol}/quotes")
        if symbol in resp:
            quote = resp[symbol].get("quote", resp[symbol])
            return {
                "symbol": symbol,
                "lastPrice": quote.get("lastPrice", 0),
                "bidPrice": quote.get("bidPrice", 0),
                "askPrice": quote.get("askPrice", 0),
                "openPrice": quote.get("openPrice", 0),
                "highPrice": quote.get("highPrice", 0),
                "lowPrice": quote.get("lowPrice", 0),
                "totalVolume": quote.get("totalVolume", 0),
                "netChange": quote.get("netChange", 0),
                "netPercentChange": quote.get("netPercentChangeInDouble", 0),
            }
        return {"symbol": symbol, "lastPrice": 0}

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols."""
        symbol_str = ",".join(symbols)
        resp = self._request("GET", f"{self.MARKETDATA_URL}/quotes", params={"symbols": symbol_str})
        results = {}
        for sym in symbols:
            if sym in resp:
                quote = resp[sym].get("quote", resp[sym])
                results[sym] = {
                    "lastPrice": quote.get("lastPrice", 0),
                    "bidPrice": quote.get("bidPrice", 0),
                    "askPrice": quote.get("askPrice", 0),
                    "totalVolume": quote.get("totalVolume", 0),
                }
        return results

    def get_price_history(
        self,
        symbol: str,
        period_type: str = "month",
        period: int = 6,
        frequency_type: str = "daily",
        frequency: int = 1,
    ) -> List[Dict[str, Any]]:
        """Get historical price data (OHLCV candles).

        Args:
            symbol: Ticker symbol.
            period_type: "day", "month", "year", "ytd"
            period: Number of periods.
            frequency_type: "minute", "daily", "weekly", "monthly"
            frequency: Frequency interval.

        Returns:
            List of candle dicts with: open, high, low, close, volume, datetime.
        """
        params = {
            "periodType": period_type,
            "period": period,
            "frequencyType": frequency_type,
            "frequency": frequency,
        }
        resp = self._request("GET", f"{self.MARKETDATA_URL}/pricehistory", params={"symbol": symbol, **params})
        candles = resp.get("candles", [])
        return candles

    # ─── Orders ───

    def _submit_order(self, order_body: Dict[str, Any], account_id: str = None) -> str:
        """Submit an order and extract the order ID from the Location header.

        The Schwab API returns 201 with the order ID in the Location header,
        not in the response body.

        Args:
            order_body: Schwab order payload.
            account_id: Target account. Defaults to the client's own account.

        Returns:
            Order ID string.
        """
        self._rate_limit()
        headers = self._headers()
        acct = account_id or self._account_id
        url = f"{self.TRADER_URL}/accounts/{acct}/orders"

        try:
            resp = self._session.post(url, headers=headers, json=order_body, timeout=30)
        except requests.RequestException as e:
            raise SchwabAPIError(f"Network error: {e}") from e

        if resp.status_code == 401:
            self._authenticate()
            headers = self._headers()
            resp = self._session.post(url, headers=headers, json=order_body, timeout=30)

        if resp.status_code not in (200, 201, 204):
            raise SchwabAPIError(
                f"Order placement failed {resp.status_code}: {resp.text}",
                status_code=resp.status_code, body=resp.text,
            )

        order_id = resp.headers.get("Location", "").rsplit("/", 1)[-1]
        if not order_id:
            order_id = uuid.uuid4().hex[:12]
        return str(order_id)

    def place_market_order(self, symbol: str, action: str, quantity: int) -> str:
        """Place a market order.

        Args:
            symbol: Ticker symbol.
            action: "BUY" or "SELL".
            quantity: Number of shares.

        Returns:
            Order ID.

        Raises:
            SchwabAPIError: If daily loss limit reached or RiskManager blocks the trade.
        """
        if self._daily_pnl <= -self._max_daily_loss:
            raise SchwabAPIError(
                f"Daily loss limit reached (P&L: ${self._daily_pnl:.2f}, limit: -${self._max_daily_loss:.2f}). "
                "No new trades allowed. Call reset_daily_pnl() to reset."
            )

        if self._risk_manager is not None:
            if not self._risk_manager.can_trade(symbol):
                raise SchwabAPIError(
                    f"RiskManager blocked trade for {symbol}"
                )

        order_body = {
            "orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": action.upper(),
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "EQUITY",
                    },
                }
            ],
        }

        logger.info("Schwab: placing market %s order for %d %s", action, quantity, symbol)
        order_id = self._submit_order(order_body)
        logger.info("Schwab market order placed: %s", order_id)
        return order_id

    def place_limit_order(
        self, symbol: str, action: str, quantity: int, price: float,
    ) -> str:
        """Place a limit order.

        Returns:
            Order ID.

        Raises:
            SchwabAPIError: If daily loss limit reached or RiskManager blocks the trade.
        """
        if self._daily_pnl <= -self._max_daily_loss:
            raise SchwabAPIError(
                f"Daily loss limit reached (P&L: ${self._daily_pnl:.2f}, limit: -${self._max_daily_loss:.2f}). "
                "No new trades allowed. Call reset_daily_pnl() to reset."
            )

        if self._risk_manager is not None:
            if not self._risk_manager.can_trade(symbol):
                raise SchwabAPIError(
                    f"RiskManager blocked trade for {symbol}"
                )

        order_body = {
            "orderType": "LIMIT",
            "session": "NORMAL",
            "price": round(float(price), 2),
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": action.upper(),
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "EQUITY",
                    },
                }
            ],
        }

        logger.info("Schwab: placing limit %s %d %s @ $%.2f", action, quantity, symbol, order_body["price"])
        order_id = self._submit_order(order_body)
        logger.info("Schwab limit order placed: %s", order_id)
        return order_id

    def place_stop_order(self, symbol: str, action: str, quantity: int, stop_price: float, account_id: str = None) -> str:
        """Place a stop-market order via Schwab API."""
        acct = account_id or self._account_id
        order_body = {
            "orderType": "STOP",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "stopPrice": round(float(stop_price), 2),
            "orderLegCollection": [{
                "instruction": action.upper(),
                "quantity": quantity,
                "instrument": {"symbol": symbol, "assetType": "EQUITY"}
            }]
        }
        logger.info("Schwab: placing stop %s %d %s @ $%.2f", action, quantity, symbol, order_body["stopPrice"])
        order_id = self._submit_order(order_body, account_id=acct)
        logger.info("Schwab stop order placed: %s", order_id)
        return order_id

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order."""
        logger.info("Schwab: cancelling order %s", order_id)
        self._request(
            "DELETE",
            f"{self.TRADER_URL}/accounts/{self._account_id}/orders/{order_id}",
        )

    def get_orders(self, status: str = "WORKING") -> List[Dict[str, Any]]:
        """Get orders for the account.

        Args:
            status: Filter by status — "WORKING", "FILLED", "CANCELED", etc.

        Returns:
            List of order dicts.
        """
        resp = self._request(
            "GET",
            f"{self.TRADER_URL}/accounts/{self._account_id}/orders",
            params={"status": status},
        )
        return resp if isinstance(resp, list) else resp.get("orders", [])

    # ─── Account ───

    def get_account_info(self) -> Dict[str, Any]:
        """Get account balances and summary."""
        resp = self._request(
            "GET",
            f"{self.TRADER_URL}/accounts/{self._account_id}",
            params={"fields": "positions"},
        )
        account = resp.get("securitiesAccount", resp)
        balances = account.get("currentBalances", {})
        return {
            "account_id": self._account_id,
            "account_type": account.get("type", ""),
            "net_liquidation": balances.get("liquidationValue", 0),
            "cash_balance": balances.get("cashBalance", 0),
            "buying_power": balances.get("buyingPower", 0),
            "equity": balances.get("equity", 0),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions.

        Returns:
            List of position dicts with: symbol, quantity, avgPrice, marketValue, pnl.
        """
        resp = self._request(
            "GET",
            f"{self.TRADER_URL}/accounts/{self._account_id}",
            params={"fields": "positions"},
        )
        account = resp.get("securitiesAccount", resp)
        positions = account.get("positions", [])

        return [
            {
                "symbol": p.get("instrument", {}).get("symbol", ""),
                "quantity": p.get("longQuantity", 0) - p.get("shortQuantity", 0),
                "avg_price": p.get("averagePrice", 0),
                "market_value": p.get("marketValue", 0),
                "current_price": p.get("currentPrice", p.get("marketValue", 0)),
                "daily_pnl_pct": p.get("currentDayProfitLossPercentage", 0),
                "pnl": p.get("currentDayProfitLoss", 0),
            }
            for p in positions
        ]

    # ─── Daily P&L Tracking ───

    def record_trade_pnl(self, symbol: str, pnl: float, quantity: int = 0) -> None:
        """Record a trade's P&L and update daily tracking.

        Args:
            symbol: Ticker symbol.
            pnl: Realized profit/loss for the trade.
            quantity: Number of shares traded.
        """
        self._daily_pnl += pnl

        # Consecutive loss tracking
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        logger.info(
            "Trade P&L recorded: %s %.2f (daily total: %.2f, consecutive_losses: %d)",
            symbol, pnl, self._daily_pnl, self._consecutive_losses,
        )

        if self._risk_manager is not None:
            # RiskManager.record_trade takes (symbol, pnl) only. Passing
            # quantity= raised TypeError on every call, and the except below
            # downgraded it to a warning — so the shared risk state (daily
            # P&L, loss streak, cooldown, equity) never moved.
            try:
                self._risk_manager.record_trade(symbol=symbol, pnl=pnl)
            except Exception as e:
                logger.warning("RiskManager.record_trade failed: %s", e)

        if self._daily_pnl <= -self._max_daily_loss:
            logger.warning(
                "DAILY LOSS LIMIT REACHED: P&L $%.2f exceeds -$%.2f. Trading halted.",
                self._daily_pnl, self._max_daily_loss,
            )

        self._save_risk_state()

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L tracking (call at start of each trading day)."""
        logger.info("Daily P&L reset from $%.2f to $0.00", self._daily_pnl)
        self._daily_pnl = 0.0

    @property
    def daily_pnl(self) -> float:
        """Current daily P&L."""
        return self._daily_pnl

    @property
    def is_trading_halted(self) -> bool:
        """True if daily loss limit has been reached."""
        return self._daily_pnl <= -self._max_daily_loss

    # ─── Options Greeks ───

    def get_option_chain(self, symbol: str, strike_count: int = 10) -> Dict[str, Any]:
        """Fetch option chain with Greeks from Schwab API.

        Args:
            symbol: Underlying ticker symbol.
            strike_count: Number of strikes above/below ATM to include.

        Returns:
            Dict with 'calls' and 'puts' lists, each entry containing
            strike, expiry, bid, ask, last, volume, openInterest,
            and greeks (delta, gamma, theta, vega, impliedVolatility).
        """
        params = {
            "symbol": symbol,
            "strikeCount": strike_count,
            "includeUnderlyingQuote": True,
            "strategy": "SINGLE",
        }

        resp = self._request("GET", f"{self.MARKETDATA_URL}/chains", params=params)

        calls: List[Dict[str, Any]] = []
        puts: List[Dict[str, Any]] = []

        for exp_date, strikes in resp.get("callExpDateMap", {}).items():
            for strike_str, contracts in strikes.items():
                for c in contracts:
                    calls.append({
                        "symbol": c.get("symbol", ""),
                        "strike": float(strike_str),
                        "expiry": exp_date.split(":")[0],
                        "bid": c.get("bid", 0),
                        "ask": c.get("ask", 0),
                        "last": c.get("last", 0),
                        "volume": c.get("totalVolume", 0),
                        "openInterest": c.get("openInterest", 0),
                        "greeks": {
                            "delta": c.get("delta", 0),
                            "gamma": c.get("gamma", 0),
                            "theta": c.get("theta", 0),
                            "vega": c.get("vega", 0),
                            "impliedVolatility": c.get("volatility", 0),
                        },
                    })

        for exp_date, strikes in resp.get("putExpDateMap", {}).items():
            for strike_str, contracts in strikes.items():
                for c in contracts:
                    puts.append({
                        "symbol": c.get("symbol", ""),
                        "strike": float(strike_str),
                        "expiry": exp_date.split(":")[0],
                        "bid": c.get("bid", 0),
                        "ask": c.get("ask", 0),
                        "last": c.get("last", 0),
                        "volume": c.get("totalVolume", 0),
                        "openInterest": c.get("openInterest", 0),
                        "greeks": {
                            "delta": c.get("delta", 0),
                            "gamma": c.get("gamma", 0),
                            "theta": c.get("theta", 0),
                            "vega": c.get("vega", 0),
                            "impliedVolatility": c.get("volatility", 0),
                        },
                    })

        logger.info(
            "Option chain for %s: %d calls, %d puts",
            symbol, len(calls), len(puts),
        )
        return {"symbol": symbol, "calls": calls, "puts": puts}

    @staticmethod
    def _is_option_position(position: Dict[str, Any]) -> bool:
        """Check if a position is an options position.

        Args:
            position: Position dict from get_positions().

        Returns:
            True if the position symbol matches option format (e.g. AAPL_012025C150).
        """
        symbol = position.get("symbol", "")
        # Schwab option symbols contain underscores and end with C/P + strike
        return bool(symbol) and ("_" in symbol or len(symbol) > 10)

    def get_portfolio_greeks(self) -> Dict[str, Any]:
        """Calculate aggregated portfolio Greeks across all option positions.

        Fetches current positions, identifies option positions, retrieves
        their Greeks from the option chain, and returns aggregated totals.

        Returns:
            Dict with per-position greeks and aggregated portfolio-level
            delta, gamma, theta, and vega.
        """
        positions = self.get_positions()
        option_positions = [p for p in positions if self._is_option_position(p)]

        if not option_positions:
            logger.info("No option positions found in portfolio")
            return {
                "positions": [],
                "portfolio_delta": 0.0,
                "portfolio_gamma": 0.0,
                "portfolio_theta": 0.0,
                "portfolio_vega": 0.0,
            }

        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        position_greeks: List[Dict[str, Any]] = []

        # Group by underlying symbol to minimize API calls
        underlyings: Dict[str, List[Dict[str, Any]]] = {}
        for pos in option_positions:
            underlying = pos["symbol"].split("_")[0] if "_" in pos["symbol"] else pos["symbol"][:4]
            underlyings.setdefault(underlying, []).append(pos)

        for underlying, opts in underlyings.items():
            try:
                chain = self.get_option_chain(underlying)
                all_contracts = chain.get("calls", []) + chain.get("puts", [])
                contract_map = {c["symbol"]: c for c in all_contracts}

                for pos in opts:
                    contract = contract_map.get(pos["symbol"])
                    qty = pos.get("quantity", 0)
                    multiplier = 100

                    if contract:
                        greeks = contract.get("greeks", {})
                        pos_delta = greeks.get("delta", 0) * qty * multiplier
                        pos_gamma = greeks.get("gamma", 0) * qty * multiplier
                        pos_theta = greeks.get("theta", 0) * qty * multiplier
                        pos_vega = greeks.get("vega", 0) * qty * multiplier

                        total_delta += pos_delta
                        total_gamma += pos_gamma
                        total_theta += pos_theta
                        total_vega += pos_vega

                        position_greeks.append({
                            "symbol": pos["symbol"],
                            "underlying": underlying,
                            "quantity": qty,
                            "delta": round(pos_delta, 4),
                            "gamma": round(pos_gamma, 4),
                            "theta": round(pos_theta, 4),
                            "vega": round(pos_vega, 4),
                            "iv": greeks.get("impliedVolatility", 0),
                        })
                    else:
                        logger.warning(
                            "Greeks not found for option position %s",
                            pos["symbol"],
                        )
                        position_greeks.append({
                            "symbol": pos["symbol"],
                            "underlying": underlying,
                            "quantity": qty,
                            "delta": 0, "gamma": 0, "theta": 0, "vega": 0,
                            "iv": 0,
                            "warning": "contract not found in chain",
                        })
            except Exception as e:
                logger.error("Failed to fetch Greeks for %s: %s", underlying, e)

        result = {
            "positions": position_greeks,
            "portfolio_delta": round(total_delta, 4),
            "portfolio_gamma": round(total_gamma, 4),
            "portfolio_theta": round(total_theta, 4),
            "portfolio_vega": round(total_vega, 4),
        }
        logger.info(
            "Portfolio Greeks: Δ=%.2f Γ=%.4f Θ=%.2f ν=%.2f",
            total_delta, total_gamma, total_theta, total_vega,
        )
        return result

    def __repr__(self) -> str:
        return f"SchwabClient(account={self._account_id[:8]}...)"
