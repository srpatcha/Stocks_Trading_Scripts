"""
TradingView Webhook Server
FastAPI server that receives TradingView alert webhooks, validates them,
and routes orders to configured broker adapters.

stocks_plugin - tradingview/webhooks/webhook_server.py
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import re
import statistics
import threading
import time
import traceback
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ─── Graceful imports from shared modules ───
try:
    from shared.config import Config
except ImportError:
    Config = None

try:
    from shared.notifier import AlertDispatcher
except ImportError:
    AlertDispatcher = None

# ─── Logging Setup ───
logger = logging.getLogger("webhook_server")

# ─── Sector Mapping for Risk Limits (Fix 8: import from unified sector map) ───
try:
    from shared.config.sector_map import SECTOR_MAP
except ImportError:
    SECTOR_MAP: Dict[str, str] = {
        "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
        "GOOG": "Technology", "META": "Technology", "AMZN": "Technology",
        "NVDA": "Technology", "TSM": "Technology", "AVGO": "Technology",
        "ORCL": "Technology", "CRM": "Technology", "ADBE": "Technology",
        "AMD": "Technology", "INTC": "Technology", "CSCO": "Technology",
        "QCOM": "Technology", "IBM": "Technology", "TXN": "Technology",
        "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
        "GS": "Financials", "MS": "Financials", "C": "Financials",
        "BLK": "Financials", "SCHW": "Financials", "AXP": "Financials",
        "V": "Financials", "MA": "Financials", "PYPL": "Financials",
        "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
        "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
        "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
        "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
        "NKE": "Consumer Discretionary", "MCD": "Consumer Discretionary",
        "SBUX": "Consumer Discretionary", "LOW": "Consumer Discretionary",
        "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
        "COST": "Consumer Staples", "WMT": "Consumer Staples",
        "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
        "SLB": "Energy", "EOG": "Energy", "OXY": "Energy",
        "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
        "HON": "Industrials", "UPS": "Industrials", "RTX": "Industrials",
        "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
        "DIS": "Communication Services", "NFLX": "Communication Services",
        "CMCSA": "Communication Services", "T": "Communication Services",
        "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
        "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
        "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    }



# ─── Client identity and request limits ───

#: Reject bodies larger than this before reading them. An alert is a few
#: hundred bytes; `await request.body()` buffered the whole payload into
#: memory before the HMAC was even computed, so an unauthenticated caller
#: could stream an arbitrarily large body at the process.
MAX_BODY_BYTES = 64 * 1024


def _parse_allowlist(entries: List[str]):
    """Parse allowlist entries into networks. Accepts plain IPs and CIDRs.

    The allowlist was exact string matching, so a CIDR could not be expressed
    at all — and TradingView publishes ranges, not a fixed set of addresses.
    """
    networks = []
    for raw in entries or []:
        text = str(raw).strip()
        if not text:
            continue
        try:
            networks.append(ipaddress.ip_network(text, strict=False))
        except ValueError:
            logger.error("Ignoring malformed allowlist entry %r", text)
    return networks


def client_ip_of(request: Request, config: dict) -> str:
    """Best available client address.

    ``request.client.host`` is the immediate peer. Behind the reverse proxy
    this server needs anyway (it speaks plaintext HTTP and the passphrase is
    in the body), that is the PROXY's address — so the IP allowlist admitted
    everyone or no one, and every client shared a single rate-limit bucket.

    X-Forwarded-For is only honoured when the peer is a configured trusted
    proxy; otherwise any caller could spoof their way past the allowlist by
    setting a header.
    """
    peer = request.client.host if request.client else "unknown"
    trusted = _parse_allowlist(
        config.get("security", {}).get("trusted_proxies", []))
    if not trusted:
        return peer
    try:
        peer_addr = ipaddress.ip_address(peer)
    except ValueError:
        return peer
    if not any(peer_addr in net for net in trusted):
        return peer

    forwarded = request.headers.get("X-Forwarded-For", "")
    if not forwarded:
        return peer
    # Left-most entry is the original client; the proxy appends as it forwards.
    candidate = forwarded.split(",")[0].strip()
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        logger.warning("Malformed X-Forwarded-For from trusted proxy: %r", forwarded)
        return peer
    return candidate


def ip_allowed(ip: str, allowed_entries: List[str]) -> bool:
    """True when ``ip`` falls inside the allowlist, or the list is empty."""
    networks = _parse_allowlist(allowed_entries)
    if not networks:
        return True
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in net for net in networks)


# ─── Configuration Loading ───
def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")

    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return _default_config()
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config: {e}")
        return _default_config()


def _default_config() -> dict:
    """Return default configuration.

    Binds to loopback. This server terminates plaintext HTTP and the webhook
    passphrase travels in the request body, so exposing it directly is only
    safe behind a TLS-terminating reverse proxy. Set server.host explicitly in
    config.yaml once that proxy is in place.
    """
    return {
        "server": {"host": "127.0.0.1", "port": 5000, "debug": False},
        "security": {
            "hmac_secret": "",
            "hmac_algorithm": "sha256",
            "require_hmac": True,
            "allowed_ips": ["127.0.0.1"],
            "require_passphrase": False,
            "passphrase": "",
            "cors_origins": [],
        },
        "rate_limiting": {
            "enabled": True,
            "max_requests_per_minute": 60,
            "window_seconds": 60,
        },
        "broker_routing": {"default_broker": "interactive_brokers", "routes": []},
        "logging": {"level": "INFO", "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        "risk_management": {
            "max_daily_loss": 5000.0,
            "daily_reset_hour_utc": 0,
            "max_consecutive_losses": 3,
            "cooldown_minutes": 30,
            "max_drawdown_pct": 10.0,
            "drawdown_lockout_hours": 24,
            "max_trade_value": 50000.0,
        },
        "health_monitor": {
            "enabled": True,
            "ping_url": "",
            "ping_interval_seconds": 60,
            "max_silence_minutes": 30,
        },
    }


def setup_logging(config: dict) -> None:
    """Configure logging from config."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handlers = []

    if log_config.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)

    log_file = log_config.get("file")
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=log_config.get("max_bytes", 10485760),
            backupCount=log_config.get("backup_count", 5),
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    logging.basicConfig(level=log_level, handlers=handlers, force=True)


# ─── Pydantic Models ───

# Absolute ceiling on a single inbound order. The dollar cap in
# _check_risk_gates is the real control; this is a schema-level backstop so a
# malformed or hostile payload cannot ask for an unbounded share count.
MAX_ORDER_QUANTITY = 1_000_000.0


class AlertPayload(BaseModel):
    """TradingView alert webhook payload."""
    symbol: str = Field(
        ..., min_length=1, max_length=12, pattern=r"^[A-Za-z0-9.\-:]+$",
        description="Trading symbol (e.g., AAPL)",
    )
    action: str = Field(..., description="Trade action: buy, sell, close")
    # gt=0, not ge=0. With price=0 the notional cap computed
    # quantity * 0 == 0, which never exceeds max_trade_value, so the position
    # size limit was bypassed entirely and the order stayed a market order.
    price: float = Field(..., gt=0, description="Current price")
    quantity: Optional[float] = Field(
        None, gt=0, le=MAX_ORDER_QUANTITY, description="Order quantity",
    )
    order_type: str = Field("market", description="Order type: market, limit, stop")
    passphrase: Optional[str] = Field(None, description="Webhook passphrase for auth")
    timestamp: Optional[str] = Field(None, description="Alert timestamp")
    strategy: Optional[str] = Field(None, description="Strategy name")
    timeframe: Optional[str] = Field(None, description="Chart timeframe")
    message: Optional[str] = Field(None, description="Additional message")
    regime: Optional[str] = Field(None, description="Market regime: TRENDING, RANGING, VOLATILE")
    signal: Optional[str] = Field(None, description="Signal type from strategy")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    uptime_seconds: float
    last_alert_time: Optional[str]
    total_alerts_processed: int
    version: str


class OrderResult(BaseModel):
    """Result of order execution attempt."""
    success: bool
    broker: str
    order_id: Optional[str] = None
    message: str
    timestamp: str


# ─── Broker Adapter Pattern ───
class BrokerAdapter(ABC):
    """Abstract base class for broker adapters."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker."""
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        order_type: str,
        price: float,
    ) -> OrderResult:
        """Place an order with the broker."""
        ...

    @abstractmethod
    def get_account_info(self) -> dict:
        """Get account information."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Broker adapter name."""
        ...


class IBBrokerAdapter(BrokerAdapter):
    """Interactive Brokers adapter — delegates to the real IBAdapter from broker_bridge."""

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._adapter = None
        self._init_adapter()

    def _init_adapter(self) -> None:
        try:
            import sys, os
            _ib_path = os.path.join(os.path.dirname(__file__), "..", "..")
            if _ib_path not in sys.path:
                sys.path.insert(0, _ib_path)
            from shared.daemon.broker_bridge import IBAdapter
            self._adapter = IBAdapter(
                host=self._config.get("host", "127.0.0.1"),
                port=self._config.get("port", 7497),
                client_id=self._config.get("client_id", 1),
            )
        except Exception as e:
            logger.warning("IBBrokerAdapter init failed: %s", e)

    @property
    def name(self) -> str:
        return "interactive_brokers"

    def connect(self) -> bool:
        if self._adapter is None:
            self._init_adapter()
        if self._adapter:
            return self._adapter.connect()
        return False

    def disconnect(self) -> None:
        if self._adapter:
            self._adapter.disconnect()

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        order_type: str,
        price: float,
    ) -> OrderResult:
        if self._adapter is None:
            return OrderResult(
                success=False, broker=self.name, order_id=None,
                message="IBAdapter not initialised",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        if not self._adapter.is_connected():
            if not self._adapter.connect():
                return OrderResult(
                    success=False, broker=self.name, order_id=None,
                    message="Could not connect to IB Gateway",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
        try:
            if order_type == "limit" and price > 0:
                result = self._adapter.place_limit_order(symbol, action.upper(), int(quantity), price)
            else:
                result = self._adapter.place_market_order(symbol, action.upper(), int(quantity))

            return OrderResult(
                success=result.success,
                broker=self.name,
                order_id=result.order_id or None,
                message=result.message,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            logger.error("IB place_order error: %s", e)
            return OrderResult(
                success=False, broker=self.name, order_id=None, message=str(e),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    def get_account_info(self) -> dict:
        if self._adapter and self._adapter.is_connected():
            return self._adapter.get_account_info()
        return {"broker": self.name, "connected": False}


class TradeStationBrokerAdapter(BrokerAdapter):
    """TradeStation adapter — delegates to the real TradeStationAdapter from broker_bridge."""

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._adapter = None
        self._init_adapter()

    def _init_adapter(self) -> None:
        try:
            import sys, os
            _ts_path = os.path.join(os.path.dirname(__file__), "..", "..")
            if _ts_path not in sys.path:
                sys.path.insert(0, _ts_path)
            from shared.daemon.broker_bridge import TradeStationAdapter
            self._adapter = TradeStationAdapter(
                self._config,
                self._config.get("account_id", ""),
            )
        except Exception as e:
            logger.warning("TradeStationBrokerAdapter init failed: %s", e)

    @property
    def name(self) -> str:
        return "tradestation"

    def connect(self) -> bool:
        if self._adapter is None:
            self._init_adapter()
        if self._adapter:
            return self._adapter.connect()
        return False

    def disconnect(self) -> None:
        if self._adapter:
            self._adapter.disconnect()

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        order_type: str,
        price: float,
    ) -> OrderResult:
        if self._adapter is None:
            return OrderResult(
                success=False, broker=self.name, order_id=None,
                message="TradeStationAdapter not initialised",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        if not self._adapter.is_connected():
            if not self._adapter.connect():
                return OrderResult(
                    success=False, broker=self.name, order_id=None,
                    message="Could not connect to TradeStation",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
        try:
            if order_type == "limit" and price > 0:
                result = self._adapter.place_limit_order(symbol, action.upper(), int(quantity), price)
            else:
                result = self._adapter.place_market_order(symbol, action.upper(), int(quantity))

            return OrderResult(
                success=result.success,
                broker=self.name,
                order_id=result.order_id or None,
                message=result.message,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            logger.error("TradeStation place_order error: %s", e)
            return OrderResult(
                success=False, broker=self.name, order_id=None, message=str(e),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    def get_account_info(self) -> dict:
        if self._adapter:
            return self._adapter.get_account_info()
        return {"broker": self.name, "connected": False}


class SchwabBrokerAdapter(BrokerAdapter):
    """Schwab/thinkorswim adapter — delegates to the real SchwabAdapter from broker_bridge."""

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._adapter = None
        self._init_adapter()

    def _init_adapter(self) -> None:
        try:
            import sys, os
            _schwab_path = os.path.join(os.path.dirname(__file__), "..", "..")
            if _schwab_path not in sys.path:
                sys.path.insert(0, _schwab_path)
            from shared.daemon.broker_bridge import SchwabAdapter
            self._adapter = SchwabAdapter(self._config)
        except Exception as e:
            logger.warning("SchwabBrokerAdapter init failed: %s", e)

    @property
    def name(self) -> str:
        return "schwab"

    def connect(self) -> bool:
        if self._adapter is None:
            self._init_adapter()
        if self._adapter:
            return self._adapter.connect()
        return False

    def disconnect(self) -> None:
        if self._adapter:
            self._adapter.disconnect()

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        order_type: str,
        price: float,
    ) -> OrderResult:
        if self._adapter is None:
            return OrderResult(
                success=False, broker=self.name, order_id=None,
                message="SchwabAdapter not initialised",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        if not self._adapter.is_connected():
            if not self._adapter.connect():
                return OrderResult(
                    success=False, broker=self.name, order_id=None,
                    message="Could not connect to Schwab",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
        try:
            if order_type == "limit" and price > 0:
                result = self._adapter.place_limit_order(symbol, action.upper(), int(quantity), price)
            else:
                result = self._adapter.place_market_order(symbol, action.upper(), int(quantity))

            return OrderResult(
                success=result.success,
                broker=self.name,
                order_id=result.order_id or None,
                message=result.message,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            logger.error("Schwab place_order error: %s", e)
            return OrderResult(
                success=False, broker=self.name, order_id=None, message=str(e),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    def get_account_info(self) -> dict:
        if self._adapter:
            return self._adapter.get_account_info()
        return {"broker": self.name, "connected": False}


# ─── Rate Limiter ───
class RateLimiter:
    """Per-IP rate limiter using sliding window.

    LIMITATION: This rate limiter uses an in-memory dict that is NOT shared
    across multiple worker processes (e.g. uvicorn --workers N). Each worker
    maintains its own counter, so effective rate limits are multiplied by the
    worker count. For multi-worker deployments, switch to a Redis-based rate
    limiter (e.g. slowapi with Redis backend) or a shared file lock counter.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request from client_ip is within rate limit."""
        now = time.time()
        cutoff = now - self.window_seconds

        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > cutoff
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            return False

        self._requests[client_ip].append(now)
        return True

    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for client_ip."""
        now = time.time()
        cutoff = now - self.window_seconds
        current = len([t for t in self._requests.get(client_ip, []) if t > cutoff])
        return max(0, self.max_requests - current)


# ─── Broker Router ───
class BrokerRouter:
    """Routes symbols to appropriate broker adapters based on config patterns."""

    def __init__(self, config: dict):
        self._routes = config.get("broker_routing", {}).get("routes", [])
        self._default = config.get("broker_routing", {}).get(
            "default_broker", "interactive_brokers"
        )
        broker_configs = config.get("broker_configs", {})
        self._adapters: dict[str, BrokerAdapter] = {
            "interactive_brokers": IBBrokerAdapter(broker_configs.get("interactive_brokers", {})),
            "tradestation": TradeStationBrokerAdapter(broker_configs.get("tradestation", {})),
            "schwab": SchwabBrokerAdapter(broker_configs.get("schwab", {})),
            "thinkorswim": SchwabBrokerAdapter(broker_configs.get("schwab", {})),
        }

    def get_broker(self, symbol: str) -> BrokerAdapter:
        """Get the appropriate broker adapter for a symbol."""
        for route in self._routes:
            pattern = route.get("pattern", "")
            broker_name = route.get("broker", self._default)
            if re.match(pattern, symbol, re.IGNORECASE):
                adapter = self._adapters.get(broker_name)
                if adapter:
                    logger.debug(
                        f"Routing {symbol} to {broker_name} (pattern: {pattern})"
                    )
                    return adapter

        default_adapter = self._adapters.get(self._default)
        if default_adapter:
            return default_adapter

        raise ValueError(f"No broker adapter found for symbol: {symbol}")

    def get_all_brokers(self) -> dict[str, BrokerAdapter]:
        """Return all registered broker adapters."""
        return self._adapters


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 1 — HealthMonitor (heartbeat + silence detection)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HealthMonitor:
    """Background daemon that pings a health-check URL, warns on silence,
    tracks alert-to-order latency, and detects stale TradingView alerts."""

    def __init__(
        self,
        ping_url: str = "",
        ping_interval_seconds: int = 60,
        max_silence_minutes: int = 30,
        alert_dispatcher: Any = None,
        alert_timeout_hours: float = 12.0,
    ):
        self._ping_url = ping_url
        self._ping_interval = ping_interval_seconds
        self._max_silence = timedelta(minutes=max_silence_minutes)
        self._dispatcher = alert_dispatcher
        self.alerts_processed: int = 0
        self.last_alert_time: Optional[datetime] = None
        self._start_time = time.time()
        self._silence_warned = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # GAP 3: Latency tracking
        self._latency_history: deque = deque(maxlen=100)
        self._last_latency_ms: Optional[float] = None

        # GAP 6: Alert freshness / expiration detection
        self._alert_timeout = timedelta(hours=alert_timeout_hours)
        self._alert_freshness_warned = False

    def record_alert(self) -> None:
        self.alerts_processed += 1
        self.last_alert_time = datetime.now(timezone.utc)
        self._silence_warned = False
        self._alert_freshness_warned = False

    def record_latency(self, latency_ms: float) -> None:
        """Record alert-to-order latency in milliseconds."""
        self._last_latency_ms = latency_ms
        self._latency_history.append(latency_ms)
        if len(self._latency_history) >= 10:
            avg = statistics.mean(self._latency_history)
            if avg > 2000:
                logger.warning(
                    "HIGH LATENCY WARNING: average alert-to-order latency %.0fms "
                    "(last %d samples) exceeds 2000ms threshold",
                    avg, len(self._latency_history),
                )

    def get_latency_stats(self) -> Dict[str, Any]:
        """Return rolling latency statistics."""
        if not self._latency_history:
            return {"samples": 0}
        data = list(self._latency_history)
        return {
            "samples": len(data),
            "last_ms": round(data[-1], 1),
            "avg_ms": round(statistics.mean(data), 1),
            "min_ms": round(min(data), 1),
            "max_ms": round(max(data), 1),
            "p95_ms": round(sorted(data)[int(len(data) * 0.95)], 1) if len(data) >= 2 else round(data[-1], 1),
        }

    def check_alert_freshness(self) -> Dict[str, str]:
        """Check if alerts are arriving within the expected timeout window.

        Returns a dict with ``status`` ("ok" or "stale") and a human message.
        """
        if self.last_alert_time is None:
            elapsed = timedelta(seconds=self.uptime_seconds)
        else:
            elapsed = datetime.now(timezone.utc) - self.last_alert_time

        if elapsed > self._alert_timeout:
            hours = elapsed.total_seconds() / 3600
            msg = f"stale - no alerts for {hours:.1f}h"
            if not self._alert_freshness_warned:
                self._alert_freshness_warned = True
                warn_msg = (
                    f"⚠️ Possible alert expiration detected: no alerts received for "
                    f"{hours:.1f}h (timeout={self._alert_timeout.total_seconds()/3600:.0f}h). "
                    f"Check TradingView alert settings."
                )
                logger.warning(warn_msg)
                if self._dispatcher:
                    try:
                        self._dispatcher.dispatch(
                            title="Alert Freshness Warning",
                            message=warn_msg,
                            priority="WARNING",
                        )
                    except Exception:
                        pass
            return {"status": "stale", "detail": msg}
        return {"status": "ok", "detail": "ok"}

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="HealthMonitor")
        self._thread.start()
        logger.info("HealthMonitor started (ping_interval=%ds, max_silence=%s)",
                     self._ping_interval, self._max_silence)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._ping_health_url()
                self._check_silence()
                self.check_alert_freshness()
            except Exception:
                logger.debug("HealthMonitor tick error: %s", traceback.format_exc())
            self._stop_event.wait(timeout=self._ping_interval)

    def _ping_health_url(self) -> None:
        if not self._ping_url:
            return
        # urlopen honours file:, ftp: and custom schemes, so a mistyped or
        # tampered ping_url could read a local file rather than make an HTTP
        # request. Restrict it to the two schemes this is meant to use.
        scheme = urllib.parse.urlparse(self._ping_url).scheme.lower()
        if scheme not in ("http", "https"):
            logger.error(
                "HealthMonitor ping_url must be http or https, got %r — not pinging",
                scheme or "(none)",
            )
            return
        try:
            req = urllib.request.Request(self._ping_url, method="GET")
            with urllib.request.urlopen(req, timeout=10):  # nosec B310 - scheme checked above
                pass
        except Exception as exc:
            logger.warning("HealthMonitor ping failed (%s): %s", self._ping_url, exc)

    def _check_silence(self) -> None:
        if self._silence_warned:
            return
        if self.last_alert_time is None:
            elapsed = timedelta(seconds=self.uptime_seconds)
        else:
            elapsed = datetime.now(timezone.utc) - self.last_alert_time
        if elapsed > self._max_silence:
            self._silence_warned = True
            msg = (f"⚠️ No webhook alerts received for {elapsed}. "
                   f"Last alert: {self.last_alert_time or 'never'}")
            logger.warning(msg)
            if self._dispatcher:
                try:
                    self._dispatcher.dispatch(title="Webhook Silence Warning",
                                              message=msg, priority="WARNING")
                except Exception:
                    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 2 — DailyPnLTracker (daily loss-limit enforcement)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DailyPnLTracker:
    """Tracks cumulative daily P&L and blocks trading when the loss limit is hit."""

    def __init__(self, max_daily_loss: float = 5000.0, reset_hour_utc: int = 0):
        self._max_daily_loss = max_daily_loss
        self._reset_hour_utc = reset_hour_utc
        self._daily_pnl: float = 0.0
        self._trade_count: int = 0
        self._last_reset_date: Optional[datetime] = None
        self._lock = threading.Lock()
        self._reset_if_new_day()

    def record_trade(self, symbol: str, pnl: float) -> None:
        with self._lock:
            self._reset_if_new_day()
            self._daily_pnl += pnl
            self._trade_count += 1
            logger.info("DailyPnL: %s %.2f → cumulative %.2f (limit %.2f)",
                        symbol, pnl, self._daily_pnl, self._max_daily_loss)

    def can_trade(self) -> bool:
        with self._lock:
            self._reset_if_new_day()
            if self._daily_pnl < 0 and abs(self._daily_pnl) >= self._max_daily_loss:
                logger.warning("Daily loss limit reached: %.2f (max %.2f)",
                               self._daily_pnl, self._max_daily_loss)
                return False
            return True

    def reset_daily(self) -> None:
        with self._lock:
            prev = self._daily_pnl
            self._daily_pnl = 0.0
            self._trade_count = 0
            self._last_reset_date = datetime.now(timezone.utc).date()
            logger.info("DailyPnL reset (previous cumulative: %.2f)", prev)

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def trade_count(self) -> int:
        return self._trade_count

    def _reset_if_new_day(self) -> None:
        now = datetime.now(timezone.utc)
        reset_boundary = now.replace(
            hour=self._reset_hour_utc, minute=0, second=0, microsecond=0)
        today = (reset_boundary.date() if now >= reset_boundary
                 else (reset_boundary - timedelta(days=1)).date())
        if self._last_reset_date is None or today > self._last_reset_date:
            self._daily_pnl = 0.0
            self._trade_count = 0
            self._last_reset_date = today


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 3 — CooldownManager (per-strategy consecutive-loss cooldown)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CooldownManager:
    """Enforces a trading cooldown after N consecutive losses for a strategy."""

    def __init__(self, max_consecutive_losses: int = 3, cooldown_minutes: int = 30):
        self._max_losses = max_consecutive_losses
        self._cooldown_duration = timedelta(minutes=cooldown_minutes)
        self._loss_streaks: Dict[str, int] = defaultdict(int)
        self._cooldown_until: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def record_result(self, strategy: str, won: bool) -> None:
        with self._lock:
            if won:
                self._loss_streaks[strategy] = 0
            else:
                self._loss_streaks[strategy] += 1
                streak = self._loss_streaks[strategy]
                logger.info("CooldownManager: %s loss streak = %d", strategy, streak)
                if streak >= self._max_losses:
                    until = datetime.now(timezone.utc) + self._cooldown_duration
                    self._cooldown_until[strategy] = until
                    logger.warning(
                        "CooldownManager: %s entering cooldown until %s "
                        "(%d consecutive losses)",
                        strategy, until.isoformat(), streak)

    def is_in_cooldown(self, strategy: str) -> bool:
        with self._lock:
            until = self._cooldown_until.get(strategy)
            if until is None:
                return False
            if datetime.now(timezone.utc) >= until:
                self._cooldown_until.pop(strategy, None)
                self._loss_streaks[strategy] = 0
                logger.info("CooldownManager: %s cooldown expired, resetting", strategy)
                return False
            return True

    def get_status(self, strategy: str) -> dict:
        with self._lock:
            streak = self._loss_streaks.get(strategy, 0)
            until = self._cooldown_until.get(strategy)
            in_cd = until is not None and datetime.now(timezone.utc) < until
            return {
                "strategy": strategy,
                "loss_streak": streak,
                "in_cooldown": in_cd,
                "cooldown_until": until.isoformat() if until else None,
            }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PositionLedger — realized P&L for the risk controls
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PositionLedger:
    """Tracks open entries so a closing alert yields a real realized P&L.

    DailyPnLTracker, CooldownManager and DrawdownCircuitBreaker are all
    outcome-driven, but a webhook only ever sees an instruction — never a
    result. Without somewhere to remember the entry price, the server fed them
    ``pnl=0.0`` and ``won=True`` on every order, which pinned the daily loss at
    zero and reset the loss streak after each trade, so none of the three could
    ever fire. This is the missing half: remember the open, and compute the
    realized P&L when it is closed out.

    Average-cost basis, long-only-or-short-only per symbol. Partial closes are
    supported; an unmatched close (no recorded entry) returns None rather than
    inventing a number.
    """

    def __init__(self) -> None:
        # symbol -> {"quantity": signed float, "avg_price": float}
        self._positions: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

    def record_open(self, symbol: str, quantity: float, price: float) -> None:
        """Record an entry (or an add to an existing position)."""
        if quantity == 0 or price <= 0:
            return
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is None or pos["quantity"] == 0:
                self._positions[symbol] = {"quantity": quantity, "avg_price": price}
                return
            # Same direction: blend into the average cost.
            if (pos["quantity"] > 0) == (quantity > 0):
                total_qty = pos["quantity"] + quantity
                pos["avg_price"] = (
                    pos["avg_price"] * abs(pos["quantity"]) + price * abs(quantity)
                ) / abs(total_qty)
                pos["quantity"] = total_qty
            else:
                # Opposite direction is a reduction, not an add.
                pos["quantity"] += quantity

    def record_close(
        self, symbol: str, quantity: float, price: float,
    ) -> Optional[float]:
        """Close (part of) a position and return its realized P&L.

        Returns None when there is no recorded entry to close against, so the
        caller can tell "flat trade" from "unknown outcome".
        """
        if price <= 0:
            return None
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is None or pos["quantity"] == 0:
                return None

            held = pos["quantity"]
            closing = min(abs(quantity), abs(held)) if quantity else abs(held)
            if closing <= 0:
                return None

            if held > 0:
                pnl = (price - pos["avg_price"]) * closing
                pos["quantity"] = held - closing
            else:
                pnl = (pos["avg_price"] - price) * closing
                pos["quantity"] = held + closing

            if pos["quantity"] == 0:
                del self._positions[symbol]
            return pnl

    def get_open_symbols(self) -> List[str]:
        with self._lock:
            return sorted(self._positions)


def _update_risk_controls(app, alert: "AlertPayload", quantity: float, broker=None) -> None:
    """Feed the risk controls the real outcome of a filled order.

    Opening an position has no outcome yet, so only the ledger is touched.
    Closing one produces a realized P&L, which is what the daily-loss tracker,
    the loss-streak cooldown and the drawdown breaker all need.

    Replaces an earlier version that passed ``pnl=0.0`` and ``won=True`` on
    every order and re-fed the breaker its own unchanged equity — three
    constants that made all three controls mathematically unable to trip.
    """
    action = alert.action.lower()
    ledger = getattr(app.state, "position_ledger", None)
    if ledger is None:
        return

    if action in ("buy", "long"):
        ledger.record_open(alert.symbol, quantity, alert.price)
        return

    if action not in ("sell", "close", "short", "exit"):
        return

    signed_qty = -quantity if action == "short" else quantity
    if action == "short":
        ledger.record_open(alert.symbol, signed_qty, alert.price)
        return

    realized = ledger.record_close(alert.symbol, quantity, alert.price)
    if realized is None:
        # No recorded entry — the outcome is genuinely unknown. Recording a
        # guess here is what the old code did; skipping is the honest choice.
        logger.warning(
            "Close for %s has no recorded entry; realized P&L unknown, "
            "risk controls not updated", alert.symbol,
        )
        return

    app.state.pnl_tracker.record_trade(alert.symbol, realized)
    app.state.cooldown_mgr.record_result(alert.strategy or "default", won=realized > 0)

    # Prefer the broker's own equity; fall back to starting equity plus
    # cumulative realized P&L so the breaker still has a moving series.
    app.state.realized_pnl_total = (
        getattr(app.state, "realized_pnl_total", 0.0) + realized
    )
    equity = _read_broker_equity(broker)
    if equity is None and getattr(app.state, "starting_equity", 0.0) > 0:
        equity = app.state.starting_equity + app.state.realized_pnl_total
    if equity is not None:
        app.state.drawdown_breaker.update_equity(equity)


def _read_broker_equity(broker) -> Optional[float]:
    """Best-effort read of live account equity. None when unavailable."""
    if broker is None:
        return None
    try:
        info = broker.get_account_info() or {}
    except Exception as e:
        logger.warning("Could not read broker equity: %s", e)
        return None
    for key in ("net_liquidation", "NetLiquidation", "equity", "Equity"):
        value = info.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 4 — DrawdownCircuitBreaker (max drawdown kill-switch)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DrawdownCircuitBreaker:
    """Blocks ALL trading when drawdown from peak equity exceeds a threshold."""

    def __init__(self, max_drawdown_pct: float = 10.0, lockout_hours: int = 24):
        self._max_dd_pct = max_drawdown_pct
        self._lockout_duration = timedelta(hours=lockout_hours)
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._tripped_until: Optional[datetime] = None
        self._lock = threading.Lock()

    def update_equity(self, equity: float) -> None:
        with self._lock:
            self._current_equity = equity
            if equity > self._peak_equity:
                self._peak_equity = equity
            if self._peak_equity <= 0:
                return
            dd_pct = ((self._peak_equity - equity) / self._peak_equity) * 100.0
            if dd_pct >= self._max_dd_pct and self._tripped_until is None:
                self._tripped_until = datetime.now(timezone.utc) + self._lockout_duration
                logger.critical(
                    "DrawdownCircuitBreaker TRIPPED: %.2f%% drawdown "
                    "(peak=%.2f, current=%.2f). Blocked until %s",
                    dd_pct, self._peak_equity, equity,
                    self._tripped_until.isoformat())

    def can_trade(self) -> bool:
        with self._lock:
            if self._tripped_until is None:
                return True
            if datetime.now(timezone.utc) >= self._tripped_until:
                logger.info("DrawdownCircuitBreaker lockout expired, resetting")
                self.reset()
                return True
            return False

    def reset(self) -> None:
        self._tripped_until = None
        self._peak_equity = self._current_equity

    @property
    def drawdown_pct(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return ((self._peak_equity - self._current_equity) / self._peak_equity) * 100.0

    def get_status(self) -> dict:
        tripped = (self._tripped_until is not None
                   and datetime.now(timezone.utc) < self._tripped_until)
        return {
            "peak_equity": self._peak_equity,
            "current_equity": self._current_equity,
            "drawdown_pct": round(self.drawdown_pct, 2),
            "tripped": tripped,
            "tripped_until": (self._tripped_until.isoformat()
                              if self._tripped_until else None),
        }


# ─── Risk-gate helper (shared by /webhook and /ai-webhook) ───
def _check_risk_gates(
    app: FastAPI, alert: AlertPayload, quantity: float,
) -> Optional[JSONResponse]:
    """Run all risk checks before order execution.
    Returns a JSONResponse to short-circuit if blocked, else None.
    """
    # FIX 2: Daily loss limit
    if not app.state.pnl_tracker.can_trade():
        logger.warning("BLOCKED by daily loss limit: %s %s", alert.action, alert.symbol)
        return JSONResponse(status_code=200, content={
            "status": "blocked", "reason": "daily_loss_limit",
            "daily_pnl": app.state.pnl_tracker.daily_pnl,
        })

    # FIX 3: Strategy cooldown
    strategy = alert.strategy or "default"
    if app.state.cooldown_mgr.is_in_cooldown(strategy):
        status = app.state.cooldown_mgr.get_status(strategy)
        logger.warning("BLOCKED by cooldown: strategy=%s", strategy)
        return JSONResponse(status_code=200, content={
            "status": "blocked", "reason": "strategy_cooldown", **status,
        })

    # FIX 4: Drawdown circuit breaker
    if not app.state.drawdown_breaker.can_trade():
        dd_status = app.state.drawdown_breaker.get_status()
        logger.warning("BLOCKED by drawdown circuit breaker")
        return JSONResponse(status_code=200, content={
            "status": "blocked", "reason": "drawdown_circuit_breaker", **dd_status,
        })

    # FIX 5: Position size dollar cap
    max_trade_value: float = app.state.config.get(
        "risk_management", {}).get("max_trade_value", 50000.0)
    trade_value = quantity * alert.price
    if trade_value > max_trade_value:
        capped_qty = max_trade_value / alert.price if alert.price > 0 else quantity
        logger.warning(
            "Position size capped: %.2f × %.2f = $%.2f > $%.2f → qty=%.2f",
            quantity, alert.price, trade_value, max_trade_value, capped_qty)
        alert.quantity = capped_qty

    # GAP 12: Sector exposure limit, measured in dollars.
    #
    # This counted POSITIONS, not exposure: with one Technology position open,
    # sector_count/total_positions was 100% and the second tech trade was
    # refused outright, while a single enormous position in one sector passed
    # freely. max_sector_pct is a share of capital, so compare dollars.
    sector = SECTOR_MAP.get(alert.symbol.upper(), "Unknown")
    if alert.action.lower() in ("buy", "long"):
        max_sector_pct = app.state.config.get(
            "risk_management", {}).get("max_sector_pct", 30.0)
        sector_exposure = getattr(app.state, "sector_exposure", {})
        capital = float(app.state.config.get(
            "risk_management", {}).get("starting_equity", 0.0)) or None

        proposed = quantity * alert.price
        sector_total = sector_exposure.get(sector, 0.0) + proposed

        # The denominator must be capital, not the book's own size. Using
        # total exposure makes the first position 100% of the book and blocks
        # everything — the same defect as the old position-count version, just
        # denominated in dollars. With no configured capital the limit has no
        # meaning, so skip it rather than block.
        basis = capital
        if not basis:
            if not getattr(app.state, "_warned_no_sector_basis", False):
                logger.warning(
                    "risk_management.starting_equity is not set — the sector "
                    "concentration limit cannot be evaluated and is not enforced"
                )
                app.state._warned_no_sector_basis = True
        elif (sector_total / basis * 100) > max_sector_pct:
            logger.warning(
                "BLOCKED by sector limit: %s at $%.0f of $%.0f (%.1f%% > %.1f%%)",
                sector, sector_total, basis,
                sector_total / basis * 100, max_sector_pct,
            )
            return JSONResponse(status_code=200, content={
                "status": "blocked",
                "reason": "sector_concentration",
                "sector": sector,
                "sector_exposure": round(sector_total, 2),
                "basis": round(basis, 2),
                "sector_pct": round(sector_total / basis * 100, 1),
                "max_sector_pct": max_sector_pct,
            })

    return None


def validate_hmac_signature(
    payload: bytes, signature: str, secret: str, algorithm: str = "sha256"
) -> bool:
    """Validate HMAC signature from X-Webhook-Signature header."""
    if not signature or not secret:
        return False

    hash_func = getattr(hashlib, algorithm, None)
    if hash_func is None:
        logger.warning(
            "Invalid HMAC algorithm '%s', falling back to sha256", algorithm
        )
        hash_func = hashlib.sha256
    expected = hmac.new(secret.encode(), payload, hash_func).hexdigest()

    return hmac.compare_digest(expected, signature)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 1 — Webhook Idempotency (dedup cache)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_processed_alerts: Dict[str, datetime] = {}
_DEDUP_TTL = timedelta(minutes=5)
_dedup_lock = threading.Lock()


def _get_alert_id(payload_dict: dict, raw_body: bytes) -> str:
    """Extract or compute a unique alert ID from the payload."""
    alert_id = payload_dict.get("alert_id") or payload_dict.get("id")
    if alert_id:
        return str(alert_id)
    return hashlib.sha256(raw_body).hexdigest()[:16]


def _is_duplicate_alert(alert_id: str) -> bool:
    """Check if alert_id was already processed within the TTL window."""
    with _dedup_lock:
        _cleanup_dedup_cache()
        if alert_id in _processed_alerts:
            return True
        _processed_alerts[alert_id] = datetime.now(timezone.utc)
        return False


def _cleanup_dedup_cache() -> None:
    """Remove entries older than the TTL from the dedup cache."""
    now = datetime.now(timezone.utc)
    expired = [k for k, v in _processed_alerts.items() if now - v > _DEDUP_TTL]
    for k in expired:
        del _processed_alerts[k]


class StaleAlertError(ValueError):
    """Raised when an alert's timestamp is missing, unparseable or too old."""


def _check_alert_freshness(
    timestamp: Optional[str],
    max_skew_seconds: int,
    require_timestamp: bool,
) -> None:
    """Reject alerts whose timestamp is outside the accepted skew window.

    ``AlertPayload.timestamp`` existed but was never read anywhere, so the only
    replay protection was a five-minute dedup cache keyed on alert_id. A
    captured signed request stayed valid forever and could be replayed once the
    cache entry expired — and varying alert_id defeated the cache outright.
    The timestamp is inside the signed body, so binding freshness to it closes
    the window without a second signature.

    Args:
        timestamp: ISO-8601 string from the payload, or None.
        max_skew_seconds: Accepted age. 0 disables the check entirely.
        require_timestamp: Whether a missing timestamp is itself a rejection.

    Raises:
        StaleAlertError: If the alert is missing a required timestamp, the
            timestamp cannot be parsed, or it falls outside the window.
    """
    if max_skew_seconds <= 0:
        return

    if not timestamp:
        if require_timestamp:
            raise StaleAlertError("timestamp required but not provided")
        return

    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except (TypeError, ValueError) as e:
        raise StaleAlertError(f"unparseable timestamp: {timestamp!r}") from e

    if parsed.tzinfo is None:
        # TradingView's {{timenow}} is UTC but carries no offset.
        parsed = parsed.replace(tzinfo=timezone.utc)

    age = (datetime.now(timezone.utc) - parsed).total_seconds()
    if age > max_skew_seconds:
        raise StaleAlertError(
            f"alert is {age:.0f}s old, limit is {max_skew_seconds}s"
        )
    # A timestamp far in the future is equally suspect — it would keep an alert
    # replayable well past the window.
    if age < -max_skew_seconds:
        raise StaleAlertError(
            f"alert is {-age:.0f}s in the future, limit is {max_skew_seconds}s"
        )


# ─── Application Factory ───
def create_app(config_path: str = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = load_config(config_path)
    setup_logging(config)

    # FIX 7: Validate HMAC secret — refuse to start with weak/empty secret
    security_cfg = config.get("security", {})
    hmac_secret = security_cfg.get("hmac_secret", "")
    require_hmac = security_cfg.get("require_hmac", True)
    weak_secrets = {"", "default_secret", "CHANGE_ME", "CHANGE_ME_TO_A_SECURE_SECRET_KEY"}

    if hmac_secret in weak_secrets:
        if require_hmac:
            import os as _os
            generated_secret = _os.urandom(32).hex()
            security_cfg["hmac_secret"] = generated_secret
            config["security"] = security_cfg
            logger.critical(
                "HMAC secret was empty/default. Generated a random secret for "
                "this session. Set security.hmac_secret in config.yaml — until "
                "you do, every restart invalidates previously-signed alerts and "
                "each worker generates a different one."
            )
        else:
            logger.warning(
                "HMAC secret is empty/default and require_hmac=False. "
                "Webhook signature validation is DISABLED. This is insecure!"
            )

    # The passphrase had no equivalent guard: the shipped placeholder is in the
    # repository, so require_passphrase=true with the default value is exactly
    # as good as no passphrase at all, while looking configured.
    weak_passphrases = {
        "", "CHANGE_ME", "CHANGE_ME_WEBHOOK_PASSPHRASE", "changeme", "password",
    }
    if security_cfg.get("require_passphrase", False):
        if security_cfg.get("passphrase", "") in weak_passphrases:
            raise RuntimeError(
                "security.require_passphrase is true but security.passphrase is "
                "empty or still the shipped placeholder. That placeholder is "
                "public in this repository, so it authenticates nobody. Set a "
                "real passphrase in config.yaml, or set require_passphrase: "
                "false if you are relying on HMAC and the IP allowlist alone."
            )

    server_config = config.get("server", {})
    app = FastAPI(
        title=server_config.get("title", "TradingView Webhook Server"),
        version=server_config.get("version", "1.0.0"),
        debug=server_config.get("debug", False),
    )

    # Fix 7: CORS restriction — configurable origins, default empty (no browser access)
    cors_origins = config.get("security", {}).get("cors_origins", [])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Application State ───
    app.state.config = config
    app.state.start_time = time.time()
    app.state.last_alert_time = None
    app.state.total_alerts = 0
    app.state.broker_router = BrokerRouter(config)
    app.state.sector_tracker: Dict[str, int] = defaultdict(int)
    # Dollar exposure per sector. max_sector_pct is a share of capital,
    # so the limit has to be measured in dollars; the count above is kept
    # only for the position tally reported by /status.
    app.state.sector_exposure: Dict[str, float] = defaultdict(float)

    rate_config = config.get("rate_limiting", {})
    app.state.rate_limiter = RateLimiter(
        max_requests=rate_config.get("max_requests_per_minute", 60),
        window_seconds=rate_config.get("window_seconds", 60),
    )

    # Initialize AlertDispatcher if available
    app.state.alert_dispatcher = None
    if AlertDispatcher is not None:
        try:
            # AlertDispatcher requires its config section; calling it with no
            # argument raised TypeError straight into the handler below, so no
            # notification channel was ever active.
            app.state.alert_dispatcher = AlertDispatcher(
                config.get("notifications", {})
            )
            logger.info("AlertDispatcher initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize AlertDispatcher: {e}")

    # ─── Risk Management State (FIX 2-5) ───
    risk_cfg = config.get("risk_management", {})

    app.state.pnl_tracker = DailyPnLTracker(
        max_daily_loss=risk_cfg.get("max_daily_loss", 5000.0),
        reset_hour_utc=risk_cfg.get("daily_reset_hour_utc", 0),
    )
    app.state.cooldown_mgr = CooldownManager(
        max_consecutive_losses=risk_cfg.get("max_consecutive_losses", 3),
        cooldown_minutes=risk_cfg.get("cooldown_minutes", 30),
    )
    app.state.drawdown_breaker = DrawdownCircuitBreaker(
        max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 10.0),
        lockout_hours=risk_cfg.get("drawdown_lockout_hours", 24),
    )

    # Entry tracking, so a closing alert produces a real realized P&L to feed
    # the three controls above.
    app.state.position_ledger = PositionLedger()
    app.state.starting_equity = float(risk_cfg.get("starting_equity", 0.0))
    app.state.realized_pnl_total = 0.0
    if app.state.starting_equity > 0:
        # Seed the peak so the drawdown breaker has a baseline to measure from.
        app.state.drawdown_breaker.update_equity(app.state.starting_equity)

    # ─── Health Monitor (FIX 1) ───
    hm_cfg = config.get("health_monitor", {})
    app.state.health_monitor = HealthMonitor(
        ping_url=hm_cfg.get("ping_url", ""),
        ping_interval_seconds=hm_cfg.get("ping_interval_seconds", 60),
        max_silence_minutes=hm_cfg.get("max_silence_minutes", 30),
        alert_dispatcher=app.state.alert_dispatcher,
        alert_timeout_hours=hm_cfg.get("alert_timeout_hours", 12.0),
    )

    # GAP 3: Latency tracking config
    app.state.prefer_limit_orders: bool = config.get(
        "risk_management", {}).get("prefer_limit_orders", True)
    app.state.limit_order_offset_pct: float = config.get(
        "risk_management", {}).get("limit_order_offset_pct", 0.02)

    # ─── Logging Middleware ───
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        client_ip = client_ip_of(request, config)
        logger.info(f"→ {request.method} {request.url.path} from {client_ip}")

        response = await call_next(request)

        duration = time.time() - start
        logger.info(
            f"← {request.method} {request.url.path} → {response.status_code} ({duration:.3f}s)"
        )
        return response

    # ─── Health Endpoint (enhanced — FIX 1) ───
    def _authorize_read_endpoint(request: Request) -> None:
        """Apply the webhook's IP allowlist and HMAC gate to a read endpoint.

        /status returns live brokerage balances and account identifiers, so it
        needs the same gate as the order endpoints. Raises HTTPException when
        the caller is not authorised.
        """
        security_config = config.get("security", {})
        client_ip = client_ip_of(request, config)

        allowed_ips = security_config.get("allowed_ips", [])
        if not ip_allowed(client_ip, allowed_ips):
            logger.warning("Rejected read request from unauthorized IP: %s", client_ip)
            raise HTTPException(status_code=403, detail="Not authorized")

        hmac_secret = security_config.get("hmac_secret", "")
        if hmac_secret and hmac_secret != "CHANGE_ME_TO_A_SECURE_SECRET_KEY":
            signature = request.headers.get("X-Webhook-Signature", "")
            algorithm = security_config.get("hmac_algorithm", "sha256")
            # GET has no body; the signature is taken over the path so a
            # signature captured for one endpoint cannot be replayed at another.
            if not validate_hmac_signature(
                request.url.path.encode(), signature, hmac_secret, algorithm
            ):
                logger.warning("Invalid signature on read endpoint from %s", client_ip)
                raise HTTPException(status_code=401, detail="Not authorized")

    @app.get("/health")
    async def health_check():
        """Liveness probe. Deliberately carries no financial information.

        This endpoint is unauthenticated so uptime monitors and container
        healthchecks can reach it, so daily P&L, equity, drawdown and sector
        exposure are not reported here — they moved to /status, which is
        authenticated. Reporting an account's P&L to anonymous callers is not
        something a liveness probe needs to do.
        """
        hm = app.state.health_monitor

        return JSONResponse(content={
            "status": "ok",
            "uptime": round(hm.uptime_seconds, 2),
            "version": server_config.get("version", "1.0.0"),
        })

    # ─── Webhook Endpoint ───
    @app.post("/webhook")
    async def receive_webhook(request: Request):
        """
        Receive TradingView alert webhook.

        Validates HMAC signature, checks rate limits, authenticates passphrase,
        routes to appropriate broker, and dispatches notifications.
        """
        client_ip = client_ip_of(request, config)
        security_config = config.get("security", {})

        # ── IP Allowlist Check ──
        allowed_ips = security_config.get("allowed_ips", [])
        if not ip_allowed(client_ip, allowed_ips):
            logger.warning(f"Rejected request from unauthorized IP: {client_ip}")
            raise HTTPException(status_code=403, detail="IP not allowed")

        # ── Rate Limiting ──
        if config.get("rate_limiting", {}).get("enabled", True):
            if not app.state.rate_limiter.is_allowed(client_ip):
                remaining = app.state.rate_limiter.get_remaining(client_ip)
                logger.warning(f"Rate limit exceeded for {client_ip}")
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Try again later. Remaining: {remaining}",
                )

        # ── Read Raw Body ──
        # Bound the body BEFORE buffering it: this runs ahead of the HMAC
        # check, so the caller is still unauthenticated at this point.
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            logger.warning("Oversized body (%s bytes) from %s", declared, client_ip)
            raise HTTPException(status_code=413, detail="Payload too large")
        raw_body = await request.body()
        if len(raw_body) > MAX_BODY_BYTES:
            logger.warning("Oversized body (%d bytes) from %s", len(raw_body), client_ip)
            raise HTTPException(status_code=413, detail="Payload too large")

        # ── HMAC Signature Validation ──
        hmac_secret = security_config.get("hmac_secret", "")
        signature = request.headers.get("X-Webhook-Signature", "")

        if hmac_secret and hmac_secret != "CHANGE_ME_TO_A_SECURE_SECRET_KEY":
            algorithm = security_config.get("hmac_algorithm", "sha256")
            if not validate_hmac_signature(raw_body, signature, hmac_secret, algorithm):
                logger.warning(f"Invalid HMAC signature from {client_ip}")
                raise HTTPException(status_code=401, detail="Invalid signature")

        # ── Parse Payload ──
        try:
            payload_dict = json.loads(raw_body)
            alert = AlertPayload(**payload_dict)
        except Exception as e:
            logger.error(f"Invalid payload from {client_ip}: {e}")
            raise HTTPException(status_code=422, detail=f"Invalid payload: {str(e)}")

        # ── Passphrase Validation ──
        # Fix 6: constant-time passphrase comparison
        #
        # Must run BEFORE the dedup cache is touched. With dedup first, an
        # unauthenticated caller could pre-poison the cache with a guessed
        # alert_id so the genuine TradingView alert that followed was dropped
        # as a "duplicate" — suppression of a real trading signal without ever
        # knowing the passphrase. /ai-webhook already had the correct order.
        if security_config.get("require_passphrase", False):
            expected_passphrase = security_config.get("passphrase", "")
            received = alert.passphrase or ""
            if not hmac.compare_digest(received, expected_passphrase):
                logger.warning(f"Invalid passphrase from {client_ip}")
                raise HTTPException(status_code=401, detail="Invalid passphrase")

        # ── Replay window ──
        # The timestamp is inside the HMAC-signed body, so checking its age
        # bounds how long a captured signed request stays usable. Without it
        # the only limit was a 5-minute dedup cache keyed on alert_id, which a
        # replayer defeats simply by varying alert_id.
        try:
            _check_alert_freshness(
                alert.timestamp,
                security_config.get("max_timestamp_skew_seconds", 300),
                security_config.get("require_timestamp", False),
            )
        except StaleAlertError as e:
            logger.warning("%sRejected stale alert from %s: %s", "", client_ip, e)
            raise HTTPException(status_code=401, detail="Stale or invalid timestamp")

        # ── FIX 1: Idempotency — deduplicate alerts ──
        alert_id = _get_alert_id(payload_dict, raw_body)
        if _is_duplicate_alert(alert_id):
            logger.info("Duplicate alert skipped: %s", alert_id)
            return JSONResponse(status_code=200, content={
                "status": "duplicate", "alert_id": alert_id,
            })

        # ── Validate Action ──
        valid_actions = {"buy", "sell", "close", "long", "short"}
        if alert.action.lower() not in valid_actions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {alert.action}. Must be one of: {valid_actions}",
            )

        # ── Risk Gates (FIX 2-5) ──
        quantity = alert.quantity or 1.0
        risk_block = _check_risk_gates(app, alert, quantity)
        if risk_block is not None:
            return risk_block
        quantity = alert.quantity or quantity

        # ── GAP 3: Track alert-received time ──
        alert_received_time = datetime.now(timezone.utc)

        # ── GAP 3: Prefer limit orders with price offset ──
        effective_order_type = alert.order_type
        effective_price = alert.price
        if (
            app.state.prefer_limit_orders
            and alert.order_type == "market"
            and alert.price > 0
        ):
            offset_pct = app.state.limit_order_offset_pct / 100.0
            if alert.action.lower() in ("buy", "long"):
                effective_price = round(alert.price * (1 + offset_pct), 2)
            else:
                effective_price = round(alert.price * (1 - offset_pct), 2)
            effective_order_type = "limit"
            logger.info(
                "Prefer limit orders: converted market → limit @ $%.2f "
                "(offset=%.4f%% from $%.2f)",
                effective_price, app.state.limit_order_offset_pct, alert.price,
            )

        # ── Route to Broker ──
        try:
            broker = app.state.broker_router.get_broker(alert.symbol)

            order_result = broker.place_order(
                symbol=alert.symbol,
                action=alert.action.lower(),
                quantity=quantity,
                order_type=effective_order_type,
                price=effective_price,
            )
        except Exception as e:
            logger.error(f"Broker execution error: {e}")
            # FIX 1: Return HTTP 200 on broker errors to prevent TradingView retries
            return JSONResponse(status_code=200, content={
                "status": "error", "detail": str(e),
            })

        # ── GAP 3: Track order-placed time and compute latency ──
        order_placed_time = datetime.now(timezone.utc)
        latency_ms = (order_placed_time - alert_received_time).total_seconds() * 1000
        app.state.health_monitor.record_latency(latency_ms)

        # ── Update State ──
        app.state.last_alert_time = datetime.now(timezone.utc)
        app.state.total_alerts += 1
        app.state.health_monitor.record_alert()

        # Track sector exposure (GAP 12) — both the count and the dollars.
        sector = SECTOR_MAP.get(alert.symbol.upper(), "Unknown")
        notional = quantity * alert.price
        if alert.action.lower() in ("buy", "long"):
            app.state.sector_tracker[sector] = app.state.sector_tracker.get(sector, 0) + 1
            app.state.sector_exposure[sector] = (
                app.state.sector_exposure.get(sector, 0.0) + notional
            )
        elif alert.action.lower() in ("sell", "close"):
            app.state.sector_tracker[sector] = max(0, app.state.sector_tracker.get(sector, 0) - 1)
            app.state.sector_exposure[sector] = max(
                0.0, app.state.sector_exposure.get(sector, 0.0) - notional
            )

        # ── Dispatch Notifications ──
        if app.state.alert_dispatcher:
            try:
                app.state.alert_dispatcher.dispatch(
                    title=f"Trade Alert: {alert.action.upper()} {alert.symbol}",
                    message=(
                        f"Action: {alert.action}\n"
                        f"Symbol: {alert.symbol}\n"
                        f"Price: {alert.price}\n"
                        f"Quantity: {quantity}\n"
                        f"Broker: {broker.name}\n"
                        f"Order ID: {order_result.order_id}"
                    ),
                    priority="INFO",
                )
            except Exception as e:
                logger.warning(f"Notification dispatch failed: {e}")

        # ── FIX 5: Wire risk controls to actual trade results ──
        if order_result.success:
            try:
                _update_risk_controls(app, alert, quantity, broker)
            except Exception as risk_exc:
                logger.warning("Risk control update failed: %s", risk_exc)

        logger.info(
            f"Alert processed: {alert.action} {alert.symbol} @ {alert.price} "
            f"via {broker.name} → {order_result.order_id}"
            f"{f' | regime={alert.regime}' if alert.regime else ''}"
            f"{f' | strategy={alert.strategy}' if alert.strategy else ''}"
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "alert": {
                    "symbol": alert.symbol,
                    "action": alert.action,
                    "price": alert.price,
                    "quantity": quantity,
                    "regime": alert.regime,
                    "strategy": alert.strategy,
                },
                "order": order_result.model_dump(),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ─── AI-Enhanced Webhook Endpoint ───
    @app.post("/ai-webhook")
    async def receive_ai_webhook(request: Request):
        """
        AI-enhanced TradingView webhook.

        Routes alerts through the SelfLearningAgent before executing.
        The agent can confirm, override, or block the Pine Script signal
        based on ML analysis, news sentiment, and trade history.

        Flow: TradingView Alert → AI Agent.decide() → Broker Execution
        """
        client_ip = client_ip_of(request, config)
        security_config = config.get("security", {})

        # ── FIX 3: HMAC / passphrase authentication (same as /webhook) ──
        allowed_ips = security_config.get("allowed_ips", [])
        if not ip_allowed(client_ip, allowed_ips):
            logger.warning("AI-webhook rejected from unauthorized IP: %s", client_ip)
            raise HTTPException(status_code=403, detail="IP not allowed")

        # ── Rate Limiting ──
        if config.get("rate_limiting", {}).get("enabled", True):
            if not app.state.rate_limiter.is_allowed(client_ip):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # ── Parse Payload ──
        # Bound the body BEFORE buffering it: this runs ahead of the HMAC
        # check, so the caller is still unauthenticated at this point.
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            logger.warning("Oversized body (%s bytes) from %s", declared, client_ip)
            raise HTTPException(status_code=413, detail="Payload too large")
        raw_body = await request.body()
        if len(raw_body) > MAX_BODY_BYTES:
            logger.warning("Oversized body (%d bytes) from %s", len(raw_body), client_ip)
            raise HTTPException(status_code=413, detail="Payload too large")

        # ── HMAC Signature Validation ──
        hmac_secret = security_config.get("hmac_secret", "")
        signature = request.headers.get("X-Webhook-Signature", "")
        if hmac_secret and hmac_secret != "CHANGE_ME_TO_A_SECURE_SECRET_KEY":
            algorithm = security_config.get("hmac_algorithm", "sha256")
            if not validate_hmac_signature(raw_body, signature, hmac_secret, algorithm):
                logger.warning("AI-webhook invalid HMAC from %s", client_ip)
                raise HTTPException(status_code=401, detail="Invalid signature")

        try:
            payload_dict = json.loads(raw_body)
            alert = AlertPayload(**payload_dict)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid payload: {str(e)}")

        # ── Passphrase Validation ──
        if security_config.get("require_passphrase", False):
            expected_passphrase = security_config.get("passphrase", "")
            if not hmac.compare_digest(str(alert.passphrase or ""), str(expected_passphrase or "")):
                logger.warning("AI-webhook invalid passphrase from %s", client_ip)
                raise HTTPException(status_code=401, detail="Invalid passphrase")

        # ── Replay window ──
        # The timestamp is inside the HMAC-signed body, so checking its age
        # bounds how long a captured signed request stays usable. Without it
        # the only limit was a 5-minute dedup cache keyed on alert_id, which a
        # replayer defeats simply by varying alert_id.
        try:
            _check_alert_freshness(
                alert.timestamp,
                security_config.get("max_timestamp_skew_seconds", 300),
                security_config.get("require_timestamp", False),
            )
        except StaleAlertError as e:
            logger.warning("%sRejected stale alert from %s: %s", "AI-webhook: ", client_ip, e)
            raise HTTPException(status_code=401, detail="Stale or invalid timestamp")

        # ── FIX 1: Idempotency — deduplicate alerts ──
        alert_id = _get_alert_id(payload_dict, raw_body)
        if _is_duplicate_alert(alert_id):
            logger.info("AI-webhook duplicate alert skipped: %s", alert_id)
            return JSONResponse(status_code=200, content={
                "status": "duplicate", "alert_id": alert_id,
            })

        # ── Fetch Data for AI Agent ──
        ai_result = {"agent_used": False, "agent_action": None, "confidence": 0}

        try:
            import sys, os
            _ai_path = os.path.join(os.path.dirname(__file__), "..", "..")
            if _ai_path not in sys.path:
                sys.path.insert(0, _ai_path)

            from shared.data.public_data_fetcher import PublicDataFetcher
            from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

            # Initialize agent (lazy singleton)
            if not hasattr(app.state, "ai_agent") or app.state.ai_agent is None:
                import tempfile
                db_path = os.path.join(tempfile.gettempdir(), "webhook_agent_memory.db")
                app.state.ai_agent = SelfLearningAgent(AgentConfig(db_path=db_path))

            # Fetch market data
            fetcher = PublicDataFetcher(cache_enabled=True)
            df = fetcher.fetch_ohlcv(alert.symbol, period="6mo", interval="1d")

            if df is not None and len(df) >= 60:
                decision = app.state.ai_agent.decide(df, symbol=alert.symbol)
                ai_action = decision.get("action", "HOLD")
                ai_confidence = decision.get("confidence", 0)
                ai_regime = decision.get("regime", "UNKNOWN")

                ai_result = {
                    "agent_used": True,
                    "agent_action": ai_action,
                    "confidence": ai_confidence,
                    "regime": ai_regime,
                    "tv_action": alert.action,
                    "agreement": ai_action.lower() == alert.action.lower(),
                }

                # AI can override: if agent strongly disagrees, block the trade
                if not ai_result["agreement"] and ai_confidence > 0.7:
                    logger.warning(
                        "AI OVERRIDE: TradingView says %s but agent says %s (confidence=%.2f). BLOCKING.",
                        alert.action, ai_action, ai_confidence,
                    )
                    return JSONResponse(
                        status_code=200,
                        content={
                            "status": "blocked_by_ai",
                            "tv_action": alert.action,
                            "ai_action": ai_action,
                            "ai_confidence": ai_confidence,
                            "ai_regime": ai_regime,
                            "reason": "AI agent strongly disagrees with TradingView signal",
                        },
                    )

        except Exception as e:
            logger.warning("AI agent processing failed (executing TV signal anyway): %s", e)

        # ── Risk Gates (FIX 2-5) ──
        quantity = alert.quantity or 1.0
        risk_block = _check_risk_gates(app, alert, quantity)
        if risk_block is not None:
            return risk_block
        quantity = alert.quantity or quantity

        # ── GAP 3: Track alert-received time ──
        ai_alert_received_time = datetime.now(timezone.utc)

        # ── GAP 3: Prefer limit orders with price offset ──
        effective_order_type = alert.order_type
        effective_price = alert.price
        if (
            app.state.prefer_limit_orders
            and alert.order_type == "market"
            and alert.price > 0
        ):
            offset_pct = app.state.limit_order_offset_pct / 100.0
            if alert.action.lower() in ("buy", "long"):
                effective_price = round(alert.price * (1 + offset_pct), 2)
            else:
                effective_price = round(alert.price * (1 - offset_pct), 2)
            effective_order_type = "limit"

        # ── Route to Broker (real execution) ──
        try:
            broker = app.state.broker_router.get_broker(alert.symbol)

            order_result = broker.place_order(
                symbol=alert.symbol,
                action=alert.action.lower(),
                quantity=quantity,
                order_type=effective_order_type,
                price=effective_price,
            )
        except Exception as e:
            logger.error("AI-webhook broker error: %s", e)
            # FIX 1: Return HTTP 200 on broker errors to prevent TradingView retries
            return JSONResponse(status_code=200, content={
                "status": "error", "detail": str(e),
            })

        # ── GAP 3: Track order-placed time and compute latency ──
        ai_order_placed_time = datetime.now(timezone.utc)
        ai_latency_ms = (ai_order_placed_time - ai_alert_received_time).total_seconds() * 1000
        app.state.health_monitor.record_latency(ai_latency_ms)

        app.state.last_alert_time = datetime.now(timezone.utc)
        app.state.total_alerts += 1
        app.state.health_monitor.record_alert()

        # ── FIX 5: Wire risk controls to actual trade results ──
        if order_result.success:
            try:
                _update_risk_controls(app, alert, quantity, broker)
            except Exception as risk_exc:
                logger.warning("AI-webhook risk control update failed: %s", risk_exc)

        # ── Notifications ──
        if app.state.alert_dispatcher:
            try:
                ai_note = ""
                if ai_result["agent_used"]:
                    ai_note = f"\nAI: {ai_result['agent_action']} (conf={ai_result['confidence']:.0%})"
                app.state.alert_dispatcher.dispatch(
                    title=f"AI Trade: {alert.action.upper()} {alert.symbol}",
                    message=(
                        f"TV Signal: {alert.action} {alert.symbol} @ {alert.price}"
                        f"{ai_note}\nBroker: {broker.name} → {order_result.order_id}"
                    ),
                    priority="INFO",
                )
            except Exception:
                pass

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "alert": {
                    "symbol": alert.symbol,
                    "action": alert.action,
                    "price": alert.price,
                    "quantity": quantity,
                },
                "ai": ai_result,
                "order": order_result.model_dump(),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ─── Status Endpoint ──
    @app.get("/status")
    async def server_status(request: Request):
        """Detailed status, including live broker balances. Authenticated.

        get_account_info() returns net liquidation, buying power and the full
        brokerage account identifier. This endpoint had no authentication at
        all, so a plain ``curl http://host:5000/status`` disclosed the account
        number and its net worth.
        """
        _authorize_read_endpoint(request)

        brokers = app.state.broker_router.get_all_brokers()
        broker_status = {}
        for name, adapter in brokers.items():
            try:
                info = adapter.get_account_info()
                broker_status[name] = info
            except Exception as e:
                # Broker error text can carry request/response detail; log it
                # in full, return only that something failed.
                logger.warning("get_account_info failed for %s: %s", name, e)
                broker_status[name] = {"error": "unavailable"}

        return {
            "server": "running",
            "uptime_seconds": round(time.time() - app.state.start_time, 2),
            "total_alerts": app.state.total_alerts,
            "brokers": broker_status,
            "risk": {
                "daily_pnl": app.state.pnl_tracker.daily_pnl,
                "daily_trade_count": app.state.pnl_tracker.trade_count,
                "daily_loss_limit_ok": app.state.pnl_tracker.can_trade(),
                "drawdown": app.state.drawdown_breaker.get_status(),
                "sector_positions": dict(app.state.sector_tracker),
                "sector_exposure_usd": dict(app.state.sector_exposure),
                "open_positions": app.state.position_ledger.get_open_symbols(),
            },
            "latency": app.state.health_monitor.get_latency_stats(),
            "alert_freshness": app.state.health_monitor.check_alert_freshness()["detail"],
            "rate_limiting": {
                "enabled": config.get("rate_limiting", {}).get("enabled", True),
                "max_per_minute": config.get("rate_limiting", {}).get(
                    "max_requests_per_minute", 60
                ),
            },
        }

    # Fix 15: validate broker package installed at startup
    default_broker = config.get("broker_routing", {}).get("default_broker", "interactive_brokers")
    broker_pkg_map = {
        "interactive_brokers": "interactive_brokers",
        "tradestation": "tradestation",
        "schwab": "thinkorswim",
        "thinkorswim": "thinkorswim",
    }
    pkg = broker_pkg_map.get(default_broker)
    if pkg:
        import importlib
        try:
            importlib.import_module(pkg)
        except ImportError:
            logger.warning(
                "Default broker '%s' package '%s' not installed. "
                "Order routing may fail.",
                default_broker, pkg,
            )

    return app


# ─── Application Instance ───
#
# Built on first access rather than at import. `app = create_app()` here meant
# that merely importing this module — which pytest does during collection —
# read config.yaml from disk, instantiated broker adapters, started the health
# monitor and enforced the startup guards. PEP 562 module __getattr__ keeps the
# `webhook_server:app` import string that uvicorn.run() needs working, because
# uvicorn resolves it with getattr().
_app_instance: Optional[FastAPI] = None


def get_app() -> FastAPI:
    """Return the singleton application, constructing it on first call."""
    global _app_instance
    if _app_instance is None:
        _app_instance = create_app()
    return _app_instance


def __getattr__(name: str):
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ─── Main Entry Point (with auto-restart + startup notification) ───
if __name__ == "__main__":
    import uvicorn

    config = load_config()
    server_config = config.get("server", {})
    # Loopback unless config.yaml says otherwise — see _default_config().
    host = server_config.get("host", "127.0.0.1")
    port = server_config.get("port", 5000)

    # Bind the lazily-built app to a local name. Module-level __getattr__
    # resolves `webhook_server.app` for uvicorn's import string, but it does
    # NOT resolve a bare `app` inside this module — that would NameError.
    app = get_app()

    # Start health monitor background thread
    app.state.health_monitor.start()

    # Startup notification
    startup_msg = f"🚀 Webhook server started on {host}:{port}"
    logger.info(startup_msg)
    if app.state.alert_dispatcher:
        try:
            app.state.alert_dispatcher.dispatch(
                title="Webhook Server Started",
                message=startup_msg,
                priority="INFO",
            )
        except Exception:
            pass

    # Auto-restart wrapper
    max_restarts = server_config.get("max_restarts", 5)
    restart_count = 0
    last_restart_time = time.time()
    while restart_count < max_restarts:
        try:
            uvicorn.run(
                "webhook_server:app",
                host=host,
                port=port,
                reload=server_config.get("debug", False),
                workers=server_config.get("workers", 1),
                log_level=config.get("logging", {}).get("level", "info").lower(),
            )
            break  # clean shutdown
        except Exception as exc:
            # Fix 13: reset restart counter after 1 hour of uptime
            now = time.time()
            if now - last_restart_time >= 3600:
                restart_count = 0
                logger.info("Server ran >1h, resetting restart counter")
            last_restart_time = now

            restart_count += 1
            logger.critical(
                "Server crashed (%d/%d): %s — restarting in 5s",
                restart_count, max_restarts, exc,
            )
            if app.state.alert_dispatcher:
                try:
                    app.state.alert_dispatcher.dispatch(
                        title="Webhook Server CRASH",
                        message=f"Crash #{restart_count}: {exc}. Restarting\u2026",
                        priority="CRITICAL",
                    )
                except Exception:
                    pass
            time.sleep(5)

    if restart_count >= max_restarts:
        logger.critical("Max restarts (%d) exhausted. Server NOT restarting.", max_restarts)
        if app.state.alert_dispatcher:
            try:
                app.state.alert_dispatcher.dispatch(
                    title="Webhook Server DEAD",
                    message=f"Exhausted {max_restarts} restarts. Manual intervention required.",
                    priority="CRITICAL",
                )
            except Exception:
                pass
