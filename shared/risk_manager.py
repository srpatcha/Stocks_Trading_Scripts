"""
Standalone Risk Manager
========================

Provides position sizing, daily loss limits, cooldown after consecutive
losses, portfolio heat limits, trade frequency throttling, and drawdown
circuit breakers. Used by all Python trading strategies.

Usage:
    rm = RiskManager(config=RiskManagerConfig(max_daily_loss=5000))
    size = rm.calculate_position_size("AAPL", 150.0, stop_price=145.0)
    if rm.can_trade():
        # place order
        rm.record_trade(pnl=-200)
"""

from __future__ import annotations

import json
import functools
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from shared.utils.trading_clock import trading_day

def _synchronized(method):
    """Run ``method`` holding ``self._state_lock``.

    The writers (record_trade, add_position, remove_position) were already
    locked, but every read-side gate — can_trade, validate_order, get_status,
    calculate_position_size, check_portfolio_heat, can_pyramid — was not.
    Worse, can_trade and get_status call _reset_daily_if_needed, which WRITES
    (`self._daily_pnl = 0.0`); a reset racing record_trade's
    `self._daily_pnl += pnl` silently lost the update. "Thread-safe" covered
    the writers only.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._state_lock:
            return method(self, *args, **kwargs)
    return wrapper



class SizingMethod(Enum):
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY = "kelly"
    FIXED_SHARES = "fixed_shares"
    FIXED_DOLLAR = "fixed_dollar"


@dataclass
class RiskManagerConfig:
    """Configuration for the RiskManager."""

    # Position sizing
    sizing_method: SizingMethod = SizingMethod.FIXED_FRACTIONAL
    risk_per_trade_pct: float = 2.0
    fixed_shares: int = 100
    fixed_dollar_amount: float = 10000.0

    # Kelly criterion parameters
    kelly_win_rate: float = 0.55
    kelly_avg_win: float = 1.5
    kelly_avg_loss: float = 1.0
    kelly_fraction: float = 0.5  # half-Kelly for safety

    # Daily loss limit
    max_daily_loss: float = 5000.0
    auto_flatten_on_daily_loss: bool = True

    # Consecutive loss cooldown
    max_consecutive_losses: int = 3
    cooldown_seconds: int = 1800  # 30 minutes

    # Portfolio heat (total exposure)
    max_portfolio_heat_pct: float = 20.0  # max % of portfolio at risk
    max_open_positions: int = 10

    # Trade frequency
    max_trades_per_hour: int = 10
    min_seconds_between_trades: float = 30.0

    # Drawdown circuit breaker
    max_drawdown_pct: float = 10.0
    circuit_breaker_pause_hours: float = 24.0

    # Capital
    total_capital: float = 100000.0

    # Pyramiding (Livermore — adding to winners)
    enable_pyramiding: bool = False
    max_pyramid_levels: int = 3
    pyramid_threshold_pct: float = 2.0
    pyramid_scale_factor: float = 0.5
    pyramid_trail_stop_pct: float = 1.0

    # Monthly risk cap (Elder's 6% rule)
    max_monthly_loss: float = 0.0  # 0 = disabled; e.g., 6000.0 for 6% of $100k
    monthly_reset_day: int = 1  # day of month to reset

    # ─── Production Safety (Critical) ───

    # Max position size caps
    max_position_pct_equity: float = 25.0  # max % of equity in a single position
    max_position_notional: float = 0.0  # 0 = disabled; hard dollar cap per position
    max_shares_per_order: int = 10000  # fat-finger protection

    # Pre-order price validation
    max_price_deviation_pct: float = 10.0  # reject orders >10% from last price

    # Short-selling limits
    max_short_positions: int = 5
    max_short_exposure_pct: float = 30.0  # max % of equity in total short exposure
    require_short_stop: bool = True  # force buy-stop on all shorts

    # Market hours enforcement
    enforce_market_hours: bool = False  # True = block signals outside NYSE hours
    allow_premarket: bool = False
    allow_afterhours: bool = False

    # Liquidity filter
    min_avg_volume: int = 50000  # skip stocks with avg daily volume below this
    max_position_pct_adv: float = 5.0  # max position as % of avg daily volume

    # State persistence (crash recovery)
    persist_path: Optional[str] = None


@dataclass
class TradeRecord:
    """Record of a completed trade for risk tracking."""
    symbol: str
    pnl: float
    timestamp: datetime = field(default_factory=datetime.now)
    is_win: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_win = self.pnl > 0


class RiskManager:
    """Manages risk controls across all trading strategies.

    Tracks daily P&L, consecutive losses, trade frequency, and
    portfolio exposure. All strategies should check can_trade()
    before placing orders and record_trade() after fills.

    Args:
        config: RiskManagerConfig with all risk parameters.
    """

    def __init__(self, config: Optional[RiskManagerConfig] = None) -> None:
        self.config = config or RiskManagerConfig()

        # Daily tracking
        self._daily_pnl: float = 0.0
        self._daily_pnl_date: date = trading_day()
        self._daily_trade_count: int = 0

        # Consecutive loss tracking
        self._consecutive_losses: int = 0
        self._cooldown_until: float = 0.0

        # Trade frequency tracking
        self._trade_timestamps: List[float] = []
        self._last_trade_time: float = 0.0

        # Drawdown tracking
        self._peak_equity: float = self.config.total_capital
        self._current_equity: float = self.config.total_capital
        self._circuit_breaker_until: float = 0.0

        # Open positions tracking
        self._open_positions: dict[str, float] = {}  # symbol -> risk_amount

        # Trade history
        self._trade_history: List[TradeRecord] = []

        # Pyramiding tracking
        self._pyramid_counts: dict[str, int] = {}  # symbol -> number of adds

        # Monthly risk cap tracking
        self._monthly_pnl: float = 0.0
        self._monthly_reset_date: date = self._next_monthly_reset()

        # Thread safety — protects ALL mutable state
        # RLock, not Lock: the read-side gates nest (get_status and
        # validate_order both call can_trade), so a plain mutex would
        # self-deadlock once they are synchronised.
        self._state_lock = threading.RLock()

        # State persistence
        self._persist_conn: Optional[sqlite3.Connection] = None
        self._persist_lock = threading.Lock()
        if self.config.persist_path:
            self._init_persistence(self.config.persist_path)
            self._load_state()

    # ─── State Persistence ───

    def _init_persistence(self, db_path: str) -> None:
        """Initialize the SQLite database for state persistence."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._persist_conn = sqlite3.connect(db_path, check_same_thread=False)
        # FIX 8: SQLite WAL mode for better concurrency
        self._persist_conn.execute("PRAGMA journal_mode=WAL")
        self._persist_conn.execute("PRAGMA busy_timeout=5000")
        self._persist_conn.execute(
            "CREATE TABLE IF NOT EXISTS risk_state "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        self._persist_conn.commit()
        logger.info("RiskManager persistence initialized: %s", db_path)

    def _save_state(self) -> None:
        """Persist current risk state to SQLite."""
        if self._persist_conn is None:
            return

        now = datetime.now().isoformat()
        state = {
            "daily_pnl": self._daily_pnl,
            "daily_pnl_date": self._daily_pnl_date.isoformat(),
            "consecutive_losses": self._consecutive_losses,
            "current_equity": self._current_equity,
            "peak_equity": self._peak_equity,
            "circuit_breaker_until": self._circuit_breaker_until,
            "cooldown_until": self._cooldown_until,
            "daily_trade_count": self._daily_trade_count,
            "open_positions": self._open_positions,
            "pyramid_counts": self._pyramid_counts,
            "monthly_pnl": self._monthly_pnl,
            "monthly_reset_date": self._monthly_reset_date.isoformat(),
        }

        with self._persist_lock:
            try:
                for key, value in state.items():
                    self._persist_conn.execute(
                        "INSERT OR REPLACE INTO risk_state (key, value, updated_at) "
                        "VALUES (?, ?, ?)",
                        (key, json.dumps(value), now),
                    )
                self._persist_conn.commit()
            except Exception as e:
                logger.error("Failed to save risk state: %s", e)

    def _load_state(self) -> None:
        """Restore risk state from SQLite on startup."""
        if self._persist_conn is None:
            return

        with self._persist_lock:
            try:
                rows = self._persist_conn.execute(
                    "SELECT key, value FROM risk_state"
                ).fetchall()
            except Exception as e:
                logger.error("Failed to load risk state: %s", e)
                return

        if not rows:
            logger.info("No persisted risk state found — starting fresh")
            return

        state = {key: json.loads(value) for key, value in rows}

        saved_date_str = state.get("daily_pnl_date")
        if saved_date_str:
            saved_date = date.fromisoformat(saved_date_str)
            if saved_date == trading_day():
                self._daily_pnl = float(state.get("daily_pnl", 0.0))
                self._daily_trade_count = int(state.get("daily_trade_count", 0))
                self._daily_pnl_date = saved_date
            else:
                logger.info("Persisted state is from %s — resetting daily counters", saved_date_str)

        self._consecutive_losses = int(state.get("consecutive_losses", 0))
        self._current_equity = float(state.get("current_equity", self.config.total_capital))
        self._peak_equity = float(state.get("peak_equity", self.config.total_capital))
        self._circuit_breaker_until = float(state.get("circuit_breaker_until", 0.0))
        self._cooldown_until = float(state.get("cooldown_until", 0.0))

        positions = state.get("open_positions")
        if isinstance(positions, dict):
            self._open_positions = {k: float(v) for k, v in positions.items()}

        # Restore pyramid counts
        pyramid_counts = state.get("pyramid_counts")
        if isinstance(pyramid_counts, dict):
            self._pyramid_counts = {k: int(v) for k, v in pyramid_counts.items()}

        # Restore monthly P&L
        monthly_pnl = state.get("monthly_pnl")
        if monthly_pnl is not None:
            self._monthly_pnl = float(monthly_pnl)
        monthly_reset = state.get("monthly_reset_date")
        if monthly_reset:
            try:
                self._monthly_reset_date = date.fromisoformat(monthly_reset)
            except (ValueError, TypeError):
                pass

        logger.info(
            "Risk state restored: equity=$%.2f, daily_pnl=$%.2f, "
            "consecutive_losses=%d, positions=%d",
            self._current_equity, self._daily_pnl,
            self._consecutive_losses, len(self._open_positions),
        )

    def _reset_daily_if_needed(self) -> None:
        """Reset daily counters if a new day has started."""
        if self._daily_pnl_date != trading_day():
            logger.info(
                "New day — resetting daily counters. Previous day P&L: $%.2f",
                self._daily_pnl,
            )
            self._daily_pnl = 0.0
            self._daily_trade_count = 0
            self._daily_pnl_date = trading_day()

    # ─── Position Sizing ───

    @_synchronized
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_price: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> int:
        """Calculate position size based on configured sizing method.

        Args:
            symbol: Ticker symbol.
            entry_price: Planned entry price.
            stop_price: Stop loss price (required for fixed_fractional).
            atr: Average True Range (alternative to stop_price).

        Returns:
            Number of shares to trade.
        """
        method = self.config.sizing_method

        if method == SizingMethod.FIXED_SHARES:
            return self._apply_position_caps(self.config.fixed_shares, entry_price)

        if method == SizingMethod.FIXED_DOLLAR:
            if entry_price <= 0:
                return 0
            raw = max(1, int(self.config.fixed_dollar_amount / entry_price))
            return self._apply_position_caps(raw, entry_price)

        if method == SizingMethod.KELLY:
            raw = self._kelly_size(entry_price)
            return self._apply_position_caps(raw, entry_price)

        # FIXED_FRACTIONAL (default)
        raw_size = self._fixed_fractional_size(entry_price, stop_price, atr)
        return self._apply_position_caps(raw_size, entry_price)

    def _apply_position_caps(self, shares: int, entry_price: float) -> int:
        """Apply production safety caps to any position size.

        Enforces:
        - Max position as % of equity
        - Max notional dollar cap
        - Max shares per order (fat-finger)

        Returns 0 when no whole share fits inside the caps. Callers must treat
        0 as "do not trade" rather than assuming a minimum of one share: for an
        instrument priced above the cap, one share already breaches it. This
        previously skipped the cap whenever it rounded down to zero and then
        floored the result at 1, so a $30,000 stock on a $100k account with a
        25% cap returned 3 shares — $90,000, or 90% of equity.
        """
        if entry_price <= 0 or shares <= 0:
            return 0

        # Cap: max % of equity
        max_by_equity = int(
            self._current_equity * (self.config.max_position_pct_equity / 100.0) / entry_price
        )
        shares = min(shares, max_by_equity)

        # Cap: max notional dollars
        if self.config.max_position_notional > 0:
            max_by_notional = int(self.config.max_position_notional / entry_price)
            shares = min(shares, max_by_notional)

        # Cap: max shares per order (fat-finger)
        shares = min(shares, self.config.max_shares_per_order)

        if shares <= 0:
            logger.warning(
                "Position rejected: no whole share of a $%.2f instrument fits within "
                "the position caps (%.1f%% of $%.2f equity)",
                entry_price, self.config.max_position_pct_equity, self._current_equity,
            )
            return 0

        return shares

    def _fixed_fractional_size(
        self,
        entry_price: float,
        stop_price: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> int:
        """Risk a fixed % of capital per trade.

        risk_amount = capital * risk_pct
        shares = risk_amount / risk_per_share
        """
        risk_amount = self._current_equity * (self.config.risk_per_trade_pct / 100.0)

        if stop_price is not None and stop_price != entry_price:
            risk_per_share = abs(entry_price - stop_price)
        elif atr is not None and atr > 0:
            risk_per_share = atr * 2.0
        else:
            risk_per_share = entry_price * 0.02  # default 2% of price

        if risk_per_share <= 0:
            return 0

        shares = int(risk_amount / risk_per_share)
        return max(1, shares)

    def _kelly_size(self, entry_price: float) -> int:
        """Kelly criterion position sizing (half-Kelly by default).

        f* = (W * R - L) / R
        where W = win rate, R = avg_win / avg_loss, L = loss rate
        """
        w = self.config.kelly_win_rate
        r = (
            self.config.kelly_avg_win / self.config.kelly_avg_loss
            if self.config.kelly_avg_loss > 0
            else 1.0
        )
        loss_rate = 1.0 - w

        kelly_pct = (w * r - loss_rate) / r if r > 0 else 0
        kelly_pct = max(0, kelly_pct) * self.config.kelly_fraction

        dollar_amount = self._current_equity * kelly_pct
        if entry_price <= 0:
            return 0

        return max(1, int(dollar_amount / entry_price))

    # ─── Trade Validation ───

    @_synchronized
    def can_trade(self) -> bool:
        """Check all risk gates. Returns True if trading is allowed.

        Checks: daily loss, cooldown, circuit breaker, trade frequency.
        """
        self._reset_daily_if_needed()

        # Daily loss limit
        if self._daily_pnl <= -self.config.max_daily_loss:
            logger.warning(
                "Daily loss limit reached: $%.2f (max: $%.2f)",
                self._daily_pnl,
                self.config.max_daily_loss,
            )
            return False

        # Consecutive loss cooldown
        now = time.time()
        if now < self._cooldown_until:
            remaining = self._cooldown_until - now
            logger.warning(
                "Cooldown active: %d consecutive losses. %.0fs remaining.",
                self._consecutive_losses,
                remaining,
            )
            return False

        # Drawdown circuit breaker
        if now < self._circuit_breaker_until:
            remaining = self._circuit_breaker_until - now
            logger.warning(
                "Circuit breaker active. %.0fs remaining.",
                remaining,
            )
            return False

        # Trade frequency: max per hour
        cutoff = now - 3600
        recent_trades = [t for t in self._trade_timestamps if t > cutoff]
        if len(recent_trades) >= self.config.max_trades_per_hour:
            logger.warning(
                "Trade frequency limit: %d trades in last hour (max: %d)",
                len(recent_trades),
                self.config.max_trades_per_hour,
            )
            return False

        # Minimum time between trades
        if now - self._last_trade_time < self.config.min_seconds_between_trades:
            logger.debug("Too soon since last trade. Waiting.")
            return False

        # Max open positions
        if len(self._open_positions) >= self.config.max_open_positions:
            logger.warning(
                "Max open positions reached: %d (max: %d)",
                len(self._open_positions),
                self.config.max_open_positions,
            )
            return False

        # Monthly loss cap (Elder's 6% rule)
        if self.config.max_monthly_loss > 0:
            self._reset_monthly_if_needed()
            if self._monthly_pnl <= -self.config.max_monthly_loss:
                logger.warning(
                    "Monthly loss limit reached: $%.2f (max: $%.2f)",
                    self._monthly_pnl,
                    self.config.max_monthly_loss,
                )
                return False

        return True

    @_synchronized
    def check_portfolio_heat(self, additional_risk: float = 0.0) -> bool:
        """Check if adding a new position would exceed portfolio heat limit.

        Heat is the *gross* dollar risk at stake. Short positions are stored in
        ``_open_positions`` as negative amounts (see ``validate_order``, which
        counts negative values as shorts), so summing them signed let a short
        cancel out a long: a book of $15k long risk and $15k short risk
        reported 0% heat and approved every further position. Risk does not
        net — both legs can lose at once — so magnitudes are summed.

        Args:
            additional_risk: Dollar risk of the proposed new position.

        Returns:
            True if within limits.
        """
        current_heat = sum(abs(v) for v in self._open_positions.values())
        total_heat = current_heat + abs(additional_risk)
        heat_pct = (total_heat / self._current_equity) * 100 if self._current_equity > 0 else 100

        if heat_pct > self.config.max_portfolio_heat_pct:
            logger.warning(
                "Portfolio heat would be %.1f%% (max: %.1f%%)",
                heat_pct,
                self.config.max_portfolio_heat_pct,
            )
            return False

        return True

    # ─── Trade Recording ───

    def record_trade(self, symbol: str = "UNKNOWN", pnl: float = 0.0) -> None:
        """Record a completed trade for risk tracking.

        Thread-safe: acquires _state_lock to protect all mutable state.
        """
        with self._state_lock:
            self._record_trade_unlocked(symbol, pnl)

    def _record_trade_unlocked(self, symbol: str, pnl: float) -> None:
        """Internal record_trade without lock (called under _state_lock)."""
        self._reset_daily_if_needed()

        self._daily_pnl += pnl
        self._daily_trade_count += 1

        now = time.time()
        self._trade_timestamps.append(now)
        self._last_trade_time = now

        # Trim old timestamps (keep last 2 hours)
        cutoff = now - 7200
        self._trade_timestamps = [t for t in self._trade_timestamps if t > cutoff]

        # Update equity
        self._current_equity += pnl
        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity

        # Consecutive loss tracking
        if pnl < 0:
            self._consecutive_losses += 1
            logger.info(
                "Loss #%d (streak): $%.2f on %s",
                self._consecutive_losses,
                pnl,
                symbol,
            )

            if self._consecutive_losses >= self.config.max_consecutive_losses:
                self._cooldown_until = now + self.config.cooldown_seconds
                logger.warning(
                    "Cooldown triggered: %d consecutive losses. Pausing %ds.",
                    self._consecutive_losses,
                    self.config.cooldown_seconds,
                )
        else:
            if self._consecutive_losses > 0:
                logger.info(
                    "Win streak reset after %d consecutive losses.",
                    self._consecutive_losses,
                )
            self._consecutive_losses = 0

        # Drawdown circuit breaker
        drawdown_pct = (
            (self._peak_equity - self._current_equity) / self._peak_equity * 100
            if self._peak_equity > 0
            else 0
        )
        if drawdown_pct >= self.config.max_drawdown_pct:
            pause_seconds = self.config.circuit_breaker_pause_hours * 3600
            self._circuit_breaker_until = now + pause_seconds
            logger.warning(
                "CIRCUIT BREAKER: Drawdown %.1f%% exceeds max %.1f%%. "
                "Pausing %.0f hours.",
                drawdown_pct,
                self.config.max_drawdown_pct,
                self.config.circuit_breaker_pause_hours,
            )

        # Monthly P&L tracking
        if self.config.max_monthly_loss > 0:
            self._reset_monthly_if_needed()
            self._monthly_pnl += pnl

        # Record in history
        self._trade_history.append(TradeRecord(symbol=symbol, pnl=pnl))
        # FIX 9: Trim trade history to prevent unbounded growth
        if len(self._trade_history) > 1000:
            self._trade_history = self._trade_history[-500:]
        logger.info(
            "Trade recorded: %s P&L=$%.2f | Daily P&L=$%.2f | Equity=$%.2f",
            symbol,
            pnl,
            self._daily_pnl,
            self._current_equity,
        )
        self._save_state()

    # ─── Position Tracking ───

    def add_position(self, symbol: str, risk_amount: float) -> bool:
        """Register an open position's risk amount. Thread-safe.

        Enforces max_open_positions inside the lock and returns whether the
        slot was granted.

        can_trade() already refuses once the limit is reached, but checking it
        and then calling this method is two operations: concurrent callers can
        all observe can_trade() as True before any of them adds a position. A
        test doing exactly that opened 15 positions against a limit of 10:

            AssertionError: Accepted 15 positions, max is 10

        Both methods were individually thread-safe, which is not the same as
        the pair being atomic. Enforcing the ceiling here makes the limit hold
        regardless of how a caller sequences its checks.

        Args:
            symbol: Ticker symbol. Re-registering an existing symbol updates
                its risk amount and never consumes a new slot.
            risk_amount: Dollar risk carried by the position.

        Returns:
            True if the position was recorded; False if the account is already
            at max_open_positions and this symbol is not among them.
        """
        with self._state_lock:
            is_new = symbol not in self._open_positions
            if is_new and len(self._open_positions) >= self.config.max_open_positions:
                logger.warning(
                    "Position rejected: %s — already at max_open_positions (%d)",
                    symbol, self.config.max_open_positions,
                )
                return False

            self._open_positions[symbol] = risk_amount
            logger.info(
                "Position added: %s risk=$%.2f | Total positions: %d",
                symbol, risk_amount, len(self._open_positions),
            )
        self._save_state()
        return True

    def remove_position(self, symbol: str) -> None:
        """Remove a closed position from tracking. Thread-safe."""
        with self._state_lock:
            if symbol in self._open_positions:
                del self._open_positions[symbol]
                logger.info(
                    "Position removed: %s | Remaining positions: %d",
                    symbol, len(self._open_positions),
                )
        self._save_state()

    # ─── Status ───

    @_synchronized
    def get_status(self) -> dict:
        """Get current risk manager status.

        Returns:
            Dictionary with all risk metrics.
        """
        self._reset_daily_if_needed()

        now = time.time()
        drawdown_pct = (
            (self._peak_equity - self._current_equity) / self._peak_equity * 100
            if self._peak_equity > 0
            else 0
        )
        # Gross, matching check_portfolio_heat — see the note there on why
        # short and long risk must not net against each other.
        total_heat = sum(abs(v) for v in self._open_positions.values())
        heat_pct = (total_heat / self._current_equity) * 100 if self._current_equity > 0 else 0

        recent_trades_1h = len([t for t in self._trade_timestamps if t > now - 3600])

        return {
            "can_trade": self.can_trade(),
            "current_equity": round(self._current_equity, 2),
            "peak_equity": round(self._peak_equity, 2),
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_trade_count": self._daily_trade_count,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_active": now < self._cooldown_until,
            "cooldown_remaining_s": max(0, round(self._cooldown_until - now)),
            "circuit_breaker_active": now < self._circuit_breaker_until,
            "drawdown_pct": round(drawdown_pct, 2),
            "open_positions": len(self._open_positions),
            "portfolio_heat_pct": round(heat_pct, 2),
            "trades_last_hour": recent_trades_1h,
            "total_trades": len(self._trade_history),
            "monthly_pnl": round(self._monthly_pnl, 2),
            "pyramid_positions": len(self._pyramid_counts),
            "max_position_pct_equity": self.config.max_position_pct_equity,
            "max_shares_per_order": self.config.max_shares_per_order,
            "enforce_market_hours": self.config.enforce_market_hours,
        }

    # ─── Production Safety Gates ───

    @_synchronized
    def validate_order(
        self,
        symbol: str,
        shares: int,
        price: float,
        last_price: float,
        direction: str = "LONG",
        avg_daily_volume: Optional[float] = None,
    ) -> tuple:
        """Pre-order validation — catches fat-finger errors and unsafe orders.

        Args:
            symbol: Ticker symbol.
            shares: Number of shares to order.
            price: Intended order price.
            last_price: Most recent market price for sanity check.
            direction: "LONG" or "SHORT".
            avg_daily_volume: Average daily volume for liquidity check.

        Returns:
            Tuple of (approved: bool, reason: str).
            If approved, reason is "OK". Otherwise, describes the rejection.
        """
        # 1. Fat-finger: max shares per order
        if abs(shares) > self.config.max_shares_per_order:
            return False, (
                f"Order size {shares} exceeds max_shares_per_order "
                f"({self.config.max_shares_per_order})"
            )

        # 2. Price sanity: reject if price deviates >X% from last known price
        if last_price > 0 and price > 0:
            deviation_pct = abs(price - last_price) / last_price * 100
            if deviation_pct > self.config.max_price_deviation_pct:
                return False, (
                    f"Price ${price:.2f} deviates {deviation_pct:.1f}% from "
                    f"last price ${last_price:.2f} (max {self.config.max_price_deviation_pct}%)"
                )

        # 3. Max notional check
        notional = abs(shares) * price
        if self.config.max_position_notional > 0 and notional > self.config.max_position_notional:
            return False, (
                f"Notional ${notional:,.0f} exceeds max_position_notional "
                f"(${self.config.max_position_notional:,.0f})"
            )

        # 4. Max position as % of equity
        max_notional_by_equity = self._current_equity * (self.config.max_position_pct_equity / 100.0)
        if notional > max_notional_by_equity:
            return False, (
                f"Notional ${notional:,.0f} exceeds {self.config.max_position_pct_equity}% "
                f"of equity (${max_notional_by_equity:,.0f})"
            )

        # 5. Short-selling checks
        if direction.upper() == "SHORT":
            short_count = sum(
                1 for v in self._open_positions.values() if v < 0
            )
            if short_count >= self.config.max_short_positions:
                return False, (
                    f"Max short positions reached ({self.config.max_short_positions})"
                )

            total_short_exposure = sum(
                abs(v) for v in self._open_positions.values() if v < 0
            )
            new_short_pct = (
                (total_short_exposure + notional) / self._current_equity * 100
                if self._current_equity > 0 else 100
            )
            if new_short_pct > self.config.max_short_exposure_pct:
                return False, (
                    f"Short exposure would be {new_short_pct:.1f}% "
                    f"(max {self.config.max_short_exposure_pct}%)"
                )

        # 6. Liquidity check
        if avg_daily_volume is not None and avg_daily_volume > 0:
            if avg_daily_volume < self.config.min_avg_volume:
                return False, (
                    f"Avg daily volume {avg_daily_volume:,.0f} below minimum "
                    f"({self.config.min_avg_volume:,})"
                )
            position_pct_adv = abs(shares) / avg_daily_volume * 100
            if position_pct_adv > self.config.max_position_pct_adv:
                return False, (
                    f"Position is {position_pct_adv:.1f}% of avg daily volume "
                    f"(max {self.config.max_position_pct_adv}%)"
                )

        # 7. Standard risk gates
        if not self.can_trade():
            return False, "Risk gates blocked (can_trade=False)"

        # The notional is pre-formatted rather than passed as %,.0f: the comma
        # flag is a str.format feature and is not valid in %-style formatting,
        # so logging raised ValueError("unsupported format character ','")
        # every time an order passed validation and a handler formatted the
        # record. With no formatting handler attached the record is never
        # rendered, which is why this stayed hidden.
        logger.info(
            "Order validated: %s %s %d shares @ $%.2f (notional=$%s)",
            direction, symbol, shares, price, f"{notional:,.0f}",
        )
        return True, "OK"

    def check_market_hours(self) -> tuple:
        """Check if trading is allowed based on current market hours.

        Returns:
            Tuple of (allowed: bool, reason: str).
        """
        if not self.config.enforce_market_hours:
            return True, "Market hours enforcement disabled"

        try:
            from shared.data.public_data_fetcher import PublicDataFetcher
            fetcher = PublicDataFetcher(cache_enabled=False)
            if fetcher.is_market_open():
                return True, "Market is open"

            if self.config.allow_premarket or self.config.allow_afterhours:
                return True, "Extended hours trading allowed"

            return False, "Market is closed — trading blocked"

        except Exception as e:
            logger.warning("Market hours check failed: %s — allowing trade", e)
            return True, "Market hours check failed (allowing)"

    # ─── Pyramiding (Livermore) ───

    @_synchronized
    def can_pyramid(
        self,
        symbol: str,
        current_price: float,
        avg_entry: float,
        pyramid_count: Optional[int] = None,
    ) -> bool:
        """Check if a pyramid add is allowed for a winning position.

        Conditions:
        - Pyramiding must be enabled in config.
        - All standard risk gates (can_trade) must pass.
        - Unrealised profit must exceed threshold (scaled by level).
        - Max pyramid levels not exceeded.

        Args:
            symbol: Ticker symbol.
            current_price: Current market price.
            avg_entry: Average entry price of the existing position.
            pyramid_count: Override for current pyramid level (uses tracked count if None).

        Returns:
            True if pyramiding is allowed.
        """
        if not self.config.enable_pyramiding:
            return False

        # Check ALL standard risk gates before allowing pyramid add
        if not self.can_trade():
            logger.debug("%s: pyramid blocked — risk gates not clear", symbol)
            return False

        count = pyramid_count if pyramid_count is not None else self._pyramid_counts.get(symbol, 0)
        if count >= self.config.max_pyramid_levels:
            logger.debug("%s: max pyramid levels reached (%d)", symbol, count)
            return False

        if avg_entry <= 0:
            return False

        unrealised_pct = (current_price - avg_entry) / avg_entry * 100
        threshold = self.config.pyramid_threshold_pct * (1 + count)
        if unrealised_pct < threshold:
            logger.debug(
                "%s: unrealised profit %.2f%% < threshold %.2f%% for level %d",
                symbol, unrealised_pct, threshold, count + 1,
            )
            return False

        return True

    def calculate_pyramid_size(self, base_size: int, pyramid_level: int) -> int:
        """Calculate position size for a pyramid add (scales down each level).

        Level 0 = base_size, Level 1 = base * scale_factor,
        Level 2 = base * scale_factor^2, etc.

        Args:
            base_size: Original position size in shares.
            pyramid_level: Current pyramid level (0-based).

        Returns:
            Number of shares for this pyramid add.
        """
        factor = self.config.pyramid_scale_factor ** pyramid_level
        return max(1, int(base_size * factor))

    def record_pyramid(self, symbol: str) -> int:
        """Record a pyramid add for a symbol, returns new pyramid count."""
        with self._state_lock:
            self._pyramid_counts[symbol] = self._pyramid_counts.get(symbol, 0) + 1
            logger.info(
                "%s: pyramid level %d recorded",
                symbol, self._pyramid_counts[symbol],
            )
            count = self._pyramid_counts[symbol]
        self._save_state()
        return count

    def reset_pyramid(self, symbol: str) -> None:
        """Reset pyramid count when a position is fully closed."""
        with self._state_lock:
            self._pyramid_counts.pop(symbol, None)
        self._save_state()

    def get_pyramid_count(self, symbol: str) -> int:
        """Return current pyramid level for a symbol."""
        return self._pyramid_counts.get(symbol, 0)

    # ─── Monthly Risk Cap (Elder 6% Rule) ───

    def _next_monthly_reset(self) -> date:
        """Calculate the next monthly reset date.

        Clamps the reset day to the last day of the month to avoid
        ValueError when monthly_reset_day > days in month (e.g., 31 in Feb).
        """
        import calendar
        today = trading_day()
        reset_day = self.config.monthly_reset_day
        if today.day >= reset_day:
            # Next month
            month = today.month + 1
            year = today.year
            if month > 12:
                month = 1
                year += 1
            max_day = calendar.monthrange(year, month)[1]
            return date(year, month, min(reset_day, max_day))
        max_day = calendar.monthrange(today.year, today.month)[1]
        return date(today.year, today.month, min(reset_day, max_day))

    def _reset_monthly_if_needed(self) -> None:
        """Reset monthly P&L counter if the reset date has passed."""
        today = trading_day()
        if today >= self._monthly_reset_date:
            logger.info(
                "Monthly reset — previous month P&L: $%.2f",
                self._monthly_pnl,
            )
            self._monthly_pnl = 0.0
            self._monthly_reset_date = self._next_monthly_reset()

    def get_monthly_pnl(self) -> float:
        """Return current month's accumulated P&L."""
        self._reset_monthly_if_needed()
        return self._monthly_pnl

    def __repr__(self) -> str:
        status = self.get_status()
        return (
            f"RiskManager(equity=${status['current_equity']:,.2f}, "
            f"daily_pnl=${status['daily_pnl']:+,.2f}, "
            f"positions={status['open_positions']}, "
            f"can_trade={status['can_trade']})"
        )
