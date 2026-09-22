"""The trading day, in exchange time.

Daily and monthly counters were reset against ``date.today()`` — the host's
local date. On a UTC-configured host (every CI runner, most containers, most
cloud VMs) that rolls over at 19:00 or 20:00 New York time, i.e. *during* the
US session on the preceding evening's date. The accumulated daily loss was
wiped mid-session, so the daily stop silently re-armed at zero while positions
were still open.

Everything that asks "what trading day is it?" should use ``trading_day()``.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9
    try:
        from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]
    except ImportError:
        ZoneInfo = None  # type: ignore[assignment,misc]

#: Exchange the trading day is measured against. Override for non-US venues.
ENV_VAR = "STOCKS_PLUGIN_EXCHANGE_TZ"
DEFAULT_EXCHANGE_TZ = "America/New_York"

_warned = False


def exchange_timezone():
    """Return the exchange tzinfo, or None if zoneinfo is unavailable."""
    global _warned
    if ZoneInfo is None:
        if not _warned:
            logger.warning(
                "zoneinfo unavailable — falling back to local time for the "
                "trading day. Install tzdata to get correct session boundaries."
            )
            _warned = True
        return None
    name = os.environ.get(ENV_VAR, DEFAULT_EXCHANGE_TZ)
    try:
        return ZoneInfo(name)
    except Exception as e:  # pragma: no cover - bad tz name
        logger.error("Unknown exchange timezone %r (%s) — using local time", name, e)
        return None


def exchange_now() -> datetime:
    """Current time at the exchange."""
    tz = exchange_timezone()
    if tz is None:
        return datetime.now()
    return datetime.now(timezone.utc).astimezone(tz)


def trading_day() -> date:
    """Today's date at the exchange.

    Use in place of ``date.today()`` for anything that resets per session.
    """
    return exchange_now().date()
