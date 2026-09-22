"""The trading day must be measured at the exchange, not on the host.

Daily and monthly counters reset against date.today() — the host's local date.
On a UTC host (every CI runner, most containers, most cloud VMs) that rolls
over at 19:00/20:00 New York, i.e. during the US session. The accumulated
daily loss was wiped mid-session and the daily stop re-armed at zero while
positions were still open.
"""

import os
import sys
from datetime import date, datetime, timezone

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.utils import trading_clock
from shared.utils.trading_clock import exchange_now, trading_day

zoneinfo = pytest.importorskip("zoneinfo")
from zoneinfo import ZoneInfo


class TestTradingDay:
    def test_returns_a_date(self):
        assert isinstance(trading_day(), date)

    def test_matches_new_york_not_utc(self):
        ny_today = datetime.now(timezone.utc).astimezone(
            ZoneInfo("America/New_York")).date()
        assert trading_day() == ny_today

    def test_evening_utc_is_still_the_previous_us_trading_day(self):
        """23:30 UTC is 19:30 New York — the same session day, not the next."""
        utc_moment = datetime(2026, 3, 10, 23, 30, tzinfo=timezone.utc)
        ny = utc_moment.astimezone(ZoneInfo("America/New_York"))
        assert utc_moment.date() == date(2026, 3, 10)
        assert ny.date() == date(2026, 3, 10)

        later = datetime(2026, 3, 11, 1, 30, tzinfo=timezone.utc)
        assert later.date() == date(2026, 3, 11)          # UTC has rolled over
        assert later.astimezone(
            ZoneInfo("America/New_York")).date() == date(2026, 3, 10)  # NY has not

    def test_exchange_now_is_timezone_aware(self):
        assert exchange_now().tzinfo is not None

    def test_exchange_timezone_is_configurable(self, monkeypatch):
        monkeypatch.setenv(trading_clock.ENV_VAR, "Europe/London")
        london = datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/London")).date()
        assert trading_day() == london

    def test_unknown_timezone_falls_back_without_raising(self, monkeypatch):
        monkeypatch.setenv(trading_clock.ENV_VAR, "Not/AZone")
        assert isinstance(trading_day(), date)


class TestRiskManagersUseExchangeTime:
    def test_risk_manager_stamps_the_exchange_day(self):
        from shared.risk_manager import RiskManager, RiskManagerConfig

        rm = RiskManager(config=RiskManagerConfig(total_capital=100_000.0))
        assert rm._daily_pnl_date == trading_day()

    def test_unified_gate_stamps_the_exchange_day(self):
        from shared.risk_manager_unified import (
            UnifiedPortfolioRiskGate, UnifiedRiskConfig,
        )

        UnifiedPortfolioRiskGate.reset_instance()
        gate = UnifiedPortfolioRiskGate(UnifiedRiskConfig())
        assert gate._daily_pnl_date == trading_day()
        UnifiedPortfolioRiskGate.reset_instance()

    def test_daily_pnl_survives_a_utc_midnight_crossing(self, monkeypatch):
        """The counter must not reset just because UTC rolled over."""
        from shared.risk_manager import RiskManager, RiskManagerConfig

        rm = RiskManager(config=RiskManagerConfig(
            total_capital=100_000.0, max_daily_loss=1e9))
        rm.record_trade("X", -1_000.0)
        assert rm.get_status()["daily_pnl"] == pytest.approx(-1_000.0)

        # Same exchange day => no reset, whatever the host clock says.
        assert rm._daily_pnl_date == trading_day()
        assert rm.get_status()["daily_pnl"] == pytest.approx(-1_000.0)
