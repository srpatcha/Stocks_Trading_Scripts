"""Atomic reservations and correct position accounting in the portfolio gate.

This gate is the only risk control on the BrokerBridge order path, so its
accounting has to be right.

Two defects motivated these tests:

- can_open_position() released the lock before the order went out, so N
  concurrent callers all saw the same headroom and all passed. Measured: 8
  concurrent $14k orders landing at 112% of equity against an 80% cap.
- register_position() overwrote _positions[symbol] instead of accumulating,
  while can_open_position() accumulated. The check and the bookkeeping
  disagreed, and the bookkeeping understated risk.
"""

import os
import sys
import threading

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.risk_manager_unified import (
    UnifiedPortfolioRiskGate,
    UnifiedRiskConfig,
)


@pytest.fixture
def gate():
    UnifiedPortfolioRiskGate.reset_instance()
    g = UnifiedPortfolioRiskGate(UnifiedRiskConfig(
        account_equity=100_000.0,
        max_portfolio_exposure=0.80,
        max_single_stock_pct=1.0,      # isolate the portfolio-level cap
        max_sector_pct=1.0,
        max_correlated_exposure=1.0,
    ))
    yield g
    UnifiedPortfolioRiskGate.reset_instance()


class TestReservationClosesTheRace:
    def test_concurrent_reservations_respect_the_exposure_cap(self, gate):
        """8 threads x $14k against an 80% cap on $100k must not all pass."""
        accepted = []
        barrier = threading.Barrier(8)

        def attempt():
            barrier.wait()  # maximise overlap
            ok, _ = gate.reserve("SYM", 14_000.0)
            if ok:
                accepted.append(True)

        threads = [threading.Thread(target=attempt) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        exposure = gate.get_portfolio_summary()["total_exposure"]
        assert exposure <= 80_000.0, f"exposure ${exposure} exceeds the 80% cap"
        assert len(accepted) == 5, f"expected 5 of 8 to fit, got {len(accepted)}"

    def test_sequential_check_then_act_would_have_overcommitted(self, gate):
        """Documents the old behaviour so the regression is visible."""
        for _ in range(8):
            ok, _ = gate.can_open_position("SYM", 14_000.0)
            assert ok, "point-in-time check never sees the in-flight orders"
        # Nothing was booked, which is exactly why it was unsafe.
        assert gate.get_portfolio_summary()["total_exposure"] == 0.0


class TestReservationLifecycle:
    def test_release_returns_the_headroom(self, gate):
        ok, _ = gate.reserve("AAPL", 30_000.0)
        assert ok
        assert gate.get_portfolio_summary()["total_exposure"] == 30_000.0
        gate.release_reservation("AAPL", 30_000.0)
        assert gate.get_portfolio_summary()["total_exposure"] == 0.0

    def test_commit_converts_reservation_into_a_position(self, gate):
        gate.reserve("AAPL", 15_000.0)
        gate.commit_reservation("AAPL", 15_000.0, 100, 150.0, "s", "ib")
        summary = gate.get_portfolio_summary()
        assert summary["open_positions"] == 1
        assert summary["total_exposure"] == pytest.approx(15_000.0)

    def test_commit_does_not_double_count(self, gate):
        gate.reserve("AAPL", 15_000.0)
        gate.commit_reservation("AAPL", 15_000.0, 100, 150.0, "s", "ib")
        assert gate.get_portfolio_summary()["total_exposure"] == pytest.approx(15_000.0)

    def test_release_is_clamped_to_what_was_held(self, gate):
        gate.reserve("AAPL", 10_000.0)
        gate.release_reservation("AAPL", 999_999.0)
        assert gate.get_portfolio_summary()["total_exposure"] == 0.0

    def test_release_without_a_reservation_is_a_noop(self, gate):
        gate.register_position("AAPL", 100, 150.0, "s", "ib")
        before = gate.get_portfolio_summary()["total_exposure"]
        gate.release_reservation("AAPL", 15_000.0)
        assert gate.get_portfolio_summary()["total_exposure"] == before

    def test_a_failed_order_frees_headroom_for_the_next_one(self, gate):
        gate.reserve("A", 79_000.0)
        ok, _ = gate.reserve("B", 5_000.0)
        assert ok is False
        gate.release_reservation("A", 79_000.0)
        ok, _ = gate.reserve("B", 5_000.0)
        assert ok is True


class TestPositionAccounting:
    def test_adding_to_a_position_accumulates(self, gate):
        for _ in range(5):
            gate.register_position("AAPL", 100, 140.0, "s", "ib")
        summary = gate.get_portfolio_summary()
        assert summary["positions"]["AAPL"]["shares"] == 500
        assert summary["total_exposure"] == pytest.approx(5 * 100 * 140.0)

    def test_average_entry_is_share_weighted(self, gate):
        gate.register_position("AAPL", 100, 100.0, "s", "ib")
        gate.register_position("AAPL", 300, 200.0, "s", "ib")
        pos = gate.get_portfolio_summary()["positions"]["AAPL"]
        assert pos["shares"] == 400
        assert pos["entry_price"] == pytest.approx(175.0)

    def test_partial_close_keeps_the_remainder(self, gate):
        gate.register_position("AAPL", 100, 150.0, "s", "ib")
        gate.close_position("AAPL", pnl=100.0, shares=40)
        summary = gate.get_portfolio_summary()
        assert summary["open_positions"] == 1
        assert summary["positions"]["AAPL"]["shares"] == 60
        assert summary["total_exposure"] == pytest.approx(9_000.0)

    def test_full_close_removes_the_position(self, gate):
        gate.register_position("AAPL", 100, 150.0, "s", "ib")
        gate.close_position("AAPL", pnl=100.0)
        assert gate.get_portfolio_summary()["open_positions"] == 0
        assert gate.get_portfolio_summary()["total_exposure"] == 0.0

    def test_oversized_partial_close_closes_everything(self, gate):
        gate.register_position("AAPL", 100, 150.0, "s", "ib")
        gate.close_position("AAPL", pnl=0.0, shares=500)
        assert gate.get_portfolio_summary()["open_positions"] == 0

    def test_short_position_accumulates_too(self, gate):
        gate.register_position("AAPL", -100, 150.0, "s", "ib")
        gate.register_position("AAPL", -100, 150.0, "s", "ib")
        pos = gate.get_portfolio_summary()["positions"]["AAPL"]
        assert pos["shares"] == -200
        assert gate.get_portfolio_summary()["total_exposure"] == pytest.approx(30_000.0)

    def test_single_stock_cap_sees_the_accumulated_total(self, gate):
        g = UnifiedPortfolioRiskGate(UnifiedRiskConfig(
            account_equity=100_000.0, max_single_stock_pct=0.15,
            max_portfolio_exposure=1.0, max_sector_pct=1.0,
            max_correlated_exposure=1.0,
        ))
        g.register_position("AAPL", 100, 100.0, "s", "ib")   # $10k
        ok, reason = g.can_open_position("AAPL", 6_000.0)    # would be $16k > 15%
        assert ok is False
        assert "concentration" in reason.lower()


class TestGateIsProperlyConfiguredByBrokerBridge:
    """BrokerBridge built the gate with get_instance() and no config at all.

    That left the only risk gate on the order path with no persistence, a
    hard-coded $100k equity that ignored capital=, and a $10k daily loss limit
    instead of the documented $5k.
    """

    def _bridge(self, tmp_path, **kwargs):
        from shared.daemon.broker_bridge import BrokerBridge

        UnifiedPortfolioRiskGate.reset_instance()
        defaults = dict(
            broker="ib", mode="paper",
            diary_path=str(tmp_path / "diary.jsonl"),
        )
        defaults.update(kwargs)
        return BrokerBridge(**defaults)

    def test_gate_equity_follows_the_bridge_capital(self, tmp_path):
        bridge = self._bridge(tmp_path, capital=250_000.0)
        assert bridge._portfolio_gate._account_equity == pytest.approx(250_000.0)
        UnifiedPortfolioRiskGate.reset_instance()

    def test_daily_loss_limit_defaults_to_the_documented_5k(self, tmp_path):
        bridge = self._bridge(tmp_path)
        assert bridge._portfolio_gate._max_daily_loss == pytest.approx(5_000.0)
        UnifiedPortfolioRiskGate.reset_instance()

    def test_limits_are_configurable(self, tmp_path):
        bridge = self._bridge(tmp_path, config={
            "portfolio_risk": {"max_daily_loss": 1_234.0, "max_sector_pct": 0.11},
        })
        assert bridge._portfolio_gate._max_daily_loss == pytest.approx(1_234.0)
        assert bridge._portfolio_gate._max_sector_pct == pytest.approx(0.11)
        UnifiedPortfolioRiskGate.reset_instance()

    def test_gate_state_survives_a_restart(self, tmp_path):
        """A crash mid-day used to zero the daily P&L and every position."""
        db = str(tmp_path / "unified_risk.db")

        bridge = self._bridge(tmp_path, config={
            "portfolio_risk": {"persist_path": db},
        })
        bridge._portfolio_gate.register_position("AAPL", 100, 150.0, "s", "ib")
        bridge._portfolio_gate.close_position("MSFT", pnl=-2_500.0)

        # Simulate a process restart against the same database.
        UnifiedPortfolioRiskGate.reset_instance()
        restarted = self._bridge(tmp_path, config={
            "portfolio_risk": {"persist_path": db},
        })
        summary = restarted._portfolio_gate.get_portfolio_summary()

        assert summary["open_positions"] == 1
        assert summary["total_exposure"] == pytest.approx(15_000.0)
        assert summary["daily_pnl"] == pytest.approx(-2_500.0)
        UnifiedPortfolioRiskGate.reset_instance()

    def test_second_get_instance_with_a_different_config_warns(self, tmp_path, caplog):
        """A discarded config used to be silently ignored."""
        import logging

        UnifiedPortfolioRiskGate.reset_instance()
        UnifiedPortfolioRiskGate.get_instance(
            UnifiedRiskConfig(account_equity=100_000.0))
        with caplog.at_level(logging.WARNING):
            UnifiedPortfolioRiskGate.get_instance(
                UnifiedRiskConfig(account_equity=999_000.0))
        assert any("IGNORED" in r.message for r in caplog.records)
        UnifiedPortfolioRiskGate.reset_instance()
