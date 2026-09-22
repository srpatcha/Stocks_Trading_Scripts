"""
Comprehensive tests for OrderManager (interactive_brokers.utils.order_manager).
"""

import sys
import os
import uuid
import time
from datetime import date, datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interactive_brokers.utils.order_manager import (
    OrderManager,
    OrderSide,
    OrderState,
    OrderStatus,
    RiskConfig,
)
# trading_day(), not date.today(): the reset logic measures the session
# at the exchange, so on a UTC host the host date is a day ahead and
# stamping state with it looks like a brand new trading day.
from shared.utils.trading_clock import trading_day


# ── Fixtures ──


@pytest.fixture
def mock_connection():
    conn = MagicMock()
    conn.qualifyContracts = MagicMock()
    conn.placeOrder = MagicMock()
    conn.cancelOrder = MagicMock()
    return conn


@pytest.fixture
def mock_connection_with_ib():
    conn = MagicMock()
    conn.qualifyContracts = MagicMock()
    conn.placeOrder = MagicMock()
    conn.cancelOrder = MagicMock()
    conn.ib = MagicMock()
    return conn


@pytest.fixture
def mock_notifier():
    notifier = MagicMock()
    notifier.info = MagicMock()
    notifier.warning = MagicMock()
    notifier.error = MagicMock()
    return notifier


@pytest.fixture
def default_manager(mock_connection):
    return OrderManager(mock_connection)


@pytest.fixture
def manager_with_notifier(mock_connection, mock_notifier):
    return OrderManager(mock_connection, notifier=mock_notifier)


def _make_trade(order_id=1):
    trade = MagicMock()
    trade.order = MagicMock()
    trade.order.orderId = order_id
    return trade


# ── Constructor / Config Tests ──


class TestOrderManagerInit:
    def test_init_default_risk_config(self, mock_connection):
        mgr = OrderManager(mock_connection)
        assert isinstance(mgr.risk_config, RiskConfig)
        assert mgr.risk_config.max_position_pct == 0.05

    def test_init_dict_config(self, mock_connection):
        mgr = OrderManager(mock_connection, config={"max_order_value": 10000.0})
        assert mgr.risk_config.max_order_value == 10000.0
        assert mgr.risk_config.max_position_pct == 0.05  # default kept

    def test_init_dict_config_ignores_unknown_keys(self, mock_connection):
        mgr = OrderManager(mock_connection, config={"max_order_value": 10000.0, "bogus": 42})
        assert mgr.risk_config.max_order_value == 10000.0

    def test_init_risk_config_instance(self, mock_connection):
        rc = RiskConfig(max_daily_loss=999.0)
        mgr = OrderManager(mock_connection, config=rc)
        assert mgr.risk_config.max_daily_loss == 999.0

    def test_init_none_config(self, mock_connection):
        mgr = OrderManager(mock_connection, config=None)
        assert isinstance(mgr.risk_config, RiskConfig)


# ── _create_contract Tests ──


class TestCreateContract:
    @patch("interactive_brokers.utils.order_manager.OrderManager._create_contract")
    def test_create_stock_contract(self, mock_create, default_manager):
        mock_create.return_value = MagicMock()
        contract = default_manager._create_contract("AAPL")
        mock_create.assert_called_once_with("AAPL")

    def test_create_contract_stock_via_ib_async(self, mock_connection):
        mgr = OrderManager(mock_connection)
        with patch.dict("sys.modules", {"ib_async": MagicMock()}):
            import sys as _sys
            ib_async = _sys.modules["ib_async"]
            mock_stock = MagicMock()
            ib_async.Stock.return_value = mock_stock
            contract = mgr._create_contract("AAPL", "STK", "SMART", "USD")
            ib_async.Stock.assert_called_once_with("AAPL", "SMART", "USD")
            mock_connection.qualifyContracts.assert_called_with(mock_stock)

    def test_create_contract_option_via_ib_async(self, mock_connection):
        mgr = OrderManager(mock_connection)
        with patch.dict("sys.modules", {"ib_async": MagicMock()}):
            import sys as _sys
            ib_async = _sys.modules["ib_async"]
            mock_opt = MagicMock()
            ib_async.Option.return_value = mock_opt
            contract = mgr._create_contract(
                "AAPL", "OPT", "SMART", "USD",
                expiry="20250620", strike=150, right="C",
            )
            ib_async.Option.assert_called_once()

    def test_create_contract_future_via_ib_async(self, mock_connection):
        mgr = OrderManager(mock_connection)
        with patch.dict("sys.modules", {"ib_async": MagicMock()}):
            import sys as _sys
            ib_async = _sys.modules["ib_async"]
            mock_fut = MagicMock()
            ib_async.Future.return_value = mock_fut
            contract = mgr._create_contract(
                "ES", "FUT", "CME", "USD", expiry="20250620",
            )
            ib_async.Future.assert_called_once()


# ── place_order / market_order Tests ──


class TestPlaceOrder:
    def test_market_order_places_and_tracks(self, mock_connection):
        mgr = OrderManager(mock_connection)
        trade = _make_trade(order_id=42)
        mock_connection.placeOrder.return_value = trade

        with patch.object(mgr, "_create_contract", return_value=MagicMock()):
            with patch.object(mgr, "_create_order", return_value=MagicMock()):
                result = mgr.market_order("AAPL", "BUY", 10)

        assert result == trade
        assert 42 in mgr._orders
        assert mgr._orders[42].symbol == "AAPL"
        assert mgr._orders[42].action == "BUY"
        assert mgr._orders[42].quantity == 10

    def test_limit_order_places_and_tracks(self, mock_connection):
        mgr = OrderManager(mock_connection)
        trade = _make_trade(order_id=55)
        mock_connection.placeOrder.return_value = trade

        with patch.object(mgr, "_create_contract", return_value=MagicMock()):
            with patch.object(mgr, "_create_order", return_value=MagicMock()):
                result = mgr.limit_order("AAPL", "BUY", 10, 150.0)

        assert result == trade
        assert mgr._orders[55].limit_price == 150.0

    def test_limit_price_zero_handled(self, mock_connection):
        """Verify fix: limit_price=0.0 is correctly rejected (must be > 0)."""
        mgr = OrderManager(mock_connection)

        with pytest.raises(ValueError, match="Limit price must be positive"):
            mgr.limit_order("AAPL", "BUY", 10, limit_price=0.0)

    def test_stop_order_places_and_tracks(self, mock_connection):
        mgr = OrderManager(mock_connection)
        trade = _make_trade(order_id=77)
        mock_connection.placeOrder.return_value = trade

        with patch.object(mgr, "_create_contract", return_value=MagicMock()):
            with patch.object(mgr, "_create_order", return_value=MagicMock()):
                result = mgr.stop_order("AAPL", "SELL", 10, 140.0)

        assert mgr._orders[77].stop_price == 140.0


# ── cancel_order Tests ──


class TestCancelOrder:
    def test_cancel_active_order(self, mock_connection_with_ib):
        mgr = OrderManager(mock_connection_with_ib)
        mgr._orders[1] = OrderStatus(
            order_id=1, symbol="AAPL", action="BUY",
            order_type="MKT", quantity=10, status=OrderState.SUBMITTED,
        )
        open_order = MagicMock()
        open_order.orderId = 1
        mock_connection_with_ib.ib.openOrders.return_value = [open_order]

        result = mgr.cancel_order(1)
        assert result is True
        assert mgr._orders[1].status == OrderState.CANCELLED

    def test_cancel_inactive_order_returns_false(self, mock_connection_with_ib):
        mgr = OrderManager(mock_connection_with_ib)
        mgr._orders[1] = OrderStatus(
            order_id=1, symbol="AAPL", action="BUY",
            order_type="MKT", quantity=10, status=OrderState.FILLED,
        )
        result = mgr.cancel_order(1)
        assert result is False

    def test_cancel_order_not_found(self, mock_connection_with_ib):
        mgr = OrderManager(mock_connection_with_ib)
        mgr._orders[1] = OrderStatus(
            order_id=1, symbol="AAPL", action="BUY",
            order_type="MKT", quantity=10, status=OrderState.SUBMITTED,
        )
        mock_connection_with_ib.ib.openOrders.return_value = []
        result = mgr.cancel_order(1)
        assert result is False

    def test_cancel_order_hasattr_guard_no_ib(self, mock_connection):
        """Verify fix: hasattr guard for .ib attribute."""
        del mock_connection.ib  # ensure no .ib
        mgr = OrderManager(mock_connection)
        mgr._orders[1] = OrderStatus(
            order_id=1, symbol="AAPL", action="BUY",
            order_type="MKT", quantity=10, status=OrderState.SUBMITTED,
        )
        result = mgr.cancel_order(1)
        assert result is False


# ── modify_order Tests ──


class TestModifyOrder:
    def test_modify_order_price(self, mock_connection_with_ib):
        mgr = OrderManager(mock_connection_with_ib)
        mgr._orders[1] = OrderStatus(
            order_id=1, symbol="AAPL", action="BUY",
            order_type="LMT", quantity=10, limit_price=150.0,
            status=OrderState.SUBMITTED,
        )
        trade_mock = MagicMock()
        trade_mock.order.orderId = 1
        trade_mock.order.lmtPrice = 150.0
        mock_connection_with_ib.ib.openTrades.return_value = [trade_mock]

        result = mgr.modify_order(1, new_limit_price=155.0)
        assert result is True
        assert mgr._orders[1].limit_price == 155.0
        assert trade_mock.order.lmtPrice == 155.0

    def test_modify_order_quantity(self, mock_connection_with_ib):
        mgr = OrderManager(mock_connection_with_ib)
        mgr._orders[1] = OrderStatus(
            order_id=1, symbol="AAPL", action="BUY",
            order_type="LMT", quantity=10,
            status=OrderState.SUBMITTED,
        )
        trade_mock = MagicMock()
        trade_mock.order.orderId = 1
        mock_connection_with_ib.ib.openTrades.return_value = [trade_mock]

        result = mgr.modify_order(1, new_quantity=20)
        assert result is True
        assert mgr._orders[1].quantity == 20

    def test_modify_untracked_order(self, mock_connection_with_ib):
        mgr = OrderManager(mock_connection_with_ib)
        result = mgr.modify_order(999)
        assert result is False

    def test_modify_inactive_order(self, mock_connection_with_ib):
        mgr = OrderManager(mock_connection_with_ib)
        mgr._orders[1] = OrderStatus(
            order_id=1, symbol="AAPL", action="BUY",
            order_type="MKT", quantity=10, status=OrderState.FILLED,
        )
        result = mgr.modify_order(1, new_limit_price=200.0)
        assert result is False

    def test_modify_order_hasattr_guard_no_ib(self, mock_connection):
        """Verify fix: hasattr guard for .ib attribute on modify_order."""
        del mock_connection.ib
        mgr = OrderManager(mock_connection)
        mgr._orders[1] = OrderStatus(
            order_id=1, symbol="AAPL", action="BUY",
            order_type="LMT", quantity=10,
            status=OrderState.SUBMITTED,
        )
        result = mgr.modify_order(1, new_limit_price=200.0)
        assert result is False


# ── _check_daily_loss Tests ──


class TestCheckDailyLoss:
    def test_daily_loss_not_exceeded_allows_trade(self, default_manager):
        default_manager._daily_pnl = -100.0
        default_manager._daily_pnl_date = trading_day()
        default_manager._check_daily_loss()  # should not raise

    def test_daily_loss_exceeded_blocks_trade(self, default_manager):
        default_manager._daily_pnl = -5000.0
        default_manager._daily_pnl_date = trading_day()
        with pytest.raises(ValueError, match="Daily loss limit reached"):
            default_manager._check_daily_loss()

    def test_profitable_day_not_blocked(self, default_manager):
        """Verify fix: profitable days NOT blocked by _check_daily_loss."""
        default_manager._daily_pnl = 5000.0
        default_manager._daily_pnl_date = trading_day()
        default_manager._check_daily_loss()  # must NOT raise

    def test_zero_pnl_not_blocked(self, default_manager):
        default_manager._daily_pnl = 0.0
        default_manager._daily_pnl_date = trading_day()
        default_manager._check_daily_loss()  # must NOT raise

    def test_daily_pnl_resets_on_new_day(self, default_manager):
        default_manager._daily_pnl = -9999.0
        default_manager._daily_pnl_date = date(2000, 1, 1)
        default_manager._check_daily_loss()  # should reset and pass
        assert default_manager._daily_pnl == 0.0
        assert default_manager._daily_pnl_date == trading_day()


# ── _track_order / UUID Fallback Tests ──


class TestTrackOrder:
    def test_track_order_uses_order_id_from_trade(self, default_manager):
        trade = _make_trade(order_id=123)
        status = default_manager._track_order(
            trade, "AAPL", "BUY", "MKT", 10,
        )
        assert status.order_id == 123
        assert 123 in default_manager._orders

    def test_track_order_uuid_fallback_when_no_order_attr(self, default_manager):
        """Verify fix: uuid fallback instead of id()."""
        trade = MagicMock(spec=[])  # no .order attribute
        status = default_manager._track_order(
            trade, "AAPL", "BUY", "MKT", 10,
        )
        assert isinstance(status.order_id, int)
        assert status.order_id > 0
        assert status.order_id in default_manager._orders


# ── Order Lifecycle Tests ──


class TestOrderLifecycle:
    def test_full_lifecycle_place_fill_pnl(self, mock_connection):
        mgr = OrderManager(mock_connection)
        trade = _make_trade(order_id=10)
        mock_connection.placeOrder.return_value = trade

        with patch.object(mgr, "_create_contract", return_value=MagicMock()):
            with patch.object(mgr, "_create_order", return_value=MagicMock()):
                mgr.market_order("AAPL", "BUY", 100)

        # Simulate fill
        status = mgr._orders[10]
        assert status.is_active
        assert not status.is_filled

        status.status = OrderState.FILLED
        status.filled_quantity = 100
        status.avg_fill_price = 150.0
        assert status.is_filled
        assert not status.is_active

        # Update daily PnL
        mgr.update_daily_pnl(500.0)
        assert mgr._daily_pnl == 500.0

    def test_order_status_properties(self):
        os_ = OrderStatus(
            order_id=1, symbol="AAPL", action="BUY",
            order_type="MKT", quantity=10,
        )
        assert os_.is_active  # default PENDING
        assert not os_.is_filled

        os_.status = OrderState.FILLED
        assert os_.is_filled
        assert not os_.is_active

        os_.status = OrderState.CANCELLED
        assert not os_.is_active
        assert not os_.is_filled


# ── Notification Tests ──


class TestNotify:
    def test_notify_info(self, manager_with_notifier, mock_notifier):
        manager_with_notifier._notify("test message")
        mock_notifier.info.assert_called_once_with("test message")

    def test_notify_warning(self, manager_with_notifier, mock_notifier):
        manager_with_notifier._notify("warn msg", level="warning")
        mock_notifier.warning.assert_called_once_with("warn msg")

    def test_notify_error(self, manager_with_notifier, mock_notifier):
        manager_with_notifier._notify("err msg", level="error")
        mock_notifier.error.assert_called_once_with("err msg")

    def test_notify_handles_exception(self, manager_with_notifier, mock_notifier):
        mock_notifier.info.side_effect = RuntimeError("boom")
        manager_with_notifier._notify("test")  # should not raise


# ── Risk Checks ──


class TestRiskChecks:
    def test_position_size_exceeds_max(self, default_manager):
        with pytest.raises(ValueError, match="exceeds max position size"):
            default_manager._check_position_size(9999)

    def test_position_size_zero_rejected(self, default_manager):
        with pytest.raises(ValueError, match="must be positive"):
            default_manager._check_position_size(0)

    def test_position_size_negative_rejected(self, default_manager):
        with pytest.raises(ValueError, match="must be positive"):
            default_manager._check_position_size(-5)

    def test_cooldown_blocks_trading(self, default_manager):
        default_manager._cooldown_until = time.time() + 1000
        default_manager._loss_streak = 3
        with pytest.raises(ValueError, match="Cooldown active"):
            default_manager._check_cooldown()

    def test_cooldown_expired_allows_trading(self, default_manager):
        default_manager._cooldown_until = time.time() - 1
        default_manager._check_cooldown()  # should not raise

    def test_validate_order_max_open_orders(self, mock_connection):
        mgr = OrderManager(mock_connection, config={"max_open_orders": 2})
        for i in range(2):
            mgr._orders[i] = OrderStatus(
                order_id=i, symbol="AAPL", action="BUY",
                order_type="MKT", quantity=1,
                status=OrderState.SUBMITTED,
            )
        with pytest.raises(ValueError, match="Open order count"):
            mgr._validate_order("AAPL", "BUY", 1)

    def test_validate_order_max_value(self, mock_connection):
        mgr = OrderManager(mock_connection, config={"max_order_value": 1000.0})
        with pytest.raises(ValueError, match="exceeds max"):
            mgr._validate_order("AAPL", "BUY", 100, estimated_price=20.0)


# ── record_trade_result / Loss Streak ──


class TestRecordTradeResult:
    def test_loss_streak_increments(self, default_manager):
        default_manager.record_trade_result(-100)
        assert default_manager._loss_streak == 1
        default_manager.record_trade_result(-200)
        assert default_manager._loss_streak == 2

    def test_profitable_trade_resets_loss_streak(self, default_manager):
        default_manager._loss_streak = 2
        default_manager.record_trade_result(50)
        assert default_manager._loss_streak == 0

    def test_cooldown_activates_on_streak(self, default_manager):
        default_manager.risk_config.cooldown_after_losses = 3
        default_manager.record_trade_result(-1)
        default_manager.record_trade_result(-1)
        default_manager.record_trade_result(-1)
        assert default_manager._cooldown_until > time.time()

    def test_get_open_orders(self, default_manager):
        default_manager._orders[1] = OrderStatus(
            order_id=1, symbol="AAPL", action="BUY",
            order_type="MKT", quantity=10, status=OrderState.SUBMITTED,
        )
        default_manager._orders[2] = OrderStatus(
            order_id=2, symbol="MSFT", action="SELL",
            order_type="MKT", quantity=5, status=OrderState.FILLED,
        )
        open_orders = default_manager.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].order_id == 1

    def test_get_all_orders(self, default_manager):
        default_manager._orders[1] = OrderStatus(
            order_id=1, symbol="AAPL", action="BUY",
            order_type="MKT", quantity=10,
        )
        result = default_manager.get_all_orders()
        assert 1 in result
