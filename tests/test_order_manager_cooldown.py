"""Tests for OrderManager cooldown and record_trade_result functionality."""

import pytest
import time
from unittest.mock import MagicMock, patch

from interactive_brokers.utils.order_manager import (
    OrderManager,
    RiskConfig,
    OrderStatus,
    OrderState,
    OrderSide,
)


# ─── Fixtures ───


@pytest.fixture
def mock_connection():
    """Mock IB connection object."""
    conn = MagicMock()
    conn.qualifyContracts = MagicMock()
    conn.placeOrder = MagicMock(return_value=MagicMock(
        order=MagicMock(orderId=1001)
    ))
    return conn


@pytest.fixture
def default_om(mock_connection):
    """OrderManager with default RiskConfig."""
    return OrderManager(mock_connection)


@pytest.fixture
def custom_om(mock_connection):
    """OrderManager with tight cooldown settings for testing."""
    config = RiskConfig(
        cooldown_after_losses=2,
        cooldown_pause_seconds=60,
        max_daily_loss=1000.0,
    )
    return OrderManager(mock_connection, config=config)


# ─── RiskConfig Tests ───


class TestRiskConfig:
    """Tests for the RiskConfig dataclass with new cooldown fields."""

    def test_default_cooldown_fields(self):
        config = RiskConfig()
        assert config.cooldown_after_losses == 3
        assert config.cooldown_pause_seconds == 1800

    def test_custom_cooldown_fields(self):
        config = RiskConfig(cooldown_after_losses=5, cooldown_pause_seconds=300)
        assert config.cooldown_after_losses == 5
        assert config.cooldown_pause_seconds == 300

    def test_config_from_dict(self, mock_connection):
        config_dict = {
            "max_daily_loss": 3000.0,
            "cooldown_after_losses": 4,
            "cooldown_pause_seconds": 600,
        }
        om = OrderManager(mock_connection, config=config_dict)
        assert om.risk_config.cooldown_after_losses == 4
        assert om.risk_config.cooldown_pause_seconds == 600
        assert om.risk_config.max_daily_loss == 3000.0

    def test_config_from_dict_ignores_unknown_keys(self, mock_connection):
        config_dict = {
            "max_daily_loss": 2000.0,
            "unknown_field": "should_be_ignored",
        }
        om = OrderManager(mock_connection, config=config_dict)
        assert om.risk_config.max_daily_loss == 2000.0


# ─── OrderManager Initialization Tests ───


class TestOrderManagerInit:
    """Tests for OrderManager initialization with cooldown fields."""

    def test_loss_streak_initialized(self, default_om):
        assert default_om._loss_streak == 0

    def test_cooldown_until_initialized(self, default_om):
        assert default_om._cooldown_until == 0.0

    def test_accepts_risk_config_object(self, mock_connection):
        config = RiskConfig(cooldown_after_losses=10)
        om = OrderManager(mock_connection, config=config)
        assert om.risk_config.cooldown_after_losses == 10

    def test_no_config_uses_defaults(self, mock_connection):
        om = OrderManager(mock_connection)
        assert om.risk_config.cooldown_after_losses == 3
        assert om.risk_config.max_position_size == 1000


# ─── record_trade_result() Tests ───


class TestRecordTradeResult:
    """Tests for the record_trade_result method."""

    def test_single_loss_increments_streak(self, custom_om):
        custom_om.record_trade_result(pnl=-100.0)
        assert custom_om._loss_streak == 1

    def test_multiple_losses_accumulate(self, custom_om):
        custom_om.record_trade_result(pnl=-100.0)
        custom_om.record_trade_result(pnl=-200.0)
        assert custom_om._loss_streak == 2

    def test_winning_trade_resets_streak(self, custom_om):
        custom_om.record_trade_result(pnl=-100.0)
        assert custom_om._loss_streak == 1
        custom_om.record_trade_result(pnl=200.0)
        assert custom_om._loss_streak == 0

    def test_cooldown_triggered_at_threshold(self, custom_om):
        # cooldown_after_losses=2
        custom_om.record_trade_result(pnl=-100.0)
        assert custom_om._cooldown_until == 0.0  # not yet
        custom_om.record_trade_result(pnl=-100.0)
        assert custom_om._cooldown_until > time.time()  # now triggered

    def test_cooldown_duration(self, custom_om):
        # cooldown_pause_seconds=60
        before = time.time()
        custom_om.record_trade_result(pnl=-100.0)
        custom_om.record_trade_result(pnl=-100.0)
        after = time.time()
        assert custom_om._cooldown_until >= before + 60
        assert custom_om._cooldown_until <= after + 60 + 1  # small tolerance

    def test_zero_pnl_resets_streak(self, custom_om):
        custom_om.record_trade_result(pnl=-100.0)
        assert custom_om._loss_streak == 1
        custom_om.record_trade_result(pnl=0.0)
        assert custom_om._loss_streak == 0

    def test_profit_after_cooldown_trigger_resets_streak(self, custom_om):
        custom_om.record_trade_result(pnl=-100.0)
        custom_om.record_trade_result(pnl=-100.0)
        assert custom_om._loss_streak == 2
        custom_om.record_trade_result(pnl=500.0)
        assert custom_om._loss_streak == 0

    def test_notifier_called_on_cooldown(self, mock_connection):
        notifier = MagicMock()
        config = RiskConfig(cooldown_after_losses=1, cooldown_pause_seconds=30)
        om = OrderManager(mock_connection, config=config, notifier=notifier)
        om.record_trade_result(pnl=-100.0)
        notifier.warning.assert_called_once()
        call_args = notifier.warning.call_args[0][0]
        assert "Cooldown" in call_args


# ─── _check_cooldown() Tests ───


class TestCheckCooldown:
    """Tests for the _check_cooldown validation method."""

    def test_no_cooldown_does_not_raise(self, custom_om):
        custom_om._check_cooldown()   # nothing set: allowed

        # And the gate is reachable, so the above is not vacuous.
        custom_om._cooldown_until = time.time() + 600
        with pytest.raises(ValueError):
            custom_om._check_cooldown()

    def test_active_cooldown_raises(self, custom_om):
        custom_om._cooldown_until = time.time() + 3600
        custom_om._loss_streak = 2
        with pytest.raises(ValueError, match="Cooldown active"):
            custom_om._check_cooldown()

    def test_expired_cooldown_does_not_raise(self, custom_om):
        custom_om._cooldown_until = time.time() - 10   # expired 10s ago
        custom_om._check_cooldown()

        custom_om._cooldown_until = time.time() + 10   # 10s still to run
        with pytest.raises(ValueError):
            custom_om._check_cooldown()


# ─── Integration: _validate_order with cooldown ───


class TestValidateOrderWithCooldown:
    """Tests that _validate_order includes cooldown checks."""

    def test_validate_passes_normally(self, custom_om):
        custom_om._validate_order("AAPL", "BUY", 100, estimated_price=150.0)

        # An order far beyond the size cap must still be refused, so a passing
        # validation means something.
        with pytest.raises(ValueError):
            custom_om._validate_order(
                "AAPL", "BUY", custom_om.risk_config.max_position_size + 1,
                estimated_price=150.0,
            )

    def test_validate_rejects_during_cooldown(self, custom_om):
        custom_om._cooldown_until = time.time() + 3600
        custom_om._loss_streak = 2
        with pytest.raises(ValueError, match="Cooldown active"):
            custom_om._validate_order("AAPL", "BUY", 100)

    def test_validate_still_checks_position_size(self, custom_om):
        with pytest.raises(ValueError, match="exceeds max position size"):
            custom_om._validate_order("AAPL", "BUY", 9999)

    def test_validate_still_checks_daily_loss(self, custom_om):
        custom_om._daily_pnl = -1000.0  # hit the limit
        with pytest.raises(ValueError, match="Daily loss limit"):
            custom_om._validate_order("AAPL", "BUY", 100)


# ─── OrderStatus Tests ───


class TestOrderStatus:
    """Tests for the OrderStatus dataclass."""

    def test_active_states(self):
        status = OrderStatus(
            order_id=1, symbol="AAPL", action="BUY",
            order_type="MKT", quantity=100,
            status=OrderState.SUBMITTED,
        )
        assert status.is_active is True
        assert status.is_filled is False

    def test_filled_state(self):
        status = OrderStatus(
            order_id=1, symbol="AAPL", action="BUY",
            order_type="MKT", quantity=100,
            status=OrderState.FILLED,
        )
        assert status.is_active is False
        assert status.is_filled is True

    def test_cancelled_state(self):
        status = OrderStatus(
            order_id=1, symbol="AAPL", action="BUY",
            order_type="MKT", quantity=100,
            status=OrderState.CANCELLED,
        )
        assert status.is_active is False
        assert status.is_filled is False


# ─── Webhook Server Regime Field Tests ───


class TestWebhookRegimeField:
    """Tests for the new regime and signal fields in AlertPayload."""

    def test_regime_field_accepted(self):
        """Verify the AlertPayload model accepts the regime field."""
        from tradingview.webhooks.webhook_server import AlertPayload

        payload = AlertPayload(
            symbol="AAPL",
            action="buy",
            price=150.0,
            regime="TRENDING",
            signal="trend_long",
        )
        assert payload.regime == "TRENDING"
        assert payload.signal == "trend_long"

    def test_regime_field_optional(self):
        """Verify regime field is optional (backward compatible)."""
        from tradingview.webhooks.webhook_server import AlertPayload

        payload = AlertPayload(
            symbol="AAPL",
            action="buy",
            price=150.0,
        )
        assert payload.regime is None
        assert payload.signal is None

    def test_regime_field_in_json(self):
        """Verify regime field serializes correctly."""
        from tradingview.webhooks.webhook_server import AlertPayload

        payload = AlertPayload(
            symbol="SPY",
            action="sell",
            price=450.0,
            regime="VOLATILE",
            signal="squeeze_short",
            strategy="vol_breakout",
        )
        data = payload.model_dump()
        assert data["regime"] == "VOLATILE"
        assert data["signal"] == "squeeze_short"
        assert data["strategy"] == "vol_breakout"
