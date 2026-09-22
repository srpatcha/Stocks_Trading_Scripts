"""
Tests for shared/daemon/live_runner.py
"""
import sys, os, signal
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import numpy as np, pandas as pd, pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_ohlcv(n=100, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    price = 100.0
    rows = []
    for i in range(n):
        price *= 1 + rng.randn() * 0.015
        rows.append({"date": dates[i], "open": price * 1.001, "high": price * 1.005,
                      "low": price * 0.995, "close": price, "volume": 1_000_000})
    return pd.DataFrame(rows)


@pytest.fixture
def mock_deps():
    with patch("shared.data.public_data_fetcher.PublicDataFetcher") as mock_pdf, \
         patch("shared.ml.self_learning_agent.SelfLearningAgent") as mock_agent_cls, \
         patch("shared.ml.self_learning_agent.AgentConfig"), \
         patch("shared.daemon.live_runner.signal.signal"):
        mock_fetcher = MagicMock()
        mock_pdf.return_value = mock_fetcher
        mock_agent = MagicMock()
        mock_agent.decide.return_value = {"action": "HOLD", "confidence": 0.5, "regime": "TRENDING", "price": 150.0}
        mock_agent.get_performance.return_value = {"total_trades": 0, "win_rate": 0}
        mock_agent.get_weight_summary.return_value = {}
        mock_agent_cls.return_value = mock_agent
        yield {"fetcher_cls": mock_pdf, "fetcher": mock_fetcher, "agent_cls": mock_agent_cls, "agent": mock_agent}


def _create_runner(deps, **kw):
    from shared.daemon.live_runner import LiveRunner
    d = dict(symbols=["AAPL", "MSFT"], mode="monitor", interval_seconds=60, use_news=False)
    d.update(kw)
    return LiveRunner(**d)


class TestLiveRunnerInit:
    def test_basic_init(self, mock_deps):
        r = _create_runner(mock_deps)
        assert r._symbols == ["AAPL", "MSFT"]
        assert r._mode == "monitor"
        assert r._interval == 60
        assert r._running is False
        assert r._cycle_count == 0

    def test_init_paper_mode(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        assert r._mode == "paper"
        assert r._paper_capital == 100_000.0
        assert r._paper_pnl == 0.0

    def test_init_custom_models(self, mock_deps):
        r = _create_runner(mock_deps, models=["regime", "lstm"])
        assert r._models_to_train == ["regime", "lstm"]

    def test_init_default_models(self, mock_deps):
        r = _create_runner(mock_deps)
        assert r._models_to_train == ["regime"]

    def test_init_news_disabled(self, mock_deps):
        r = _create_runner(mock_deps, use_news=False)
        assert r._sentiment_analyzer is None

    def test_llm_reasoner_none_by_default(self, mock_deps):
        r = _create_runner(mock_deps)
        assert r._llm_reasoner is None

    def test_tick_count_starts_zero(self, mock_deps):
        r = _create_runner(mock_deps)
        assert r._tick_count == 0

    def test_entry_bars_dict_empty(self, mock_deps):
        r = _create_runner(mock_deps)
        assert r._entry_bars == {}


class TestPrintBanner:
    def test_banner_monitor(self, mock_deps, capsys):
        r = _create_runner(mock_deps, mode="monitor")
        r._print_banner()
        out = capsys.readouterr().out
        assert "MONITOR" in out
        assert "AAPL" in out

    def test_banner_paper(self, mock_deps, capsys):
        r = _create_runner(mock_deps, mode="paper")
        r._print_banner()
        assert "PAPER" in capsys.readouterr().out

    def test_banner_with_broker(self, mock_deps, capsys):
        r = _create_runner(mock_deps, mode="paper")
        r._broker_name = "ib"
        r._print_banner()
        assert "IB" in capsys.readouterr().out

    def test_banner_no_broker(self, mock_deps, capsys):
        r = _create_runner(mock_deps)
        r._broker_name = None
        r._print_banner()
        assert "NONE" in capsys.readouterr().out


class TestLLMInitBeforeStart:
    def test_llm_set_before_start(self, mock_deps):
        r = _create_runner(mock_deps)
        mock_llm = MagicMock()
        r._llm_reasoner = mock_llm
        assert r._llm_reasoner is mock_llm

    def test_llm_used_in_process_symbol(self, mock_deps):
        r = _create_runner(mock_deps, mode="monitor")
        mock_llm = MagicMock()
        mock_llm.reason.return_value = {"action": "BUY", "confidence": 0.9, "tp_price": 160.0,
                                         "sl_price": 140.0, "exit_plan": "trail", "reasoning": "bullish"}
        r._llm_reasoner = mock_llm
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        with patch.object(r, "_log_decision"):
            r._process_symbol("AAPL", market_open=True)
        mock_llm.reason.assert_called_once()

    def test_llm_failure_does_not_stop_the_cycle_or_open_a_trade(self, mock_deps):
        """Asserted nothing: it ran the method and passed if it returned.

        What matters is that a dead LLM endpoint degrades — the symbol is
        still processed and no position is opened off a failed reasoner.
        """
        r = _create_runner(mock_deps, mode="monitor")
        mock_llm = MagicMock()
        mock_llm.reason.side_effect = RuntimeError("API error")
        r._llm_reasoner = mock_llm
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)

        with patch.object(r, "_log_decision"):
            r._process_symbol("AAPL", market_open=True)

        mock_llm.reason.assert_called()          # the failing path really ran
        assert r._paper_positions == {}          # nothing opened off a failure

    def test_llm_override_blocked_by_risk(self, mock_deps):
        """LLM says BUY but risk manager blocks the trade."""
        r = _create_runner(mock_deps, mode="monitor")
        mock_llm = MagicMock()
        mock_llm.reason.return_value = {
            "action": "BUY", "confidence": 0.9,
            "tp_price": 160.0, "sl_price": 140.0,
        }
        r._llm_reasoner = mock_llm
        mock_risk = MagicMock()
        mock_risk.can_trade.return_value = False
        r._risk_manager = mock_risk

        mock_deps["agent"].decide.return_value = {
            "action": "HOLD", "confidence": 0.3, "regime": "TRENDING", "price": 150.0,
        }
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        with patch.object(r, "_log_decision") as ml:
            r._process_symbol("AAPL", market_open=True)
            call_args = ml.call_args
            assert call_args[0][1] == "HOLD"


class TestDatetimeUTC:
    def test_run_cycle_uses_utc(self, mock_deps):
        r = _create_runner(mock_deps, mode="monitor")
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        mock_deps["fetcher"].is_market_open.return_value = True
        with patch("shared.daemon.live_runner.datetime") as mock_dt:
            mock_now = MagicMock()
            mock_now.strftime.return_value = "2024-01-01 10:00:00"
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            r._run_cycle()
            mock_dt.now.assert_any_call(timezone.utc)


class TestSentimentFilter:
    def test_block_buy_negative(self, mock_deps):
        r = _create_runner(mock_deps)
        assert r._apply_sentiment_filter("BUY", -0.5, 0.8) == "HOLD"

    def test_allow_buy_neutral(self, mock_deps):
        r = _create_runner(mock_deps)
        assert r._apply_sentiment_filter("BUY", 0.0, 0.8) == "BUY"

    def test_block_sell_positive(self, mock_deps):
        r = _create_runner(mock_deps)
        assert r._apply_sentiment_filter("SELL", 0.5, 0.8) == "HOLD"

    def test_allow_sell_neutral(self, mock_deps):
        r = _create_runner(mock_deps)
        assert r._apply_sentiment_filter("SELL", 0.0, 0.8) == "SELL"

    def test_hold_passthrough(self, mock_deps):
        r = _create_runner(mock_deps)
        assert r._apply_sentiment_filter("HOLD", -0.9, 0.8) == "HOLD"


class TestPaperTrading:
    def test_buy_opens_long(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        r._execute_paper_trade("AAPL", "BUY", 150.0, 0.8)
        assert "AAPL" in r._paper_positions
        assert r._paper_positions["AAPL"]["direction"] == "long"
        assert r._paper_positions["AAPL"]["entry_price"] == 150.0

    def test_sell_closes_long(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        r._paper_positions["AAPL"] = {"shares": 100, "entry_price": 140.0,
                                       "entry_time": "2024-01-01", "direction": "long"}
        r._entry_bars["AAPL"] = 0
        r._execute_paper_trade("AAPL", "SELL", 150.0, 0.8)
        assert "AAPL" not in r._paper_positions
        # raw P&L = 1000.0, commission = 100 * 0.005 * 2 = 1.0
        assert r._paper_pnl == pytest.approx(1000.0 - 100 * 0.005 * 2)

    def test_sell_opens_short(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        r._execute_paper_trade("AAPL", "SELL", 150.0, 0.8)
        assert "AAPL" in r._paper_positions
        assert r._paper_positions["AAPL"]["direction"] == "short"

    def test_pnl_accumulates(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        r._paper_positions["A"] = {"shares": 50, "entry_price": 100.0,
                                    "entry_time": "2024-01-01", "direction": "long"}
        r._entry_bars["A"] = 0
        r._execute_paper_trade("A", "SELL", 110.0, 0.8)
        # raw = 500.0, commission = 50 * 0.005 * 2 = 0.50
        comm = 50 * 0.005 * 2
        assert r._paper_pnl == pytest.approx(500.0 - comm)
        r._paper_positions["B"] = {"shares": 50, "entry_price": 100.0,
                                    "entry_time": "2024-01-02", "direction": "long"}
        r._entry_bars["B"] = 0
        r._execute_paper_trade("B", "SELL", 90.0, 0.8)
        # raw A = 500, raw B = -500, total raw = 0, total comm = 0.50 * 2 = 1.0
        assert r._paper_pnl == pytest.approx(0.0 - 2 * comm)

    def test_paper_short_pnl_correct(self, mock_deps):
        """Short: entry=100, exit=95 → PnL = (100-95)*50 - commission."""
        r = _create_runner(mock_deps, mode="paper")
        r._paper_positions["X"] = {"shares": 50, "entry_price": 100.0,
                                    "entry_time": "2024-01-01", "direction": "short"}
        r._entry_bars["X"] = 0
        r._close_paper_position("X", 95.0)
        # raw = 250.0, commission = 50 * 0.005 * 2 = 0.50
        assert r._paper_pnl == pytest.approx(250.0 - 50 * 0.005 * 2)

    def test_paper_short_close_on_buy(self, mock_deps):
        """BUY signal with an open short → closes the short."""
        r = _create_runner(mock_deps, mode="paper")
        r._paper_positions["AAPL"] = {"shares": 50, "entry_price": 100.0,
                                       "entry_time": "2024-01-01", "direction": "short"}
        r._entry_bars["AAPL"] = 0
        r._execute_paper_trade("AAPL", "BUY", 95.0, 0.8)
        assert r._paper_pnl == pytest.approx(250.0 - 50 * 0.005 * 2)

    def test_holding_period_tracked(self, mock_deps):
        """entry_bars recorded, correct bars passed to record_outcome."""
        r = _create_runner(mock_deps, mode="paper")
        r._tick_count = 5
        r._execute_paper_trade("AAPL", "BUY", 150.0, 0.8)
        assert r._entry_bars["AAPL"] == 5

        r._tick_count = 15
        r._close_paper_position("AAPL", 160.0)
        call_kwargs = mock_deps["agent"].record_outcome.call_args[1]
        assert call_kwargs["holding_period_bars"] == 10

    def test_tick_count_increments(self, mock_deps):
        """_tick_count increases each cycle."""
        r = _create_runner(mock_deps, mode="monitor")
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        mock_deps["fetcher"].is_market_open.return_value = True
        initial = r._tick_count
        r._run_cycle()
        assert r._tick_count == initial + 1
        r._run_cycle()
        assert r._tick_count == initial + 2


class TestStartStopLifecycle:
    def test_signal_handler(self, mock_deps):
        r = _create_runner(mock_deps)
        r._running = True
        r._signal_handler(signal.SIGINT, None)
        assert r._running is False

    def test_shutdown_calls_close(self, mock_deps):
        r = _create_runner(mock_deps)
        r._agent = mock_deps["agent"]
        r._paper_positions = {}
        r._paper_pnl = 0.0
        r._cycle_count = 5
        with patch.object(mock_deps["agent"], "save_models"):
            r._shutdown()
        mock_deps["agent"].close.assert_called_once()

    def test_start_with_training(self, mock_deps):
        r = _create_runner(mock_deps, mode="monitor")
        def stop(r=r): r._running = False
        with patch.object(r, "_run_cycle", side_effect=lambda: stop()):
            with patch.object(r, "_initial_training") as mt:
                with patch.object(r, "_shutdown"):
                    with patch.object(r, "_smart_sleep"):
                        r.start(train_first=True)
                        mt.assert_called_once()

    def test_start_without_training(self, mock_deps):
        r = _create_runner(mock_deps, mode="monitor")
        def stop(r=r): r._running = False
        with patch.object(r, "_run_cycle", side_effect=lambda: stop()):
            with patch.object(r, "_initial_training") as mt:
                with patch.object(r, "_shutdown"):
                    with patch.object(r, "_smart_sleep"):
                        r.start(train_first=False)
                        mt.assert_not_called()

    def test_start_with_broker_connect_success(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        mock_bridge = MagicMock()
        mock_bridge.connect.return_value = True
        mock_bridge.get_account_info.return_value = {"balance": 100000}
        mock_bridge.is_connected.return_value = True
        r._broker_bridge = mock_bridge
        def stop(r=r): r._running = False
        with patch.object(r, "_run_cycle", side_effect=lambda: stop()):
            with patch.object(r, "_shutdown"):
                with patch.object(r, "_smart_sleep"):
                    r.start(train_first=False)
        mock_bridge.connect.assert_called_once()

    def test_start_with_broker_connect_failure(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        mock_bridge = MagicMock()
        mock_bridge.connect.return_value = False
        r._broker_bridge = mock_bridge
        def stop(r=r): r._running = False
        with patch.object(r, "_run_cycle", side_effect=lambda: stop()):
            with patch.object(r, "_shutdown"):
                with patch.object(r, "_smart_sleep"):
                    r.start(train_first=False)
        assert r._broker_bridge is None


class TestInitialTraining:
    def test_fetches_and_trains(self, mock_deps):
        r = _create_runner(mock_deps)
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(200)
        mock_deps["agent"].train.return_value = {"regime": "ok"}
        r._initial_training()
        mock_deps["fetcher"].fetch_ohlcv.assert_called_once()
        mock_deps["agent"].train.assert_called_once()

    def test_handles_empty_data(self, mock_deps):
        r = _create_runner(mock_deps)
        mock_deps["fetcher"].fetch_ohlcv.return_value = None
        r._initial_training()
        mock_deps["agent"].train.assert_not_called()

    def test_training_failure_leaves_the_runner_usable(self, mock_deps):
        """A failed train must not leave the runner in a started state."""
        r = _create_runner(mock_deps)
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(200)
        mock_deps["agent"].train.side_effect = RuntimeError("train fail")

        r._initial_training()

        mock_deps["agent"].train.assert_called()
        assert r._running is False
        assert r._cycle_count == 0


class TestProcessSymbol:
    def test_insufficient_data_skipped(self, mock_deps):
        r = _create_runner(mock_deps, mode="monitor")
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(n=20)
        with patch.object(r, "_log_decision") as ml:
            r._process_symbol("AAPL", market_open=True)
            ml.assert_not_called()

    def test_monitor_mode_logs_decision(self, mock_deps):
        r = _create_runner(mock_deps, mode="monitor")
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        with patch.object(r, "_log_decision") as ml:
            r._process_symbol("AAPL", market_open=True)
            ml.assert_called_once()

    def test_paper_mode_executes_trade(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        mock_deps["agent"].decide.return_value = {"action": "BUY", "confidence": 0.8, "regime": "TRENDING", "price": 150.0}
        with patch.object(r, "_execute_paper_trade") as mt:
            r._process_symbol("AAPL", market_open=True)
            mt.assert_called_once()


class TestStatusReport:
    @staticmethod
    def _capture_report(runner):
        """Capture what _print_status_report logs.

        A dedicated handler rather than caplog: webhook_server.setup_logging()
        calls logging.basicConfig(force=True), which removes caplog's handler
        if that module was imported earlier in the session. Attaching straight
        to the logger under test makes this independent of collection order.
        """
        import io
        import logging

        import shared.daemon.live_runner as live_runner_module

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        # "live_runner", not "shared.daemon.live_runner": this module (like
        # paper_trader and webhook_server) names its logger explicitly rather
        # than using __name__. Left as-is because deployment log config may
        # reference those names.
        log = logging.getLogger(live_runner_module.logger.name)
        previous_level, previous_propagate = log.level, log.propagate
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        try:
            runner._print_status_report()
        finally:
            log.removeHandler(handler)
            log.setLevel(previous_level)
            log.propagate = previous_propagate
        return stream.getvalue()

    def test_report_states_there_are_no_positions(self, mock_deps):
        r = _create_runner(mock_deps)
        r._paper_positions = {}
        r._paper_pnl = 0.0
        r._cycle_count = 10

        out = self._capture_report(r)

        assert "No open positions" in out
        assert "10" in out, "cycle count missing from the report"

    def test_report_names_the_open_position_and_pnl(self, mock_deps):
        """A status report that omits your positions is not a status report."""
        r = _create_runner(mock_deps)
        r._paper_positions = {
            "AAPL": {"shares": 100, "entry_price": 150.0, "direction": "long"}}
        r._paper_pnl = 500.0
        r._cycle_count = 10

        out = self._capture_report(r)

        assert "AAPL" in out
        assert "100 shares" in out
        assert "500" in out


class TestSmartSleep:
    # These four asserted nothing at all — they ran _smart_sleep() and passed
    # if it returned. The whole point of the method is WHICH interval it
    # picks, so that is measured, by observing the time.sleep() calls.

    @staticmethod
    def _sleep_budget(runner):
        """Total seconds _smart_sleep would sleep for, without sleeping."""
        calls = []
        with patch("shared.daemon.live_runner.time.sleep", side_effect=calls.append):
            runner._running = True
            # Stop after the first chunk so the loop terminates promptly.
            original = calls.append

            def stop_after_first(seconds):
                original(seconds)
                runner._running = False

            with patch("shared.daemon.live_runner.time.sleep",
                       side_effect=stop_after_first):
                runner._smart_sleep()
        return calls

    def test_smart_sleep_market_open_uses_configured_interval(self, mock_deps):
        r = _create_runner(mock_deps, interval_seconds=5)
        mock_deps["fetcher"].is_market_open.return_value = True
        with patch("shared.daemon.live_runner.datetime") as dt:
            dt.now.return_value = MagicMock(weekday=lambda: 2)  # Wednesday
            calls = self._sleep_budget(r)
        assert calls, "no sleep happened at all"
        assert calls[0] == 5, f"expected the configured 5s interval, slept {calls[0]}"

    def test_smart_sleep_after_hours_backs_off_to_15_minutes(self, mock_deps):
        r = _create_runner(mock_deps, interval_seconds=5)
        mock_deps["fetcher"].is_market_open.return_value = False
        with patch("shared.daemon.live_runner.datetime") as dt:
            dt.now.return_value = MagicMock(weekday=lambda: 2)
            calls = self._sleep_budget(r)
        assert calls[0] == 10, "after hours should sleep in 10s chunks toward 900s"

    def test_smart_sleep_is_responsive_to_shutdown(self, mock_deps):
        """Chunked sleeping is what makes Ctrl-C prompt; verify the chunks."""
        r = _create_runner(mock_deps, interval_seconds=3600)
        mock_deps["fetcher"].is_market_open.return_value = True
        with patch("shared.daemon.live_runner.datetime") as dt:
            dt.now.return_value = MagicMock(weekday=lambda: 2)
            calls = self._sleep_budget(r)
        assert calls[0] <= 10, (
            f"a shutdown would wait {calls[0]}s; sleep must be chunked"
        )

    def test_smart_sleep_returns_immediately_when_stopped(self, mock_deps):
        r = _create_runner(mock_deps, interval_seconds=5)
        r._running = False
        mock_deps["fetcher"].is_market_open.return_value = True
        with patch("shared.daemon.live_runner.time.sleep") as slept:
            r._smart_sleep()
        slept.assert_not_called()


class TestRunCycle:
    def test_run_cycle_increments_tick(self, mock_deps):
        r = _create_runner(mock_deps, mode="monitor")
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        mock_deps["fetcher"].is_market_open.return_value = True
        assert r._tick_count == 0
        r._run_cycle()
        assert r._tick_count == 1

    def test_run_cycle_periodic_report(self, mock_deps):
        r = _create_runner(mock_deps, mode="monitor")
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        mock_deps["fetcher"].is_market_open.return_value = True
        r._cycle_count = 12
        r._market_hours = MagicMock()
        r._market_hours.is_market_open.return_value = True
        r._market_hours.is_trading_allowed.return_value = True
        r._market_hours.should_flatten_eod.return_value = False
        with patch.object(r, "_print_status_report") as mock_report:
            r._run_cycle()
            mock_report.assert_called_once()

    def test_one_symbol_failing_does_not_abort_the_cycle(self, mock_deps):
        """Asserted nothing. A network blip on one symbol must not stop the
        others from being processed, and must still advance the cycle."""
        r = _create_runner(mock_deps, mode="monitor")
        mock_deps["fetcher"].fetch_ohlcv.side_effect = RuntimeError("network")
        # Market state comes from _market_hours, not the data fetcher —
        # mocking the fetcher's is_market_open had no effect on the cycle.
        # should_flatten_eod must be explicitly False: a bare MagicMock is
        # truthy, which trips the EOD branch and returns before any symbol is
        # looked at.
        r._market_hours = MagicMock(
            is_market_open=lambda: True,
            is_trading_allowed=lambda: True,
            should_flatten_eod=lambda: False,
        )
        before = r._tick_count   # _cycle_count is advanced by start(), not here

        r._run_cycle()

        # Both configured symbols were attempted despite the first failing.
        attempted = [c.args[0] for c in mock_deps["fetcher"].fetch_ohlcv.call_args_list]
        assert set(attempted) >= {"AAPL", "MSFT"}
        assert r._tick_count == before + 1
        assert r._paper_positions == {}

    def test_run_cycle_with_broker_reconciliation(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        mock_bridge = MagicMock()
        mock_bridge.is_connected.return_value = True
        mock_bridge.reconcile_positions.return_value = {"removed": [], "added": [], "matched": 0}
        mock_bridge.check_and_force_close.return_value = []
        r._broker_bridge = mock_bridge
        r._market_hours = MagicMock()
        r._market_hours.is_market_open.return_value = True
        r._market_hours.is_trading_allowed.return_value = True
        r._market_hours.should_flatten_eod.return_value = False
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        mock_deps["fetcher"].is_market_open.return_value = True
        r._run_cycle()
        mock_bridge.reconcile_positions.assert_called_once()

    def test_reconciliation_failure_does_not_abort_the_cycle(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        mock_bridge = MagicMock()
        mock_bridge.is_connected.return_value = True
        mock_bridge.reconcile_positions.side_effect = RuntimeError("err")
        r._broker_bridge = mock_bridge
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        r._market_hours = MagicMock(
            is_market_open=lambda: True,
            is_trading_allowed=lambda: True,
            should_flatten_eod=lambda: False,
        )
        before = r._tick_count

        r._run_cycle()

        # _sync_positions() is what the cycle calls; it consults the bridge.
        mock_bridge.is_connected.assert_called()
        assert r._tick_count == before + 1, "cycle aborted on a reconcile error"


class TestProcessSymbolAdvanced:
    def test_sentiment_analysis_called(self, mock_deps):
        r = _create_runner(mock_deps, mode="monitor")
        mock_sentiment = MagicMock()
        mock_sentiment.analyze.return_value = {
            "sentiment_score": 0.5, "sentiment_label": "bullish",
            "headlines_analyzed": 5, "bullish_count": 3, "bearish_count": 1,
        }
        r._sentiment_analyzer = mock_sentiment
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        with patch.object(r, "_log_decision"):
            r._process_symbol("AAPL", market_open=True)
        mock_sentiment.analyze.assert_called_once()

    def test_sentiment_failure_still_reaches_the_agent(self, mock_deps):
        """A dead news feed must not silently skip the decision entirely."""
        r = _create_runner(mock_deps, mode="monitor")
        mock_sentiment = MagicMock()
        mock_sentiment.analyze.side_effect = RuntimeError("API down")
        r._sentiment_analyzer = mock_sentiment
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)

        with patch.object(r, "_log_decision"):
            r._process_symbol("AAPL", market_open=True)

        mock_deps["agent"].decide.assert_called()
        assert r._paper_positions == {}

    def test_paper_mode_sell_skips_no_position(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        mock_deps["agent"].decide.return_value = {
            "action": "HOLD", "confidence": 0.3, "regime": "RANGING", "price": 100.0,
        }
        r._process_symbol("AAPL", market_open=True)
        assert "AAPL" not in r._paper_positions


class TestPaperTradingEdge:
    def test_buy_while_already_long_skips(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        r._paper_positions["AAPL"] = {"shares": 100, "entry_price": 150.0,
                                       "entry_time": "2024-01-01", "direction": "long"}
        r._execute_paper_trade("AAPL", "BUY", 155.0, 0.8)
        assert r._paper_positions["AAPL"]["entry_price"] == 150.0

    def test_sell_while_already_short_skips(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        r._paper_positions["AAPL"] = {"shares": 100, "entry_price": 150.0,
                                       "entry_time": "2024-01-01", "direction": "short"}
        r._execute_paper_trade("AAPL", "SELL", 145.0, 0.8)
        assert r._paper_positions["AAPL"]["entry_price"] == 150.0

    def test_close_paper_position_no_entry_bar(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        r._paper_positions["X"] = {"shares": 50, "entry_price": 100.0,
                                    "entry_time": "2024-01-01", "direction": "long"}
        r._close_paper_position("X", 110.0)
        # raw = 500.0, commission = 50 * 0.005 * 2 = 0.50
        assert r._paper_pnl == pytest.approx(500.0 - 50 * 0.005 * 2)

    def test_closing_an_unknown_symbol_changes_nothing(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        r._paper_positions = {"AAPL": {"shares": 10, "entry_price": 100.0,
                                       "direction": "long"}}
        pnl_before = r._paper_pnl

        r._close_paper_position("NOPE", 100.0)

        assert r._paper_pnl == pnl_before, "P&L moved on a phantom close"
        assert set(r._paper_positions) == {"AAPL"}


class TestLogDecision:
    def test_log_decision_writes_file(self, mock_deps, tmp_path):
        r = _create_runner(mock_deps, mode="monitor")
        import shared.daemon.live_runner as lr_module
        original_log_dir = lr_module.LOG_DIR
        lr_module.LOG_DIR = str(tmp_path)
        try:
            r._log_decision("AAPL", "BUY", 150.0, 0.8, "TRENDING")
            log_file = tmp_path / "decisions.jsonl"
            assert log_file.exists()
            import json
            with open(log_file) as f:
                entry = json.loads(f.readline())
            assert entry["symbol"] == "AAPL"
            assert entry["action"] == "BUY"
            assert entry["price"] == 150.0
            assert entry["mode"] == "monitor"
        finally:
            lr_module.LOG_DIR = original_log_dir


class TestExecuteLiveTrade:
    def test_live_trade_without_a_broker_opens_nothing(self, mock_deps):
        """Asserted nothing — and this is the live-money path."""
        r = _create_runner(mock_deps, mode="live")
        r._broker_bridge = None

        r._execute_live_trade("AAPL", "BUY", 150.0, 0.8)

        assert r._paper_positions == {}

    def test_live_trade_does_not_order_when_the_broker_is_down(self, mock_deps):
        """A failed reconnect must not be followed by an order attempt."""
        r = _create_runner(mock_deps, mode="live")
        mock_bridge = MagicMock()
        mock_bridge.is_connected.return_value = False
        mock_bridge.connect.return_value = False
        r._broker_bridge = mock_bridge

        r._execute_live_trade("AAPL", "BUY", 150.0, 0.8)

        mock_bridge.execute_decision.assert_not_called()

    def test_live_trade_success(self, mock_deps):
        r = _create_runner(mock_deps, mode="live")
        mock_bridge = MagicMock()
        mock_bridge.is_connected.return_value = True
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.shares = 100
        mock_result.broker = "ib"
        mock_result.order_id = "123"
        mock_bridge.execute_decision.return_value = mock_result
        r._broker_bridge = mock_bridge
        r._execute_live_trade("AAPL", "BUY", 150.0, 0.8)
        mock_bridge.execute_decision.assert_called_once()

    def test_live_trade_failure(self, mock_deps):
        r = _create_runner(mock_deps, mode="live")
        mock_bridge = MagicMock()
        mock_bridge.is_connected.return_value = True
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.message = "rejected"
        mock_bridge.execute_decision.return_value = mock_result
        r._broker_bridge = mock_bridge
        r._execute_live_trade("AAPL", "BUY", 150.0, 0.8)


class TestShutdownAdvanced:
    def test_shutdown_with_open_positions(self, mock_deps, capsys):
        r = _create_runner(mock_deps)
        r._agent = mock_deps["agent"]
        r._paper_positions = {"AAPL": {"shares": 100, "entry_price": 150.0}}
        r._paper_pnl = 500.0
        r._cycle_count = 10
        with patch.object(mock_deps["agent"], "save_models"):
            r._shutdown()

    def test_shutdown_completes_even_if_saving_models_fails(self, mock_deps):
        """A full disk must not leave the runner stuck in a running state."""
        r = _create_runner(mock_deps)
        r._agent = mock_deps["agent"]
        r._paper_positions = {}
        r._paper_pnl = 0.0
        r._cycle_count = 5
        r._running = True
        mock_deps["agent"].save_models.side_effect = RuntimeError("disk full")

        r._shutdown()

        mock_deps["agent"].save_models.assert_called()
        assert r._running is False


class TestInitAdvanced:
    def test_init_with_news_success(self, mock_deps):
        with patch("shared.ml.news_sentiment.NewsSentimentAnalyzer") as mock_ns:
            mock_ns.return_value = MagicMock()
            r = _create_runner(mock_deps, use_news=True)
            assert r._sentiment_analyzer is not None

    def test_init_with_news_failure(self, mock_deps):
        with patch("shared.ml.news_sentiment.NewsSentimentAnalyzer", side_effect=ImportError("no")):
            r = _create_runner(mock_deps, use_news=True)
            assert r._sentiment_analyzer is None

    def test_init_with_broker(self, mock_deps):
        with patch("shared.daemon.broker_bridge.BrokerBridge") as mock_bb:
            mock_bb.return_value = MagicMock()
            r = _create_runner(mock_deps, mode="paper", broker="ib", broker_config={"host": "127.0.0.1"})
            assert r._broker_bridge is not None

    def test_init_broker_failure(self, mock_deps):
        with patch("shared.daemon.broker_bridge.BrokerBridge", side_effect=RuntimeError("no broker")):
            r = _create_runner(mock_deps, mode="paper", broker="ib")
            assert r._broker_bridge is None


class TestProcessSymbolLiveMode:
    def test_live_mode_open_market(self, mock_deps):
        r = _create_runner(mock_deps, mode="live")
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        mock_deps["agent"].decide.return_value = {
            "action": "BUY", "confidence": 0.8, "regime": "TRENDING", "price": 150.0,
        }
        with patch.object(r, "_execute_live_trade") as mock_lt:
            r._process_symbol("AAPL", market_open=True)
            mock_lt.assert_called_once()

    def test_live_mode_closed_market_skips(self, mock_deps):
        r = _create_runner(mock_deps, mode="live")
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        mock_deps["agent"].decide.return_value = {
            "action": "BUY", "confidence": 0.8, "regime": "TRENDING", "price": 150.0,
        }
        with patch.object(r, "_execute_live_trade") as mock_lt:
            r._process_symbol("AAPL", market_open=False)
            mock_lt.assert_not_called()

    def test_paper_mode_with_broker_bridge(self, mock_deps):
        r = _create_runner(mock_deps, mode="paper")
        mock_bridge = MagicMock()
        mock_bridge.is_connected.return_value = True
        r._broker_bridge = mock_bridge
        mock_deps["fetcher"].fetch_ohlcv.return_value = _make_ohlcv(100)
        mock_deps["agent"].decide.return_value = {
            "action": "BUY", "confidence": 0.8, "regime": "TRENDING", "price": 150.0,
        }
        with patch.object(r, "_execute_live_trade") as mock_lt:
            r._process_symbol("AAPL", market_open=True)
            mock_lt.assert_called_once()


class TestSmartSleepLoop:
    def test_smart_sleep_runs_briefly(self, mock_deps):
        """Sleep with _running=True but very short interval."""
        r = _create_runner(mock_deps, interval_seconds=1)
        r._running = True
        mock_deps["fetcher"].is_market_open.return_value = True
        with patch("shared.daemon.live_runner.time.sleep") as mock_sleep:
            with patch("shared.daemon.live_runner.datetime") as mock_dt:
                mock_now = MagicMock()
                mock_now.weekday.return_value = 2
                mock_dt.now.return_value = mock_now
                # After first sleep call, stop running
                def stop_after_call(*a):
                    r._running = False
                mock_sleep.side_effect = stop_after_call
                r._smart_sleep()
                mock_sleep.assert_called()
