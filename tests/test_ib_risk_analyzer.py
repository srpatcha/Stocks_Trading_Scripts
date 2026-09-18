"""
Comprehensive tests for RiskAnalyzer (interactive_brokers.analytics.risk_analyzer).
"""

import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interactive_brokers.analytics.risk_analyzer import (
    RiskAnalyzer,
    VaRResult,
    DrawdownResult,
    StressScenario,
    StressTestResult,
)


# ── Fixtures ──


@pytest.fixture
def analyzer():
    return RiskAnalyzer(connection=None, risk_free_rate=0.05)


@pytest.fixture
def daily_returns():
    """200 days of synthetic daily returns (mean ~0.05%, std ~1%)."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.01, 200))


@pytest.fixture
def equity_curve():
    """Synthetic equity curve starting at 100000."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 200)
    prices = 100000 * np.cumprod(1 + returns)
    idx = pd.date_range("2025-01-01", periods=200, freq="B")
    return pd.Series(prices, index=idx)


@pytest.fixture
def positions_for_stress():
    return [
        {"symbol": "AAPL", "market_value": 30000, "beta": 1.2, "sector": "Technology"},
        {"symbol": "JPM", "market_value": 20000, "beta": 1.1, "sector": "Financials"},
        {"symbol": "XOM", "market_value": 15000, "beta": 0.8, "sector": "Energy"},
        {"symbol": "NEE", "market_value": 10000, "beta": 0.5, "sector": "Utilities"},
    ]


# ── VaR: Historical ──


class TestCalculateVarHistorical:
    def test_var_historical_basic(self, analyzer, daily_returns):
        result = analyzer.calculate_var(daily_returns, confidence=0.95, method="historical")
        assert isinstance(result, VaRResult)
        assert result.var_dollar > 0
        assert result.var_pct > 0
        assert result.method == "historical"
        assert result.confidence == 0.95

    def test_var_historical_cvar(self, analyzer, daily_returns):
        result = analyzer.calculate_var(daily_returns, confidence=0.95, method="historical")
        assert result.cvar_dollar >= result.var_dollar
        assert result.cvar_pct >= result.var_pct

    def test_var_historical_holding_period(self, analyzer, daily_returns):
        r1 = analyzer.calculate_var(daily_returns, holding_period=1, method="historical")
        r5 = analyzer.calculate_var(daily_returns, holding_period=5, method="historical")
        assert r5.var_dollar > r1.var_dollar

    def test_var_insufficient_data(self, analyzer):
        """Verify fix: <30 data points returns empty/zero result."""
        short = pd.Series(np.random.normal(0, 0.01, 10))
        result = analyzer.calculate_var(short)
        assert result.var_dollar == 0.0
        assert result.var_pct == 0.0
        assert result.cvar_dollar == 0.0

    def test_var_exactly_30_points(self, analyzer):
        returns = pd.Series(np.random.normal(0, 0.01, 30))
        result = analyzer.calculate_var(returns, method="historical")
        assert result.var_dollar > 0

    def test_var_with_nan_values(self, analyzer):
        """NaN values are dropped; if enough remain, result is valid."""
        data = list(np.random.normal(0, 0.01, 50)) + [np.nan] * 10
        returns = pd.Series(data)
        result = analyzer.calculate_var(returns, method="historical")
        assert result.var_dollar > 0


# ── VaR: Parametric ──


class TestCalculateVarParametric:
    def test_var_parametric_basic(self, analyzer, daily_returns):
        result = analyzer.calculate_var(daily_returns, method="parametric")
        assert isinstance(result, VaRResult)
        assert result.var_dollar > 0
        assert result.method == "parametric"

    def test_var_parametric_cvar(self, analyzer, daily_returns):
        result = analyzer.calculate_var(daily_returns, method="parametric")
        assert result.cvar_dollar > 0

    def test_var_parametric_needs_no_scipy(self, analyzer, daily_returns):
        """Parametric VaR must work without scipy installed.

        scipy is an optional dependency that CI does not install, so this
        branch used to raise ImportError in a default environment. It now uses
        statistics.NormalDist from the standard library.
        """
        with patch.dict("sys.modules", {"scipy": None, "scipy.stats": None}):
            result = analyzer.calculate_var(daily_returns, method="parametric")
        assert result.var_dollar > 0
        assert result.method == "parametric"

    def test_var_parametric_matches_known_z_score(self, analyzer):
        """At 95% the z-score is -1.6449; check the arithmetic, not just >0."""
        import pandas as pd

        # Zero-mean series with a known standard deviation.
        returns = pd.Series([-0.01, 0.01] * 50)
        result = analyzer.calculate_var(returns, confidence=0.95, method="parametric")
        sigma = float(returns.std())
        assert result.var_pct == pytest.approx(1.6448536 * sigma, rel=1e-4)

    def test_var_unknown_method(self, analyzer, daily_returns):
        with pytest.raises(ValueError, match="Unknown VaR method"):
            analyzer.calculate_var(daily_returns, method="monte_carlo")


# ── Max Drawdown ──


class TestCalculateMaxDrawdown:
    def test_max_drawdown_basic(self, analyzer, equity_curve):
        result = analyzer.calculate_max_drawdown(equity_curve)
        assert isinstance(result, DrawdownResult)
        assert result.max_drawdown_pct > 0
        assert result.max_drawdown_dollar > 0

    def test_max_drawdown_insufficient_data(self, analyzer):
        short = pd.Series([100000])
        result = analyzer.calculate_max_drawdown(short)
        assert result.max_drawdown_pct == 0.0

    def test_max_drawdown_monotonic_up(self, analyzer):
        curve = pd.Series([100, 110, 120, 130, 140])
        result = analyzer.calculate_max_drawdown(curve)
        assert result.max_drawdown_pct == pytest.approx(0.0)

    def test_max_drawdown_dates(self, analyzer, equity_curve):
        result = analyzer.calculate_max_drawdown(equity_curve)
        assert result.peak_date is not None
        assert result.trough_date is not None


# ── Sharpe / Sortino ──


class TestSharpe:
    def test_sharpe_basic(self, analyzer, daily_returns):
        sharpe = analyzer.calculate_sharpe(daily_returns)
        assert isinstance(sharpe, float)

    def test_sharpe_insufficient_data(self, analyzer):
        assert analyzer.calculate_sharpe(pd.Series([0.01])) == 0.0

    def test_sharpe_zero_std(self, analyzer):
        returns = pd.Series([0.001] * 50)
        assert analyzer.calculate_sharpe(returns) == 0.0

    def test_sharpe_custom_rf(self, analyzer, daily_returns):
        s1 = analyzer.calculate_sharpe(daily_returns, risk_free_rate=0.0)
        s2 = analyzer.calculate_sharpe(daily_returns, risk_free_rate=0.10)
        assert s1 > s2


class TestSortino:
    def test_sortino_basic(self, analyzer, daily_returns):
        sortino = analyzer.calculate_sortino(daily_returns)
        assert isinstance(sortino, float)

    def test_sortino_insufficient_data(self, analyzer):
        assert analyzer.calculate_sortino(pd.Series([0.01])) == 0.0

    def test_sortino_all_positive(self, analyzer):
        returns = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02])
        sortino = analyzer.calculate_sortino(returns)
        assert sortino == float("inf") or sortino > 0


# ── Portfolio Beta ──


class TestCalculateBeta:
    def test_beta_basic(self, analyzer):
        np.random.seed(42)
        bench = pd.Series(np.random.normal(0.0005, 0.01, 100), index=range(100))
        port = bench * 1.2 + pd.Series(np.random.normal(0, 0.002, 100), index=range(100))
        beta = analyzer.calculate_portfolio_beta(port, bench)
        assert 0.8 < beta < 1.6

    def test_beta_insufficient_data(self, analyzer):
        port = pd.Series([0.01] * 5, index=range(5))
        bench = pd.Series([0.01] * 5, index=range(5))
        beta = analyzer.calculate_portfolio_beta(port, bench)
        assert beta == 1.0

    def test_beta_mismatched_index(self, analyzer):
        np.random.seed(42)
        port = pd.Series(np.random.normal(0, 0.01, 50), index=range(0, 50))
        bench = pd.Series(np.random.normal(0, 0.01, 50), index=range(25, 75))
        beta = analyzer.calculate_portfolio_beta(port, bench)
        assert isinstance(beta, float)


# ── Correlation Matrix ──


class TestCorrelationMatrix:
    def test_correlation_matrix(self, analyzer):
        np.random.seed(42)
        data = {
            "AAPL": pd.Series(np.random.normal(0, 0.01, 50)),
            "MSFT": pd.Series(np.random.normal(0, 0.01, 50)),
        }
        corr = analyzer.correlation_matrix(data)
        assert corr.shape == (2, 2)
        assert corr.loc["AAPL", "AAPL"] == pytest.approx(1.0)


# ── Stress Test ──


class TestStressTest:
    def test_stress_test_default_scenarios(self, analyzer, positions_for_stress):
        results = analyzer.stress_test(positions_for_stress)
        assert len(results) == 6
        assert all(isinstance(r, StressTestResult) for r in results)

    def test_stress_test_custom_scenario(self, analyzer, positions_for_stress):
        scenarios = [
            StressScenario(
                name="Custom Crash",
                description="Custom -30% crash",
                market_move_pct=-30.0,
            ),
        ]
        results = analyzer.stress_test(positions_for_stress, scenarios=scenarios)
        assert len(results) == 1
        assert results[0].scenario == "Custom Crash"
        assert results[0].portfolio_impact_dollar < 0

    def test_stress_test_sector_shocks(self, analyzer, positions_for_stress):
        scenarios = [
            StressScenario(
                name="Tech Crash",
                description="Tech-specific crash",
                market_move_pct=-5.0,
                sector_shocks={"Technology": -25.0},
            ),
        ]
        results = analyzer.stress_test(positions_for_stress, scenarios=scenarios)
        assert results[0].portfolio_impact_dollar < 0

    def test_stress_test_empty_positions(self, analyzer):
        results = analyzer.stress_test([])
        assert len(results) == 6
        for r in results:
            assert r.portfolio_impact_dollar == 0.0

    def test_stress_test_worst_position(self, analyzer, positions_for_stress):
        results = analyzer.stress_test(positions_for_stress)
        crash = [r for r in results if "Crash" in r.scenario][0]
        assert crash.worst_position != ""
        assert crash.worst_position_loss < 0


# ── Options Risk Summary ──


class TestOptionsRiskSummary:
    def test_options_risk_summary_basic(self, analyzer):
        positions = [
            {"symbol": "AAPL", "delta": 0.5, "gamma": 0.02, "theta": -0.05,
             "vega": 0.1, "quantity": 10, "multiplier": 100},
        ]
        summary = analyzer.options_risk_summary(positions)
        assert summary["total_delta"] == 0.5 * 10 * 100
        assert summary["total_gamma"] == 0.02 * 10 * 100
        assert summary["total_theta_daily"] == -0.05 * 10 * 100
        assert summary["total_vega"] == 0.1 * 10 * 100
        assert summary["net_direction"] == "bullish"

    def test_options_risk_summary_bearish(self, analyzer):
        positions = [
            {"symbol": "AAPL", "delta": -0.3, "gamma": 0.01, "theta": -0.02,
             "vega": 0.05, "quantity": 5, "multiplier": 100},
        ]
        summary = analyzer.options_risk_summary(positions)
        assert summary["net_direction"] == "bearish"
        assert summary["total_delta"] < 0

    def test_options_risk_summary_neutral(self, analyzer):
        positions = [
            {"symbol": "AAPL", "delta": 0.5, "quantity": 10, "multiplier": 100},
            {"symbol": "AAPL", "delta": -0.5, "quantity": 10, "multiplier": 100},
        ]
        summary = analyzer.options_risk_summary(positions)
        assert summary["net_direction"] == "neutral"

    def test_options_risk_summary_empty(self, analyzer):
        summary = analyzer.options_risk_summary([])
        assert summary["total_delta"] == 0.0
        assert summary["net_direction"] == "neutral"

    def test_theta_weekly_monthly(self, analyzer):
        positions = [
            {"symbol": "AAPL", "delta": 0.5, "gamma": 0.02, "theta": -1.0,
             "vega": 0.1, "quantity": 1, "multiplier": 100},
        ]
        summary = analyzer.options_risk_summary(positions)
        assert summary["theta_weekly"] == summary["total_theta_daily"] * 5
        assert summary["theta_monthly"] == summary["total_theta_daily"] * 21


# ── Information Ratio / Calmar ──


class TestInformationRatio:
    def test_information_ratio(self, analyzer):
        np.random.seed(42)
        port = pd.Series(np.random.normal(0.001, 0.01, 100), index=range(100))
        bench = pd.Series(np.random.normal(0.0005, 0.01, 100), index=range(100))
        ir = analyzer.calculate_information_ratio(port, bench)
        assert isinstance(ir, float)

    def test_information_ratio_insufficient(self, analyzer):
        port = pd.Series([0.01], index=[0])
        bench = pd.Series([0.005], index=[0])
        assert analyzer.calculate_information_ratio(port, bench) == 0.0


class TestCalmar:
    def test_calmar_ratio(self, analyzer, daily_returns, equity_curve):
        calmar = analyzer.calculate_calmar(daily_returns, equity_curve)
        assert isinstance(calmar, float)

    def test_calmar_no_drawdown(self, analyzer):
        returns = pd.Series([0.01] * 50)
        curve = pd.Series(np.cumprod(1 + returns.values) * 100000)
        calmar = analyzer.calculate_calmar(returns, curve)
        assert calmar == 0.0
