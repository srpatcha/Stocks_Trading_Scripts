"""
Risk Analyzer for Interactive Brokers Portfolios
==================================================

Provides comprehensive risk analytics: VaR, drawdown, Sharpe/Sortino,
portfolio beta, correlation matrix, stress testing, and options risk.

Usage:
    analyzer = RiskAnalyzer(connection)
    var_95 = analyzer.calculate_var(returns, confidence=0.95)
    sharpe = analyzer.calculate_sharpe(returns, risk_free_rate=0.05)
    stress = analyzer.stress_test(portfolio, scenarios)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_STANDARD_NORMAL = NormalDist()


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    confidence: float
    var_dollar: float
    var_pct: float
    method: str
    holding_period_days: int = 1
    portfolio_value: float = 0.0
    cvar_dollar: float = 0.0  # Conditional VaR (Expected Shortfall)
    cvar_pct: float = 0.0


@dataclass
class DrawdownResult:
    """Maximum drawdown analysis result."""
    max_drawdown_pct: float
    max_drawdown_dollar: float
    peak_date: Optional[str] = None
    trough_date: Optional[str] = None
    recovery_date: Optional[str] = None
    current_drawdown_pct: float = 0.0
    drawdown_duration_days: int = 0


@dataclass
class StressScenario:
    """Definition of a stress test scenario."""
    name: str
    description: str
    market_move_pct: float
    vol_multiplier: float = 1.0
    sector_shocks: Dict[str, float] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Result of a stress test scenario."""
    scenario: str
    portfolio_impact_dollar: float
    portfolio_impact_pct: float
    worst_position: str = ""
    worst_position_loss: float = 0.0
    positions_affected: int = 0


class RiskAnalyzer:
    """Comprehensive risk analytics for IB portfolios.

    Calculates VaR, drawdown, risk-adjusted returns, correlations,
    and stress test scenarios.

    Args:
        connection: Optional IBInsyncConnection for live portfolio data.
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino (default 5%).
    """

    def __init__(
        self,
        connection: Any = None,
        risk_free_rate: float = 0.05,
    ) -> None:
        self.connection = connection
        self.risk_free_rate = risk_free_rate

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        portfolio_value: float = 100000.0,
        holding_period: int = 1,
        method: str = "historical",
    ) -> VaRResult:
        """Calculate Value at Risk using historical simulation.

        Args:
            returns: Series of historical returns (daily).
            confidence: Confidence level (e.g., 0.95 for 95%).
            portfolio_value: Current portfolio value in dollars.
            holding_period: Holding period in trading days.
            method: Calculation method ("historical" or "parametric").

        Returns:
            VaRResult with dollar and percentage VaR.
        """
        returns = returns.dropna()
        if len(returns) < 30:
            logger.warning("Insufficient data for VaR: %d returns", len(returns))
            return VaRResult(
                confidence=confidence,
                var_dollar=0.0,
                var_pct=0.0,
                method=method,
                holding_period_days=holding_period,
                portfolio_value=portfolio_value,
                cvar_dollar=0.0,
                cvar_pct=0.0,
            )

        if method == "historical":
            var_pct = -np.percentile(returns, (1 - confidence) * 100)

            # Conditional VaR (Expected Shortfall)
            tail = returns[returns <= -var_pct]
            cvar_pct = -tail.mean() if len(tail) > 0 else var_pct

        elif method == "parametric":
            # statistics.NormalDist replaces scipy.stats.norm here. scipy is an
            # optional dependency that CI does not install, so this branch used
            # to raise ImportError in a default environment. The stdlib gives
            # the same values to full double precision with no dependency.
            mu = returns.mean()
            sigma = returns.std()
            z_score = _STANDARD_NORMAL.inv_cdf(1 - confidence)
            var_pct = -(mu + z_score * sigma)

            # Parametric CVaR
            cvar_pct = -(
                mu - sigma * _STANDARD_NORMAL.pdf(z_score) / (1 - confidence)
            )
        else:
            raise ValueError(f"Unknown VaR method: {method}")

        # Scale for holding period (square root of time)
        if holding_period > 1:
            var_pct *= np.sqrt(holding_period)
            cvar_pct *= np.sqrt(holding_period)

        var_dollar = var_pct * portfolio_value
        cvar_dollar = cvar_pct * portfolio_value

        result = VaRResult(
            confidence=confidence,
            var_dollar=var_dollar,
            var_pct=var_pct,
            method=method,
            holding_period_days=holding_period,
            portfolio_value=portfolio_value,
            cvar_dollar=cvar_dollar,
            cvar_pct=cvar_pct,
        )

        logger.info(
            "VaR (%.0f%%, %s, %dd): $%.2f (%.2f%%) | "
            "CVaR: $%.2f (%.2f%%)",
            confidence * 100, method, holding_period,
            var_dollar, var_pct * 100,
            cvar_dollar, cvar_pct * 100,
        )
        return result

    def calculate_max_drawdown(
        self,
        equity_curve: pd.Series,
    ) -> DrawdownResult:
        """Calculate maximum drawdown from an equity curve.

        Args:
            equity_curve: Series of portfolio values over time.

        Returns:
            DrawdownResult with max drawdown percentage, dates, and duration.
        """
        equity_curve = equity_curve.dropna()
        if len(equity_curve) < 2:
            return DrawdownResult(
                max_drawdown_pct=0.0, max_drawdown_dollar=0.0,
            )

        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        drawdown_dollar = equity_curve - cumulative_max

        max_dd_idx = drawdown.idxmin()
        max_dd_pct = drawdown.min()
        max_dd_dollar = drawdown_dollar.min()

        peak_mask = cumulative_max.loc[:max_dd_idx]
        peak_idx = peak_mask.idxmax()

        # Find recovery (if any)
        recovery_idx = None
        post_trough = equity_curve.loc[max_dd_idx:]
        recovered = post_trough[post_trough >= cumulative_max.loc[max_dd_idx]]
        if len(recovered) > 0:
            recovery_idx = recovered.index[0]

        # Current drawdown
        current_dd = drawdown.iloc[-1]

        # Duration
        duration = 0
        if recovery_idx is not None:
            duration = (recovery_idx - peak_idx).days if hasattr(recovery_idx - peak_idx, 'days') else 0
        elif hasattr(equity_curve.index[-1] - peak_idx, 'days'):
            duration = (equity_curve.index[-1] - peak_idx).days

        result = DrawdownResult(
            max_drawdown_pct=abs(max_dd_pct) * 100,
            max_drawdown_dollar=abs(max_dd_dollar),
            peak_date=str(peak_idx),
            trough_date=str(max_dd_idx),
            recovery_date=str(recovery_idx) if recovery_idx else None,
            current_drawdown_pct=abs(current_dd) * 100,
            drawdown_duration_days=duration,
        )

        logger.info(
            "Max drawdown: %.2f%% ($%.2f) peak=%s trough=%s",
            result.max_drawdown_pct, result.max_drawdown_dollar,
            result.peak_date, result.trough_date,
        )
        return result

    def calculate_sharpe(
        self,
        returns: pd.Series,
        risk_free_rate: Optional[float] = None,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> float:
        """Calculate Sharpe ratio.

        Sharpe = (mean_return - risk_free) / std_return

        Args:
            returns: Series of periodic returns.
            risk_free_rate: Annual risk-free rate (uses instance default if None).
            annualize: If True, annualize the ratio.
            trading_days: Number of trading days per year.

        Returns:
            Sharpe ratio (annualized by default).
        """
        returns = returns.dropna()
        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        daily_rf = rf / trading_days

        excess_returns = returns - daily_rf
        sharpe = excess_returns.mean() / excess_returns.std()

        if annualize:
            sharpe *= np.sqrt(trading_days)

        logger.info("Sharpe ratio: %.4f", sharpe)
        return float(sharpe)

    def calculate_sortino(
        self,
        returns: pd.Series,
        risk_free_rate: Optional[float] = None,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> float:
        """Calculate Sortino ratio (penalizes only downside volatility).

        Sortino = (mean_return - risk_free) / downside_std

        Args:
            returns: Series of periodic returns.
            risk_free_rate: Annual risk-free rate.
            annualize: If True, annualize the ratio.
            trading_days: Number of trading days per year.

        Returns:
            Sortino ratio (annualized by default).
        """
        returns = returns.dropna()
        if len(returns) < 2:
            return 0.0

        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        daily_rf = rf / trading_days

        excess_returns = returns - daily_rf
        downside = excess_returns[excess_returns < 0]

        if len(downside) == 0 or downside.std() == 0:
            return float("inf") if excess_returns.mean() > 0 else 0.0

        downside_std = np.sqrt(np.mean(downside ** 2))
        sortino = excess_returns.mean() / downside_std

        if annualize:
            sortino *= np.sqrt(trading_days)

        logger.info("Sortino ratio: %.4f", sortino)
        return float(sortino)

    def calculate_portfolio_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Calculate portfolio beta relative to a benchmark.

        beta = cov(portfolio, benchmark) / var(benchmark)

        Args:
            portfolio_returns: Portfolio return series.
            benchmark_returns: Benchmark return series (e.g., SPY).

        Returns:
            Portfolio beta.
        """
        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        port = portfolio_returns.loc[common_idx].dropna()
        bench = benchmark_returns.loc[common_idx].dropna()

        common_idx2 = port.index.intersection(bench.index)
        port = port.loc[common_idx2]
        bench = bench.loc[common_idx2]

        if len(port) < 10 or bench.var() == 0:
            logger.warning("Insufficient data for beta calculation")
            return 1.0

        covariance = np.cov(port, bench)[0][1]
        beta = covariance / bench.var()

        logger.info("Portfolio beta: %.4f", beta)
        return float(beta)

    def correlation_matrix(
        self,
        returns_dict: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Calculate correlation matrix across multiple assets.

        Args:
            returns_dict: Dictionary mapping symbol → return series.

        Returns:
            DataFrame correlation matrix.
        """
        df = pd.DataFrame(returns_dict)
        corr = df.corr()

        logger.info(
            "Correlation matrix: %d assets, "
            "avg correlation: %.4f",
            len(returns_dict),
            corr.values[np.triu_indices_from(corr.values, k=1)].mean()
            if len(returns_dict) > 1 else 0.0,
        )
        return corr

    def stress_test(
        self,
        positions: List[Dict[str, Any]],
        scenarios: Optional[List[StressScenario]] = None,
        portfolio_value: float = 100000.0,
    ) -> List[StressTestResult]:
        """Run stress test scenarios on the portfolio.

        Args:
            positions: List of position dicts with keys:
                symbol, market_value, beta, sector.
            scenarios: List of StressScenario definitions.
                If None, uses default scenarios.
            portfolio_value: Total portfolio value.

        Returns:
            List of StressTestResult for each scenario.
        """
        if scenarios is None:
            scenarios = [
                StressScenario(
                    name="Market Crash (-20%)",
                    description="Broad market decline of 20%",
                    market_move_pct=-20.0,
                    vol_multiplier=3.0,
                ),
                StressScenario(
                    name="Correction (-10%)",
                    description="Standard market correction",
                    market_move_pct=-10.0,
                    vol_multiplier=2.0,
                ),
                StressScenario(
                    name="Flash Crash (-5%)",
                    description="Sudden intraday decline",
                    market_move_pct=-5.0,
                    vol_multiplier=4.0,
                ),
                StressScenario(
                    name="Rally (+10%)",
                    description="Strong market rally",
                    market_move_pct=10.0,
                    vol_multiplier=0.8,
                ),
                StressScenario(
                    name="Tech Selloff",
                    description="Technology sector selloff",
                    market_move_pct=-5.0,
                    sector_shocks={"Technology": -15.0, "Communication Services": -10.0},
                ),
                StressScenario(
                    name="Rate Shock",
                    description="Interest rate spike affecting rate-sensitive sectors",
                    market_move_pct=-3.0,
                    sector_shocks={
                        "Real Estate": -12.0, "Utilities": -8.0,
                        "Financials": 5.0,
                    },
                ),
            ]

        results = []

        for scenario in scenarios:
            total_impact = 0.0
            worst_sym = ""
            worst_loss = 0.0
            affected = 0

            for pos in positions:
                mv = pos.get("market_value", 0)
                beta = pos.get("beta", 1.0)
                sector = pos.get("sector", "Unknown")

                # Apply sector-specific shock if defined
                if sector in scenario.sector_shocks:
                    move_pct = scenario.sector_shocks[sector] / 100.0
                else:
                    move_pct = (scenario.market_move_pct / 100.0) * beta

                position_impact = mv * move_pct
                total_impact += position_impact
                affected += 1

                if position_impact < worst_loss:
                    worst_loss = position_impact
                    worst_sym = pos.get("symbol", "?")

            result = StressTestResult(
                scenario=scenario.name,
                portfolio_impact_dollar=total_impact,
                portfolio_impact_pct=(total_impact / portfolio_value * 100)
                if portfolio_value > 0 else 0.0,
                worst_position=worst_sym,
                worst_position_loss=worst_loss,
                positions_affected=affected,
            )
            results.append(result)

            logger.info(
                "Stress test [%s]: impact=$%.2f (%.2f%%) worst=%s ($%.2f)",
                scenario.name, total_impact,
                result.portfolio_impact_pct, worst_sym, worst_loss,
            )

        return results

    def options_risk_summary(
        self,
        positions: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate aggregate options risk (portfolio-level Greeks).

        Args:
            positions: List of position dicts with keys:
                symbol, delta, gamma, theta, vega, quantity, multiplier.

        Returns:
            Dictionary with total_delta, total_gamma, total_theta,
            total_vega, delta_dollars, gamma_dollars, theta_dollars.
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0

        for pos in positions:
            qty = pos.get("quantity", 0)
            mult = pos.get("multiplier", 100)

            total_delta += pos.get("delta", 0) * qty * mult
            total_gamma += pos.get("gamma", 0) * qty * mult
            total_theta += pos.get("theta", 0) * qty * mult
            total_vega += pos.get("vega", 0) * qty * mult

        summary = {
            "total_delta": total_delta,
            "total_gamma": total_gamma,
            "total_theta_daily": total_theta,
            "total_vega": total_vega,
            "delta_exposure_1pct": total_delta * 0.01,
            "gamma_exposure_1pct": 0.5 * total_gamma * (0.01 ** 2),
            "theta_weekly": total_theta * 5,
            "theta_monthly": total_theta * 21,
            "net_direction": "bullish" if total_delta > 0 else "bearish"
            if total_delta < 0 else "neutral",
        }

        logger.info(
            "Options risk: Δ=%.2f Γ=%.4f Θ=%.2f V=%.2f [%s]",
            total_delta, total_gamma, total_theta, total_vega,
            summary["net_direction"],
        )
        return summary

    def calculate_information_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> float:
        """Calculate Information Ratio (active return / tracking error).

        Args:
            portfolio_returns: Portfolio return series.
            benchmark_returns: Benchmark return series.
            annualize: If True, annualize.
            trading_days: Trading days per year.

        Returns:
            Information Ratio.
        """
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        active = portfolio_returns.loc[common] - benchmark_returns.loc[common]
        active = active.dropna()

        if len(active) < 2 or active.std() == 0:
            return 0.0

        ir = active.mean() / active.std()
        if annualize:
            ir *= np.sqrt(trading_days)

        logger.info("Information ratio: %.4f", ir)
        return float(ir)

    def calculate_calmar(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        trading_days: int = 252,
    ) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown).

        Args:
            returns: Daily return series.
            equity_curve: Portfolio value series.
            trading_days: Trading days per year.

        Returns:
            Calmar ratio.
        """
        ann_return = returns.mean() * trading_days
        dd = self.calculate_max_drawdown(equity_curve)

        if dd.max_drawdown_pct == 0:
            return 0.0

        calmar = (ann_return * 100) / dd.max_drawdown_pct
        logger.info("Calmar ratio: %.4f", calmar)
        return float(calmar)
