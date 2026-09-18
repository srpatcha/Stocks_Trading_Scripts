"""
Broker-Agnostic Portfolio Analytics
=====================================

Provides Value-at-Risk, Sharpe/Sortino ratios, max-drawdown analysis,
stress testing, and beta calculation.  Any platform (IB, TradeStation,
thinkorswim, TradingView webhook) can import and use these.

Usage:
    from shared.analytics import PortfolioAnalytics

    pa = PortfolioAnalytics()
    var = pa.calculate_var(returns_series, confidence=0.95)
    sharpe = pa.calculate_sharpe(returns_series)
    dd = pa.calculate_max_drawdown(equity_curve)
    stress = pa.stress_test(positions_dict)
"""

from __future__ import annotations

import logging
from statistics import NormalDist
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_STANDARD_NORMAL = NormalDist()


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF.

    Uses the standard library rather than ``scipy.stats.norm.ppf``. scipy is
    listed as an optional dependency and is not installed by CI, but the
    parametric VaR path imported it unconditionally, so that method raised
    ModuleNotFoundError in any environment that took the "optional" label at
    face value. ``NormalDist.inv_cdf`` is exact to full double precision and
    ships with Python.
    """
    return _STANDARD_NORMAL.inv_cdf(p)


class PortfolioAnalytics:
    """Broker-agnostic portfolio risk analytics."""

    def __init__(self, risk_free_rate: float = 0.05) -> None:
        self._risk_free_rate = risk_free_rate

    # ─── Value at Risk ───

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> Dict[str, Any]:
        """Value at Risk — historical or parametric.

        Args:
            returns: Series of periodic returns (e.g. daily log-returns).
            confidence: Confidence level (0.90, 0.95, 0.99).
            method: ``"historical"`` or ``"parametric"``.

        Returns:
            Dict with ``var``, ``cvar``, ``confidence``, ``method``.
        """
        clean = returns.dropna()
        if len(clean) < 2:
            return {"var": 0.0, "cvar": 0.0, "confidence": confidence, "method": method}

        if method == "historical":
            var = float(np.percentile(clean, (1 - confidence) * 100))
        else:  # parametric
            mu, sigma = float(clean.mean()), float(clean.std())
            var = float(mu + sigma * _norm_ppf(1 - confidence))

        tail = clean[clean <= var]
        cvar = float(tail.mean()) if len(tail) > 0 else var

        return {
            "var": round(var, 6),
            "cvar": round(cvar, 6),
            "confidence": confidence,
            "method": method,
        }

    # ─── Sharpe Ratio ───

    def calculate_sharpe(self, returns: pd.Series, periods: int = 252) -> float:
        """Annualised Sharpe ratio.

        Args:
            returns: Series of periodic returns.
            periods: Annualisation factor (252 for daily, 12 for monthly).

        Returns:
            Sharpe ratio as float.
        """
        clean = returns.dropna()
        if len(clean) < 2 or clean.std() == 0:
            return 0.0
        excess = clean.mean() - self._risk_free_rate / periods
        return float(excess / clean.std() * np.sqrt(periods))

    # ─── Sortino Ratio ───

    def calculate_sortino(self, returns: pd.Series, periods: int = 252) -> float:
        """Annualised Sortino ratio (penalises downside volatility only).

        Args:
            returns: Series of periodic returns.
            periods: Annualisation factor.

        Returns:
            Sortino ratio as float.
        """
        clean = returns.dropna()
        if len(clean) < 2:
            return 0.0
        excess = clean.mean() - self._risk_free_rate / periods
        downside = clean[clean < 0].std()
        if downside == 0 or np.isnan(downside):
            return 0.0
        return float(excess / downside * np.sqrt(periods))

    # ─── Max Drawdown ───

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Compute maximum drawdown from an equity curve.

        Args:
            equity_curve: Series indexed by date with portfolio value.

        Returns:
            Dict with ``max_drawdown``, ``peak_date``, ``trough_date``,
            ``recovery_date`` (None if not yet recovered).
        """
        if len(equity_curve) < 2:
            return {
                "max_drawdown": 0.0,
                "peak_date": None,
                "trough_date": None,
                "recovery_date": None,
            }

        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_dd = float(drawdown.min())
        max_dd_idx = drawdown.idxmin()
        peak_idx = equity_curve[:max_dd_idx].idxmax()

        recovery_idx = None
        post_dd = equity_curve[max_dd_idx:]
        recovered = post_dd[post_dd >= equity_curve[peak_idx]]
        if len(recovered) > 0:
            recovery_idx = recovered.index[0]

        return {
            "max_drawdown": round(max_dd, 6),
            "peak_date": str(peak_idx),
            "trough_date": str(max_dd_idx),
            "recovery_date": str(recovery_idx) if recovery_idx is not None else None,
        }

    # ─── Stress Testing ───

    def stress_test(
        self,
        positions: Dict[str, Dict[str, float]],
        scenarios: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run stress scenarios on a portfolio.

        Args:
            positions: ``{symbol: {"shares": N, "price": P, "beta": B}}``.
                       ``beta`` defaults to 1.0 if omitted.
            scenarios: ``{name: market_move_pct}``.  Defaults to standard shocks.

        Returns:
            ``{scenario_name: {"market_move", "portfolio_pnl", "pnl_pct"}}``.
        """
        if scenarios is None:
            scenarios = {
                "market_crash_10pct": -0.10,
                "correction_5pct": -0.05,
                "flash_crash_3pct": -0.03,
                "rally_5pct": 0.05,
                "bear_market_20pct": -0.20,
            }

        total_value = sum(p["shares"] * p["price"] for p in positions.values())
        results: Dict[str, Dict[str, Any]] = {}

        for name, market_move in scenarios.items():
            scenario_pnl = 0.0
            for _sym, pos in positions.items():
                beta = pos.get("beta", 1.0)
                stock_move = market_move * beta
                scenario_pnl += pos["shares"] * pos["price"] * stock_move

            pnl_pct = round(scenario_pnl / total_value * 100, 2) if total_value > 0 else 0.0
            results[name] = {
                "market_move": market_move,
                "portfolio_pnl": round(scenario_pnl, 2),
                "pnl_pct": pnl_pct,
            }

        return results

    # ─── Beta ───

    def calculate_beta(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """Calculate beta of an asset relative to a benchmark.

        Args:
            returns: Asset return series.
            benchmark_returns: Benchmark (e.g. SPY) return series.

        Returns:
            Beta coefficient.  Returns 1.0 if insufficient data.
        """
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 10:
            return 1.0
        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return float(cov[0][1] / cov[1][1]) if cov[1][1] > 0 else 1.0

    def __repr__(self) -> str:
        return f"PortfolioAnalytics(risk_free_rate={self._risk_free_rate})"
