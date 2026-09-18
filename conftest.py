"""
Root conftest.py — shared fixtures for all tests.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path so all imports resolve
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(autouse=True)
def neutral_strategy_enricher(request, monkeypatch):
    """Keep StrategyEnricher offline and neutral for the whole suite.

    Strategies enable the enricher by default, and it fetches news,
    fundamentals and earnings from Yahoo Finance. Tests use synthetic tickers
    like "TEST", so in any environment where yfinance is installed the fetch
    404s, ``should_block_entry`` gates the entry, and a strategy that should
    emit BUY emits HOLD instead. That made roughly 25 strategy tests fail on
    CI while passing locally purely because yfinance was absent — and it meant
    unit tests were making live network calls on every run.

    A default EnrichedData has every ``*_available`` flag False, so
    ``should_block_entry`` returns (False, "OK") and no confidence boost is
    applied: the strategy's own conditional logic is what gets exercised,
    which is what these tests are for.

    Opt out with ``@pytest.mark.live_enricher`` to test the enricher itself.
    """
    if request.node.get_closest_marker("live_enricher"):
        return
    try:
        from shared import strategy_enricher
    except ImportError:  # pragma: no cover - enricher is optional
        return
    monkeypatch.setattr(
        strategy_enricher.StrategyEnricher,
        "enrich",
        lambda self, symbol, df: strategy_enricher.EnrichedData(),
    )


@pytest.fixture
def synthetic_ohlcv_df() -> pd.DataFrame:
    """400-bar synthetic OHLCV DataFrame for single-symbol strategy tests."""
    rng = np.random.RandomState(42)
    n_bars = 400
    dates = pd.bdate_range("2022-01-01", periods=n_bars)
    price = 100.0
    rows = []
    for i in range(n_bars):
        regime = np.sin(2 * np.pi * i / 100)
        drift = 0.0003 * regime
        ret = drift + rng.randn() * 0.015
        price *= 1 + ret
        high = price * (1 + abs(rng.randn()) * 0.006)
        low = price * (1 - abs(rng.randn()) * 0.006)
        rows.append({
            "date": dates[i],
            "open": price * (1 + rng.randn() * 0.002),
            "high": high,
            "low": low,
            "close": price,
            "volume": int(rng.uniform(500_000, 2_000_000)),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_universe() -> pd.DataFrame:
    """5-stock × 300-bar universe DataFrame for factor strategy tests."""
    rng = np.random.RandomState(42)
    n_bars = 300
    dates = pd.bdate_range("2019-01-01", periods=n_bars)
    tickers = [f"STOCK_{chr(65 + i)}" for i in range(5)]
    prices = {}
    for t in tickers:
        drift = rng.uniform(-0.0002, 0.001)
        vol = rng.uniform(0.01, 0.025)
        p = 50.0 + rng.uniform(0, 100)
        series = [p]
        for _ in range(n_bars - 1):
            p *= 1 + drift + rng.randn() * vol
            series.append(p)
        prices[t] = series
    return pd.DataFrame(prices, index=dates)


@pytest.fixture
def backtest_engine(synthetic_ohlcv_df):
    """Pre-configured BacktestEngineV2 with synthetic data loaded."""
    from shared.backtesting.backtest_engine_v2 import BacktestEngineV2

    engine = BacktestEngineV2(initial_capital=100_000)
    engine.load_data(synthetic_ohlcv_df)
    return engine
