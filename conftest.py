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
def isolate_persistent_state(tmp_path, monkeypatch):
    """Give every test its own on-disk state directory.

    Two problems this solves. First, the portfolio risk gate persists to
    SQLite, so without an override the suite reads and writes the operator's
    real ~/.stocks_plugin/data/unified_risk.db — observed as a $9,975 position
    and $3,306 of daily P&L appearing in tests that had opened nothing.

    Second, it has to be per-test rather than per-session: the gate is a
    singleton, and reset_instance() only drops the Python object. The next
    construction reloads whatever the previous test wrote, so a shared
    database leaks state between tests just as effectively as a shared object.
    """
    monkeypatch.setenv("STOCKS_PLUGIN_DATA_DIR", str(tmp_path / "state"))


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


def pytest_configure(config):
    """Warn loudly when this machine can segfault mid-run.

    On macOS, LightGBM and PyTorch bundle separate OpenMP runtimes and crash
    the interpreter when both are used in one process (measured in both
    orders — see shared/__init__.py). That takes the whole pytest run down
    with no traceback and no report for anything after it.

    This warns rather than skipping: the crash is not confined to a list of
    test names that can be enumerated reliably, so pretending to know which
    ones are affected would be worse than saying plainly that the environment
    is unsound. Uninstall one of the two locally, or run on Linux/CI, to get a
    complete result.
    """
    import sys
    from importlib.util import find_spec

    if sys.platform != "darwin":
        return
    try:
        if find_spec("lightgbm") is None or find_spec("torch") is None:
            return
    except (ImportError, ValueError):       # pragma: no cover
        return

    config.stash  # noqa: B018 - touch to keep the reference explicit
    sys.stderr.write(
        "\n"
        "=" * 78 + "\n"
        "WARNING: LightGBM and PyTorch are both installed on macOS.\n"
        "They bundle separate OpenMP runtimes and SEGFAULT when used in one\n"
        "process, which will abort this run partway through with no report.\n"
        "Uninstall one of them locally, or run the suite on Linux/CI, for a\n"
        "complete result. See shared/__init__.py for the measurements.\n"
        + "=" * 78 + "\n\n"
    )
