# -*- coding: utf-8 -*-
"""Generate a sample GraphMemory demo and print the results."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.ml.graph_memory import GraphMemory

gm = GraphMemory("/tmp/demo_graph_memory.json", save_interval=100)

trades = [
    {"regime": "TRENDING", "symbol": "AAPL", "action": "BUY", "pnl": 320, "pnl_pct": 0.021, "is_winner": True, "decision_source": "ensemble", "ensemble_confidence": 0.82, "features_snapshot": json.dumps({"rsi_14": 0.65, "adx_14": 0.85, "volatility": -0.2, "momentum_20d": 0.7})},
    {"regime": "TRENDING", "symbol": "MSFT", "action": "BUY", "pnl": 180, "pnl_pct": 0.015, "is_winner": True, "decision_source": "lstm", "ensemble_confidence": 0.71, "features_snapshot": json.dumps({"rsi_14": 0.55, "adx_14": 0.9, "volatility": -0.1, "momentum_20d": 0.6})},
    {"regime": "TRENDING", "symbol": "AAPL", "action": "BUY", "pnl": -150, "pnl_pct": -0.01, "is_winner": False, "decision_source": "ensemble", "ensemble_confidence": 0.55, "features_snapshot": json.dumps({"rsi_14": 0.72, "adx_14": 0.6, "volatility": 0.3, "momentum_20d": 0.4})},
    {"regime": "VOLATILE", "symbol": "AAPL", "action": "SELL", "pnl": 450, "pnl_pct": 0.03, "is_winner": True, "decision_source": "rl", "ensemble_confidence": 0.65, "features_snapshot": json.dumps({"rsi_14": 0.3, "adx_14": 0.4, "volatility": 0.9, "momentum_20d": -0.5})},
    {"regime": "VOLATILE", "symbol": "TSLA", "action": "SELL", "pnl": -200, "pnl_pct": -0.013, "is_winner": False, "decision_source": "ensemble", "ensemble_confidence": 0.48, "features_snapshot": json.dumps({"rsi_14": 0.25, "adx_14": 0.35, "volatility": 0.95, "momentum_20d": -0.7})},
    {"regime": "VOLATILE", "symbol": "AAPL", "action": "HOLD", "pnl": 0, "pnl_pct": 0.0, "is_winner": False, "decision_source": "ensemble", "ensemble_confidence": 0.3, "features_snapshot": json.dumps({"rsi_14": 0.4, "adx_14": 0.3, "volatility": 0.85, "momentum_20d": -0.3})},
    {"regime": "RANGING", "symbol": "MSFT", "action": "BUY", "pnl": 90, "pnl_pct": 0.007, "is_winner": True, "decision_source": "transformer", "ensemble_confidence": 0.6, "features_snapshot": json.dumps({"rsi_14": 0.45, "adx_14": 0.15, "volatility": -0.4, "momentum_20d": 0.1})},
    {"regime": "RANGING", "symbol": "AAPL", "action": "BUY", "pnl": 60, "pnl_pct": 0.004, "is_winner": True, "decision_source": "ensemble", "ensemble_confidence": 0.58, "features_snapshot": json.dumps({"rsi_14": 0.42, "adx_14": 0.18, "volatility": -0.35, "momentum_20d": 0.05})},
    {"regime": "TRENDING", "symbol": "AAPL", "action": "BUY", "pnl": 500, "pnl_pct": 0.033, "is_winner": True, "decision_source": "lstm", "ensemble_confidence": 0.88, "features_snapshot": json.dumps({"rsi_14": 0.6, "adx_14": 0.92, "volatility": -0.15, "momentum_20d": 0.8})},
    {"regime": "TRENDING", "symbol": "SPY", "action": "BUY", "pnl": 220, "pnl_pct": 0.018, "is_winner": True, "decision_source": "ensemble", "ensemble_confidence": 0.79, "features_snapshot": json.dumps({"rsi_14": 0.58, "adx_14": 0.78, "volatility": -0.25, "momentum_20d": 0.55})},
    {"regime": "VOLATILE", "symbol": "AAPL", "action": "SELL", "pnl": 300, "pnl_pct": 0.02, "is_winner": True, "decision_source": "rl", "ensemble_confidence": 0.7, "features_snapshot": json.dumps({"rsi_14": 0.28, "adx_14": 0.38, "volatility": 0.88, "momentum_20d": -0.6})},
    {"regime": "RANGING", "symbol": "MSFT", "action": "BUY", "pnl": -80, "pnl_pct": -0.006, "is_winner": False, "decision_source": "transformer", "ensemble_confidence": 0.45, "features_snapshot": json.dumps({"rsi_14": 0.5, "adx_14": 0.12, "volatility": -0.5, "momentum_20d": -0.1})},
]

corr = {
    "AAPL": {"MSFT": 0.82, "SPY": 0.91, "TSLA": 0.45},
    "MSFT": {"AAPL": 0.82, "SPY": 0.88, "TSLA": 0.35},
    "SPY": {"AAPL": 0.91, "MSFT": 0.88, "TSLA": 0.40},
    "TSLA": {"AAPL": 0.45, "MSFT": 0.35, "SPY": 0.40},
}

for i, t in enumerate(trades):
    gm.record_trade(t, trade_id=i + 1)

gm.update_symbol_correlations(corr)
gm.save()

# -- Print the memory view --
print("=" * 70)
print("  GRAPHIFY MEMORY -- SAMPLE VIEW")
print("=" * 70)

stats = gm.get_stats()
print("")
print("[STATS] Graph: %d nodes, %d edges" % (stats["total_nodes"], stats["total_edges"]))
print("   Node types: %s" % json.dumps(stats["node_types"], indent=6))
print("   Edge types: %s" % json.dumps(stats["edge_types"], indent=6))

print("")
print("-" * 70)
print("  REGIME TRANSITIONS (Markov Chain)")
print("-" * 70)
for regime in ("TRENDING", "VOLATILE", "RANGING"):
    probs = gm.get_regime_transition_probs(regime)
    if probs:
        parts = []
        for r in sorted(probs, key=lambda x: -probs[x]):
            parts.append("%s: %.0f%%" % (r, probs[r] * 100))
        print("  %-10s -> %s" % (regime, ", ".join(parts)))
    else:
        print("  %-10s -> (no transitions recorded)" % regime)

print("")
print("-" * 70)
print("  BEST STRATEGY PER REGIME")
print("-" * 70)
for regime in ("TRENDING", "VOLATILE", "RANGING"):
    best = gm.get_best_strategy_for_regime(regime)
    if best:
        print("  %-10s -> %-12s | WR: %.0f%% | Avg PnL: $%7.0f | Trades: %d | Score: %.1f" % (
            regime, best["strategy"], best["win_rate"] * 100, best["avg_pnl"],
            best["trade_count"], best["score"]))
    else:
        print("  %-10s -> (no data)" % regime)

print("")
print("-" * 70)
print("  CORRELATED SYMBOLS (from AAPL)")
print("-" * 70)
corr_syms = gm.get_correlated_symbols("AAPL")
for cs in corr_syms:
    print("  %-6s -- correlation: %.2f" % (cs["symbol"], cs["correlation"]))

print("")
print("-" * 70)
print("  COMPOSITE INSIGHT (TRENDING + AAPL)")
print("-" * 70)
insight = gm.get_graph_enhanced_insight(
    "TRENDING", "AAPL",
    {"rsi_14": 0.6, "adx_14": 0.85, "volatility": -0.2, "momentum_20d": 0.7},
)
print(json.dumps(insight, indent=4, default=str))

print("")
print("-" * 70)
print("  RAW GRAPH JSON (first 60 lines)")
print("-" * 70)
raw = json.load(open("/tmp/demo_graph_memory.json"))
lines = json.dumps(raw, indent=2, default=str).split("\n")
for line in lines[:60]:
    print("  %s" % line)
if len(lines) > 60:
    print("  ... (%d more lines)" % (len(lines) - 60))

gm.close()
