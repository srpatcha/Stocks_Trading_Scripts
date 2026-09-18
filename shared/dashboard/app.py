# -*- coding: utf-8 -*-
"""
Graph Memory Dashboard -- FastAPI Backend
==========================================

Serves an interactive web dashboard for visualising the GraphMemory graph.
Includes WebSocket support for real-time graph updates and performance monitoring.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.ml.graph_memory import EdgeType, GraphMemory, NodeType, _node_id

app = FastAPI(title="Graphify Memory Dashboard")

_gm: GraphMemory | None = None

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

NODE_STYLE = {
    NodeType.REGIME.value: {"color": "#e74c3c", "shape": "diamond", "size": 30},
    NodeType.SYMBOL.value: {"color": "#3498db", "shape": "dot", "size": 25},
    NodeType.STRATEGY.value: {"color": "#9b59b6", "shape": "triangle", "size": 22},
    NodeType.TRADE.value: {"color": "#2ecc71", "shape": "dot", "size": 12},
    NodeType.FEATURE_STATE.value: {"color": "#95a5a6", "shape": "square", "size": 15},
}

EDGE_STYLE = {
    EdgeType.TRANSITIONS_TO.value: {"color": "#e74c3c", "dashes": False, "width": 2, "arrows": "to"},
    EdgeType.PRODUCED.value: {"color": "#9b59b6", "dashes": False, "width": 1, "arrows": "to"},
    EdgeType.TRADED_IN.value: {"color": "#7f8c8d", "dashes": [5, 5], "width": 1, "arrows": "to"},
    EdgeType.TRADED_SYMBOL.value: {"color": "#3498db", "dashes": [5, 5], "width": 1, "arrows": "to"},
    EdgeType.USED_STRATEGY.value: {"color": "#8e44ad", "dashes": [5, 5], "width": 1, "arrows": "to"},
    EdgeType.SIMILAR_TO.value: {"color": "#f39c12", "dashes": [2, 4], "width": 1, "arrows": ""},
    EdgeType.CORRELATES_WITH.value: {"color": "#95a5a6", "dashes": [2, 4], "width": 1, "arrows": "to"},
}


# ─── Performance Monitor ───


class PerformanceMonitor:
    """Tracks API latency, WebSocket throughput, and uptime metrics."""

    def __init__(self, max_samples: int = 500) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._max_samples = max_samples
        self._api_latencies: Dict[str, collections.deque] = {}
        self._ws_messages_sent = 0
        self._ws_messages_failed = 0
        self._ws_broadcast_latencies: collections.deque = collections.deque(maxlen=max_samples)
        self._total_requests = 0
        self._errors = 0

    def record_api_latency(self, endpoint: str, latency_ms: float) -> None:
        with self._lock:
            self._total_requests += 1
            if endpoint not in self._api_latencies:
                self._api_latencies[endpoint] = collections.deque(maxlen=self._max_samples)
            self._api_latencies[endpoint].append(latency_ms)

    def record_api_error(self) -> None:
        with self._lock:
            self._errors += 1

    def record_ws_broadcast(self, latency_ms: float, sent: int, failed: int) -> None:
        with self._lock:
            self._ws_messages_sent += sent
            self._ws_messages_failed += failed
            self._ws_broadcast_latencies.append(latency_ms)

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            uptime = time.time() - self._start_time

            api_stats: Dict[str, Any] = {}
            for endpoint, latencies in self._api_latencies.items():
                if not latencies:
                    continue
                sorted_lat = sorted(latencies)
                n = len(sorted_lat)
                api_stats[endpoint] = {
                    "count": n,
                    "avg_ms": round(sum(sorted_lat) / n, 2),
                    "p50_ms": round(sorted_lat[n // 2], 2),
                    "p95_ms": round(sorted_lat[int(n * 0.95)], 2) if n >= 2 else round(sorted_lat[-1], 2),
                    "p99_ms": round(sorted_lat[int(n * 0.99)], 2) if n >= 2 else round(sorted_lat[-1], 2),
                    "min_ms": round(sorted_lat[0], 2),
                    "max_ms": round(sorted_lat[-1], 2),
                }

            ws_lat = sorted(self._ws_broadcast_latencies) if self._ws_broadcast_latencies else []
            ws_n = len(ws_lat)

            return {
                "uptime_seconds": round(uptime, 1),
                "total_requests": self._total_requests,
                "total_errors": self._errors,
                "error_rate": round(self._errors / max(self._total_requests, 1), 4),
                "api_latency": api_stats,
                "websocket": {
                    "messages_sent": self._ws_messages_sent,
                    "messages_failed": self._ws_messages_failed,
                    "broadcasts": ws_n,
                    "avg_broadcast_ms": round(sum(ws_lat) / ws_n, 2) if ws_n else 0,
                    "p50_broadcast_ms": round(ws_lat[ws_n // 2], 2) if ws_n else 0,
                    "p95_broadcast_ms": round(ws_lat[int(ws_n * 0.95)], 2) if ws_n >= 2 else (round(ws_lat[-1], 2) if ws_lat else 0),
                    "throughput_per_sec": round(self._ws_messages_sent / max(uptime, 1), 2),
                },
                "active_ws_connections": manager.active_count,
            }


perf = PerformanceMonitor()


# ─── WebSocket Connection Manager ───


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self) -> None:
        self._connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        start = time.perf_counter()
        stale: List[WebSocket] = []
        sent = 0
        for ws in self._connections:
            try:
                await ws.send_json(message)
                sent += 1
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)
        elapsed_ms = (time.perf_counter() - start) * 1000
        perf.record_ws_broadcast(elapsed_ms, sent, len(stale))

    @property
    def active_count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()
_event_loop: asyncio.AbstractEventLoop | None = None


def _on_graph_change(event_type: str, data: Dict[str, Any]) -> None:
    """Bridge sync GraphMemory callback -> async WebSocket broadcast."""
    if _event_loop is None or _event_loop.is_closed():
        return
    message = {"type": "graph_update", "event": event_type, "data": data}
    asyncio.run_coroutine_threadsafe(manager.broadcast(message), _event_loop)


# ─── File-watch background task ───

_graph_path: str = ""
_last_mtime: float = 0.0


async def _file_watch_loop() -> None:
    """Periodically check if the graph JSON was modified externally."""
    global _last_mtime
    while True:
        await asyncio.sleep(2)
        try:
            if _graph_path and os.path.exists(_graph_path):
                mtime = os.path.getmtime(_graph_path)
                if _last_mtime > 0 and mtime > _last_mtime:
                    await manager.broadcast({
                        "type": "graph_update",
                        "event": "file_changed",
                        "data": {"path": _graph_path},
                    })
                _last_mtime = mtime
        except Exception:
            pass


@app.on_event("startup")
async def _startup() -> None:
    global _event_loop, _last_mtime
    _event_loop = asyncio.get_running_loop()
    if _graph_path and os.path.exists(_graph_path):
        _last_mtime = os.path.getmtime(_graph_path)
    asyncio.create_task(_file_watch_loop())


# ─── Configuration ───


def configure(graph_path: str) -> None:
    """Load the GraphMemory instance from disk."""
    global _gm, _graph_path, _last_mtime
    _graph_path = graph_path
    _gm = GraphMemory(path=graph_path)
    _gm.add_on_change(_on_graph_change)
    if os.path.exists(graph_path):
        _last_mtime = os.path.getmtime(graph_path)


def _gm_or_error() -> GraphMemory:
    if _gm is None:
        raise HTTPException(status_code=503, detail="GraphMemory not loaded")
    return _gm


# ─── HTTP Endpoints ───


@app.get("/", response_class=HTMLResponse)
async def index():
    template = TEMPLATE_DIR / "index.html"
    if not template.exists():
        raise HTTPException(status_code=500, detail="Template not found")
    return HTMLResponse(content=template.read_text(encoding="utf-8"))


@app.get("/api/graph")
async def get_graph() -> Dict[str, Any]:
    """Return vis.js-compatible nodes + edges."""
    start = time.perf_counter()
    gm = _gm_or_error()
    g = gm._graph

    nodes: List[Dict[str, Any]] = []
    for nid, data in g.nodes(data=True):
        ntype = data.get("type", "unknown")
        style = NODE_STYLE.get(ntype, {"color": "#ffffff", "shape": "dot", "size": 10})

        color = style["color"]
        if ntype == NodeType.TRADE.value:
            color = "#2ecc71" if data.get("is_winner", False) else "#e67e22"

        label = data.get("label", nid)
        title_parts = ["<b>" + label + "</b>", "Type: " + ntype]
        for key in ("trade_count", "total_pnl", "wins", "pnl", "pnl_pct", "action", "confidence"):
            if key in data:
                val = data[key]
                if isinstance(val, float):
                    val = round(val, 4)
                title_parts.append(f"{key}: {val}")
        if data.get("trade_count", 0) > 0 and data.get("wins") is not None:
            wr = data["wins"] / data["trade_count"]
            title_parts.append(f"win_rate: {wr:.1%}")

        nodes.append({
            "id": nid,
            "label": label,
            "group": ntype,
            "title": "<br>".join(title_parts),
            "color": color,
            "shape": style["shape"],
            "size": style["size"],
            "raw": {k: v for k, v in data.items()},
        })

    edges: List[Dict[str, Any]] = []
    for u, v, data in g.edges(data=True):
        etype = data.get("type", "unknown")
        style = EDGE_STYLE.get(etype, {"color": "#555", "dashes": False, "width": 1, "arrows": "to"})

        edge_label = etype.replace("_", " ")
        if etype == EdgeType.TRANSITIONS_TO.value:
            count = data.get("count", 1)
            edge_label = f"{count}x"

        edges.append({
            "from": u,
            "to": v,
            "label": edge_label,
            "color": {"color": style["color"]},
            "dashes": style["dashes"],
            "width": style.get("width", 1),
            "arrows": style["arrows"],
            "title": etype + ": " + u + " -> " + v,
            "raw": {k: v_ for k, v_ in data.items()},
        })

    perf.record_api_latency("/api/graph", (time.perf_counter() - start) * 1000)
    return {"nodes": nodes, "edges": edges}


@app.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    start = time.perf_counter()
    gm = _gm_or_error()
    stats = gm.get_stats()
    g = gm._graph
    n = g.number_of_nodes()
    stats["density"] = round(g.number_of_edges() / (n * (n - 1)), 6) if n > 1 else 0
    perf.record_api_latency("/api/stats", (time.perf_counter() - start) * 1000)
    return stats


@app.get("/api/transitions")
async def get_transitions() -> Dict[str, Any]:
    """Transition probabilities for all regime nodes."""
    start = time.perf_counter()
    gm = _gm_or_error()
    g = gm._graph
    regimes = [
        data.get("label", nid)
        for nid, data in g.nodes(data=True)
        if data.get("type") == NodeType.REGIME.value
    ]
    result: Dict[str, Dict[str, float]] = {}
    for regime in regimes:
        probs = gm.get_regime_transition_probs(regime)
        if probs:
            result[regime] = probs
    perf.record_api_latency("/api/transitions", (time.perf_counter() - start) * 1000)
    return {"transitions": result}


@app.get("/api/strategies")
async def get_strategies() -> Dict[str, Any]:
    """Best strategy per regime."""
    start = time.perf_counter()
    gm = _gm_or_error()
    g = gm._graph
    regimes = [
        data.get("label", nid)
        for nid, data in g.nodes(data=True)
        if data.get("type") == NodeType.REGIME.value
    ]
    result: Dict[str, Any] = {}
    for regime in regimes:
        best = gm.get_best_strategy_for_regime(regime)
        if best:
            result[regime] = best
    perf.record_api_latency("/api/strategies", (time.perf_counter() - start) * 1000)
    return {"strategies": result}


@app.get("/api/correlations/{symbol}")
async def get_correlations(symbol: str) -> Dict[str, Any]:
    start = time.perf_counter()
    gm = _gm_or_error()
    result = {"symbol": symbol, "correlations": gm.get_correlated_symbols(symbol)}
    perf.record_api_latency("/api/correlations", (time.perf_counter() - start) * 1000)
    return result


@app.get("/api/insight/{regime}/{symbol}")
async def get_insight(regime: str, symbol: str) -> Dict[str, Any]:
    start = time.perf_counter()
    gm = _gm_or_error()
    result = gm.get_graph_enhanced_insight(regime, symbol)
    perf.record_api_latency("/api/insight", (time.perf_counter() - start) * 1000)
    return result


@app.get("/api/perf")
async def get_perf() -> Dict[str, Any]:
    """Return performance monitoring snapshot."""
    return perf.get_snapshot()


# ─── WebSocket Endpoint ───


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ─── CLI ───


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Graph Memory Dashboard")
    parser.add_argument("--graph-path", default="graph_memory.json", help="Path to graph_memory.json")
    parser.add_argument("--port", type=int, default=8050, help="Server port")
    # Loopback by default: the dashboard has no authentication. Pass
    # --host 0.0.0.0 deliberately, and only on a trusted network.
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    args = parser.parse_args()

    configure(args.graph_path)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
