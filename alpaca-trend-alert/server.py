# Alpaca Trend Alert System
# Copyright (C) 2026 Your Name
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Alpaca Trading Server
- OAuth login (pending approval) / API key fallback
- Background trend watcher
- WebSocket alert channel to client
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, HTMLResponse
import requests

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.requests import StockSnapshotRequest
from alpaca.data.historical import StockHistoricalDataClient

# ─── Config ─────────────────────────────────────────────────────────────────
CLIENT_ID = os.environ.get("ALPACA_CLIENT_ID", "").strip()
CLIENT_SECRET = os.environ.get("ALPACA_CLIENT_SECRET", "").strip()
API_KEY = os.environ.get("ALPACA_API_KEY", "").strip()
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "").strip()

REDIRECT_URI = "http://localhost:8000/oauth/callback"
PAPER = True  # Set False for live trading

BASE_URL = "https://paper-api.alpaca.markets" if PAPER else "https://api.alpaca.markets"
TOKEN_URL = f"{BASE_URL}/oauth/token"
AUTH_URL = "https://app.alpaca.markets/oauth/authorize"

# Trend-watching config
WATCH_SYMBOLS = ["AAPL", "TSLA", "NVDA"]
PRICE_CHANGE_THRESHOLD = 0.02  # 2% move triggers alert
CHECK_INTERVAL = 10  # seconds

# ─── State ──────────────────────────────────────────────────────────────────
oauth_token: str | None = None
trading_client: TradingClient | None = None
data_client: StockHistoricalDataClient | None = None
ws_clients: list[WebSocket] = []
last_prices: dict[str, float] = {}
alerts_pending: dict[str, dict] = {}


# ─── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[DEBUG] Server starting. PAPER={PAPER}, BASE_URL={BASE_URL}")
    if CLIENT_ID:
        print(f"[DEBUG] CLIENT_ID={CLIENT_ID[:8]}... (len={len(CLIENT_ID)})")
    if API_KEY:
        print(f"[DEBUG] API_KEY={API_KEY[:8]}... (len={len(API_KEY)})")
    print(f"[DEBUG] REDIRECT_URI={REDIRECT_URI}")

    task = asyncio.create_task(trend_watcher())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)


# ─── OAuth Routes ───────────────────────────────────────────────────────────
@app.get("/login")
async def login():
    if not CLIENT_ID or not CLIENT_SECRET:
        return HTMLResponse(
            "<h1>OAuth not configured</h1>"
            "<p>Set ALPACA_CLIENT_ID and ALPACA_CLIENT_SECRET in .env</p>"
        )

    state = secrets.token_urlsafe(16)
    auth_url = (
        f"{AUTH_URL}"
        f"?response_type=code"
        f"&client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&state={state}"
        f"&scope=account%3Awrite%20trading"
    )
    print(f"[DEBUG] Authorize URL: {auth_url}")
    return RedirectResponse(url=auth_url)


@app.get("/oauth/callback")
async def oauth_callback(request: Request):
    global oauth_token, trading_client, data_client

    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")

    print(f"[DEBUG] Callback hit: code={code[:10] if code else None}... state={state} error={error}")

    if error:
        print(f"[DEBUG] OAuth ERROR from Alpaca: {error}")
        return HTMLResponse(f"<h1>OAuth Error</h1><p>{error}</p>")

    if not code:
        return HTMLResponse("<h1>No authorization code received</h1>")

    token_payload = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
    }
    print(f"[DEBUG] Token exchange: client_id={CLIENT_ID[:8]}... secret={CLIENT_SECRET[:4]}****")

    resp = requests.post(TOKEN_URL, data=token_payload)
    print(f"[DEBUG] Token response: status={resp.status_code}")
    print(f"[DEBUG] Token body: {resp.text[:200]}")

    if resp.status_code != 200:
        return HTMLResponse(f"<h1>Token exchange failed</h1><pre>{resp.text}</pre>")

    data = resp.json()
    oauth_token = data["access_token"]
    print(f"[DEBUG] Got token: {oauth_token[:20]}... (expires in {data.get('expires_in', '?')}s)")

    trading_client = TradingClient(oauth_token=oauth_token, paper=PAPER)
    data_client = StockHistoricalDataClient(oauth_token, "")

    account = trading_client.get_account()
    print(f"[DEBUG] Account: id={account.id} status={account.status} equity=${account.equity}")
    print(f"[DEBUG] Buying power: ${account.buying_power}")

    return HTMLResponse(
        f"<h1>✅ Connected (OAuth)</h1>"
        f"<p>Account: {account.id}</p>"
        f"<p>Status: {account.status}</p>"
        f"<p>Equity: ${account.equity}</p>"
        f"<p>Buying Power: ${account.buying_power}</p>"
        f"<p>Token expires in ~15 min.</p>"
    )


# ─── API Key Fallback ───────────────────────────────────────────────────────
@app.get("/login-key")
async def login_key():
    """Use API keys directly (no OAuth needed)."""
    global trading_client, data_client

    if not API_KEY or not SECRET_KEY:
        return HTMLResponse(
            "<h1>API keys not configured</h1>"
            "<p>Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env</p>"
        )

    trading_client = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=PAPER)
    data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    account = trading_client.get_account()
    print(f"[DEBUG] API Key login: id={account.id} status={account.status}")
    print(f"[DEBUG] Account equity: ${account.equity}")
    print(f"[DEBUG] Account cash: ${account.cash}")
    print(f"[DEBUG] Buying power: ${account.buying_power}")

    return HTMLResponse(
        f"<h1>✅ Connected (API Key)</h1>"
        f"<p>Account: {account.id}</p>"
        f"<p>Status: {account.status}</p>"
        f"<p>Equity: ${account.equity}</p>"
        f"<p>Buying Power: ${account.buying_power}</p>"
    )


# ─── WebSocket (alert channel) ──────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    print(f"[DEBUG] WebSocket connected. Active clients: {len(ws_clients)}")
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            print(f"[DEBUG] WS received: {msg}")

            if msg["action"] == "buy" and msg.get("symbol") in alerts_pending:
                symbol = msg["symbol"]
                alerts_pending.pop(symbol, None)
                await place_order(symbol)
            elif msg["action"] == "ignore":
                alerts_pending.pop(msg.get("symbol", ""), None)
    except WebSocketDisconnect:
        ws_clients.remove(ws)
        print(f"[DEBUG] WebSocket disconnected. Active clients: {len(ws_clients)}")


async def broadcast(message: dict):
    for ws in list(ws_clients):
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            ws_clients.remove(ws)


# ─── Trading ────────────────────────────────────────────────────────────────
async def place_order(symbol: str):
    req = MarketOrderRequest(
        symbol=symbol,
        qty=1,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    order = trading_client.submit_order(req)
    await broadcast({"type": "order_filled", "symbol": symbol, "order_id": str(order.id)})
    print(f"[ORDER] Bought 1 {symbol} → {order.id}")


# ─── Background Trend Watcher ───────────────────────────────────────────────
async def trend_watcher():
    """Background loop — never crashes the event loop."""
    while True:
        try:
            if trading_client is None or data_client is None:
                await asyncio.sleep(5)
                continue

            for symbol in WATCH_SYMBOLS:
                snapshot = data_client.get_stock_snapshot(
                    StockSnapshotRequest(symbol_or_symbols=[symbol])
                )
                # snapshot is Dict[str, Snapshot] — access by symbol
                snap = snapshot[symbol]
                if snap.latest_trade is None:
                    continue
                price = float(snap.latest_trade.price)

                if symbol in last_prices:
                    change = (price - last_prices[symbol]) / last_prices[symbol]
                    if abs(change) >= PRICE_CHANGE_THRESHOLD and symbol not in alerts_pending:
                        alerts_pending[symbol] = {"price": price, "change": change}
                        await broadcast({
                            "type": "alert",
                            "symbol": symbol,
                            "price": price,
                            "change_pct": round(change * 100, 2),
                        })
                        print(f"[ALERT] {symbol} moved {change*100:+.2f}% → ${price}")
                last_prices[symbol] = price

        except Exception as e:
            print(f"[WATCHER] {type(e).__name__}: {e}")

        await asyncio.sleep(CHECK_INTERVAL)


# ─── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("→ Open http://localhost:8000/login in your browser to connect (OAuth).")
    print("→ Or http://localhost:8000/login-key for API key auth.")
    uvicorn.run(app, host="0.0.0.0", port=8000)   