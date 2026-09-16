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
Simple CLI alert receiver.
Connects to the server's WebSocket, displays alerts, prompts for buy/ignore.
"""

import asyncio
import json
import websockets

SERVER_WS = "ws://localhost:8000/ws"


async def main():
    async with websockets.connect(SERVER_WS) as ws:
        print("✅ Connected to trading server. Waiting for alerts...\n")

        while True:
            raw = await ws.recv()
            msg = json.loads(raw)

            if msg["type"] == "alert":
                symbol = msg["symbol"]
                price = msg["price"]
                change = msg["change_pct"]
                print(f"🔔 ALERT: {symbol} moved {change:+.2f}% → ${price:.2f}")
                choice = input("   Buy 1 share? [y/n]: ").strip().lower()

                if choice == "y":
                    await ws.send(json.dumps({"action": "buy", "symbol": symbol}))
                    print("   → Order sent.\n")
                else:
                    await ws.send(json.dumps({"action": "ignore", "symbol": symbol}))
                    print("   → Ignored.\n")

            elif msg["type"] == "order_filled":
                print(f"✅ Order filled: {msg['symbol']} (id: {msg['order_id']})\n")


if __name__ == "__main__":
    asyncio.run(main())   