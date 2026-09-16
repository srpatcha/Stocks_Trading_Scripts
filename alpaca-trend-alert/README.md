# Alpaca Trend Alert System

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/youruser/alpaca-trend-alert/blob/main/LICENSE)
A lightweight client-server system that monitors stock price movements and
delivers real-time alerts to a CLI client, where the user decides to buy or
ignore. Authentication uses Alpaca's OAuth Authorization Code flow (pending
approval) or API keys (immediate).

## Architecture

| Component | Role |
|-----------|------|
| **Server** (`server.py`) | Handles auth, runs background trend watcher, exposes WebSocket for alerts, places orders on user approval |
| **Client** (`client.py`) | Connects via WebSocket, displays alerts, prompts user for buy/ignore |
| **Alpaca API** | Market data (snapshots) + Trading API (order execution) |

## How It Works

1. **Authentication**
   - **API Key (immediate):** Visit `http://localhost:8000/login-key` in your browser.
   - **OAuth (pending approval):** Visit `http://localhost:8000/login`, authorize in browser.
   - Token/keys are held **in memory only** — never written to disk at runtime.

2. **Trend Watching (Background)**
   - A background asyncio task polls Alpaca's snapshot API every 10 seconds.
   - If a watched symbol moves ≥ 2% since the last check, an alert is queued.

3. **Alert Delivery**
   - The server broadcasts the alert to all connected WebSocket clients.
   - The client prints the alert and prompts: `Buy 1 share? [y/n]`.

4. **Execution**
   - `y` → server submits a market buy order via Alpaca Trading API.
   - `n` → alert is dismissed.
   - Order confirmation is broadcast back to the client.

5. **Token Expiry (OAuth only)**
   - After ~15 minutes the token lapses. Re-run `/login` to re-authenticate.
   - API keys don't expire (until you regenerate them).

## Setup

### Prerequisites

- Python 3.11+
- An Alpaca account (paper or live)
- (Optional) An OAuth app registered for the full OAuth flow

### Step 1: Obtain Your API Keys

1. Go to [app.alpaca.markets](https://app.alpaca.markets) and log in.
2. On the **Home / Overview** page, look at the **right sidebar** → **API Keys** section.
3. Click **Generate New Key** (or **View** if you already have one).
4. Copy both the **Key ID** and **Secret Key** immediately — the secret is
   **only shown once** and cannot be retrieved later.

> **Note:** Paper and live API keys are separate. Make sure you're on the
> correct dashboard:
> - **Paper:** [paper-trading.alpaca.markets](https://paper-trading.alpaca.markets)
> - **Live:** [alpaca.markets](https://alpaca.markets)

### Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
ALPACA_API_KEY=your_key_id_here
ALPACA_SECRET_KEY=your_secret_key_here

# Only needed for OAuth (after approval)
ALPACA_CLIENT_ID=
ALPACA_CLIENT_SECRET=
```

### Step 3: Install

```bash
conda create -n alpaca-trend python=3.11 -y
conda activate alpaca-trend
pip install -r requirements.txt
```

Or from the environment file:

```bash
conda env create -f environment.yml
conda activate alpaca-trend
```

### Step 4: Run

```bash
# Terminal 1 — Server
python server.py

# Terminal 2 — Browser
# Visit http://localhost:8000/login-key

# Terminal 3 — Client
python client.py
```

### Configure

Edit constants at the top of `server.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPER` | `True` | Set `False` for live trading (must match your key type) |
| `WATCH_SYMBOLS` | `["AAPL","TSLA","NVDA"]` | Tickers to monitor |
| `PRICE_CHANGE_THRESHOLD` | `0.02` | % move that triggers an alert |
| `CHECK_INTERVAL` | `10` | Seconds between polls |

## Testing

```bash
pytest tests/ -v
```

## Security Model

- **No keys written to disk at runtime.** The server reads from `.env` once at
  startup and holds credentials in memory.
- **Client sees no secrets.** The client communicates exclusively over
  WebSocket to your server.
- **`.env` is git-ignored.** Only `.env.example` is committed.

## Running on a Private Cloud / Tailscale

Yes — this architecture is well-suited to a **self-hosted or private cloud
server** accessed over **Tailscale**:

- **Zero port forwarding.** Install the Tailscale agent on your server and
  your laptop/phone. They get private `100.x.x.x` IPs on an encrypted
  WireGuard mesh.
- **Works behind CGNAT / NAT.** Even a home server with no public IP is
  reachable from anywhere.
- **WebSocket support.** Tailscale carries all protocols transparently.

```bash
# On your server
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Then point the client at the server's Tailscale IP or MagicDNS hostname:

```python
# client.py
SERVER_WS = "ws://100.x.x.x:8000/ws"
# or:
SERVER_WS = "ws://my-trading-box:8000/ws"
```

### Hardening checklist

| Layer | Action |
|-------|--------|
| Firewall | `ufw default deny incoming` + allow only the Tailscale interface |
| Secrets | Keep `.env` with `chmod 600`, not in code |
| Process | Run under `systemd` with `Restart=always` |
| Logging | Add structured logging for audit trail of orders |
```