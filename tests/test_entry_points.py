"""Every CLI entry point must answer --help without doing anything.

Four of these scripts had no argument parsing at all. `--help` was simply an
unrecognised argument that nothing ever looked at, so the script fell through
and ran: three launchers started the trading daemon, and the webhook server
bound its port and fired a "Webhook Server Started" alert to Discord/SMS.

Asking a program what it does must never deploy it, so that invariant is
asserted here rather than left to the next person to rediscover. These run as
subprocesses on purpose: importing the module would not catch a script whose
side effects live under ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Scripts that place, or can place, orders — plus the dashboards that serve
# operator state. Anything here that starts on --help is a live incident.
ENTRY_POINTS = [
    "launch_dashboard.py",
    "launch_ib_paper.py",
    "launch_monitor.py",
    "launch_paper_sim.py",
    "paper_trader.py",
    "setup_trading.py",
    "shared/daemon/live_runner.py",
    "shared/dashboard/app.py",
    "strategies/runner.py",
    "tradingview/webhooks/webhook_server.py",
]

# Subset that also supports --dry-run: print the resolved configuration and
# exit, so an operator can confirm what would happen before it happens.
DRY_RUN_CAPABLE = [
    "launch_ib_paper.py",
    "launch_monitor.py",
    "launch_paper_sim.py",
    "setup_trading.py",
    "tradingview/webhooks/webhook_server.py",
]


def _run(script: str, *args: str, tmp_data_dir: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    # Never let an entry point touch the operator's real state while we probe it.
    env["STOCKS_PLUGIN_DATA_DIR"] = tmp_data_dir
    env["STOCKS_PLUGIN_SUPPRESS_OPENMP_WARNING"] = "1"
    return subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        capture_output=True,
        text=True,
        # A script that ignores --help blocks forever instead of exiting, so the
        # timeout IS the assertion for that failure mode.
        timeout=180,
        cwd=str(ROOT),
        env=env,
    )


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_help_exits_cleanly_without_starting(script, tmp_path):
    proc = _run(script, "--help", tmp_data_dir=str(tmp_path))
    assert proc.returncode == 0, (
        f"{script} --help exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )
    assert "usage:" in proc.stdout.lower(), (
        f"{script} --help printed no usage line — it probably ignored the flag "
        f"and ran instead.\nstdout:\n{proc.stdout[-2000:]}"
    )


@pytest.mark.parametrize("script", DRY_RUN_CAPABLE)
def test_dry_run_exits_cleanly(script, tmp_path):
    proc = _run(script, "--dry-run", tmp_data_dir=str(tmp_path))
    assert proc.returncode == 0, (
        f"{script} --dry-run exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )
    assert "dry run" in proc.stdout.lower(), (
        f"{script} --dry-run produced no dry-run banner:\n{proc.stdout[-2000:]}"
    )


def test_unknown_flag_is_rejected_not_ignored(tmp_path):
    """An unrecognised flag must abort, never fall through into trading.

    This is the exact failure the launchers had: argparse was absent, so every
    flag was silently discarded and the daemon started with its defaults.
    """
    proc = _run("launch_monitor.py", "--not-a-real-flag", tmp_data_dir=str(tmp_path))
    assert proc.returncode != 0
    assert "unrecognized arguments" in proc.stderr.lower()
