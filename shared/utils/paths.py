"""Where the system keeps its on-disk state.

Everything defaults under ``~/.stocks_plugin``. ``STOCKS_PLUGIN_DATA_DIR``
overrides that root, which matters in two places:

- tests, which must not read or write the operator's real risk database — a
  persisted portfolio gate is a singleton, so leaked state from one run shows
  up as phantom exposure in the next;
- containers, where ``$HOME`` is often not writable or not persistent.
"""

from __future__ import annotations

import os

ENV_VAR = "STOCKS_PLUGIN_DATA_DIR"


def stocks_plugin_root() -> str:
    """Root directory for all persistent state."""
    override = os.environ.get(ENV_VAR)
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".stocks_plugin")


def stocks_plugin_data_dir(create: bool = True) -> str:
    """Directory for databases and other durable state."""
    path = os.path.join(stocks_plugin_root(), "data")
    if create:
        os.makedirs(path, exist_ok=True)
    return path
