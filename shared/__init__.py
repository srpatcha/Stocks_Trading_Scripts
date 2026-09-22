"""Shared infrastructure package for stocks_plugin trading platform.

Provides configuration management, notification dispatching,
and backtesting utilities used across all strategy modules.
"""

import os
import sys

# ─── macOS: keep LightGBM and PyTorch from loading two OpenMP runtimes ───
#
# SelfLearningAgent.train() uses LightGBM (regime classifier) and PyTorch
# (LSTM/Transformer) in the same process, and requirements.txt installs both.
# On macOS each ships its own libomp; with two OpenMP runtimes loaded, the
# first parallel region in torch's LSTM kernel SEGFAULTS the interpreter —
# a hard crash, no traceback, no exception to catch.
#
# Reproduced here: importing lightgbm and then training an LSTM
# (hidden_size=128, num_layers=2) crashes; without lightgbm it completes.
# OMP_NUM_THREADS=1 avoids it; KMP_DUPLICATE_LIB_OK=TRUE does not.
#
# The variable is read when the OpenMP runtime loads, so it must be set
# before either library is imported — hence here, at package import, rather
# than inside the predictors. Only on Darwin, so Linux (one shared libgomp,
# no conflict) keeps full multi-threaded training. Set OMP_NUM_THREADS
# yourself to override.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _openmp_conflict_warning() -> str:
    """Return a warning if this process can load two OpenMP runtimes.

    On macOS, LightGBM and PyTorch each bundle their own libomp. With both
    loaded, the first parallel region segfaults the interpreter — measured in
    BOTH orders (lightgbm then a torch LSTM, and torch then a LightGBM fit).
    OMP_NUM_THREADS=1 avoids the first order only; KMP_DUPLICATE_LIB_OK does
    not help either.

    A segfault cannot be caught, so the best available behaviour is to say so
    up front. Returns an empty string when the combination is not present.
    """
    if sys.platform != "darwin":
        return ""   # Linux shares one libgomp; no conflict observed
    if os.environ.get("STOCKS_PLUGIN_SUPPRESS_OPENMP_WARNING"):
        return ""

    from importlib.util import find_spec

    try:
        has_lightgbm = find_spec("lightgbm") is not None
        has_torch = find_spec("torch") is not None
    except (ImportError, ValueError):       # pragma: no cover
        return ""
    if not (has_lightgbm and has_torch):
        return ""

    return (
        "LightGBM and PyTorch are both installed on macOS. They bundle "
        "separate OpenMP runtimes and SEGFAULT the interpreter when used in "
        "the same process — this affects SelfLearningAgent.train(), which "
        "trains the regime classifier (LightGBM) and the LSTM/Transformer "
        "(PyTorch) together. Train them in separate invocations "
        "(train(models=['regime']) then train(models=['lstm'])), or run on "
        "Linux/Docker. Set STOCKS_PLUGIN_SUPPRESS_OPENMP_WARNING=1 to silence."
    )


_warning = _openmp_conflict_warning()
if _warning:
    import logging as _logging

    _logging.getLogger(__name__).warning(_warning)
del _warning
