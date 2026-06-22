"""Suppress noisy third-party logs during eval/training workers."""
from __future__ import annotations

import contextlib
import os
import warnings


def setup_quiet_env() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    warnings.filterwarnings("ignore", category=FutureWarning)


@contextlib.contextmanager
def suppress_stdout_stderr():
    devnull = open(os.devnull, "w")
    try:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
    finally:
        devnull.close()
