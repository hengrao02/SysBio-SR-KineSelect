# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Source: https://github.com/sdascoli/odeformer

"""Bounded symbolic postprocessing without leaking signal handlers."""
from contextlib import contextmanager
import signal
import threading
import time


@contextmanager
def timeout(seconds):
    if threading.current_thread() is not threading.main_thread():
        raise TimeoutError("symbolic postprocessing requires the main thread")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(*_):
        raise TimeoutError("ODEFormer symbolic postprocessing timeout")

    signal.signal(signal.SIGALRM, expired)
    delay = min(seconds, previous_timer[0]) if previous_timer[0] else seconds
    signal.setitimer(signal.ITIMER_REAL, delay)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(1e-6, previous_timer[0] - (time.monotonic() - started)),
                previous_timer[1],
            )
