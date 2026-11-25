from __future__ import annotations


def is_nonfatal_timeout_exception(exc: BaseException) -> bool:
    """Keep process-control exceptions fatal; ordinary timeouts use Exception."""

    return isinstance(exc, TimeoutError)
