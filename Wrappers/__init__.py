"""Public wrapper API for the integrated methods."""

from .base_ode_sr_wrapper import BaseODESRWrapper, EquationResult
from .kineselectsr_wrapper import KineSelectSRWrapper

__all__ = ["BaseODESRWrapper", "EquationResult", "KineSelectSRWrapper"]


def __getattr__(name: str):
    if name == "SISRWrapper":
        from .sisr_wrapper import SISRWrapper

        return SISRWrapper
    if name == "ReactiveSINDyWrapper":
        from .reactive_sindy_wrapper import ReactiveSINDyWrapper

        return ReactiveSINDyWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
