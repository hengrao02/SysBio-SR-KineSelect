"""ODEFormer runner registration (no optional dependency import)."""

from .run import get_method_spec, load_run_config

METHOD_SPEC = get_method_spec()
__all__ = ["METHOD_SPEC", "load_run_config"]
