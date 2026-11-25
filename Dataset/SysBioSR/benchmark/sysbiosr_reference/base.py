from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Sequence

import numpy as np


def _as_1d_float_array(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {array.shape}")
    return array


@dataclass
class BaseParameters(ABC):
    """Common parameter container for one ODE system.

    Subclasses are expected to define:
    - `STATE_NAMES`
    - `CONST_NAMES`
    - `default()`
    """

    consts: Sequence[float] | np.ndarray
    init: Sequence[float] | np.ndarray

    STATE_NAMES: ClassVar[list[str]] = []
    CONST_NAMES: ClassVar[list[str]] = []

    def __post_init__(self) -> None:
        self.consts = _as_1d_float_array(self.consts, name="consts")
        self.init = _as_1d_float_array(self.init, name="init")

        if self.STATE_NAMES and len(self.init) != len(self.STATE_NAMES):
            raise ValueError(
                f"len(init)={len(self.init)} does not match len(STATE_NAMES)={len(self.STATE_NAMES)}"
            )
        if self.CONST_NAMES and len(self.consts) != len(self.CONST_NAMES):
            raise ValueError(
                f"len(consts)={len(self.consts)} does not match len(CONST_NAMES)={len(self.CONST_NAMES)}"
            )

    @classmethod
    def default(cls) -> "BaseParameters":
        raise NotImplementedError(f"{cls.__name__}.default() must be implemented by subclasses")

    @property
    def dim(self) -> int:
        return len(self.STATE_NAMES) if self.STATE_NAMES else len(self.init)

    @property
    def n_consts(self) -> int:
        return len(self.CONST_NAMES) if self.CONST_NAMES else len(self.consts)

    @property
    def state_names(self) -> list[str]:
        return list(self.STATE_NAMES)

    @property
    def const_names(self) -> list[str]:
        return list(self.CONST_NAMES)

    def copy(self) -> "BaseParameters":
        return self.__class__(consts=self.consts.copy(), init=self.init.copy())

    def with_overrides(
        self,
        *,
        consts: Sequence[float] | np.ndarray | None = None,
        init: Sequence[float] | np.ndarray | None = None,
    ) -> "BaseParameters":
        return self.__class__(
            consts=self.consts if consts is None else consts,
            init=self.init if init is None else init,
        )

    def const_dict(self) -> dict[str, float]:
        return {name: float(self.consts[i]) for i, name in enumerate(self.CONST_NAMES)}

    def init_dict(self) -> dict[str, float]:
        return {name: float(self.init[i]) for i, name in enumerate(self.STATE_NAMES)}

    def summary(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "n_consts": self.n_consts,
            "state_names": list(self.STATE_NAMES),
            "const_names": list(self.CONST_NAMES),
        }


class BaseModel(ABC):
    """Common abstract ODE model class.

    Typical subclass pattern:

    ```python
    class Parameters(BaseParameters):
        ...

    class Model(BaseModel):
        MODEL_ID = "..."
        MODEL_NAME = "..."
        DATASET = "..."
        SOURCE_GROUP = "..."
        ParametersCls = Parameters

        def _rhs(self, t, x, consts):
            ...
    ```
    """

    MODEL_ID: ClassVar[str] = ""
    MODEL_NAME: ClassVar[str] = ""
    DATASET: ClassVar[str] = ""
    SOURCE_GROUP: ClassVar[str] = ""
    ParametersCls: ClassVar[type[BaseParameters] | None] = None

    def __init__(self, params: BaseParameters | None = None, *, metadata: dict[str, Any] | None = None) -> None:
        if self.ParametersCls is None:
            raise ValueError(f"{self.__class__.__name__} must define ParametersCls")
        if params is None:
            params = self.ParametersCls.default()
        if not isinstance(params, self.ParametersCls):
            raise TypeError(
                f"params must be an instance of {self.ParametersCls.__name__}, got {type(params).__name__}"
            )
        self.params = params
        self.metadata = metadata or {}

    @property
    def dim(self) -> int:
        return self.params.dim

    @property
    def n_consts(self) -> int:
        return self.params.n_consts

    @property
    def state_names(self) -> list[str]:
        return self.params.state_names

    @property
    def const_names(self) -> list[str]:
        return self.params.const_names

    @classmethod
    def default_parameters(cls) -> BaseParameters:
        if cls.ParametersCls is None:
            raise ValueError(f"{cls.__name__} must define ParametersCls")
        return cls.ParametersCls.default()

    def get_default_parameters(self) -> BaseParameters:
        return self.__class__.default_parameters()

    def get_default_init(self) -> np.ndarray:
        return self.get_default_parameters().init.copy()

    def get_default_consts(self) -> np.ndarray:
        return self.get_default_parameters().consts.copy()

    def set_parameters(self, params: BaseParameters) -> None:
        if self.ParametersCls is None or not isinstance(params, self.ParametersCls):
            raise TypeError(
                f"params must be an instance of {self.ParametersCls.__name__ if self.ParametersCls else 'BaseParameters'}"
            )
        self.params = params

    def validate_state(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        x = _as_1d_float_array(x, name="state")
        if len(x) != self.dim:
            raise ValueError(f"state length {len(x)} does not match model dim {self.dim}")
        return x

    def validate_consts(self, consts: Sequence[float] | np.ndarray) -> np.ndarray:
        consts = _as_1d_float_array(consts, name="consts")
        if len(consts) != self.n_consts:
            raise ValueError(f"const length {len(consts)} does not match n_consts {self.n_consts}")
        return consts

    def rhs(
        self,
        t: float,
        x: Sequence[float] | np.ndarray,
        consts: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        x = self.validate_state(x)
        consts = self.params.consts if consts is None else self.validate_consts(consts)
        out = _as_1d_float_array(self._rhs(t, x, consts), name="rhs output")
        if len(out) != self.dim:
            raise ValueError(f"rhs output length {len(out)} does not match model dim {self.dim}")
        return out

    def make_rhs(self, consts: Sequence[float] | np.ndarray | None = None):
        consts = self.params.consts if consts is None else self.validate_consts(consts)
        return lambda t, x: self.rhs(t, x, consts)

    def __call__(
        self,
        t: float,
        x: Sequence[float] | np.ndarray,
        consts: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        return self.rhs(t, x, consts)

    def summary(self) -> dict[str, Any]:
        return {
            "model_id": self.MODEL_ID,
            "model_name": self.MODEL_NAME,
            "dataset": self.DATASET,
            "source_group": self.SOURCE_GROUP,
            "dim": self.dim,
            "n_consts": self.n_consts,
            "state_names": self.state_names,
            "const_names": self.const_names,
            **self.metadata,
        }

    @abstractmethod
    def _rhs(self, t: float, x: np.ndarray, consts: np.ndarray) -> np.ndarray:
        """Return dx/dt for the current state x and const vector consts."""
        raise NotImplementedError
