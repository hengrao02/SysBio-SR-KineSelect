from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1503180002'
MODEL_NAME = 'Smallbone2015'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['S', 'E', 'ES', 'P']
CONST_NAMES = ['kr', 'kf', 'kcat']

DEFAULT_INIT = np.array([1.0, 0.001, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1503180002', 'dataset_model_name': 'Smallbone2015', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Smallbone2015.xml', 'warnings': []}


class Parameters(BaseParameters):
    STATE_NAMES = STATE_NAMES
    CONST_NAMES = CONST_NAMES

    @classmethod
    def default(cls) -> "Parameters":
        return cls(
            consts=DEFAULT_CONSTS.copy(),
            init=DEFAULT_INIT.copy(),
        )


class Model(BaseModel):
    MODEL_ID = MODEL_ID
    MODEL_NAME = MODEL_NAME
    DATASET = DATASET
    SOURCE_GROUP = SOURCE_GROUP
    ParametersCls = Parameters

    def __init__(self, params: Parameters | None = None) -> None:
        super().__init__(params=params, metadata=MODEL_METADATA.copy())

    def _rhs(self, t: float, x: np.ndarray, consts: np.ndarray) -> np.ndarray:
        x_0, x_1, x_2, x_3 = x
        c_0, c_1, c_2 = consts

        # readable state aliases
        S = x_0
        E = x_1
        ES = x_2
        P = x_3

        # readable parameter aliases
        kr = c_0
        kf = c_1
        kcat = c_2

        # explicit ODE right-hand sides
        dx_0 = c_0*x_2 - c_1*x_0*x_1
        dx_1 = c_0*x_2 - c_1*x_0*x_1 + c_2*x_2
        dx_2 = -c_0*x_2 + c_1*x_0*x_1 - c_2*x_2
        dx_3 = c_2*x_2

        return np.array([dx_0, dx_1, dx_2, dx_3], dtype=float)


MODEL = Model()


__all__ = [
    "MODEL_ID",
    "MODEL_NAME",
    "DATASET",
    "SOURCE_GROUP",
    "STATE_NAMES",
    "CONST_NAMES",
    "DEFAULT_INIT",
    "DEFAULT_CONSTS",
    "MODEL_METADATA",
    "Parameters",
    "Model",
    "MODEL",
]
