from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000686'
MODEL_NAME = 'Wodarz2007'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['x', 'y0', 'y1', 'L', 'v']
CONST_NAMES = ['lambda', 'd', 'beta', 'gamma', 'phi', 'a0', 'eta', 'a1', 'k', 'u']

DEFAULT_INIT = np.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
DEFAULT_CONSTS = np.array([10.0, 0.1, 0.1, 0.5, 0.1, 0.1, 0.01, 0.2, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000686', 'dataset_model_name': 'Wodarz2007', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Wodarz2007.xml', 'warnings': []}


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
        x_0, x_1, x_2, x_3, x_4 = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9 = consts

        # readable state aliases
        x_5 = x_0
        y0 = x_1
        y1 = x_2
        L = x_3
        v = x_4

        # readable parameter aliases
        lambda_ = c_0
        d = c_1
        beta = c_2
        gamma = c_3
        phi = c_4
        a0 = c_5
        eta = c_6
        a1 = c_7
        k = c_8
        u = c_9

        # explicit ODE right-hand sides
        dx_0 = c_0 - c_1*x_0 - c_2*x_0*x_4 - c_3*x_0*x_4
        dx_1 = c_2*x_0*x_4 + c_4*x_3 - c_5*x_1 - c_6*x_1
        dx_2 = c_6*x_1 - c_7*x_2
        dx_3 = -c_1*x_3 + c_3*x_0*x_4 - c_4*x_3
        dx_4 = c_8*x_2 - c_9*x_4

        return np.array([dx_0, dx_1, dx_2, dx_3, dx_4], dtype=float)


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
