from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000967'
MODEL_NAME = 'McLean1991'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['R', 'L', 'E', 'V']
CONST_NAMES = ['Blood', 'T', 'k', 'u', 'beta', 'p', 'z', 'alpha', 'delta', 'pi', 'sigma']

DEFAULT_INIT = np.array([100000000.0, 1000000.0, 0.0, 7000000.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 200000000.0, 4.31e-06, 0.04, 2.05e-09, 0.2, 0.9, 0.288, 16.0, 300.0, 3.75], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000967', 'dataset_model_name': 'McLean1991', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'McLean1991.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10 = consts

        # readable state aliases
        R = x_0
        L = x_1
        E = x_2
        V = x_3

        # readable parameter aliases
        Blood = c_0
        T = c_1
        k = c_2
        u = c_3
        beta = c_4
        p = c_5
        z = c_6
        alpha = c_7
        delta = c_8
        pi = c_9
        sigma = c_10

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1/(c_2*x_3 + 1) - c_0*c_3*x_0 - c_0*c_4*c_5*x_0*x_3*(-c_6 + 1) - c_0*c_4*x_0*x_3*(-c_5 + 1)*(-c_6 + 1)
        dx_1 = -c_0*c_3*x_1 + c_0*c_4*c_5*x_0*x_3*(-c_6 + 1) - c_0*c_7*x_1
        dx_2 = c_0*c_4*x_0*x_3*(-c_5 + 1)*(-c_6 + 1) + c_0*c_7*x_1 - c_0*c_8*x_2
        dx_3 = -c_0*c_10*x_3 + c_0*c_9*x_2

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
