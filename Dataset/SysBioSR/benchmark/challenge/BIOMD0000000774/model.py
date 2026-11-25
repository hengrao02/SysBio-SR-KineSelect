from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000774'
MODEL_NAME = 'Wodarz2018'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['S', 'D']
CONST_NAMES = ['a', 'r_0', 'p_0', 'h1', 'h2', 'k1', 'k2', 'g_0', 'h3', 'k3', 'tme']

DEFAULT_INIT = np.array([1.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.0025, 0.01, 0.7, 0.0001, 0.0001, 1.0, 1.0, 0.0035, 0.01, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000774', 'dataset_model_name': 'Wodarz2018', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Wodarz2018.xml', 'warnings': []}


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
        x_0, x_1 = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10 = consts

        # readable state aliases
        S = x_0
        D = x_1

        # readable parameter aliases
        a = c_0
        r_0 = c_1
        p_0 = c_2
        h1 = c_3
        h2 = c_4
        k1 = c_5
        k2 = c_6
        g_0 = c_7
        h3 = c_8
        k3 = c_9
        tme = c_10

        # explicit ODE right-hand sides
        dx_0 = (1*c_10*c_1/(1 + c_3*x_1**c_5)*x_0*(2*c_2/(1 + c_4*x_1**c_6) - 1) + 1*c_10*c_7/(1 + c_8*x_0**c_9)*x_1)/c_10
        dx_1 = ((-1)*c_10*c_7/(1 + c_8*x_0**c_9)*x_1 + 1*c_10*2*c_1/(1 + c_3*x_1**c_5)*x_0*(1 - c_2/(1 + c_4*x_1**c_6)) + (-1)*c_10*c_0*x_1)/c_10

        return np.array([dx_0, dx_1], dtype=float)


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
