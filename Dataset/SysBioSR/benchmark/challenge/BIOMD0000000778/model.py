from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000778'
MODEL_NAME = 'Wei2017'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['T', 'I', 'C']
CONST_NAMES = ['r', 'K', 'd', 'm', 'beta', 'k', 'a', 'l1', 'alpha', 'b', 'u', 'l2', 'tme']

DEFAULT_INIT = np.array([0.5, 0.01, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.01, 1000.0, 0.1, 50.0, 0.1, 1000.0, 0.03, 0.0, 0.01, 100.0, 50.0, 0.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000778', 'dataset_model_name': 'Wei2017', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Wei2017.xml', 'warnings': []}


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
        x_0, x_1, x_2 = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12 = consts

        # readable state aliases
        T = x_0
        I = x_1
        C = x_2

        # readable parameter aliases
        r = c_0
        K = c_1
        d = c_2
        m = c_3
        beta = c_4
        k = c_5
        a = c_6
        l1 = c_7
        alpha = c_8
        b = c_9
        u = c_10
        l2 = c_11
        tme = c_12

        # explicit ODE right-hand sides
        dx_0 = (1*c_12*c_0*x_0*(1 - x_0/c_1) + (-1)*c_12*c_2*x_0*x_2/(c_3 + x_0))/c_12
        dx_1 = (1*c_12*c_4*x_0*x_1/(c_5 + x_0) + (-1)*c_12*c_6*x_1 + 1*c_12*c_7)/c_12
        dx_2 = (1*c_12*c_8*x_0*x_1/(c_9 + x_0) + (-1)*c_12*c_10*x_2 + 1*c_12*c_11)/c_12

        return np.array([dx_0, dx_1, dx_2], dtype=float)


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
