from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000906'
MODEL_NAME = 'Dubey2007'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['P', 'I', 'M']
CONST_NAMES = ['beta', 'gamma', 'beta0', 'mu', 'a', 'b', 'n', 'alpha', 'alpha0', 'compartment']

DEFAULT_INIT = np.array([1.0, 1.0, 1.0], dtype=float)
DEFAULT_CONSTS = np.array([0.9, 0.05, 0.2, 0.04, 0.8, 0.3, 0.1, 2.4, 0.1, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000906', 'dataset_model_name': 'Dubey2007', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Dubey2007.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9 = consts

        # readable state aliases
        P = x_0
        I = x_1
        M = x_2

        # readable parameter aliases
        beta = c_0
        gamma = c_1
        beta0 = c_2
        mu = c_3
        a = c_4
        b = c_5
        n = c_6
        alpha = c_7
        alpha0 = c_8
        compartment = c_9

        # explicit ODE right-hand sides
        dx_0 = (1*c_9*c_0*x_0 + (-1)*c_9*(c_1*x_1*x_0 + c_2*x_0*x_0))/c_9
        dx_1 = (1*c_9*(c_3 + c_5*x_1*x_0) + (-1)*c_9*(c_4*x_1 + c_6*c_1*x_1*x_0))/c_9
        dx_2 = (1*c_9*c_7*x_0 + (-1)*c_9*c_8*x_2)/c_9

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
