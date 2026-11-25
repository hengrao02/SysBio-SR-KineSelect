from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000919'
MODEL_NAME = 'Ledzewicz2013'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['y', 'x']
CONST_NAMES = ['alpha', 'beta', 'gamma', 'delta', 'mu_C', 'mu_I', 'x_inf', 'v', 'compartment']

DEFAULT_INIT = np.array([0.1, 600.0], dtype=float)
DEFAULT_CONSTS = np.array([0.1181, 0.00264, 1.0, 0.37451, 0.5599, 0.00484, 780.0, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000919', 'dataset_model_name': 'Ledzewicz2013', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Ledzewicz2013.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8 = consts

        # readable state aliases
        y = x_0
        x_2 = x_1

        # readable parameter aliases
        alpha = c_0
        beta = c_1
        gamma = c_2
        delta = c_3
        mu_C = c_4
        mu_I = c_5
        x_inf = c_6
        v = c_7
        compartment = c_8

        # explicit ODE right-hand sides
        dx_0 = (1*c_8*c_0 + (-1)*c_8*c_3*x_0 + 1*c_8*c_5*(1 - c_1*x_1)*x_1*x_0)/c_8
        dx_1 = ((-1)*c_8*c_2*x_1*x_0 + 1*c_8*c_4*x_1*(1 - (x_1/c_6)**c_7))/c_8

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
