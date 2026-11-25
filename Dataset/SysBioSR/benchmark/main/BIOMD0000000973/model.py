from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000973'
MODEL_NAME = 'Dasgupta2020'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['P', 'N']
CONST_NAMES = ['Cell', 'k_1', 'k2', 'k1', 'k3', 'k_2']

DEFAULT_INIT = np.array([10000.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 6.6e-09, 7.5e-06, 0.00021, 1.16, 0.03], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000973', 'dataset_model_name': 'Dasgupta2020', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Dasgupta2020.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5 = consts

        # readable state aliases
        P = x_0
        N = x_1

        # readable parameter aliases
        Cell = c_0
        k_1 = c_1
        k2 = c_2
        k1 = c_3
        k3 = c_4
        k_2 = c_5

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1*x_1 - 2*c_0*c_2*x_0**2 - c_0*c_3*x_0*x_1
        dx_1 = c_0*c_2*x_0**2 - c_0*c_4*x_1 - c_0*c_5*x_1**2

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
