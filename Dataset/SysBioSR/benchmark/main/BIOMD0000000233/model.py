from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000233'
MODEL_NAME = 'Wilhelm2009'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['X', 'Y']
CONST_NAMES = ['r4__k4', 'r2__k2', 'r3__k3', 'S', 'r1__k1']

DEFAULT_INIT = np.array([1.0, 1.0], dtype=float)
DEFAULT_CONSTS = np.array([1.5, 1.0, 1.0, 1.0, 8.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000233', 'dataset_model_name': 'Wilhelm2009', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Wilhelm2009.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4 = consts

        # readable state aliases
        X = x_0
        Y = x_1

        # readable parameter aliases
        r4_k4 = c_0
        r2_k2 = c_1
        r3_k3 = c_2
        S = c_3
        r1_k1 = c_4

        # explicit ODE right-hand sides
        dx_0 = -c_0*x_0 - c_1*x_0**2 - c_2*x_0*x_1 + 2*c_3*c_4*x_1
        dx_1 = c_1*x_0**2 - c_3*c_4*x_1

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
