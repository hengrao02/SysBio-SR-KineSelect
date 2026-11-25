from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000001048'
MODEL_NAME = 'Siddhartha2002'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['Ttum', 'Tplas', 'Tnew']
CONST_NAMES = ['Tumor', 'kr1', 'b', 'kf1', 'a', 'r', 'T0', 'n', 'kf2', 'kr2', 'c', 'd']

DEFAULT_INIT = np.array([1000000.0, 4000000.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.00067, 0.2, 0.067, 0.2, 0.069, 3000000.0, 10.0, 0.00067, 0.067, 0.67, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000001048', 'dataset_model_name': 'Siddhartha2002', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Siddhartha2002.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11 = consts

        # readable state aliases
        Ttum = x_0
        Tplas = x_1
        Tnew = x_2

        # readable parameter aliases
        Tumor = c_0
        kr1 = c_1
        b = c_2
        kf1 = c_3
        a = c_4
        r = c_5
        T0 = c_6
        n = c_7
        kf2 = c_8
        kr2 = c_9
        c = c_10
        d = c_11

        # explicit ODE right-hand sides
        dx_0 = c_0*c_4*c_5*x_0*(1 - x_0/c_6) - c_0*(-c_1*x_1 + c_2*c_3*x_0)
        dx_1 = -c_0*c_10*c_11*x_1 - c_0*c_7*(-c_2*c_9*x_2 + c_8*x_1) + c_0*(-c_1*x_1 + c_2*c_3*x_0)
        dx_2 = c_0*c_4*c_5*x_2*(1 - x_2/c_6) + c_0*(-c_2*c_9*x_2 + c_8*x_1)

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
