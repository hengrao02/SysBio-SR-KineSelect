from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000274'
MODEL_NAME = 'Rattanakul2003'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['x', 'y', 'z']
CONST_NAMES = ['epsilon', 'delta', 'a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'k1', 'k2', 'k3']

DEFAULT_INIT = np.array([2.0, 1.0, 0.15], dtype=float)
DEFAULT_CONSTS = np.array([0.1, 0.9, 0.05, 0.009, 0.675, 0.01, 0.005, 0.1, 0.3, 0.01, 0.1, 0.5, 0.025], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000274', 'dataset_model_name': 'Rattanakul2003', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Rattanakul2003.xml', 'warnings': []}


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
        x_3 = x_0
        y = x_1
        z = x_2

        # readable parameter aliases
        epsilon = c_0
        delta = c_1
        a1 = c_2
        a2 = c_3
        a3 = c_4
        a4 = c_5
        a5 = c_6
        b1 = c_7
        b2 = c_8
        b3 = c_9
        k1 = c_10
        k2 = c_11
        k3 = c_12

        # explicit ODE right-hand sides
        dx_0 = c_2/(c_10 + x_1) - c_7*x_0
        dx_1 = c_0*((c_3 + c_4*x_0)*x_1*x_2/(c_11 + x_0**2) - c_8*x_1)
        dx_2 = c_0*c_1*(c_5*x_0 - (c_9*x_2 + c_6*x_0*x_2/(c_12 + x_0)))

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
