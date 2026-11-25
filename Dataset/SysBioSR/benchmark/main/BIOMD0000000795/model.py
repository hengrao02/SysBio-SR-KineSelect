from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000795'
MODEL_NAME = 'Chen2011'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['H', 'T']
CONST_NAMES = ['b1', 'u1', 'b2', 'u2', 'v', 'k', 'bone_marrow']

DEFAULT_INIT = np.array([0.5, 0.0001], dtype=float)
DEFAULT_CONSTS = np.array([0.2, 0.1, 0.8, 0.1, 0.1, 0.9, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000795', 'dataset_model_name': 'Chen2011', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Chen2011.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6 = consts

        # readable state aliases
        H = x_0
        T = x_1

        # readable parameter aliases
        b1 = c_0
        u1 = c_1
        b2 = c_2
        u2 = c_3
        v = c_4
        k = c_5
        bone_marrow = c_6

        # explicit ODE right-hand sides
        dx_0 = (1*c_6*c_2*x_0*(1 - x_0 - (1 - c_4)*x_1) + (-1)*c_6*c_3*x_0 + (-1)*c_6*c_0*x_0*x_1*c_5)/c_6
        dx_1 = (1*c_6*c_0*x_1*(1 - x_1 - (1 - c_5)*x_0) + (-1)*c_6*c_1*x_1 + (-1)*c_6*c_2*x_1*x_0*c_4)/c_6

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
