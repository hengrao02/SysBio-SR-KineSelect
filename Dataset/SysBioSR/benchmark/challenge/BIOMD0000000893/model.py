from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000893'
MODEL_NAME = 'GonzalezMiranda2013'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['x', 'y', 'z']
CONST_NAMES = ['A', 'B', 'C', 'epsilon', 'delta', 'compartment', '__lp_r5_k1']

DEFAULT_INIT = np.array([0.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.007, 954.5, 0.035, 2e-05, 0.029, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000893', 'dataset_model_name': 'GonzalezMiranda2013', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'GonzalezMiranda2013.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6 = consts

        # readable state aliases
        x_3 = x_0
        y = x_1
        z = x_2

        # readable parameter aliases
        A = c_0
        B = c_1
        C = c_2
        epsilon = c_3
        delta = c_4
        compartment = c_5
        lp_r5_k1 = c_6

        # explicit ODE right-hand sides
        dx_0 = (1*c_5*c_0*(1 - x_0)/(c_3 + x_2) + (-1)*c_5*c_1*x_2*x_0/(c_4 + x_0))/c_5
        dx_1 = (1*c_5*x_0*x_0 + (-1)*c_5*c_6*x_1)/c_5
        dx_2 = (1*c_5*x_1 + (-1)*c_5*c_2*x_2*(1 - x_0)/(c_3 + x_2))/c_5

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
