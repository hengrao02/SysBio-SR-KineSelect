from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000894'
MODEL_NAME = 'Bose2011'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['x', 'y', 'z']
CONST_NAMES = ['tau', 'R', 'Dx', 'Dy', 'Dz', 'p', 'mu', 'compartment']

DEFAULT_INIT = np.array([1e-06, 0.01, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.3, 1.0, 2.1, 0.01, 1.2, 0.06, 20.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000894', 'dataset_model_name': 'Bose2011', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Bose2011.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7 = consts

        # readable state aliases
        x_3 = x_0
        y = x_1
        z = x_2

        # readable parameter aliases
        tau = c_0
        R = c_1
        Dx = c_2
        Dy = c_3
        Dz = c_4
        p = c_5
        mu = c_6
        compartment = c_7

        # explicit ODE right-hand sides
        dx_0 = (1*c_7*((1 + c_1*(1 - c_2*c_0) + 1/2*c_2*c_4*c_0)*x_0 + c_2*(1 + c_1*c_0)*x_2) + (-1)*c_7*(x_0*x_0 + x_0*x_1))/c_7
        dx_1 = (1*c_7*x_1*x_2 + (-1)*c_7*(c_5 - c_3)*x_1)/c_7
        dx_2 = (1*c_7*(c_1*(1 - c_2*c_0) + c_2*c_4*c_0)*x_0 + (-1)*c_7*(c_6 - (c_1*(1 + c_2*c_0) + c_2*(1 + 1/2*c_4*c_0)))*x_2)/c_7

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
