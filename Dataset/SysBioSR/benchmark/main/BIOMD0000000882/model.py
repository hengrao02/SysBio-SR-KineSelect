from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000882'
MODEL_NAME = 'Munz2009'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['Susceptible', 'Zombie', 'Removal']
CONST_NAMES = ['alpha', 'beta', 'delta', 'zeta', 'p', 'compartment']

DEFAULT_INIT = np.array([500.0, 1.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.005, 0.0095, 0.0001, 0.0001, 0.05, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000882', 'dataset_model_name': 'Munz2009', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Munz2009.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5 = consts

        # readable state aliases
        Susceptible = x_0
        Zombie = x_1
        Removal = x_2

        # readable parameter aliases
        alpha = c_0
        beta = c_1
        delta = c_2
        zeta = c_3
        p = c_4
        compartment = c_5

        # explicit ODE right-hand sides
        dx_0 = (1*c_5*c_4 + (-1)*c_5*(c_1*x_0*x_1 + c_2*x_0))/c_5
        dx_1 = (1*c_5*(c_1*x_0*x_1 + c_3*x_2) + (-1)*c_5*c_0*x_0*x_1)/c_5
        dx_2 = (1*c_5*(c_0*x_0*x_1 + c_2*x_0) + (-1)*c_5*c_3*x_2)/c_5

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
