from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000887'
MODEL_NAME = 'Lim2014'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['x', 'u', 'y', 'z']
CONST_NAMES = ['lambda', 'beta', 'r', 'tau', 'gamma', 'nu', 'mu_1', 'mu_2', 'mu_3', 'mu_4', 'compartment']

DEFAULT_INIT = np.array([850.0, 0.1, 0.5, 0.1], dtype=float)
DEFAULT_CONSTS = np.array([10.0, 0.001, 0.011, 0.003, 0.029, 0.036, 0.012, 0.03, 0.03, 0.03, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000887', 'dataset_model_name': 'Lim2014', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Lim2014.xml', 'warnings': []}


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
        x_0, x_1, x_2, x_3 = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10 = consts

        # readable state aliases
        x_4 = x_0
        u = x_1
        y = x_2
        z = x_3

        # readable parameter aliases
        lambda_ = c_0
        beta = c_1
        r = c_2
        tau = c_3
        gamma = c_4
        nu = c_5
        mu_1 = c_6
        mu_2 = c_7
        mu_3 = c_8
        mu_4 = c_9
        compartment = c_10

        # explicit ODE right-hand sides
        dx_0 = (1*c_10*c_0 + (-1)*c_10*c_1*x_0*x_2 + (-1)*c_10*c_6*x_0)/c_10
        dx_1 = (1*c_10*c_1*x_0*x_2 + 1*c_10*c_2*x_2 + (-1)*c_10*c_3*x_1 + (-1)*c_10*c_7*x_1)/c_10
        dx_2 = (1*c_10*c_3*x_1 + (-1)*c_10*c_4*x_2*x_3 + (-1)*c_10*c_8*x_2)/c_10
        dx_3 = (1*c_10*c_5*x_2 + (-1)*c_10*c_9*x_3)/c_10

        return np.array([dx_0, dx_1, dx_2, dx_3], dtype=float)


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
