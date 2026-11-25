from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000840'
MODEL_NAME = 'Caldwell2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['M', 'C1', 'C2', 'A', 'T']
CONST_NAMES = ['lambda',
 'rho',
 'alpha_1',
 'alpha_2',
 'beta',
 'delta',
 'epsilon',
 'gamma_1',
 'gamma_2',
 'gamma_3',
 'compartment']

DEFAULT_INIT = np.array([37600000.0, 5640000.0, 3760000.0, 2000000.0, 700000.0], dtype=float)
DEFAULT_CONSTS = np.array([3000000.0, 1e-06, 0.22, 0.45, 0.14, 0.05, 0.03, 0.24, 0.293, 8e-10, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000840', 'dataset_model_name': 'Caldwell2019', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Caldwell2019.xml', 'warnings': []}


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
        x_0, x_1, x_2, x_3, x_4 = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10 = consts

        # readable state aliases
        M = x_0
        C1 = x_1
        C2 = x_2
        A = x_3
        T = x_4

        # readable parameter aliases
        lambda_ = c_0
        rho = c_1
        alpha_1 = c_2
        alpha_2 = c_3
        beta = c_4
        delta = c_5
        epsilon = c_6
        gamma_1 = c_7
        gamma_2 = c_8
        gamma_3 = c_9
        compartment = c_10

        # explicit ODE right-hand sides
        dx_0 = (1*c_10*c_0/(1 + c_1*x_3) + (-1)*c_10*c_2*x_0 + (-1)*c_10*c_3*x_0)/c_10
        dx_1 = (1*c_10*c_2*x_0 + (-1)*c_10*c_4*x_1 + (-1)*c_10*c_5*x_1)/c_10
        dx_2 = ((-1)*c_10*c_5*x_2 + 1*c_10*c_5*x_1 + (-1)*c_10*c_4*x_2)/c_10
        dx_3 = (1*c_10*c_5*x_2 + (-1)*c_10*c_6*x_3 + 1*c_10*c_7*x_4 + 1*c_10*c_9*x_3*x_4)/c_10
        dx_4 = (1*c_10*c_6*x_3 + (-1)*c_10*c_7*x_4 + (-1)*c_10*c_9*x_3*x_4 + (-1)*c_10*c_8*x_4)/c_10

        return np.array([dx_0, dx_1, dx_2, dx_3, dx_4], dtype=float)


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
