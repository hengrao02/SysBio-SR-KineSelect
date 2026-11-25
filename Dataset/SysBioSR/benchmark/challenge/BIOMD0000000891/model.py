from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000891'
MODEL_NAME = 'Khajanchi2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['u', 'v', 'w']
CONST_NAMES = ['r_1',
 'alpha_1',
 'alpha_2',
 'k_1',
 'r_2',
 'alpha_3',
 'k_2',
 'gamma_1',
 'k_3',
 'mu_1',
 'alpha_4',
 'k_4',
 'compartment']

DEFAULT_INIT = np.array([0.1, 0.55, 0.2], dtype=float)
DEFAULT_CONSTS = np.array([0.4822,
 0.069943,
 2.74492,
 0.90305,
 0.3307,
 0.0194,
 0.030584,
 0.1245,
 2.8743,
 0.0074,
 0.01694,
 0.378918,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000891', 'dataset_model_name': 'Khajanchi2019', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Khajanchi2019.xml', 'warnings': []}


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
        u = x_0
        v = x_1
        w = x_2

        # readable parameter aliases
        r_1 = c_0
        alpha_1 = c_1
        alpha_2 = c_2
        k_1 = c_3
        r_2 = c_4
        alpha_3 = c_5
        k_2 = c_6
        gamma_1 = c_7
        k_3 = c_8
        mu_1 = c_9
        alpha_4 = c_10
        k_4 = c_11
        compartment = c_12

        # explicit ODE right-hand sides
        dx_0 = (1*c_12*c_0*x_0*(1 - x_0) + (-1)*c_12*(c_1*x_1 + c_2*x_2)/(x_0 + c_3)*x_0)/c_12
        dx_1 = (1*c_12*c_4*x_1*(1 - x_1) + (-1)*c_12*c_5*x_0*x_1/(x_0 + c_6))/c_12
        dx_2 = (1*c_12*c_7*x_0*x_2/(c_8 + x_0) + (-1)*c_12*(c_9*x_2 + c_10*x_0*x_2/(x_0 + c_11)))/c_12

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
