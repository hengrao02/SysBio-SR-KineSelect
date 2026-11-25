from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000708'
MODEL_NAME = 'Liu2017'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['S_a', 'I_a', 'S_h', 'I_h', 'R_h']
CONST_NAMES = ['r_a',
 'K_a',
 'beta_a',
 'mu_a',
 'delta_a',
 'pi_h',
 'beta_h',
 'mu_h',
 'gamma',
 'delta_h',
 'compartment',
 'Avian_population']

DEFAULT_INIT = np.array([100000.0, 1000.0, 1000000.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.005, 50000.0, 2e-08, 0.00034246, 0.0004, 30.0, 6e-09, 3.91e-05, 0.1, 0.3445, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000708', 'dataset_model_name': 'Liu2017', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Liu2017.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11 = consts

        # readable state aliases
        S_a = x_0
        I_a = x_1
        S_h = x_2
        I_h = x_3
        R_h = x_4

        # readable parameter aliases
        r_a = c_0
        K_a = c_1
        beta_a = c_2
        mu_a = c_3
        delta_a = c_4
        pi_h = c_5
        beta_h = c_6
        mu_h = c_7
        gamma = c_8
        delta_h = c_9
        compartment = c_10
        Avian_population = c_11

        # explicit ODE right-hand sides
        dx_0 = (1*c_11*c_0*x_0*(1 - x_0/c_1) + (-1)*c_11*c_2*x_0*x_1)/c_11
        dx_1 = (1*c_11*c_2*x_0*x_1 + (-1)*c_11*c_3*x_1 + (-1)*c_11*c_4*x_1)/c_11
        dx_2 = (1*c_10*c_5 + (-1)*c_6*x_1*x_2 + (-1)*c_10*c_7*x_2)/c_10
        dx_3 = (1*c_6*x_1*x_2 + (-1)*c_10*c_7*x_3 + (-1)*c_10*c_9*x_3 + (-1)*c_10*c_8*x_3)/c_10
        dx_4 = (1*c_10*c_8*x_3 + (-1)*c_10*c_7*x_4)/c_10

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
