from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000838'
MODEL_NAME = 'Tsur2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['A', 'T', 'M']
CONST_NAMES = ['alpha_A', 'b', 'mu_a', 'alpha_e', 'mu_e', 'gamma_mel', 'nu_mel', 'g', 'compartment']

DEFAULT_INIT = np.array([14119.902077922077, 66094173.0355407, 1000000.0], dtype=float)
DEFAULT_CONSTS = np.array([2986.0, 92330.0, 0.231, 831.8, 0.1777, 0.04496, 0.1245, 60100000.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000838', 'dataset_model_name': 'Tsur2019', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Tsur2019.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8 = consts

        # readable state aliases
        A = x_0
        T = x_1
        M = x_2

        # readable parameter aliases
        alpha_A = c_0
        b = c_1
        mu_a = c_2
        alpha_e = c_3
        mu_e = c_4
        gamma_mel = c_5
        nu_mel = c_6
        g = c_7
        compartment = c_8

        # explicit ODE right-hand sides
        dx_0 = (1*c_8*c_0*x_2/(x_2 + c_1) + (-1)*c_8*c_2*x_0)/c_8
        dx_1 = (1*c_8*c_3*x_0 + (-1)*c_8*c_4*x_1)/c_8
        dx_2 = (1*c_8*c_5*x_2 + (-1)*c_8*c_6*x_1*x_2/(x_2 + c_7))/c_8

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
