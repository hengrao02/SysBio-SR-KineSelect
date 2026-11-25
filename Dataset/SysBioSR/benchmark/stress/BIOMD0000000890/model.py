from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000890'
MODEL_NAME = 'Bhattacharya2014'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['m', 'h', 'c']
CONST_NAMES = ['mu_c', 'mu_m', 'delta_m', 'delta_h', 'delta_c', 'kappa_m', 'kappa_c', 'x', 'y', 'compartment']

DEFAULT_INIT = np.array([3.65e-08, 1.1e-05, 0.023], dtype=float)
DEFAULT_CONSTS = np.array([0.462, 1.9e-10, 0.00135, 0.00193, 0.0036, 0.0001, 0.001, 3.0, 4.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000890', 'dataset_model_name': 'Bhattacharya2014', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Bhattacharya2014.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9 = consts

        # readable state aliases
        m = x_0
        h = x_1
        c = x_2

        # readable parameter aliases
        mu_c = c_0
        mu_m = c_1
        delta_m = c_2
        delta_h = c_3
        delta_c = c_4
        kappa_m = c_5
        kappa_c = c_6
        x_3 = c_7
        y = c_8
        compartment = c_9

        # explicit ODE right-hand sides
        dx_0 = (1*c_9*c_1/(1 + (c_5*(1 + (x_2/c_6)**c_8))**c_7) + (-1)*c_9*c_2*x_0)/c_9
        dx_1 = (1*c_9*x_0 + (-1)*c_9*c_3*x_1)/c_9
        dx_2 = (1*c_9*c_0*x_1 + (-1)*c_9*c_4*x_2)/c_9

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
