from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000157'
MODEL_NAME = 'Zatorsky2006'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['x', 'y', 'y0']
CONST_NAMES = ['beta_x', 'psi', 'alpha_x', 'beta_y', 'alpha_y', 'alpha_0', 'k', 'alpha_k', 'compartment']

DEFAULT_INIT = np.array([0.0, 0.8, 0.1], dtype=float)
DEFAULT_CONSTS = np.array([0.9, 1.0, 0.0, 1.1, 0.8, 0.8, 0.0001, 1.7, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000157', 'dataset_model_name': 'Zatorsky2006', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Zatorsky2006.xml', 'warnings': []}


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
        x_3 = x_0
        y = x_1
        y0 = x_2

        # readable parameter aliases
        beta_x = c_0
        psi = c_1
        alpha_x = c_2
        beta_y = c_3
        alpha_y = c_4
        alpha_0 = c_5
        k = c_6
        alpha_k = c_7
        compartment = c_8

        # explicit ODE right-hand sides
        dx_0 = (1*c_8*c_0*c_1 + (-1)*c_8*c_2*x_0 + (-1)*c_8*c_7*x_1*x_0/(x_0 + c_6))/c_8
        dx_1 = (1*c_8*c_5*x_2 + (-1)*c_8*c_4*x_1)/c_8
        dx_2 = (1*c_8*c_3*x_0*c_1 + (-1)*c_8*c_5*x_2)/c_8

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
