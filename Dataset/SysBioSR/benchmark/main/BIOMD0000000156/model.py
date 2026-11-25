from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000156'
MODEL_NAME = 'Zatorsky2006'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['x', 'y', 'y0']
CONST_NAMES = ['psi', 'alpha_xy', 'beta_y', 'alpha_y', 'alpha_0', 'Theta', 'compartment']

DEFAULT_INIT = np.array([0.12986983623245899, 0.3453123597951041, 0.17001266401372103], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 3.7, 1.5, 0.9, 1.1, 2.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000156', 'dataset_model_name': 'Zatorsky2006', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Zatorsky2006.xml', 'warnings': []}


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
        y0 = x_2

        # readable parameter aliases
        psi = c_0
        alpha_xy = c_1
        beta_y = c_2
        alpha_y = c_3
        alpha_0 = c_4
        Theta = c_5
        compartment = c_6

        # explicit ODE right-hand sides
        dx_0 = (1*c_6*c_5*x_0*c_0 + (-1)*c_6*c_1*x_1*x_0)/c_6
        dx_1 = (1*c_6*c_4*x_2 + (-1)*c_6*c_3*x_1)/c_6
        dx_2 = (1*c_6*c_2*x_0*c_0 + (-1)*c_6*c_4*x_2)/c_6

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
