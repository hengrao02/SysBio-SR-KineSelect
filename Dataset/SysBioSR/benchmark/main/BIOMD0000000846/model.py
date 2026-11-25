from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000846'
MODEL_NAME = 'Gulbudak2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['S', 'C', 'V_C']
CONST_NAMES = ['compartment', 'r', 'K', 'phi_tilde', 'r_tilde', 'alpha', 'mu']

DEFAULT_INIT = np.array([830000000.0, 0.0, 33200000.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.339, 894700000.0, 5e-12, 0.2, 0.1, 0.0866], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000846', 'dataset_model_name': 'Gulbudak2019', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Gulbudak2019.xml', 'warnings': []}


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
        S = x_0
        C = x_1
        V_C = x_2

        # readable parameter aliases
        compartment = c_0
        r = c_1
        K = c_2
        phi_tilde = c_3
        r_tilde = c_4
        alpha = c_5
        mu = c_6

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1*x_0*(1 - (x_0 + x_1)/c_2) - c_0*c_3*x_0*x_2 - c_0*x_0/24
        dx_1 = c_0*c_3*x_0*x_2 + c_0*c_4*x_1*(1 - (x_0 + x_1)/c_2) - c_0*x_1/20
        dx_2 = -c_0*c_3*x_0*x_2 + c_0*c_5*x_1 - c_0*c_6*x_2

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
