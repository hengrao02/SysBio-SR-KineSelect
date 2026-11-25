from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000521'
MODEL_NAME = 'Ribba2012'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['C', 'P', 'Q', 'Qp']
CONST_NAMES = ['lambda_P', 'k_PQ', 'k_Qp_P', 'delta_QP', 'gamma', 'KDE', 'K']

DEFAULT_INIT = np.array([1.0, 7.13, 41.2, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.121, 0.00295, 0.0031, 0.0087, 0.729, 0.24, 100.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000521', 'dataset_model_name': 'Ribba2012', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Ribba2012.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6 = consts

        # readable state aliases
        C = x_0
        P = x_1
        Q = x_2
        Qp = x_3

        # readable parameter aliases
        lambda_P = c_0
        k_PQ = c_1
        k_Qp_P = c_2
        delta_QP = c_3
        gamma = c_4
        KDE = c_5
        K = c_6

        # explicit ODE right-hand sides
        dx_0 = (-c_5)*x_0
        dx_1 = c_0*x_1*(1 - (x_1 + x_2 + x_3)/c_6) + c_2*x_3 - c_1*x_1 - c_4*x_0*c_5*x_1
        dx_2 = c_1 - c_4*x_0*c_5*x_2
        dx_3 = c_4*x_0*c_5*x_2 - c_2*x_3 - c_3*x_3

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
