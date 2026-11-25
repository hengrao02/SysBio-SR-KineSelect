from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1909050002'
MODEL_NAME = 'Ribba2018'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['A_1_b', 'C_4_T']
CONST_NAMES = ['compartment',
 'k_clear_u',
 'P',
 'R_Cap',
 'xi',
 'R_K_grogh',
 'K_D',
 'V',
 'epsilon',
 'ModelValue_1',
 'ModelValue_2',
 'k_e']

DEFAULT_INIT = np.array([8.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.02, 5e-06, 8.0, 0.082, 75.0, 4e-08, 3.58, 0.25, 150.0, 0.25, 0.002], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1909050002', 'dataset_model_name': 'Ribba2018', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Ribba2018.xml', 'warnings': []}


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
        x_0, x_1 = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11 = consts

        # readable state aliases
        A_1_b = x_0
        C_4_T = x_1

        # readable parameter aliases
        compartment = c_0
        k_clear_u = c_1
        P = c_2
        R_Cap = c_3
        xi = c_4
        R_K_grogh = c_5
        K_D = c_6
        V = c_7
        epsilon = c_8
        ModelValue_1 = c_9
        ModelValue_2 = c_10
        k_e = c_11

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*x_0 + 2*c_0*c_2*c_3*c_4*(c_6*c_7*x_1/(c_8*(c_6 + c_9/c_10)) - x_0)/c_5**2
        dx_1 = 2*c_0*c_2*c_3*c_4*(-c_6*x_1/(c_8*(c_6 + c_9/c_10)) + x_0/c_7)/c_5**2 - c_0*c_11*c_9*x_1/(c_10*(c_6 + c_9/c_10))

        return np.array([dx_0, dx_1], dtype=float)


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
