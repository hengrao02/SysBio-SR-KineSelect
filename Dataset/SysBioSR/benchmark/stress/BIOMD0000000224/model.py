from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000224'
MODEL_NAME = 'Meyer1991'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['CaI', 'IP3', 'CaS', 'g']
CONST_NAMES = ['L', 'A', 'k1', 'B', 'k2', 'C', 'k3', 'R', 'D', 'F', 'E']

DEFAULT_INIT = np.array([0.1, 0.05, 1100.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.01, 20.0, 0.5, 40.0, 0.15, 1.1, 1.0, 0.09, 2.0, 0.02, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000224', 'dataset_model_name': 'Meyer1991', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Meyer1991.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10 = consts

        # readable state aliases
        CaI = x_0
        IP3 = x_1
        CaS = x_2
        g = x_3

        # readable parameter aliases
        L = c_0
        A = c_1
        k1 = c_2
        B = c_3
        k2 = c_4
        C = c_5
        k3 = c_6
        R = c_7
        D = c_8
        F = c_9
        E = c_10

        # explicit ODE right-hand sides
        dx_0 = -0.0001*c_3*x_0**2/(c_4**2 + 0.0001*x_0**2) + x_2*(-x_3 + 1)*(c_0 + 0.0625*c_1*x_1**4/(c_2 + 0.5*x_1)**4)
        dx_1 = c_5*(-c_6/((c_6 + 0.01*x_0)*(c_7 + 1)) + 1) - 0.5*c_8*x_1
        dx_2 = 0.0001*c_3*x_0**2/(c_4**2 + 0.0001*x_0**2) - x_2*(-x_3 + 1)*(c_0 + 0.0625*c_1*x_1**4/(c_2 + 0.5*x_1)**4)
        dx_3 = 1.0e-8*c_10*x_0**4*(-x_3 + 1) - c_9

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
