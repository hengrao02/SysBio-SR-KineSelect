from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000851'
MODEL_NAME = 'Ho2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['S', 'V', 'V_e', 'I', 'R']
CONST_NAMES = ['compartment', 'r', 'A', 'beta', 'ModelValue_3', 'ModelValue_5', 'gamma']

DEFAULT_INIT = np.array([0.9424, 0.0565, 0.0565, 0.0012, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.0155, 0.1155, 2.7516, 2.7516, 0.45, 2.1272], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000851', 'dataset_model_name': 'Ho2019', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Ho2019.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6 = consts

        # readable state aliases
        S = x_0
        V = x_1
        V_e = x_2
        I = x_3
        R = x_4

        # readable parameter aliases
        compartment = c_0
        r = c_1
        A = c_2
        beta = c_3
        ModelValue_3 = c_4
        ModelValue_5 = c_5
        gamma = c_6

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*(1 - x_2/c_2) - c_0*c_3*x_0*x_3
        dx_1 = c_0*c_1*(1 - x_2/c_2) - c_0*c_4*x_1*x_3*(-c_5 + 1)
        dx_2 = c_0*c_1*(1 - x_2/c_2)
        dx_3 = c_0*c_3*x_0*x_3 + c_0*c_4*x_1*x_3*(-c_5 + 1) - c_0*c_6*x_3
        dx_4 = c_0*c_6*x_3

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
