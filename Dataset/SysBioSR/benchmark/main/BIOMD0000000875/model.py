from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000875'
MODEL_NAME = 'Nelson2000'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['T', 'T_i', 'V_I', 'V_NI']
CONST_NAMES = ['k', 'delta', 'N', 'c', 'np', 'lambda', 'delta1', 'plasma']

DEFAULT_INIT = np.array([180000.0, 1675.0, 134000.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([3.43e-08, 0.5, 480.0, 3.0, 0.5, 10.0, 0.03, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000875', 'dataset_model_name': 'Nelson2000', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Nelson2000.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7 = consts

        # readable state aliases
        T = x_0
        T_i = x_1
        V_I = x_2
        V_NI = x_3

        # readable parameter aliases
        k = c_0
        delta = c_1
        N = c_2
        c = c_3
        np_2 = c_4
        lambda_ = c_5
        delta1 = c_6
        plasma = c_7

        # explicit ODE right-hand sides
        dx_0 = (1*c_7*c_5 + (-1)*c_7*c_6*x_0 + (-1)*c_7*c_0*x_2*x_0)/c_7
        dx_1 = (1*c_7*c_0*x_2*x_0 + (-1)*c_7*c_1*x_1)/c_7
        dx_2 = (1*c_7*(1 - c_4)*c_2*c_1*x_1 + (-1)*c_7*c_3*x_2)/c_7
        dx_3 = (1*c_7*c_4*c_2*c_1*x_1 + (-1)*c_7*c_3*x_3)/c_7

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
