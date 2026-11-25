from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000874'
MODEL_NAME = 'Perelson1993'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['T', 'T_1', 'T_2', 'V']
CONST_NAMES = ['s', 'r', 'T_max', 'mu_T', 'k_1', 'k_2', 'mu_b', 'mu_V', 'N', 'COMpartment']

DEFAULT_INIT = np.array([1000.0, 0.0, 0.0, 0.001], dtype=float)
DEFAULT_CONSTS = np.array([10.0, 0.03, 1500.0, 0.02, 2.4e-05, 0.003, 0.24, 2.4, 1000.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000874', 'dataset_model_name': 'Perelson1993', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Perelson1993.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9 = consts

        # readable state aliases
        T = x_0
        T_1 = x_1
        T_2 = x_2
        V = x_3

        # readable parameter aliases
        s = c_0
        r = c_1
        T_max = c_2
        mu_T = c_3
        k_1 = c_4
        k_2 = c_5
        mu_b = c_6
        mu_V = c_7
        N = c_8
        COMpartment = c_9

        # explicit ODE right-hand sides
        dx_0 = (1*c_9*(c_0 + c_1*x_0) + (-1)*c_9*(c_3*x_0 + c_4*x_3*x_0 + c_1*x_0*(x_0 + x_1 + x_2)/c_2))/c_9
        dx_1 = (1*c_9*c_4*x_3*x_0 + (-1)*c_9*(c_3*x_1 + c_5*x_1))/c_9
        dx_2 = (1*c_9*c_5*x_1 + (-1)*c_9*c_6*x_2)/c_9
        dx_3 = (1*c_9*c_8*c_6*x_2 + (-1)*c_9*(c_4*x_3*x_0 + c_7*x_3))/c_9

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
