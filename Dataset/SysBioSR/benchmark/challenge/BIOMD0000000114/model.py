from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000114'
MODEL_NAME = 'Somogyi1990'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['x', 'y']
CONST_NAMES = ['k', 'alpha', 'n', 'a', 'k1', 'beta', 'gamma', 'cytoplasm', 'ER', 'extracellular']

DEFAULT_INIT = np.array([1.0, 1.0], dtype=float)
DEFAULT_CONSTS = np.array([0.01, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000114', 'dataset_model_name': 'Somogyi1990', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Somogyi1990.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9 = consts

        # readable state aliases
        x_2 = x_0
        y = x_1

        # readable parameter aliases
        k = c_0
        alpha = c_1
        n = c_2
        a = c_3
        k1 = c_4
        beta = c_5
        gamma = c_6
        cytoplasm = c_7
        ER = c_8
        extracellular = c_9

        # explicit ODE right-hand sides
        dx_0 = ((-1)*(c_0*x_0*c_7 - c_4*x_1*c_8) + (-1)*c_1*x_1**c_2/(c_3**c_2 + x_1**c_2)*x_0*c_7)/c_8
        dx_1 = (1*c_6*c_7 + 1*(c_0*x_0*c_7 - c_4*x_1*c_8) + (-1)*c_5*x_1*c_9 + 1*c_1*x_1**c_2/(c_3**c_2 + x_1**c_2)*x_0*c_7)/c_7

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
