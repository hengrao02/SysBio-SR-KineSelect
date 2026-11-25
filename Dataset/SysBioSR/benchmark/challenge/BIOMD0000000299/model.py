from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000299'
MODEL_NAME = 'Leloup1999'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['M', 'FC', 'FN']
CONST_NAMES = ['vs', 'vm', 'Km', 'KI', 'n', 'ks', 'vd', 'Kd', 'k1', 'k2']

DEFAULT_INIT = np.array([4.95382004497908, 1.7552479757933654, 0.8331378725127524], dtype=float)
DEFAULT_CONSTS = np.array([1.6, 0.505, 0.5, 1.0, 4.0, 0.5, 1.4, 0.13, 0.5, 0.6], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000299', 'dataset_model_name': 'Leloup1999', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Leloup1999.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9 = consts

        # readable state aliases
        M = x_0
        FC = x_1
        FN = x_2

        # readable parameter aliases
        vs = c_0
        vm = c_1
        Km = c_2
        KI = c_3
        n = c_4
        ks = c_5
        vd = c_6
        Kd = c_7
        k1 = c_8
        k2 = c_9

        # explicit ODE right-hand sides
        dx_0 = c_0*c_3**c_4/(c_3**c_4 + x_2**c_4) - c_1*x_0/(c_2 + x_0)
        dx_1 = c_5*x_0 + c_9*x_2 - (c_6*x_1/(c_7 + x_1) + c_8*x_1)
        dx_2 = c_8*x_1 - c_9*x_2

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
