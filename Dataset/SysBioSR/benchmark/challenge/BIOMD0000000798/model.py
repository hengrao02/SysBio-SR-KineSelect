from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000798'
MODEL_NAME = 'Sharp2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['S', 'A', 'D', 'L', 'T']
CONST_NAMES = ['ps', 'pa', 'pl', 'ds', 'da', 'dl', 'ud', 'ut', 'k1', 'k2', 'a', 'y', 'bone_marrow']

DEFAULT_INIT = np.array([0.1, 0.0, 0.0, 0.1, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.5, 0.43, 0.27, 0.14, 0.44, 0.05, 0.275, 0.3, 1.0, 1.0, 0.015, 0.01, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000798', 'dataset_model_name': 'Sharp2019', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Sharp2019.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12 = consts

        # readable state aliases
        S = x_0
        A = x_1
        D = x_2
        L = x_3
        T = x_4

        # readable parameter aliases
        ps = c_0
        pa = c_1
        pl = c_2
        ds = c_3
        da = c_4
        dl = c_5
        ud = c_6
        ut = c_7
        k1 = c_8
        k2 = c_9
        a = c_10
        y = c_11
        bone_marrow = c_12

        # explicit ODE right-hand sides
        dx_0 = (1*c_12*c_0*x_0*(c_8 - x_0) + (-1)*c_12*c_3*x_0)/c_12
        dx_1 = (1*c_12*c_3*x_0 + 1*c_12*c_1*x_1*(c_9 - (x_1 + x_3)) + (-1)*c_12*c_4*x_1)/c_12
        dx_2 = (1*c_12*c_4*x_1 + (-1)*c_12*c_6*x_2)/c_12
        dx_3 = (1*c_12*c_2*x_3*(c_9 - (x_1 + x_3)) + (-1)*c_12*c_5*x_3 + (-1)*c_12*c_10*x_3/(c_11 + x_3))/c_12
        dx_4 = (1*c_12*c_5*x_3 + (-1)*c_12*c_7*x_4)/c_12

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
