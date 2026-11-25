from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1304300000'
MODEL_NAME = 'Hofmeyr1996'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['s1', 's2', 's3']
CONST_NAMES = ['v1__e1',
 'x0',
 'v1__p1',
 'v3__e3',
 'x4',
 'v3__p3',
 'v4__e4',
 'x5',
 'v4__p4',
 'v2__e2',
 'x6',
 'x7',
 'v2__p2']

DEFAULT_INIT = np.array([0.0, 2.5, 2.5], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 10.0, 10.0, 1.0, 0.0, 50.0, 1.0, 0.0, 10.0, 1.0, 1.0, 1.0, 10.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1304300000', 'dataset_model_name': 'Hofmeyr1996', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Hofmeyr1996.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12 = consts

        # readable state aliases
        s1 = x_0
        s2 = x_1
        s3 = x_2

        # readable parameter aliases
        v1_e1 = c_0
        x0 = c_1
        v1_p1 = c_2
        v3_e3 = c_3
        x4 = c_4
        v3_p3 = c_5
        v4_e4 = c_6
        x5 = c_7
        v4_p4 = c_8
        v2_e2 = c_9
        x6 = c_10
        x7 = c_11
        v2_p2 = c_12

        # explicit ODE right-hand sides
        dx_0 = c_0*(c_1*c_2*x_1 - x_0*x_2)/(c_1*x_1 + c_1 + x_0*x_2 + x_0 + x_1 + x_2 + 1) - c_3*(-c_4 + c_5*x_0)/(c_4 + x_0 + 1) - c_6*(-c_7 + c_8*x_0)/(c_7 + x_0 + 1)
        dx_1 = -c_0*(c_1*c_2*x_1 - x_0*x_2)/(c_1*x_1 + c_1 + x_0*x_2 + x_0 + x_1 + x_2 + 1) + c_9*(c_10*c_12*x_2 - c_11*x_1)/(c_10*x_2 + c_10 + c_11*x_1 + c_11 + x_1 + x_2 + 1)
        dx_2 = c_0*(c_1*c_2*x_1 - x_0*x_2)/(c_1*x_1 + c_1 + x_0*x_2 + x_0 + x_1 + x_2 + 1) - c_9*(c_10*c_12*x_2 - c_11*x_1)/(c_10*x_2 + c_10 + c_11*x_1 + c_11 + x_1 + x_2 + 1)

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
