from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000001060'
MODEL_NAME = 'Frank2021'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['x1', 'x2']
CONST_NAMES = ['Macrophage',
 'b1',
 'q1',
 'p2',
 'l2',
 'S1',
 'a1',
 'n1',
 'k1',
 'b2',
 'q2',
 'S2',
 'p1',
 'l1',
 'a2',
 'n2',
 'k2']

DEFAULT_INIT = np.array([1.2, 2.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.05, 5.0, 1.0, 1.0, 3.75, 5.0, 6.0, 1.0, 0.05, 5.0, 3.75, 0.5, 1.0, 5.0, 6.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000001060', 'dataset_model_name': 'Frank2021', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Frank2021.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16 = consts

        # readable state aliases
        x1 = x_0
        x2 = x_1

        # readable parameter aliases
        Macrophage = c_0
        b1 = c_1
        q1 = c_2
        p2 = c_3
        l2 = c_4
        S1 = c_5
        a1 = c_6
        n1 = c_7
        k1 = c_8
        b2 = c_9
        q2 = c_10
        S2 = c_11
        p1 = c_12
        l1 = c_13
        a2 = c_14
        n2 = c_15
        k2 = c_16

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1 - c_0*c_2*x_0 + c_0*c_3**c_4*(c_5 + c_6*x_0**c_7/(c_8**c_7 + x_0**c_7))/(c_3**c_4 + x_1**c_4)
        dx_1 = -c_0*c_10*x_1 + c_0*c_11*c_12**c_13/(c_12**c_13 + x_0**c_13) + c_0*c_14*x_1**c_15/(c_16**c_15 + x_1**c_15) + c_0*c_9

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
