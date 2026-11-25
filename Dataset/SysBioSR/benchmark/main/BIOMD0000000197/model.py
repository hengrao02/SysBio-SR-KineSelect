from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000197'
MODEL_NAME = 'Bartholome2007'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['x1', 'x2', 'x3', 'x4', 'x5']
CONST_NAMES = ['p4', 'p7', 'p1', 'p12', 'basolat', 'apical', 'p3', 'p6', 'p8', 'p10', 'p2', 'p5', 'p9', 'p11']

DEFAULT_INIT = np.array([88.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.0827,
 0.0397,
 0.0025,
 0.0003,
 1.0,
 1.5,
 0.0013,
 6.4e-05,
 1000.0,
 1.6,
 0.0784,
 0.0091,
 0.0098,
 1000.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000197', 'dataset_model_name': 'Bartholome2007', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Bartholome2007.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13 = consts

        # readable state aliases
        x1 = x_0
        x2 = x_1
        x3 = x_2
        x4 = x_3
        x5 = x_4

        # readable parameter aliases
        p4 = c_0
        p7 = c_1
        p1 = c_2
        p12 = c_3
        basolat = c_4
        apical = c_5
        p3 = c_6
        p6 = c_7
        p8 = c_8
        p10 = c_9
        p2 = c_10
        p5 = c_11
        p9 = c_12
        p11 = c_13

        # explicit ODE right-hand sides
        dx_0 = c_0*x_2 + c_1*x_1 - c_2*x_0 - c_3*(-x_4/c_5 + x_0/c_4) - c_6*x_0 - c_7*x_0*(c_8 - x_1)
        dx_1 = -c_1*x_1 + c_7*x_0*(c_8 - x_1)
        dx_2 = -c_0*x_2 - c_10*x_2 - c_11*x_2 - c_12*x_2*(c_13 - x_3) + c_2*x_0 + c_6*x_0 + c_9*x_3
        dx_3 = c_12*x_2*(c_13 - x_3) - c_9*x_3
        dx_4 = c_10*x_2 + c_11*x_2 + c_3*(-x_4/c_5 + x_0/c_4)

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
