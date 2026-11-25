from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000290'
MODEL_NAME = 'Alexander2010'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['A', 'R', 'E', 'G']
CONST_NAMES = ['v_max',
 'k',
 'f',
 'gamma',
 'beta',
 'pi1',
 'lambdaE',
 'muA',
 'muR',
 'muE',
 'muG',
 'b1',
 'sigma1',
 'body']

DEFAULT_INIT = np.array([1.0, 0.0, 0.0, 100000000.0], dtype=float)
DEFAULT_CONSTS = np.array([125000.0,
 50000000.0,
 0.0001,
 2000.0,
 200.0,
 0.016,
 1000.0,
 0.25,
 0.25,
 0.25,
 5.0,
 0.25,
 3e-06,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000290', 'dataset_model_name': 'Alexander2010', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Alexander2010.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13 = consts

        # readable state aliases
        A = x_0
        R = x_1
        E = x_2
        G = x_3

        # readable parameter aliases
        v_max = c_0
        k = c_1
        f = c_2
        gamma = c_3
        beta = c_4
        pi1 = c_5
        lambdaE = c_6
        muA = c_7
        muR = c_8
        muE = c_9
        muG = c_10
        b1 = c_11
        sigma1 = c_12
        body = c_13

        # explicit ODE right-hand sides
        dx_0 = (1*c_2*c_0/(c_1 + x_3)*x_3 + (-1)*c_7*x_0 + (-1)*c_11*x_0 + (-1)*c_12*x_0*x_1)/c_13
        dx_1 = (1*c_4*x_0 + 1*c_5*x_2*x_0 + (-1)*c_8*x_1)/c_13
        dx_2 = (1*c_6*x_0 + (-1)*c_9*x_2)/c_13
        dx_3 = ((-1)*c_0/(c_1 + x_3)*x_3 + 1*c_3*x_2 + (-1)*c_10*x_3)/c_13

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
