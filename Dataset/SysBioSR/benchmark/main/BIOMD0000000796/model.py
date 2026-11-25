from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000796'
MODEL_NAME = 'Yang2012'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['C', 'E', 'T', 'P', 'A']
CONST_NAMES = ['a1',
 'a2',
 'a3',
 'e',
 'u1',
 'u2',
 'u3',
 'u4',
 'u5',
 'k1',
 'k2',
 'k3',
 'k4',
 'd',
 'y',
 'b1',
 'b2',
 'tme']

DEFAULT_INIT = np.array([9.0, 10.0, 1.023, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.1,
 0.1,
 0.2,
 0.01,
 0.01,
 0.05,
 0.05,
 0.01,
 0.01,
 10.0,
 20.0,
 5.0,
 1.0,
 0.1,
 0.01,
 0.01,
 0.01,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000796', 'dataset_model_name': 'Yang2012', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Yang2012.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17 = consts

        # readable state aliases
        C = x_0
        E = x_1
        T = x_2
        P = x_3
        A = x_4

        # readable parameter aliases
        a1 = c_0
        a2 = c_1
        a3 = c_2
        e = c_3
        u1 = c_4
        u2 = c_5
        u3 = c_6
        u4 = c_7
        u5 = c_8
        k1 = c_9
        k2 = c_10
        k3 = c_11
        k4 = c_12
        d = c_13
        y = c_14
        b1 = c_15
        b2 = c_16
        tme = c_17

        # explicit ODE right-hand sides
        dx_0 = (1*c_17*c_0*x_0*(1 - x_0/c_9) + (-1)*c_17*c_4*x_0 + (-1)*c_17*c_15*x_2*x_0)/c_17
        dx_1 = (1*c_17*c_1*x_1*(1 - x_1/c_10) + (-1)*c_17*c_5*x_1 + (-1)*c_17*c_14*x_2*x_1)/c_17
        dx_2 = (1*c_17*c_2*x_4*x_2*(1 - x_2/c_11) + (-1)*c_17*c_6*x_2 + (-1)*c_17*c_16*x_0*x_2)/c_17
        dx_3 = (1*c_17*c_14*x_2*x_1 + (-1)*c_17*c_7*x_3 + (-1)*c_17*c_13*x_3)/c_17
        dx_4 = (1*c_17*c_13*x_3 + 1*c_17*c_3*x_2*x_4*(1 - x_4/c_12) + (-1)*c_17*c_8*x_4)/c_17

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
