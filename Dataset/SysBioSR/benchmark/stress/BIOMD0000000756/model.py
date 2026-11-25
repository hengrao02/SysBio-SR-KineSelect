from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000756'
MODEL_NAME = 'Figueredo2013'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['T', 'E', 'I', 'S']
CONST_NAMES = ['a',
 'c',
 'aa',
 'g2',
 'u2',
 'p1',
 'g1',
 'p2',
 'g3',
 'u3',
 'alpha',
 'g4',
 'gamma',
 'u1',
 'p3',
 'p4',
 'q1',
 'q2',
 'theta',
 'k',
 'tumor_microenvironment']

DEFAULT_INIT = np.array([1.0, 1.0, 10.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.18,
 0.035,
 1.0,
 100000.0,
 10.0,
 0.1245,
 20000000.0,
 0.27,
 20000000.0,
 10.0,
 0.001,
 1000.0,
 10.0,
 0.03,
 5.0,
 2.84,
 10.0,
 0.1121,
 1000000.0,
 10000000000.0,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000756', 'dataset_model_name': 'Figueredo2013', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Figueredo2013.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19, c_20 = consts

        # readable state aliases
        T = x_0
        E = x_1
        I = x_2
        S = x_3

        # readable parameter aliases
        a = c_0
        c = c_1
        aa = c_2
        g2 = c_3
        u2 = c_4
        p1 = c_5
        g1 = c_6
        p2 = c_7
        g3 = c_8
        u3 = c_9
        alpha = c_10
        g4 = c_11
        gamma = c_12
        u1 = c_13
        p3 = c_14
        p4 = c_15
        q1 = c_16
        q2 = c_17
        theta = c_18
        k = c_19
        tumor_microenvironment = c_20

        # explicit ODE right-hand sides
        dx_0 = (1*c_20*c_0*x_0 + (-1)*c_20*c_0*x_0**2/c_19 + (-1)*c_20*c_2*x_1*x_0/(c_3 + x_0) + 1*c_20*c_7*x_3*x_0/(c_8 + x_3))/c_20
        dx_1 = ((-1)*c_20*c_13*x_1 + 1*c_20*c_1*x_0/(1 + c_12*x_3) + 1*c_20*c_5*x_1*x_2/(c_6 + x_2)*(c_5 - c_16*x_3/(c_17 + x_3)))/c_20
        dx_2 = ((-1)*c_20*c_4*x_2 + 1*c_20*c_14*x_1*x_0/((c_11 + x_0)*(1 + c_10*x_3)))/c_20
        dx_3 = (1*c_20*c_15*x_0**2/(c_18**2 + x_0**2) + (-1)*c_20*c_9*x_3)/c_20

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
