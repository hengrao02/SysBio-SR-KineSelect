from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000216'
MODEL_NAME = 'Hong2009'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['M', 'TF', 'CP', 'CP2', 'IC']
CONST_NAMES = ['Dex',
 'kms',
 'n',
 'J',
 'kmd',
 'system',
 'chk2c',
 'kcp2d',
 'kicd',
 'kp2',
 'Jp',
 'kica',
 'kcps',
 'chk2',
 'kcpd',
 'kp1',
 'ka',
 'kd']

DEFAULT_INIT = np.array([1.4, 0.13, 0.037, 0.046, 0.37], dtype=float)
DEFAULT_CONSTS = np.array([0.0,
 1.0,
 2.0,
 0.3,
 0.1,
 1.0,
 0.0,
 0.0525,
 0.01,
 0.1,
 0.05,
 20.0,
 0.5,
 0.0,
 0.525,
 10.0,
 100.0,
 0.01], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000216', 'dataset_model_name': 'Hong2009', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Hong2009.xml', 'warnings': []}


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
        M = x_0
        TF = x_1
        CP = x_2
        CP2 = x_3
        IC = x_4

        # readable parameter aliases
        Dex = c_0
        kms = c_1
        n = c_2
        J = c_3
        kmd = c_4
        system = c_5
        chk2c = c_6
        kcp2d = c_7
        kicd = c_8
        kp2 = c_9
        Jp = c_10
        kica = c_11
        kcps = c_12
        chk2 = c_13
        kcpd = c_14
        kp1 = c_15
        ka = c_16
        kd = c_17

        # explicit ODE right-hand sides
        dx_0 = c_0 + c_1*x_1**c_2/(c_3**c_2 + x_1**c_2) - c_4*c_5*x_0
        dx_1 = -c_11*c_5*x_1*x_3 + c_5*c_6*x_4 + c_5*c_7*x_4 + c_5*c_8*x_4 + c_9*x_4/(c_10 + x_2 + 2*x_3 + 2*x_4)
        dx_2 = c_12*c_5*x_0 - c_13*c_5*x_2 - c_14*c_5*x_2 - c_15*x_2/(c_10 + x_2 + 2*x_3 + 2*x_4) - 2*c_16*c_5*x_2**2 + 2*c_17*c_5*x_3
        dx_3 = -c_11*c_5*x_1*x_3 - c_13*c_5*x_3 + c_16*c_5*x_2**2 - c_17*c_5*x_3 - c_5*c_7*x_3 + c_5*c_8*x_4 - c_9*x_3/(c_10 + x_2 + 2*x_3 + 2*x_4)
        dx_4 = c_11*c_5*x_1*x_3 - c_5*c_6*x_4 - c_5*c_7*x_4 - c_5*c_8*x_4 - c_9*x_4/(c_10 + x_2 + 2*x_3 + 2*x_4)

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
