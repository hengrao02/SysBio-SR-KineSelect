from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000275'
MODEL_NAME = 'Goldbeter2007'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['RA', 'M_C', 'C', 'F']
CONST_NAMES = ['PSM',
 'kd5',
 'RALDH2_0',
 'ks1',
 'x',
 'L',
 'kd1',
 'V0',
 'Vsc',
 'n',
 'Ka',
 'kd3',
 'ks2',
 'kd2',
 'kd4',
 'M_0',
 'ks3',
 'Ki',
 'm']

DEFAULT_INIT = np.array([0.1, 0.1, 0.1, 1.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.0, 7.1, 1.0, 15.0, 50.0, 1.0, 0.365, 7.1, 2.0, 0.2, 1.0, 1.0, 0.28, 1.0, 5.0, 1.0, 0.2, 2.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000275', 'dataset_model_name': 'Goldbeter2007', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Goldbeter2007.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18 = consts

        # readable state aliases
        RA = x_0
        M_C = x_1
        C = x_2
        F = x_3

        # readable parameter aliases
        PSM = c_0
        kd5 = c_1
        RALDH2_0 = c_2
        ks1 = c_3
        x_4 = c_4
        L = c_5
        kd1 = c_6
        V0 = c_7
        Vsc = c_8
        n = c_9
        Ka = c_10
        kd3 = c_11
        ks2 = c_12
        kd2 = c_13
        kd4 = c_14
        M_0 = c_15
        ks3 = c_16
        Ki = c_17
        m = c_18

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*x_0 + c_0*c_2*c_3*(-c_4/c_5 + 1) - c_0*c_6*x_0*x_2
        dx_1 = -c_0*c_11*x_1 + c_0*(c_7 + c_8*x_3**c_9/(c_10**c_9 + x_3**c_9))
        dx_2 = c_0*c_12*x_1 - c_0*c_13*x_2
        dx_3 = -c_0*c_14*x_3 + c_0*c_15*c_16*c_17**c_18*c_4/(c_5*(c_17**c_18 + x_0**c_18))

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
