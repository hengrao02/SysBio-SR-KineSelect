from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL2001160001'
MODEL_NAME = 'dePillis2007'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['T', 'N', 'C', 'M']
CONST_NAMES = ['compartment',
 'K_T',
 'c1',
 'a',
 'b',
 'alpha_1',
 'g',
 'h',
 'f',
 'K_N',
 'p',
 'alpha_2',
 'beta',
 'K_C',
 'V_M',
 'gamma']

DEFAULT_INIT = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0,
 0.8,
 3.41e-10,
 0.00431,
 1.02e-14,
 12000.0,
 0.015,
 20.2,
 0.0412,
 0.6,
 2e-11,
 750000000.0,
 0.012,
 0.6,
 0.0,
 0.9], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL2001160001', 'dataset_model_name': 'dePillis2007', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'dePillis2007.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15 = consts

        # readable state aliases
        T = x_0
        N = x_1
        C = x_2
        M = x_3

        # readable parameter aliases
        compartment = c_0
        K_T = c_1
        c1 = c_2
        a = c_3
        b = c_4
        alpha_1 = c_5
        g = c_6
        h = c_7
        f = c_8
        K_N = c_9
        p = c_10
        alpha_2 = c_11
        beta = c_12
        K_C = c_13
        V_M = c_14
        gamma = c_15

        # explicit ODE right-hand sides
        dx_0 = c_0*c_3*x_0*(-c_4*x_0 + 1) - c_0*(c_1*x_0*x_3 + c_2*x_0*x_1)
        dx_1 = c_0*(c_5 + c_6*x_0*x_1/(c_7 + x_0)) - c_0*(c_10*x_0*x_1 + c_8*x_1 + c_9*x_1*x_3)
        dx_2 = c_0*c_11 - c_0*(c_12*x_2 + c_13*x_2*x_3)
        dx_3 = c_0*c_14 - c_0*c_15*x_3

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
