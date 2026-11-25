from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1909100002'
MODEL_NAME = 'Louzoun2014'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['C', 'P', 'R', 'T']
CONST_NAMES = ['compartment',
 'C_0',
 'k_c',
 'ModelValue_2',
 'ModelValue_1',
 'lambda_c',
 'K_c',
 'lambda_p',
 'P_0',
 'k_p',
 'ModelValue_6',
 'ModelValue_0',
 'k_r',
 'lambda_r',
 'gamma_c',
 'ModelValue_11',
 'ModelValue_9',
 'lambda_t',
 'k_t',
 'K_t']

DEFAULT_INIT = np.array([200.0, 99250.0, 0.0, 0.091], dtype=float)
DEFAULT_CONSTS = np.array([1.0,
 1000000.0,
 0.075,
 0.075,
 100000.0,
 1e-07,
 0.1,
 0.15,
 100000.0,
 0.2,
 0.2,
 1000000.0,
 0.2,
 0.22,
 0.02,
 0.22,
 0.15,
 0.3,
 3300.0,
 0.1], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1909100002', 'dataset_model_name': 'Louzoun2014', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Louzoun2014.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19 = consts

        # readable state aliases
        C = x_0
        P = x_1
        R = x_2
        T = x_3

        # readable parameter aliases
        compartment = c_0
        C_0 = c_1
        k_c = c_2
        ModelValue_2 = c_3
        ModelValue_1 = c_4
        lambda_c = c_5
        K_c = c_6
        lambda_p = c_7
        P_0 = c_8
        k_p = c_9
        ModelValue_6 = c_10
        ModelValue_0 = c_11
        k_r = c_12
        lambda_r = c_13
        gamma_c = c_14
        ModelValue_11 = c_15
        ModelValue_9 = c_16
        lambda_t = c_17
        k_t = c_18
        K_t = c_19

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_5*x_0*x_3/(c_6 - x_2 + 1) + c_0*x_0**0.75*(-(x_0/c_1)**0.25 + 1)*(c_2 + 20*c_3*x_1/c_4)
        dx_1 = -c_0*c_7*x_1 + c_0*x_1*(1 - x_1/c_8)*(20*c_10*x_0/(c_11/100 + x_0) + c_9)
        dx_2 = c_0*c_12 - c_0*x_2*(c_13 + c_14*x_0 + 0.02*c_15*x_1/(c_4*(1 - c_16/c_10)))
        dx_3 = -c_0*c_17*x_3 + c_0*c_18*x_2/(c_19 - x_2 + 1)

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
