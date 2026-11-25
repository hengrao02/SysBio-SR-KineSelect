from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000008'
MODEL_NAME = 'Gardner1998'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['C', 'X', 'M', 'Y', 'Z']
CONST_NAMES = ['K6',
 'V1p',
 'V3p',
 'Cell',
 '__lp_r2_vi',
 '__lp_r3_k1',
 '__lp_r3_K5',
 '__lp_r4_kd',
 '__lp_r5_K1',
 '__lp_r6_V2',
 '__lp_r6_K2',
 '__lp_r7_K3',
 '__lp_r8_K4',
 '__lp_r8_V4',
 '__lp_r9_a1',
 '__lp_r10_a2',
 '__lp_r11_alpha',
 '__lp_r11_d1',
 '__lp_r12_kd',
 '__lp_r12_alpha',
 '__lp_r13_vs',
 '__lp_r14_d1']

DEFAULT_INIT = np.array([0.20917378933088832,
 0.15128968711425134,
 0.35691624964283825,
 3.7071924859166465,
 0.41039731629530285], dtype=float)
DEFAULT_CONSTS = np.array([0.3,
 0.75,
 0.3,
 1.0,
 0.1,
 0.5,
 0.02,
 0.02,
 0.1,
 0.25,
 0.1,
 0.2,
 0.1,
 0.1,
 0.05,
 0.05,
 0.1,
 0.05,
 0.02,
 0.1,
 0.2,
 0.05], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000008', 'dataset_model_name': 'Gardner1998', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Gardner1998.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19, c_20, c_21 = consts

        # readable state aliases
        C = x_0
        X = x_1
        M = x_2
        Y = x_3
        Z = x_4

        # readable parameter aliases
        K6 = c_0
        V1p = c_1
        V3p = c_2
        Cell = c_3
        lp_r2_vi = c_4
        lp_r3_k1 = c_5
        lp_r3_K5 = c_6
        lp_r4_kd = c_7
        lp_r5_K1 = c_8
        lp_r6_V2 = c_9
        lp_r6_K2 = c_10
        lp_r7_K3 = c_11
        lp_r8_K4 = c_12
        lp_r8_V4 = c_13
        lp_r9_a1 = c_14
        lp_r10_a2 = c_15
        lp_r11_alpha = c_16
        lp_r11_d1 = c_17
        lp_r12_kd = c_18
        lp_r12_alpha = c_19
        lp_r13_vs = c_20
        lp_r14_d1 = c_21

        # explicit ODE right-hand sides
        dx_0 = (1*c_4 + (-1)*x_0*c_5*x_1*(x_0 + c_6)**(-1) + (-1)*x_0*c_7 + (-1)*c_14*x_0*x_3 + 1*c_15*x_4 + 1*c_16*c_17*x_4)/c_3
        dx_1 = (1*x_2*c_2*(1 + (-1)*x_1)*(c_11 + (-1)*x_1 + 1)**(-1) + (-1)*c_13*x_1*(c_12 + x_1)**(-1))/c_3
        dx_2 = (1*(1 + (-1)*x_2)*x_0*c_1*(x_0 + c_0)**(-1)*(c_8 + (-1)*x_2 + 1)**(-1) + (-1)*x_2*c_9*(c_10 + x_2)**(-1))/c_3
        dx_3 = ((-1)*c_14*x_0*x_3 + 1*c_15*x_4 + 1*c_19*c_18*x_4 + 1*c_20 + (-1)*c_21*x_3)/c_3
        dx_4 = (1*c_14*x_0*x_3 + (-1)*c_15*x_4 + (-1)*c_16*c_17*x_4 + (-1)*c_19*c_18*x_4)/c_3

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
