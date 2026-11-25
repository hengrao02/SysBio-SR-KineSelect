from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000004'
MODEL_NAME = 'Goldbeter1991'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['C', 'M', 'X', 'MI', 'XI']
CONST_NAMES = ['VM1',
 'VM3',
 'Kc',
 'cell',
 '__lp_r2_vi',
 '__lp_r3_kd',
 '__lp_r4_vd',
 '__lp_r4_Kd',
 '__lp_r5_K1',
 '__lp_r6_V2',
 '__lp_r6_K2',
 '__lp_r7_K3',
 '__lp_r8_K4',
 '__lp_r8_V4']

DEFAULT_INIT = np.array([0.4535009323056436,
 0.06977168400902338,
 0.0008026743700016656,
 0.9302283159909792,
 0.9991973256299989], dtype=float)
DEFAULT_CONSTS = np.array([3.0, 1.0, 0.5, 1.0, 0.025, 0.01, 0.25, 0.02, 0.005, 1.5, 0.005, 0.005, 0.005, 0.5], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000004', 'dataset_model_name': 'Goldbeter1991', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Goldbeter1991.xml', 'warnings': []}


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
        C = x_0
        M = x_1
        X = x_2
        MI = x_3
        XI = x_4

        # readable parameter aliases
        VM1 = c_0
        VM3 = c_1
        Kc = c_2
        cell = c_3
        lp_r2_vi = c_4
        lp_r3_kd = c_5
        lp_r4_vd = c_6
        lp_r4_Kd = c_7
        lp_r5_K1 = c_8
        lp_r6_V2 = c_9
        lp_r6_K2 = c_10
        lp_r7_K3 = c_11
        lp_r8_K4 = c_12
        lp_r8_V4 = c_13

        # explicit ODE right-hand sides
        dx_0 = (1*c_3*c_4 + (-1)*x_0*c_3*c_5 + (-1)*x_0*c_3*c_6*x_2*(x_0 + c_7)**(-1))/c_3
        dx_1 = (1*c_3*x_3*x_0*c_0*(x_0 + c_2)**(-1)*(c_8 + x_3)**(-1) + (-1)*c_3*x_1*c_9*(c_10 + x_1)**(-1))/c_3
        dx_2 = (1*c_3*x_1*c_1*x_4*(c_11 + x_4)**(-1) + (-1)*c_3*c_13*x_2*(c_12 + x_2)**(-1))/c_3
        dx_3 = ((-1)*c_3*x_3*x_0*c_0*(x_0 + c_2)**(-1)*(c_8 + x_3)**(-1) + 1*c_3*x_1*c_9*(c_10 + x_1)**(-1))/c_3
        dx_4 = ((-1)*c_3*x_1*c_1*x_4*(c_11 + x_4)**(-1) + 1*c_3*c_13*x_2*(c_12 + x_2)**(-1))/c_3

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
