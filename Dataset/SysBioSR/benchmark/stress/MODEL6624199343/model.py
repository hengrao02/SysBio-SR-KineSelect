from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL6624199343'
MODEL_NAME = 'Martins2001'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['GSH', 'HTA', 'SLG']
CONST_NAMES = ['vneHTA__k2neHTA',
 'MG',
 'vneHTA__k1neHTA',
 'vGlxII__VGlxII',
 'vGlxII__KmGlxIISLG',
 'vGlxII__KmGlxIIGSH',
 'vGlxII__KiGlxIIHTA',
 'Lac',
 'vGlxII__KmGlxIILac',
 'vGlxII__KeqGlxII',
 'vGlxI__KmGlxIGSH',
 'vGlxI__KmGlxIHTA',
 'vGlxI__KmGlxISLG',
 'vGlxI__VfGlxI',
 'vGlxI__VrGlxI']

DEFAULT_INIT = np.array([8.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.01,
 0.01,
 0.34,
 0.00103,
 0.32,
 0.0228,
 0.3,
 0.0,
 10000.0,
 440000.0,
 5.0,
 0.53,
 1.9,
 0.0318,
 3.26e-05], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL6624199343', 'dataset_model_name': 'Martins2001', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Martins2001.xml', 'warnings': []}


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
        x_0, x_1, x_2 = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14 = consts

        # readable state aliases
        GSH = x_0
        HTA = x_1
        SLG = x_2

        # readable parameter aliases
        vneHTA_k2neHTA = c_0
        MG = c_1
        vneHTA_k1neHTA = c_2
        vGlxII_VGlxII = c_3
        vGlxII_KmGlxIISLG = c_4
        vGlxII_KmGlxIIGSH = c_5
        vGlxII_KiGlxIIHTA = c_6
        Lac = c_7
        vGlxII_KmGlxIILac = c_8
        vGlxII_KeqGlxII = c_9
        vGlxI_KmGlxIGSH = c_10
        vGlxI_KmGlxIHTA = c_11
        vGlxI_KmGlxISLG = c_12
        vGlxI_VfGlxI = c_13
        vGlxI_VrGlxI = c_14

        # explicit ODE right-hand sides
        dx_0 = c_0*x_1 - c_1*c_2*x_0 + c_3*(-c_7*x_0/c_9 + x_2)/(c_4*(c_7/c_8 + 1 + x_1/c_6 + x_0/c_5 + x_2/c_4))
        dx_1 = -c_0*x_1 + c_1*c_2*x_0 - (-c_14*x_2/c_12 + c_13*x_1/c_11)/(1 + x_2/c_12 + x_1/c_11 + x_0/c_10)
        dx_2 = -c_3*(-c_7*x_0/c_9 + x_2)/(c_4*(c_7/c_8 + 1 + x_1/c_6 + x_0/c_5 + x_2/c_4)) + (-c_14*x_2/c_12 + c_13*x_1/c_11)/(1 + x_2/c_12 + x_1/c_11 + x_0/c_10)

        return np.array([dx_0, dx_1, dx_2], dtype=float)


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
