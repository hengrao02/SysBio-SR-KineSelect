from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL2005070001'
MODEL_NAME = 'Hatzimanikatis1999'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['C', 'K', 'Rp']
CONST_NAMES = ['compartment',
 'VC_s',
 'E',
 'Vs_m',
 'Ks_E',
 'Kp',
 'V2_m',
 'gamma',
 'K2',
 'Vd_E_m',
 'Kd_C',
 'V1_m',
 'K1_C',
 'K1',
 'V4_m',
 'K4',
 'RE',
 'V3_m',
 'K3']

DEFAULT_INIT = np.array([1.0, 1.0, 1.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0,
 1.0,
 1.0,
 1.0,
 0.1,
 1.0,
 40.0,
 1.0,
 0.0001,
 1000.0,
 0.005,
 50.0,
 0.1,
 0.0001,
 3.0,
 0.0001,
 1.0,
 3000.0,
 0.0001], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL2005070001', 'dataset_model_name': 'Hatzimanikatis1999', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Hatzimanikatis1999.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18 = consts

        # readable state aliases
        C = x_0
        K = x_1
        Rp = x_2

        # readable parameter aliases
        compartment = c_0
        VC_s = c_1
        E = c_2
        Vs_m = c_3
        Ks_E = c_4
        Kp = c_5
        V2_m = c_6
        gamma = c_7
        K2 = c_8
        Vd_E_m = c_9
        Kd_C = c_10
        V1_m = c_11
        K1_C = c_12
        K1 = c_13
        V4_m = c_14
        K4 = c_15
        RE = c_16
        V3_m = c_17
        K3 = c_18

        # explicit ODE right-hand sides
        dx_0 = c_0*(c_1 + c_2*c_3/(c_2 + c_4) + c_5*c_6*c_7/(c_5 + c_8)) - c_0*(c_11*c_7*x_0*x_1/((c_12 + x_0)*(c_13 + x_1)) + c_2*c_9*x_0/(c_10 + x_0) + x_0)
        dx_1 = -c_0*c_11*x_0*x_1/((c_12 + x_0)*(c_13 + x_1)) + c_0*c_5*c_6/(c_5 + c_8)
        dx_2 = -c_0*c_14*x_2/(c_15 + x_2) + c_0*c_16*c_17*c_5/(c_16 + c_18)

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
