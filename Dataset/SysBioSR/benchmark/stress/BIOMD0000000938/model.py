from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000938'
MODEL_NAME = 'Gerard2013'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['Cdk', 'Anaphase_promoting_complex_Phosphorylated', 'Securin']
CONST_NAMES = ['V_scdk',
 'nuclear',
 'k_d1cdk',
 'k_dcdk',
 'K_dcdk',
 'K_dsec',
 'V_1APC',
 'K_1APC',
 'k_2APC',
 'Anaphase_promoting_complex_total',
 'K_2APC',
 'V_ssec',
 'k_d1sec',
 'k_dsec']

DEFAULT_INIT = np.array([0.3, 0.0, 0.35], dtype=float)
DEFAULT_CONSTS = np.array([0.06, 1.0, 0.01, 0.35, 0.01, 0.001, 0.15, 0.01, 0.3, 1.0, 0.01, 0.1, 0.01, 0.4], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000938', 'dataset_model_name': 'Gerard2013', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Gerard2013.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13 = consts

        # readable state aliases
        Cdk = x_0
        Anaphase_promoting_complex_Phosphorylated = x_1
        Securin = x_2

        # readable parameter aliases
        V_scdk = c_0
        nuclear = c_1
        k_d1cdk = c_2
        k_dcdk = c_3
        K_dcdk = c_4
        K_dsec = c_5
        V_1APC = c_6
        K_1APC = c_7
        k_2APC = c_8
        Anaphase_promoting_complex_total = c_9
        K_2APC = c_10
        V_ssec = c_11
        k_d1sec = c_12
        k_dsec = c_13

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1 - c_1*c_2*x_0 - c_1*c_3*x_0*x_1/(c_4*(1 + x_2/c_5) + x_0)
        dx_1 = -c_1*c_6*x_1/(c_7 + x_1) + c_1*c_8*x_0*(c_9 - x_1)/(c_10 + c_9 - x_1)
        dx_2 = c_1*c_11 - c_1*c_12*x_2 - c_1*c_13*x_1*x_2/(c_5*(1 + x_0/c_4) + x_2)

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
