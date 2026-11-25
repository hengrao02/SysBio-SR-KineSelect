from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000869'
MODEL_NAME = 'Simon2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['p100', 'p52', 'NIK', 'IkBd']
CONST_NAMES = ['compartment',
 '__lp_r2_k1',
 '__lp_r3_Km',
 '__lp_r3_kcat',
 '__lp_r4_k1',
 '__lp_r5_k1',
 '__lp_r6_k1',
 '__lp_r6_k2',
 '__lp_r7_k1',
 '__lp_r8_Km',
 '__lp_r8_kcat',
 'p100t']

DEFAULT_INIT = np.array([0.0, 0.0, 10.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.2, 10.0, 0.05, 0.00038, 0.00038, 1.6e-05, 0.00024, 0.00038, 10.0, 0.05, 2.5], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000869', 'dataset_model_name': 'Simon2019', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Simon2019.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11 = consts

        # readable state aliases
        p100 = x_0
        p52 = x_1
        NIK = x_2
        IkBd = x_3

        # readable parameter aliases
        compartment = c_0
        lp_r2_k1 = c_1
        lp_r3_Km = c_2
        lp_r3_kcat = c_3
        lp_r4_k1 = c_4
        lp_r5_k1 = c_5
        lp_r6_k1 = c_6
        lp_r6_k2 = c_7
        lp_r7_k1 = c_8
        lp_r8_Km = c_9
        lp_r8_kcat = c_10
        p100t = c_11

        # explicit ODE right-hand sides
        dx_0 = (1*c_0*c_1*c_11 + (-1)*c_0*x_2*c_3*x_0/(c_2 + x_0) + (-1)*c_0*c_5*x_0 + (-2)*c_0*(c_6*x_0**2 - c_7*x_3))/c_0
        dx_1 = (1*c_0*x_2*c_3*x_0/(c_2 + x_0) + (-1)*c_0*c_4*x_1)/c_0
        dx_2 = 0/c_0
        dx_3 = (1*c_0*(c_6*x_0**2 - c_7*x_3) + (-1)*c_0*c_8*x_3 + (-1)*c_0*x_2*c_10*x_3/(c_9 + x_3))/c_0

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
