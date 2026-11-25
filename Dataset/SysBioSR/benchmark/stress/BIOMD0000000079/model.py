from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000079'
MODEL_NAME = 'Goldbeter2006'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['P', 'Q', 'R']
CONST_NAMES = ['body',
 '__lp_r2_a',
 '__lp_r3_V',
 '__lp_r3_Km',
 '__lp_r4_V1',
 '__lp_r4_K1',
 '__lp_r5_V2',
 '__lp_r5_K2',
 '__lp_r6_V3',
 '__lp_r6_k3',
 '__lp_r7_V',
 '__lp_r7_Km']

DEFAULT_INIT = np.array([0.46962702075688995, 0.28709791375471216, 0.9350605992470157], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.1, 0.1, 0.2, 1.0, 0.01, 1.5, 0.01, 6.0, 0.01, 2.5, 0.01], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000079', 'dataset_model_name': 'Goldbeter2006', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Goldbeter2006.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11 = consts

        # readable state aliases
        P = x_0
        Q = x_1
        R = x_2

        # readable parameter aliases
        body = c_0
        lp_r2_a = c_1
        lp_r3_V = c_2
        lp_r3_Km = c_3
        lp_r4_V1 = c_4
        lp_r4_K1 = c_5
        lp_r5_V2 = c_6
        lp_r5_K2 = c_7
        lp_r6_V3 = c_8
        lp_r6_k3 = c_9
        lp_r7_V = c_10
        lp_r7_Km = c_11

        # explicit ODE right-hand sides
        dx_0 = (1*c_0*c_1*x_1 + (-1)*c_2*x_0/(c_3 + x_0))/c_0
        dx_1 = (1*c_4*(1 - x_1)/(c_5 + 1 - x_1) + (-1)*c_6*x_2*x_1/(c_7 + x_1))/c_0
        dx_2 = (1*x_0*c_8*(1 - x_2)/(c_9 + 1 - x_2) + (-1)*c_10*x_2/(c_11 + x_2))/c_0

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
