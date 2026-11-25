from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000060'
MODEL_NAME = 'Keizer1996'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['Pc1', 'Po2', 'Po1', 'Pc2']
CONST_NAMES = ['Closed_state_1__ka_minus',
 'Closed_state_1__ka_plus',
 'Closed_state_1__Ca',
 'Closed_state_1__n',
 'Open_state_2__kb_minus',
 'Open_state_2__kb_plus',
 'Open_state_2__Ca',
 'Open_state_2__m',
 'Closed_State_2__kc_minus',
 'Closed_State_2__kc_plus']

DEFAULT_INIT = np.array([0.963, 0.0, 0.0, 0.037], dtype=float)
DEFAULT_CONSTS = np.array([28.8, 1500.0, 0.9, 4.0, 385.9, 1500.0, 0.9, 3.0, 0.1, 1.75], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000060', 'dataset_model_name': 'Keizer1996', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Keizer1996.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9 = consts

        # readable state aliases
        Pc1 = x_0
        Po2 = x_1
        Po1 = x_2
        Pc2 = x_3

        # readable parameter aliases
        Closed_state_1_ka_minus = c_0
        Closed_state_1_ka_plus = c_1
        Closed_state_1_Ca = c_2
        Closed_state_1_n = c_3
        Open_state_2_kb_minus = c_4
        Open_state_2_kb_plus = c_5
        Open_state_2_Ca = c_6
        Open_state_2_m = c_7
        Closed_State_2_kc_minus = c_8
        Closed_State_2_kc_plus = c_9

        # explicit ODE right-hand sides
        dx_0 = c_0*x_2 - c_1*c_2**c_3*x_0
        dx_1 = -c_4*x_1 + c_5*c_6**c_7*x_2
        dx_2 = -c_0*x_2 + c_1*c_2**c_3*x_0 + c_4*x_1 - c_5*c_6**c_7*x_2 + c_8*x_3 - c_9*x_2
        dx_3 = -c_8*x_3 + c_9*x_2

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
