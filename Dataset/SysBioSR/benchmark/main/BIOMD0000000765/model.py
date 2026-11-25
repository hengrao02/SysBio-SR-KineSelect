from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000765'
MODEL_NAME = 'Mager2005'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['C', 'A_T', 'R', 'RC']
CONST_NAMES = ['k_el',
 'V_c',
 'k_on',
 'k_int',
 'k_deg',
 'k_syn',
 'k_off',
 'k_tp',
 'k_pt',
 'Central',
 'Peripheral_Tissue']

DEFAULT_INIT = np.array([4000.0, 0.0, 53.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 10.0, 0.1, 0.0, 0.566, 0.0, 0.1, 0.0, 0.0, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000765', 'dataset_model_name': 'Mager2005', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Mager2005.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10 = consts

        # readable state aliases
        C = x_0
        A_T = x_1
        R = x_2
        RC = x_3

        # readable parameter aliases
        k_el = c_0
        V_c = c_1
        k_on = c_2
        k_int = c_3
        k_deg = c_4
        k_syn = c_5
        k_off = c_6
        k_tp = c_7
        k_pt = c_8
        Central = c_9
        Peripheral_Tissue = c_10

        # explicit ODE right-hand sides
        dx_0 = ((-1)*c_9*c_2*x_2*x_0 + 1*c_9*c_6*x_3 + (-1)*c_9*(c_0 + c_8)*x_0 + 1*c_7*x_1/c_1)/c_9
        dx_1 = (1*c_8*x_0*c_1 + (-1)*c_10*c_7*x_1)/c_10
        dx_2 = ((-1)*c_9*c_2*x_2*x_0 + 1*c_9*c_6*x_3 + 1*c_9*c_5 + (-1)*c_9*c_4*x_2)/c_9
        dx_3 = (1*c_9*c_2*x_2*x_0 + (-1)*c_9*(c_6 + c_3)*x_3)/c_9

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
