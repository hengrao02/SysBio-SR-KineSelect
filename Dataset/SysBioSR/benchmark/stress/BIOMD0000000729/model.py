from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000729'
MODEL_NAME = 'Goldbeter1996'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['C', 'M', 'X']
CONST_NAMES = ['VM1', 'V2', 'VM3', 'V4', 'Kc', 'vi', 'vd', 'Kd', 'K1', 'K2', 'K3', 'K4', 'kd', 'compartment']

DEFAULT_INIT = np.array([0.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([3.0, 1.5, 1.0, 0.5, 0.5, 0.05, 0.25, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000729', 'dataset_model_name': 'Goldbeter1996', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Goldbeter1996.xml', 'warnings': []}


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
        C = x_0
        M = x_1
        X = x_2

        # readable parameter aliases
        VM1 = c_0
        V2 = c_1
        VM3 = c_2
        V4 = c_3
        Kc = c_4
        vi = c_5
        vd = c_6
        Kd = c_7
        K1 = c_8
        K2 = c_9
        K3 = c_10
        K4 = c_11
        kd = c_12
        compartment = c_13

        # explicit ODE right-hand sides
        dx_0 = (1*c_13*c_5 + (-1)*c_13*c_6*x_2*x_0/(c_7 + x_0) + (-1)*c_13*c_12*x_0)/c_13
        dx_1 = (1*c_13*x_0/(c_4 + x_0)*c_0*(1 - x_1)/(c_8 + 1 - x_1) + (-1)*c_13*c_1*x_1/(c_9 + x_1))/c_13
        dx_2 = (1*c_13*x_1*c_2*(1 - x_2)/(c_10 + 1 - x_2) + (-1)*c_13*c_3*x_2/(c_11 + x_2))/c_13

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
