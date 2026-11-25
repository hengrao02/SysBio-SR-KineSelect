from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000670'
MODEL_NAME = 'Owen1998'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['l', 'n', 'm']
CONST_NAMES = ['I', 'S', 'delta_l', 'K_l', 'A', 'N', 'K_m', 'xi']

DEFAULT_INIT = np.array([0.0999999999999985, 0.899999999999988, 0.899999999999988], dtype=float)
DEFAULT_CONSTS = np.array([0.01, 62.5, 0.1, 17.857, 0.025, 1.0, 25.0, 2.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000670', 'dataset_model_name': 'Owen1998', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Owen1998.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7 = consts

        # readable state aliases
        l = x_0
        n = x_1
        m = x_2

        # readable parameter aliases
        I = c_0
        S = c_1
        delta_l = c_2
        K_l = c_3
        A = c_4
        N = c_5
        K_m = c_6
        xi = c_7

        # explicit ODE right-hand sides
        dx_0 = c_0*(c_1*x_2 + 1) - c_2*x_0 - c_3*x_0*x_2**2 + c_4*x_0*x_2*(c_5 + 1)/(c_5 + x_0 + x_1 + x_2)
        dx_1 = x_1*(c_5 + 1)/(c_5 + x_0 + x_1 + x_2) - x_1
        dx_2 = -c_6*x_0*x_2**2 + c_7*x_2*(c_5 + 1)/(c_5 + x_0 + x_1 + x_2) - x_2

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
