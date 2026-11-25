from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000292'
MODEL_NAME = 'Rovers1995'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['NADPH', 'ATP', 'X']
CONST_NAMES = ['c', 'k4', 'N0', 'k2', 'k1', 'k3', 'A0']

DEFAULT_INIT = np.array([0.7, 0.5, 2.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.614, 1.2, 0.625, 0.123, 0.3, 2.5], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000292', 'dataset_model_name': 'Rovers1995', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Rovers1995.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6 = consts

        # readable state aliases
        NADPH = x_0
        ATP = x_1
        X = x_2

        # readable parameter aliases
        c = c_0
        k4 = c_1
        N0 = c_2
        k2 = c_3
        k1 = c_4
        k3 = c_5
        A0 = c_6

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1*(c_2 - x_0) - c_0*c_3*x_1
        dx_1 = -c_0*c_3*x_1 - c_0*c_4*x_2 + 2*c_0*c_5*(c_2 - x_0)*(c_6 - x_1)
        dx_2 = c_0*c_3*x_1 - c_0*c_4*x_2

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
