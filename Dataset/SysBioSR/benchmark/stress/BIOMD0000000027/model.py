from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000027'
MODEL_NAME = 'Markevich2004'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['M', 'Mp', 'Mpp']
CONST_NAMES = ['MKP3', 'k4cat', 'uVol', 'Km4', 'Km5', 'Km3', 'MAPKK', 'k1cat', 'Km1', 'Km2', 'k3cat', 'k2cat']

DEFAULT_INIT = np.array([500.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([100.0, 0.06, 1.0, 18.0, 78.0, 22.0, 50.0, 0.01, 50.0, 500.0, 0.084, 15.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000027', 'dataset_model_name': 'Markevich2004', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Markevich2004.xml', 'warnings': []}


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
        M = x_0
        Mp = x_1
        Mpp = x_2

        # readable parameter aliases
        MKP3 = c_0
        k4cat = c_1
        uVol = c_2
        Km4 = c_3
        Km5 = c_4
        Km3 = c_5
        MAPKK = c_6
        k1cat = c_7
        Km1 = c_8
        Km2 = c_9
        k3cat = c_10
        k2cat = c_11

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1*c_2*x_1/(c_3*(1 + x_2/c_5 + x_0/c_4 + x_1/c_3)) - c_2*c_6*c_7*x_0/(c_8*(1 + x_1/c_9 + x_0/c_8))
        dx_1 = -c_0*c_1*c_2*x_1/(c_3*(1 + x_2/c_5 + x_0/c_4 + x_1/c_3)) + c_0*c_10*c_2*x_2/(c_5*(1 + x_2/c_5 + x_0/c_4 + x_1/c_3)) - c_11*c_2*c_6*x_1/(c_9*(1 + x_1/c_9 + x_0/c_8)) + c_2*c_6*c_7*x_0/(c_8*(1 + x_1/c_9 + x_0/c_8))
        dx_2 = -c_0*c_10*c_2*x_2/(c_5*(1 + x_2/c_5 + x_0/c_4 + x_1/c_3)) + c_11*c_2*c_6*x_1/(c_9*(1 + x_1/c_9 + x_0/c_8))

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
