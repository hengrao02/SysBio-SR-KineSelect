from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL0910896131'
MODEL_NAME = 'Guyton1972'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['VV7', 'VV6']
CONST_NAMES = ['SRK', 'tu', 'SR', 'VVE', 'SRK2', 'SR2']

DEFAULT_INIT = np.array([0.00366525, 0.0101913], dtype=float)
DEFAULT_CONSTS = np.array([5.0, 1.0, 1.0, 0.743224, 10000.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL0910896131', 'dataset_model_name': 'Guyton1972', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Guyton1972.xml', 'warnings': []}


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
        x_0, x_1 = x
        c_0, c_1, c_2, c_3, c_4, c_5 = consts

        # readable state aliases
        VV7 = x_0
        VV6 = x_1

        # readable parameter aliases
        SRK = c_0
        tu = c_1
        SR = c_2
        VVE = c_3
        SRK2 = c_4
        SR2 = c_5

        # explicit ODE right-hand sides
        dx_0 = (c_2*(c_3 - 0.74) - x_0)/(c_0*c_1)
        dx_1 = (c_5*(c_3 - 0.74) - x_1)/(c_1*c_4)

        return np.array([dx_0, dx_1], dtype=float)


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
