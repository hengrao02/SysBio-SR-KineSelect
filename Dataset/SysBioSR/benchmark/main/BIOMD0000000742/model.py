from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000742'
MODEL_NAME = 'Victor_Garcia2018'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['T', 'E']
CONST_NAMES = ['a', 'b', 'k', 's', 'm', 'd', 'Tumor']

DEFAULT_INIT = np.array([100.0, 1000.0], dtype=float)
DEFAULT_CONSTS = np.array([0.514, 1.02e-09, 0.0001, 10.0, -1e-06, 0.02, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000742', 'dataset_model_name': 'Victor_Garcia2018', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Victor_Garcia2018.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6 = consts

        # readable state aliases
        T = x_0
        E = x_1

        # readable parameter aliases
        a = c_0
        b = c_1
        k = c_2
        s = c_3
        m = c_4
        d = c_5
        Tumor = c_6

        # explicit ODE right-hand sides
        dx_0 = (1*c_6*c_0*x_0 + (-1)*c_6*c_0*c_1*x_0*x_0 + (-1)*c_6*c_2*x_0*x_1)/c_6
        dx_1 = ((-1)*c_6*c_5*x_1 + 1*c_6*c_3 + 1*c_6*c_4*x_1*x_0)/c_6

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
