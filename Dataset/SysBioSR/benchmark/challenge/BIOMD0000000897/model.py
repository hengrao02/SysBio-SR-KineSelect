from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000897'
MODEL_NAME = 'Khajanchi2015'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['E', 'T']
CONST_NAMES = ['s', 'p', 'g', 'm', 'd', 'a', 'b', 'n', 'e1', 'e2', 'compartment']

DEFAULT_INIT = np.array([1708110.0, 8286380.0], dtype=float)
DEFAULT_CONSTS = np.array([13000.0, 0.1245, 20190000.0, 3.422e-10, 0.0412, 0.18, 2e-09, 1.101e-07, 1.0, 0.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000897', 'dataset_model_name': 'Khajanchi2015', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Khajanchi2015.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10 = consts

        # readable state aliases
        E = x_0
        T = x_1

        # readable parameter aliases
        s = c_0
        p = c_1
        g = c_2
        m = c_3
        d = c_4
        a = c_5
        b = c_6
        n = c_7
        e1 = c_8
        e2 = c_9
        compartment = c_10

        # explicit ODE right-hand sides
        dx_0 = (1*c_10*(c_0*c_8 + c_1*x_0*x_1/(c_2 + x_1)) + (-1)*c_10*(c_3*x_0*x_1 + c_4*x_0))/c_10
        dx_1 = (1*c_10*c_5*x_1*(1 - c_6*x_1) + (-1)*c_10*(c_7*x_0*x_1 + c_9*x_1))/c_10

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
