from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000752'
MODEL_NAME = 'Wilkie2013r'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['C', 'I', 'R']
CONST_NAMES = ['a', 'K', 'a0', 'y', 'Ie', 'u', 'b', 'f', 'tumor_microenvironment']

DEFAULT_INIT = np.array([9000.0, 100.0, 1000.0], dtype=float)
DEFAULT_CONSTS = np.array([0.2, 10000000000.0, 0.0001, 0.2, 100.0, 0.001, 0.2, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000752', 'dataset_model_name': 'Wilkie2013r', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Wilkie2013r.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8 = consts

        # readable state aliases
        C = x_0
        I = x_1
        R = x_2

        # readable parameter aliases
        a = c_0
        K = c_1
        a0 = c_2
        y = c_3
        Ie = c_4
        u = c_5
        b = c_6
        f = c_7
        tumor_microenvironment = c_8

        # explicit ODE right-hand sides
        dx_0 = (1*c_8*c_0*x_0*(1 - (x_0 + x_2)/c_1) + (-1)*c_8*c_2*x_0*x_1)/c_8
        dx_1 = 1*c_8*c_3*x_1*(1 - x_1/(c_4 + c_5*(x_0 + 1/1000*x_2)*x_1))/c_8
        dx_2 = (1*c_8*c_6*x_2*(1 - (x_2 + x_0)/c_1) + (-1)*c_8*c_2*c_7*x_2*x_1)/c_8

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
