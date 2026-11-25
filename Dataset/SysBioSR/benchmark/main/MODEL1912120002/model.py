from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1912120002'
MODEL_NAME = 'Sandip2013'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['T', 'I', 'VI', 'VNI']
CONST_NAMES = ['compartment', 's', 'r', 'k', 'd1', 'alpha', 'c', 'n1', 'd2', 'd3', 'beta', 'nr']

DEFAULT_INIT = np.array([17000000.0, 1.0, 1.0, 1.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 1.0, 1.99, 36000000.0, 0.01, 2.25e-07, 0.0, 0.5, 1.0, 6.0, 2.9, 0.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1912120002', 'dataset_model_name': 'Sandip2013', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Sandip2013.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11 = consts

        # readable state aliases
        T = x_0
        I = x_1
        VI = x_2
        VNI = x_3

        # readable parameter aliases
        compartment = c_0
        s = c_1
        r = c_2
        k = c_3
        d1 = c_4
        alpha = c_5
        c = c_6
        n1 = c_7
        d2 = c_8
        d3 = c_9
        beta = c_10
        nr = c_11

        # explicit ODE right-hand sides
        dx_0 = c_0*(c_1 + c_2*x_0*(1 - (x_0 + 1)/c_3)) - c_0*(c_4*x_0 + c_5*x_0*x_2*(-c_6*c_7 + 1))
        dx_1 = c_0*c_5*x_0*x_2*(-c_6*c_7 + 1) - c_0*c_8*x_1
        dx_2 = c_0*c_10*x_1*(-c_11/2 - c_7/2 + 1) - c_0*c_9*x_2
        dx_3 = c_0*c_10*x_1*(c_11/2 + c_7/2) - c_0*c_9*x_3

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
