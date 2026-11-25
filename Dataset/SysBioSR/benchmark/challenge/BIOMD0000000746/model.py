from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000746'
MODEL_NAME = 'Saad2017'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['C', 'E', 'B', 'P']
CONST_NAMES = ['r',
 'alpha1',
 'k',
 'a1',
 'a2',
 'delta',
 'alpha2',
 'u1',
 'u2',
 'b',
 'alpha3',
 'u3',
 'bladder_cancer_tme']

DEFAULT_INIT = np.array([100000.0, 0.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.0033,
 1.1e-07,
 2000.0,
 0.25,
 0.052,
 151932.0,
 3.45e-10,
 0.041,
 0.1,
 650000.0,
 1.25e-07,
 166.32,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000746', 'dataset_model_name': 'Saad2017', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Saad2017.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12 = consts

        # readable state aliases
        C = x_0
        E = x_1
        B = x_2
        P = x_3

        # readable parameter aliases
        r = c_0
        alpha1 = c_1
        k = c_2
        a1 = c_3
        a2 = c_4
        delta = c_5
        alpha2 = c_6
        u1 = c_7
        u2 = c_8
        b = c_9
        alpha3 = c_10
        u3 = c_11
        bladder_cancer_tme = c_12

        # explicit ODE right-hand sides
        dx_0 = (1*c_12*c_0*x_0 + (-1)*c_12*c_1*x_1*x_0/(x_3 + c_2))/c_12
        dx_1 = (1*c_12*c_3*x_0*x_1/(x_3 + c_2) + 1*c_12*c_4*x_2*x_1/(x_3 + c_2) + (-1)*c_12*c_6*x_1*x_0 + (-1)*c_12*c_7*x_1)/c_12
        dx_2 = (1*c_12*c_9 + (-1)*c_12*c_8*x_2 + (-1)*c_12*c_10*x_1*x_2)/c_12
        dx_3 = (1*c_12*c_5 + (-1)*c_12*c_11*x_3)/c_12

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
