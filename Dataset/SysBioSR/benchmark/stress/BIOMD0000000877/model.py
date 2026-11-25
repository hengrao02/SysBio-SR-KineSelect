from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000877'
MODEL_NAME = 'Ontah2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['U', 'I', 'V', 'C']
CONST_NAMES = ['r',
 'K',
 'beta',
 'alpha',
 'delta_u',
 'K_c',
 'delta',
 'delta_i',
 'b',
 'gamma',
 'mu',
 'psi',
 'compartment']

DEFAULT_INIT = np.array([100.0, 10.0, 10.0, 30.0], dtype=float)
DEFAULT_CONSTS = np.array([0.1, 2139.0, 0.01, 0.5, 50.0, 10000.0, 0.5, 60.0, 0.5, 0.1, 150.0, 4.17, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000877', 'dataset_model_name': 'Ontah2019', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Ontah2019.xml', 'warnings': []}


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
        U = x_0
        I = x_1
        V = x_2
        C = x_3

        # readable parameter aliases
        r = c_0
        K = c_1
        beta = c_2
        alpha = c_3
        delta_u = c_4
        K_c = c_5
        delta = c_6
        delta_i = c_7
        b = c_8
        gamma = c_9
        mu = c_10
        psi = c_11
        compartment = c_12

        # explicit ODE right-hand sides
        dx_0 = (1*c_12*c_0*x_0*(1 - (x_0 + x_1)/c_1) + (-1)*c_12*c_2*x_0*x_2/(x_0 + x_1 + c_3) + (-1)*c_12*c_4*x_0*x_3/(c_5 + x_3))/c_12
        dx_1 = (1*c_12*c_2*x_0*x_2/(x_0 + x_1 + c_3) + (-1)*c_12*c_7*x_1*x_3/(c_5 + x_3) + (-1)*c_12*c_6*x_1)/c_12
        dx_2 = ((-1)*c_12*c_2*x_0*x_2/(x_0 + x_1 + c_3) + 1*c_12*c_8*c_6*x_1 + (-1)*c_12*c_9*x_2)/c_12
        dx_3 = (1*c_12*c_10 + (-1)*c_12*c_11*x_3)/c_12

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
