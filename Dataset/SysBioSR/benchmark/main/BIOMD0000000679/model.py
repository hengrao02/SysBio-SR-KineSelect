from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000679'
MODEL_NAME = 'Waugh2006'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['phi_I', 'phi_R', 'T']
CONST_NAMES = ['alpha', 'tau4', 'tau3', 'tau1', 'tau2', 'd1', 'COMpartment', 'k1', 'k2', 'k3', 'k4', 'd2']

DEFAULT_INIT = np.array([200.0, 200.0, 6.0], dtype=float)
DEFAULT_CONSTS = np.array([0.8, 1.75, 6.41, -2.47, 21.94, 0.2, 1.0, 0.05, 0.693, 0.002, 0.07, 9.1], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000679', 'dataset_model_name': 'Waugh2006', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Waugh2006.xml', 'warnings': []}


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
        phi_I = x_0
        phi_R = x_1
        T = x_2

        # readable parameter aliases
        alpha = c_0
        tau4 = c_1
        tau3 = c_2
        tau1 = c_3
        tau2 = c_4
        d1 = c_5
        COMpartment = c_6
        k1 = c_7
        k2 = c_8
        k3 = c_9
        k4 = c_10
        d2 = c_11

        # explicit ODE right-hand sides
        dx_0 = c_0*(c_1 + c_2*x_2 + c_3*x_2**3 + c_4*x_2**2) + c_7*c_8*x_0*(1 - c_9*(x_0 + x_1)) - c_5*x_0
        dx_1 = (1 - c_0)*(c_1 + c_2*x_2 + c_3*x_2**3 + c_4*x_2**2) + c_7*c_8*x_1*(1 - c_9*(x_0 + x_1)) - c_5*x_1
        dx_2 = c_10*x_0 - c_11*x_2

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
