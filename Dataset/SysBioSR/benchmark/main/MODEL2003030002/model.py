from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL2003030002'
MODEL_NAME = 'Back2018'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['X1', 'X2', 'X3', 'X4', 'X5']
CONST_NAMES = ['Compartment', 'k_g', 'k_ma', 'E_bile', 'k_el', 'k_g_prime', 'k_out']

DEFAULT_INIT = np.array([250.0, 1e-05, 1e-05, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.067, 0.24, 0.0239, 0.27, 0.00971, 0.00972], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL2003030002', 'dataset_model_name': 'Back2018', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Back2018.xml', 'warnings': []}


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
        x_0, x_1, x_2, x_3, x_4 = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6 = consts

        # readable state aliases
        X1 = x_0
        X2 = x_1
        X3 = x_2
        X4 = x_3
        X5 = x_4

        # readable parameter aliases
        Compartment = c_0
        k_g = c_1
        k_ma = c_2
        E_bile = c_3
        k_el = c_4
        k_g_prime = c_5
        k_out = c_6

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*x_0
        dx_1 = c_0*c_1*x_0 - c_0*c_2*x_1*(c_3*x_4 + 1)
        dx_2 = c_0*c_2*x_1*(c_3*x_4 + 1) - c_0*c_4*x_2
        dx_3 = -c_0*c_5*x_3
        dx_4 = c_0*c_5*x_3 - c_0*c_6*x_4

        return np.array([dx_0, dx_1, dx_2, dx_3, dx_4], dtype=float)


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
