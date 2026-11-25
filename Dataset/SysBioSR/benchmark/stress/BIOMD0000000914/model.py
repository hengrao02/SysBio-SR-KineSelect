from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000914'
MODEL_NAME = 'Guillen2013'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['VAC', 'TRAN', 'SVAC', 'Ts', 'REG']
CONST_NAMES = ['k1', 'k3', 'k4', 'REG_50', 'gamma', 'k2_pop2', 'compartment']

DEFAULT_INIT = np.array([1.0, 0.0, 0.0, 0.324, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.0907, 1.08, 0.039, 3.18, 5.24, 0.0907, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000914', 'dataset_model_name': 'Guillen2013', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Guillen2013.xml', 'warnings': []}


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
        VAC = x_0
        TRAN = x_1
        SVAC = x_2
        Ts = x_3
        REG = x_4

        # readable parameter aliases
        k1 = c_0
        k3 = c_1
        k4 = c_2
        REG_50 = c_3
        gamma = c_4
        k2_pop2 = c_5
        compartment = c_6

        # explicit ODE right-hand sides
        dx_0 = (-1)*c_6*c_0*x_0/c_6
        dx_1 = (1*c_6*c_0*x_0 + (-1)*c_6*c_0*x_1)/c_6
        dx_2 = (1*c_6*c_0*x_1 + (-1)*c_6*c_5*x_2)/c_6
        dx_3 = (1*c_6*c_4 + (-1)*c_6*c_1*c_3**c_4/(c_3**c_4 + x_4**c_4)*x_3*x_2)/c_6
        dx_4 = (1*c_6*c_2*x_3 + (-1)*c_6*c_2*x_4)/c_6

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
