from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000001011'
MODEL_NAME = 'Triana2020'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['CAR_T_cells', 'Leukaemic_B_cells', 'B_cells_healthy']
CONST_NAMES = ['tao_B', 'rho_L', 'tao_C', 'alpha', 'compartment']

DEFAULT_INIT = np.array([10000000.0, 50000000000.0, 25000000000.0], dtype=float)
DEFAULT_CONSTS = np.array([60.0, 0.03333333333, 14.0, 4.5e-11, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000001011', 'dataset_model_name': 'Triana2020', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Triana2020.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4 = consts

        # readable state aliases
        CAR_T_cells = x_0
        Leukaemic_B_cells = x_1
        B_cells_healthy = x_2

        # readable parameter aliases
        tao_B = c_0
        rho_L = c_1
        tao_C = c_2
        alpha = c_3
        compartment = c_4

        # explicit ODE right-hand sides
        dx_0 = (1*c_4*1/4*c_3*(x_1 + x_2)*x_0 + (-1)*c_4*1/c_2*x_0)/c_4
        dx_1 = (1*c_4*c_1*x_1 + (-1)*c_4*c_3*x_1*x_0)/c_4
        dx_2 = (-1)*c_4*(c_3*x_2*x_0 + 1/c_0*x_2)/c_4

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
