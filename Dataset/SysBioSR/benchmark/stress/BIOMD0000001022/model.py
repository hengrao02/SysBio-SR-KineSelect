from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000001022'
MODEL_NAME = 'Creemers2021'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['T', 'I', 'S', 'N']
CONST_NAMES = ['rho', 'xi', 'h', 'delta', 'alpha', 'p_s', 'm_s', 'Tumor', 'Lymph_node']

DEFAULT_INIT = np.array([1.0, 0.0, 0.0, 1000000.0], dtype=float)
DEFAULT_CONSTS = np.array([6.0, 0.035, 571.0, 0.019, 0.0025, 1.0, 1.0, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000001022', 'dataset_model_name': 'Creemers2021', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Creemers2021.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8 = consts

        # readable state aliases
        T = x_0
        I = x_1
        S = x_2
        N = x_3

        # readable parameter aliases
        rho = c_0
        xi = c_1
        h = c_2
        delta = c_3
        alpha = c_4
        p_s = c_5
        m_s = c_6
        Tumor = c_7
        Lymph_node = c_8

        # explicit ODE right-hand sides
        dx_0 = (1*c_7*c_0*x_0**(4/5) + (-1)*c_7*c_1*x_1*x_0/(1 + x_1/c_2 + x_0/c_2))/c_7
        dx_1 = (1*c_6*x_2 + (-1)*c_7*c_3*x_1)/c_7
        dx_2 = ((-1)*c_6*x_2 + 1*c_4*x_0/(10000000 + x_0)*x_3 + 1*c_8*c_5*x_2)/c_8
        dx_3 = (-1)*c_4*x_0/(10000000 + x_0)*x_3/c_8

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
