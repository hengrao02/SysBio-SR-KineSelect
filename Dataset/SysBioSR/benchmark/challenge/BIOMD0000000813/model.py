from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000813'
MODEL_NAME = 'Anderson2015'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['x_Tumor_Cells', 'y_CD4_T_Cells', 'z_Cytokine']
CONST_NAMES = ['r', 'K', 'delta', 'm', 'beta', 'k', 'a', 'alpha', 'b', 'mu', 'I_1', 'I_2', 'compartment']

DEFAULT_INIT = np.array([0.5, 0.01, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.03, 1000.0, 0.1, 1.0, 0.02, 10.0, 0.02, 0.1, 0.1, 47.0, 10.0, 0.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000813', 'dataset_model_name': 'Anderson2015', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Anderson2015.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12 = consts

        # readable state aliases
        x_Tumor_Cells = x_0
        y_CD4_T_Cells = x_1
        z_Cytokine = x_2

        # readable parameter aliases
        r = c_0
        K = c_1
        delta = c_2
        m = c_3
        beta = c_4
        k = c_5
        a = c_6
        alpha = c_7
        b = c_8
        mu = c_9
        I_1 = c_10
        I_2 = c_11
        compartment = c_12

        # explicit ODE right-hand sides
        dx_0 = (1*c_12*c_0*x_0*(1 - x_0/c_1) + (-1)*c_12*c_2*x_0*x_2/(c_3 + x_0))/c_12
        dx_1 = (1*c_12*c_4*x_0*x_1/(c_5 + x_0) + (-1)*c_12*c_6*x_1 + 1*c_12*c_10)/c_12
        dx_2 = (1*c_12*c_7*x_0*x_1/(c_8 + x_0) + (-1)*c_12*c_9*x_2 + 1*c_12*c_11)/c_12

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
