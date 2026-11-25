from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000880'
MODEL_NAME = 'Trisilowati2018'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['T_Tumor_Cells', 'L_CD8_T_Cells', 'D_Dendritic_Cells', 'H_CD4_T_Cells']
CONST_NAMES = ['a_1',
 'a_2',
 'a_3',
 'b_1__1',
 'b_2__1',
 'b_3__1',
 'e',
 'alpha_1',
 'alpha_2',
 'beta_1',
 'beta_2',
 'beta_3',
 'u_1',
 'u_2',
 'compartment']

DEFAULT_INIT = np.array([1000000.0, 10.0, 10.0, 5.0], dtype=float)
DEFAULT_CONSTS = np.array([0.431,
 0.234,
 0.017,
 1.02e-09,
 1.25e-05,
 0.0005,
 1.04e-08,
 4.2e-08,
 8e-10,
 2e-05,
 2e-05,
 2e-05,
 0.0,
 0.0,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000880', 'dataset_model_name': 'Trisilowati2018', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Trisilowati2018.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14 = consts

        # readable state aliases
        T_Tumor_Cells = x_0
        L_CD8_T_Cells = x_1
        D_Dendritic_Cells = x_2
        H_CD4_T_Cells = x_3

        # readable parameter aliases
        a_1 = c_0
        a_2 = c_1
        a_3 = c_2
        b_1_1 = c_3
        b_2_1 = c_4
        b_3_1 = c_5
        e = c_6
        alpha_1 = c_7
        alpha_2 = c_8
        beta_1 = c_9
        beta_2 = c_10
        beta_3 = c_11
        u_1 = c_12
        u_2 = c_13
        compartment = c_14

        # explicit ODE right-hand sides
        dx_0 = (1*c_14*c_0*x_0*(1 - c_3*x_0) + (-1)*c_14*c_7*x_0*x_1 + (-1)*c_14*c_12*x_0)/c_14
        dx_1 = (1*c_14*c_9*(x_2 + x_3)*x_1 + (-1)*c_14*c_8*x_0*x_1 + (-1)*c_14*c_6*x_1)/c_14
        dx_2 = (1*c_14*c_1*x_2*(1 - c_4*x_2) + (-1)*c_14*c_10*x_2*x_1 + 1*c_14*c_13)/c_14
        dx_3 = (1*c_14*c_2*x_3*(1 - c_5*x_3) + (-1)*c_14*c_11*x_3*x_1)/c_14

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
