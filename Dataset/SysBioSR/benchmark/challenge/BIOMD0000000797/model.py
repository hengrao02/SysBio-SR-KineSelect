from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000797'
MODEL_NAME = 'Hu2018'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['x_Tumor_Cells', 'y_CD4_T_Cells', 'z_Cytokine', 'w_Healthy_Tissue']
CONST_NAMES = ['r_1',
 'b_1',
 'c_1',
 'a_1',
 'delta_1',
 'beta_1',
 'alpha_1',
 'mu_1',
 'delta_2',
 'beta_2',
 'alpha_2',
 'mu_2',
 'r_2',
 'b_2',
 'delta_3',
 'I_2',
 'I_1',
 'compartment']

DEFAULT_INIT = np.array([67700.0, 1000000.0, 100000.0, 1000000000.0], dtype=float)
DEFAULT_CONSTS = np.array([0.514,
 1.02e-09,
 0.2,
 100000.0,
 1.1e-10,
 0.835,
 1000.0,
 0.1,
 1e-07,
 5.4,
 1000.0,
 34.0,
 0.2822,
 1e-09,
 5.8e-11,
 0.0,
 0.0,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000797', 'dataset_model_name': 'Hu2018', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Hu2018.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17 = consts

        # readable state aliases
        x_Tumor_Cells = x_0
        y_CD4_T_Cells = x_1
        z_Cytokine = x_2
        w_Healthy_Tissue = x_3

        # readable parameter aliases
        r_1 = c_0
        b_1 = c_1
        c_1_2 = c_2
        a_1 = c_3
        delta_1 = c_4
        beta_1 = c_5
        alpha_1 = c_6
        mu_1 = c_7
        delta_2 = c_8
        beta_2 = c_9
        alpha_2 = c_10
        mu_2 = c_11
        r_2 = c_12
        b_2 = c_13
        delta_3 = c_14
        I_2 = c_15
        I_1 = c_16
        compartment = c_17

        # explicit ODE right-hand sides
        dx_0 = (1*c_17*c_0*x_0*(1 - c_1*x_0) + (-1)*c_17*c_2*x_0*x_2/(c_3 + x_0) + (-1)*c_17*c_4*x_0*x_3)/c_17
        dx_1 = (1*c_17*c_5*x_0*x_2/(c_6 + x_0) + (-1)*c_17*c_7*x_1 + (-1)*c_17*c_8*x_0*x_1 + 1*c_17*c_16)/c_17
        dx_2 = (1*c_17*c_9*x_0*x_1/(c_10 + x_0) + (-1)*c_17*c_11*x_2 + 1*c_17*c_15)/c_17
        dx_3 = (1*c_17*c_12*x_3*(1 - c_13*x_3) + (-1)*c_17*c_14*x_0*x_3)/c_17

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
