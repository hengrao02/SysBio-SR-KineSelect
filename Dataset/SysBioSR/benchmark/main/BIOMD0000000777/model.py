from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000777'
MODEL_NAME = 'Chakrabarty2010'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['M_Tumor_Cells', 'N_CTL', 'Z_THL']
CONST_NAMES = ['r_1', 'k_1', 'alpha_1', 'd_1', 'alpha_2', 'r_2', 'k_2', 'mu_1', 'mu_2', 'compartment']

DEFAULT_INIT = np.array([2750000.0, 200000.0, 7200000.0], dtype=float)
DEFAULT_CONSTS = np.array([0.18, 5000000.0, 1.101e-07, 0.0412, 3.422e-10, 0.0245, 10000000.0, 0.05, 0.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000777', 'dataset_model_name': 'Chakrabarty2010', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Chakrabarty2010.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9 = consts

        # readable state aliases
        M_Tumor_Cells = x_0
        N_CTL = x_1
        Z_THL = x_2

        # readable parameter aliases
        r_1 = c_0
        k_1 = c_1
        alpha_1 = c_2
        d_1 = c_3
        alpha_2 = c_4
        r_2 = c_5
        k_2 = c_6
        mu_1 = c_7
        mu_2 = c_8
        compartment = c_9

        # explicit ODE right-hand sides
        dx_0 = (1*c_9*c_0*x_0*(1 - x_0/c_1) + (-1)*c_9*c_2*x_0*x_1 + (-1)*c_9*c_7*x_0)/c_9
        dx_1 = (1*c_9*c_8*x_1*x_2 + (-1)*c_9*c_3*x_1 + (-1)*c_9*c_4*x_0*x_1)/c_9
        dx_2 = (1*c_9*c_5*x_2*(1 - x_2/c_6) + (-1)*c_9*c_8*x_1*x_2)/c_9

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
