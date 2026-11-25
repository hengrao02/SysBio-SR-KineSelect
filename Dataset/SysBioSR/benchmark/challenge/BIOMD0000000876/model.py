from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000876'
MODEL_NAME = 'Aavani2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['C_Uninfected_CD4', 'I_Infected_CD4', 'F_CTL', 'V_Virus']
CONST_NAMES = ['C_ast', 'N', 'beta', 'lambda', 'a', 'k', 'rho', 'b', 'e', 'compartment']

DEFAULT_INIT = np.array([1000.0, 1.0, 1.0, 200.0], dtype=float)
DEFAULT_CONSTS = np.array([1000.0, 2000.0, 5.75e-05, 0.01, 1.0, 23.0, 1.0, 0.1, 0.001, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000876', 'dataset_model_name': 'Aavani2019', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Aavani2019.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9 = consts

        # readable state aliases
        C_Uninfected_CD4 = x_0
        I_Infected_CD4 = x_1
        F_CTL = x_2
        V_Virus = x_3

        # readable parameter aliases
        C_ast = c_0
        N = c_1
        beta = c_2
        lambda_ = c_3
        a = c_4
        k = c_5
        rho = c_6
        b = c_7
        e = c_8
        compartment = c_9

        # explicit ODE right-hand sides
        dx_0 = (1*c_9*c_3*c_0 + (-1)*c_9*c_3*x_0 + (-1)*c_9*c_2*x_0*x_3)/c_9
        dx_1 = ((-1)*c_9*c_4*x_1 + 1*c_9*c_2*x_0*x_3 + (-1)*c_9*c_6*x_2*x_1)/c_9
        dx_2 = ((-1)*c_9*c_7*x_2 + 1*c_9*c_8*x_0*x_3*x_2/(c_0 + x_2))/c_9
        dx_3 = ((-1)*c_9*c_5*x_3 + 1*c_9*c_4*c_1*x_1)/c_9

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
