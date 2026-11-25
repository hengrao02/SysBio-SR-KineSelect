from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000812'
MODEL_NAME = 'Galante2012'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['X_Complex', 'P_Perforin', 'C_Cancer_Uncomplexed', 'T_ast']
CONST_NAMES = ['compartment', 'k_2', 'k_3', 'k_1', 'ModelValue_11', 'k_4', 'k_p', 'k_m_2', 'k_m_1', 'k', 'k_5']

DEFAULT_INIT = np.array([0.0, 0.315, 1.0, 1.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.0001, 0.0001, 0.0001, 1.0, 3.0, 0.097, 80.0, 2.2, 0.035, 0.003], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000812', 'dataset_model_name': 'Galante2012', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Galante2012.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10 = consts

        # readable state aliases
        X_Complex = x_0
        P_Perforin = x_1
        C_Cancer_Uncomplexed = x_2
        T_ast = x_3

        # readable parameter aliases
        compartment = c_0
        k_2 = c_1
        k_3 = c_2
        k_1 = c_3
        ModelValue_11 = c_4
        k_4 = c_5
        k_p = c_6
        k_m_2 = c_7
        k_m_1 = c_8
        k = c_9
        k_5 = c_10

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*x_0 - c_0*c_2*x_0 + c_0*c_3*x_2*(c_4 - x_0)
        dx_1 = -c_0*c_5*x_1*x_2 + c_0*c_6*(c_4 - x_0)/(c_7*x_2*(c_4 + c_8 - x_0))
        dx_2 = c_0*c_1*x_0 - c_0*c_10*x_2**2 - c_0*c_3*x_2*(c_4 - x_0) - c_0*c_5*x_1*x_2 + c_0*c_9*x_2
        dx_3 = -c_0*c_10*x_3**2 + c_0*c_9*x_3

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
