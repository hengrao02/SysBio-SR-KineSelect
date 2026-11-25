from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000001024'
MODEL_NAME = 'Chaudhury2020'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['Expanding_CAR_T_cells', 'Tumor_cells']
CONST_NAMES = ['rho', 'k_E', 'g', 'd', 'Whole_organism_blood']

DEFAULT_INIT = np.array([10.0, 900.0], dtype=float)
DEFAULT_CONSTS = np.array([0.0018, 0.16, 0.15, 0.00015, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000001024', 'dataset_model_name': 'Chaudhury2020', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Chaudhury2020.xml', 'warnings': []}


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
        x_0, x_1 = x
        c_0, c_1, c_2, c_3, c_4 = consts

        # readable state aliases
        Expanding_CAR_T_cells = x_0
        Tumor_cells = x_1

        # readable parameter aliases
        rho = c_0
        k_E = c_1
        g = c_2
        d = c_3
        Whole_organism_blood = c_4

        # explicit ODE right-hand sides
        dx_0 = (1*c_4*c_0*x_0*x_1 + (-1)*c_4*c_1*x_0)/c_4
        dx_1 = (1*c_4*c_2*x_1 + (-1)*c_4*c_3*x_0*x_1)/c_4

        return np.array([dx_0, dx_1], dtype=float)


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
