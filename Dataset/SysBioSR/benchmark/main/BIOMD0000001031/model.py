from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000001031'
MODEL_NAME = 'Tuwairqi2020'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['uninfected_cancer_cells', 'infected_cancer_cells', 'free_virus']
CONST_NAMES = ['f', 'h', 'm', 'Whole_organism', '__lp_r4_k1']

DEFAULT_INIT = np.array([0.6, 0.0, 0.5], dtype=float)
DEFAULT_CONSTS = np.array([0.0, 0.15, 0.02, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000001031', 'dataset_model_name': 'Tuwairqi2020', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Tuwairqi2020.xml', 'warnings': []}


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
        uninfected_cancer_cells = x_0
        infected_cancer_cells = x_1
        free_virus = x_2

        # readable parameter aliases
        f = c_0
        h = c_1
        m = c_2
        Whole_organism = c_3
        lp_r4_k1 = c_4

        # explicit ODE right-hand sides
        dx_0 = (1*c_3*c_0*x_0 + (-1)*c_3*x_0*x_2)/c_3
        dx_1 = (1*c_3*x_0*x_2 + (-1)*c_3*c_4*x_1)/c_3
        dx_2 = (1*c_3*c_4*x_1 + (-1)*c_3*(c_2*x_0*x_2 + c_1*x_2))/c_3

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
