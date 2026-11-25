from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000001035'
MODEL_NAME = 'Tuwairqi2020'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['uninfected_cancer_cells', 'infected_cancer_cells', 'free_virus', 'immune_cells']
CONST_NAMES = ['r_1', 'a', 'h_1', 'd_1', 'c', 'h_2', 'e', 'm_1', 'm_2', 'n', 'b', 'Whole_organism']

DEFAULT_INIT = np.array([0.5, 0.0, 0.01, 0.01], dtype=float)
DEFAULT_CONSTS = np.array([0.36, 0.5, 0.36, 0.1278, 0.48, 0.16, 0.2, 0.6, 0.29, 0.16, 2.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000001035', 'dataset_model_name': 'Tuwairqi2020', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Tuwairqi2020.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11 = consts

        # readable state aliases
        uninfected_cancer_cells = x_0
        infected_cancer_cells = x_1
        free_virus = x_2
        immune_cells = x_3

        # readable parameter aliases
        r_1 = c_0
        a = c_1
        h_1 = c_2
        d_1 = c_3
        c = c_4
        h_2 = c_5
        e = c_6
        m_1 = c_7
        m_2 = c_8
        n = c_9
        b = c_10
        Whole_organism = c_11

        # explicit ODE right-hand sides
        dx_0 = (1*c_11*c_0*x_0*(1 - (x_0 + x_1)) + (-1)*c_11*c_1*x_0*x_2 + (-1)*c_11*(c_2*x_0*x_3 + c_3*x_0))/c_11
        dx_1 = (1*c_11*c_1*x_0*x_2 + (-1)*c_11*(c_4*x_1*x_3 + x_1))/c_11
        dx_2 = ((-1)*c_11*c_1*x_0*x_2 + 1*c_11*c_10*x_1 + (-1)*c_11*(c_5*x_2*x_3 + c_6*x_2))/c_11
        dx_3 = (1*c_11*(c_7*x_1*x_3 + c_8*x_0*x_3) + (-1)*c_11*c_9*x_3)/c_11

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
