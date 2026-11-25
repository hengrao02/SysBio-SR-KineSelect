from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000001034'
MODEL_NAME = 'Mendrazitsky2007'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['BCG', 'Effector_cells', 'Tumor_infected_cells', 'Tumor_uninfected_cells']
CONST_NAMES = ['mu1', 'mu2', 'p1', 'p2', 'p3', 'p4', 'p5', 'alpha', 'beta', 'r', 'b', 'Bladder']

DEFAULT_INIT = np.array([0.0, 0.05, 0.0, 0.55], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.41, 1.25, 0.285, 1.1, 0.085, 0.003, 0.52, 0.0155, 0.12, 1.9, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000001034', 'dataset_model_name': 'Mendrazitsky2007', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Mendrazitsky2007.xml', 'warnings': []}


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
        BCG = x_0
        Effector_cells = x_1
        Tumor_infected_cells = x_2
        Tumor_uninfected_cells = x_3

        # readable parameter aliases
        mu1 = c_0
        mu2 = c_1
        p1 = c_2
        p2 = c_3
        p3 = c_4
        p4 = c_5
        p5 = c_6
        alpha = c_7
        beta = c_8
        r = c_9
        b = c_10
        Bladder = c_11

        # explicit ODE right-hand sides
        dx_0 = (1*c_11*c_10 + (-1)*c_11*c_3*x_3*x_0 + (-1)*c_11*(x_0*c_2*x_1 + x_0))/c_11
        dx_1 = (1*c_11*(x_1*c_5*x_0 + c_7*x_2) + (-1)*c_11*(x_1*c_1/c_0 + x_1*c_6*x_2))/c_11
        dx_2 = (1*c_11*c_3*x_3*x_0 + (-1)*c_11*c_4*x_1*x_2)/c_11
        dx_3 = (1*c_11*x_3*c_9*(1 - c_8*x_3) + (-1)*c_11*c_3*x_3*x_0)/c_11

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
