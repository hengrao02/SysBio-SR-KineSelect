from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000001038'
MODEL_NAME = 'Alharbi2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['Normal_cells', 'Tumor_cells', 'Vitamins']
CONST_NAMES = ['r', 'beta1', 'gamma', 'alpha1', 'alpha2', 'beta2', 'k1', 'k2', 'c1', 'c2', 'Whole_organism']

DEFAULT_INIT = np.array([1.0, 1.0, 5.0], dtype=float)
DEFAULT_CONSTS = np.array([0.4312, 2.99e-06, 0.9817, 0.4426, 0.4, 0.2291, 0.8976, 0.9611, 0.2215, 0.4975, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000001038', 'dataset_model_name': 'Alharbi2019', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Alharbi2019.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10 = consts

        # readable state aliases
        Normal_cells = x_0
        Tumor_cells = x_1
        Vitamins = x_2

        # readable parameter aliases
        r = c_0
        beta1 = c_1
        gamma = c_2
        alpha1 = c_3
        alpha2 = c_4
        beta2 = c_5
        k1 = c_6
        k2 = c_7
        c1 = c_8
        c2 = c_9
        Whole_organism = c_10

        # explicit ODE right-hand sides
        dx_0 = (1*c_10*(c_0*x_0*(1 - c_1*x_0) + c_8*x_0*x_2) + (-1)*c_10*c_2*x_0*x_1)/c_10
        dx_1 = (1*c_10*(c_3*x_1*(1 - c_4*x_1) + c_5*x_0*x_1) + (-1)*c_10*c_9*x_1*x_2)/c_10
        dx_2 = (1*c_10*c_6 + (-1)*c_10*c_7*x_2)/c_10

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
