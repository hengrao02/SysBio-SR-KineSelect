from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000815'
MODEL_NAME = 'Chrobak2011'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['x_Cancer', 'y_Immune_System']
CONST_NAMES = ['compartment']

DEFAULT_INIT = np.array([0.0005, 0.01], dtype=float)
DEFAULT_CONSTS = np.array([1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000815', 'dataset_model_name': 'Chrobak2011', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Chrobak2011.xml', 'warnings': []}


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
        c_0, = consts

        # readable state aliases
        x_Cancer = x_0
        y_Immune_System = x_1

        # readable parameter aliases
        compartment = c_0

        # explicit ODE right-hand sides
        dx_0 = (1*c_0*1/16*x_0 + (-1)*c_0*1/8*x_0*x_1 + (-1)*c_0*1/32*x_0**2)/c_0
        dx_1 = (1*c_0*1/32*x_1 + (-1)*c_0*275/3200*x_0*x_1 + (-1)*c_0*1/32*x_1**2)/c_0

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
