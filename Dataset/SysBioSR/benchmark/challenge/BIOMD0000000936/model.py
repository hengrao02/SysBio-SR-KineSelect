from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000936'
MODEL_NAME = 'ferrel2011'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['CDK1_active']
CONST_NAMES = ['a1', 'b1', 'k1', 'n1', 'compartment']

DEFAULT_INIT = np.array([0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.1, 1.0, 0.5, 8.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000936', 'dataset_model_name': 'ferrel2011', 'dim': 1, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'ferrel2011.xml', 'warnings': []}


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
        x_0, = x
        c_0, c_1, c_2, c_3, c_4 = consts

        # readable state aliases
        CDK1_active = x_0

        # readable parameter aliases
        a1 = c_0
        b1 = c_1
        k1 = c_2
        n1 = c_3
        compartment = c_4

        # explicit ODE right-hand sides
        dx_0 = (1*c_4*c_0 + (-1)*c_4*c_1*x_0**(c_3 + 1)/(c_2**c_3 + x_0**c_3))/c_4

        return np.array([dx_0], dtype=float)


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
