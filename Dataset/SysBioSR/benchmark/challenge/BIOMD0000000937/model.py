from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000937'
MODEL_NAME = 'Ferrel2011'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['CDK1_active', 'APC_active', 'Plk1_active']
CONST_NAMES = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'k1', 'k2', 'k3', 'n1', 'n2', 'n3', 'compartment']

DEFAULT_INIT = np.array([0.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.1, 3.0, 3.0, 3.0, 1.0, 1.0, 0.5, 0.5, 0.5, 8.0, 8.0, 8.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000937', 'dataset_model_name': 'Ferrel2011', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Ferrel2011.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12 = consts

        # readable state aliases
        CDK1_active = x_0
        APC_active = x_1
        Plk1_active = x_2

        # readable parameter aliases
        a1 = c_0
        a2 = c_1
        a3 = c_2
        b1 = c_3
        b2 = c_4
        b3 = c_5
        k1 = c_6
        k2 = c_7
        k3 = c_8
        n1 = c_9
        n2 = c_10
        n3 = c_11
        compartment = c_12

        # explicit ODE right-hand sides
        dx_0 = (1*c_12*c_0 + (-1)*c_12*c_3*x_0*x_1**c_9/(c_6**c_9 + x_1**c_9))/c_12
        dx_1 = (1*c_12*c_2*(1 - x_1)*x_2**c_11/(c_8**c_11 + x_2**c_11) + (-1)*c_12*c_5*x_1)/c_12
        dx_2 = (1*c_12*c_1*(1 - x_2)*x_0**c_10/(c_7**c_10 + x_0**c_10) + (-1)*c_12*c_4*x_2)/c_12

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
