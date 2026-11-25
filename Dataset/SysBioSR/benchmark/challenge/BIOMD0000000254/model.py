from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000254'
MODEL_NAME = 'Bier2000'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['G1', 'T1', 'G2', 'T2']
CONST_NAMES = ['V_in', 'k1', 'kp', 'km', 'epsilon']

DEFAULT_INIT = np.array([9.818349251052702, 0.7929095682443323, 9.324355655246805, 2.3294785591006457], dtype=float)
DEFAULT_CONSTS = np.array([0.36, 0.02, 6.0, 13.0, 0.01], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000254', 'dataset_model_name': 'Bier2000', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Bier2000.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4 = consts

        # readable state aliases
        G1 = x_0
        T1 = x_1
        G2 = x_2
        T2 = x_3

        # readable parameter aliases
        V_in = c_0
        k1 = c_1
        kp = c_2
        km = c_3
        epsilon = c_4

        # explicit ODE right-hand sides
        dx_0 = c_0 - c_1*x_0*x_1
        dx_1 = 2*c_1*x_1*x_0 - c_2*x_1/(c_3 + x_1) + c_4*(x_3 - x_1)
        dx_2 = c_0 - c_1*x_2*x_3
        dx_3 = 2*c_1*x_2*x_3 - c_2*x_3/(c_3 + x_3) - c_4*(x_3 - x_1)

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
