from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000001056'
MODEL_NAME = 'Chulian2021'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['C1', 'C2', 'C3']
CONST_NAMES = ['Bone_marrow', 'alpha1', 'rho1', 'ModelValue_39', 'alpha2', 'rho2', 'alpha3']

DEFAULT_INIT = np.array([1000000.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.008, 0.0289, 1e-10, 0.006, 0.0193, 0.02], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000001056', 'dataset_model_name': 'Chulian2021', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Chulian2021.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6 = consts

        # readable state aliases
        C1 = x_0
        C2 = x_1
        C3 = x_2

        # readable parameter aliases
        Bone_marrow = c_0
        alpha1 = c_1
        rho1 = c_2
        ModelValue_39 = c_3
        alpha2 = c_4
        rho2 = c_5
        alpha3 = c_6

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*x_0 + c_0*c_2*x_0/(c_3*(x_0 + x_1 + x_2) + 1)
        dx_1 = c_0*c_1*x_0 - c_0*c_4*x_1 + c_0*c_5*x_1/(c_3*(x_0 + x_1 + x_2) + 1)
        dx_2 = c_0*c_4*x_1 - c_0*c_6*x_2

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
