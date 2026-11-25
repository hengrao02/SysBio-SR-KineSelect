from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000854'
MODEL_NAME = 'Gray2016'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['Ac', 'Pc', 'Ap', 'Pp']
CONST_NAMES = ['compartment', 'kin', 'koff', 'alpha1', 'beta1']

DEFAULT_INIT = np.array([0.95, 0.0, 0.05, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.55, 0.35, 0.014, 2.2], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000854', 'dataset_model_name': 'Gray2016', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Gray2016.xml', 'warnings': []}


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
        Ac = x_0
        Pc = x_1
        Ap = x_2
        Pp = x_3

        # readable parameter aliases
        compartment = c_0
        kin = c_1
        koff = c_2
        alpha1 = c_3
        beta1 = c_4

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*c_3*x_0 + c_0*c_1*x_2 + c_0*c_2*x_1
        dx_1 = -c_0*c_1*c_3*x_1 + c_0*c_1*x_3 - c_0*c_2*x_1
        dx_2 = c_0*c_1*c_3*x_0 - c_0*c_1*x_2 - c_0*c_2*c_4*x_2 + c_0*c_2*x_3
        dx_3 = c_0*c_1*c_3*x_1 - c_0*c_1*x_3 + c_0*c_2*c_4*x_2 - c_0*c_2*x_3

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
