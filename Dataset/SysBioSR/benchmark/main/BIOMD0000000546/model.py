from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000546'
MODEL_NAME = 'Miao2010'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['s1', 's2', 's3']
CONST_NAMES = ['default', 'rho_E', 'beta_a', 'delta_Es', 'pi_a', 'c_V']

DEFAULT_INIT = np.array([580000.0, 0.0, 1000.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 6.2e-08, 1e-06, 0.6, 100.0, 4.2], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000546', 'dataset_model_name': 'Miao2010', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Miao2010.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5 = consts

        # readable state aliases
        s1 = x_0
        s2 = x_1
        s3 = x_2

        # readable parameter aliases
        default = c_0
        rho_E = c_1
        beta_a = c_2
        delta_Es = c_3
        pi_a = c_4
        c_V = c_5

        # explicit ODE right-hand sides
        dx_0 = -c_0**2*c_2*x_0*x_2 + c_0*c_1*x_0
        dx_1 = c_0**2*c_2*x_0*x_2 - c_0*c_3*x_1
        dx_2 = c_0*c_4*x_1 - c_0*c_5*x_2

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
