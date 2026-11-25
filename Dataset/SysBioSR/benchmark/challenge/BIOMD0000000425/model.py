from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000425'
MODEL_NAME = 'Tan2012'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['c']
CONST_NAMES = ['cell',
 '__lp_r2_alpha',
 '__lp_r3_kd',
 '__lp_r4_kappa',
 '__lp_r5_phi',
 '__lp_r5_delta',
 '__lp_r5_gamma']

DEFAULT_INIT = np.array([1.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.001, 1.0, 0.5, 5e-06, 1e-05, 1e-05], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000425', 'dataset_model_name': 'Tan2012', 'dim': 1, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Tan2012.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6 = consts

        # readable state aliases
        c = x_0

        # readable parameter aliases
        cell = c_0
        lp_r2_alpha = c_1
        lp_r3_kd = c_2
        lp_r4_kappa = c_3
        lp_r5_phi = c_4
        lp_r5_delta = c_5
        lp_r5_gamma = c_6

        # explicit ODE right-hand sides
        dx_0 = (1*c_1 + (-1)*c_2*x_0 + 1*x_0/(c_3 + x_0) + (-1)*c_4*x_0/(c_5 + c_6*x_0))/c_0

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
