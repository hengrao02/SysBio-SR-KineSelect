from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL7743212613'
MODEL_NAME = 'Radulescu2008'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['s125', 's128', 's135', 's160', 's164']
CONST_NAMES = ['k26',
 'k27',
 'k19',
 'k20p',
 'k18',
 'k14p1',
 'k14p2',
 'k15p1',
 'k15p2',
 'k15p3',
 'k22',
 'k21p1',
 'k21p2',
 'k21p3']

DEFAULT_INIT = np.array([0.0, 0.0, 0.06, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([5e-07,
 0.0004,
 0.0,
 0.000625,
 0.0003,
 0.001,
 0.0001359,
 3.397e-09,
 0.0001359,
 0.015,
 0.5,
 2e-05,
 0.0001359,
 0.015], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL7743212613', 'dataset_model_name': 'Radulescu2008', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Radulescu2008.xml', 'warnings': []}


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
        x_0, x_1, x_2, x_3, x_4 = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13 = consts

        # readable state aliases
        s125 = x_0
        s128 = x_1
        s135 = x_2
        s160 = x_3
        s164 = x_4

        # readable parameter aliases
        k26 = c_0
        k27 = c_1
        k19 = c_2
        k20p = c_3
        k18 = c_4
        k14p1 = c_5
        k14p2 = c_6
        k15p1 = c_7
        k15p2 = c_8
        k15p3 = c_9
        k22 = c_10
        k21p1 = c_11
        k21p2 = c_12
        k21p3 = c_13

        # explicit ODE right-hand sides
        dx_0 = c_0*x_4 - c_1*x_0
        dx_1 = c_2 + c_3*x_4 - c_4*x_1
        dx_2 = c_5*x_3*x_4/(c_6 + x_4) - c_7*x_2/((c_8 + x_3)*(c_9 + x_1))
        dx_3 = c_10*x_0 - c_11*x_2*x_3/((c_12 + x_3)*(c_13 + x_1)) - c_5*x_3*x_4/(c_6 + x_4)
        dx_4 = -5*c_5*x_3*x_4/(c_6 + x_4) + 5*c_7*x_2/((c_8 + x_3)*(c_9 + x_1))

        return np.array([dx_0, dx_1, dx_2, dx_3, dx_4], dtype=float)


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
