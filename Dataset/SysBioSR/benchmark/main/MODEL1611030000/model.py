from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1611030000'
MODEL_NAME = 'Guisoni2016'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['r0', 'r1', 'r2', 'r3', 'ft']
CONST_NAMES = ['c1', 'q', 'p', 'eps', 'alfa', 'deg__k1']

DEFAULT_INIT = np.array([0.0, 0.0, 0.0, 1.0, 10.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.75, 0.001, 9.0, 45.0, 0.3], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1611030000', 'dataset_model_name': 'Guisoni2016', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Guisoni2016.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5 = consts

        # readable state aliases
        r0 = x_0
        r1 = x_1
        r2 = x_2
        r3 = x_3
        ft = x_4

        # readable parameter aliases
        c1 = c_0
        q = c_1
        p = c_2
        eps = c_3
        alfa = c_4
        deg_k1 = c_5

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1*x_1 - 3*c_0*c_2*x_0*x_4
        dx_1 = -c_0*c_1*x_1 + 2*c_0*c_1*x_2 - 2*c_0*c_2*c_3*x_1*x_4 + 3*c_0*c_2*x_0*x_4
        dx_2 = -2*c_0*c_1*x_2 + 3*c_0*c_1*x_3 - c_0*c_2*c_3**2*x_2*x_4 + 2*c_0*c_2*c_3*x_1*x_4
        dx_3 = -3*c_0*c_1*x_3 + c_0*c_2*c_3**2*x_2*x_4
        dx_4 = c_0*c_1*x_1 + 2*c_0*c_1*x_2 + 3*c_0*c_1*x_3 - c_0*c_2*c_3**2*x_2*x_4 - 2*c_0*c_2*c_3*x_1*x_4 - 3*c_0*c_2*x_0*x_4 + c_0*c_4*x_0 - c_0*c_5*x_4

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
