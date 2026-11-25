from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000413'
MODEL_NAME = 'Band2012'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['auxin', 'TIR1', 'auxinTIR1', 'auxinTIR1VENUS', 'VENUS']
CONST_NAMES = ['alpha_tr', 'kd', 'mu', 'ka', 'ld', 'lm', 'la', 'delta', 'lambda']

DEFAULT_INIT = np.array([7.38, 15.4, 0.28, 2.78, 40.4], dtype=float)
DEFAULT_CONSTS = np.array([30.5, 0.334, 0.79, 0.000822, 4.49, 0.175, 1.15, 0.486, 0.00316], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000413', 'dataset_model_name': 'Band2012', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Band2012.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8 = consts

        # readable state aliases
        auxin = x_0
        TIR1 = x_1
        auxinTIR1 = x_2
        auxinTIR1VENUS = x_3
        VENUS = x_4

        # readable parameter aliases
        alpha_tr = c_0
        kd = c_1
        mu = c_2
        ka = c_3
        ld = c_4
        lm = c_5
        la = c_6
        delta = c_7
        lambda_ = c_8

        # explicit ODE right-hand sides
        dx_0 = c_0 + c_1*x_2 - c_2*x_0 - c_3*x_0*x_1
        dx_1 = c_1*x_2 - c_3*x_0*x_1
        dx_2 = -c_1*x_2 + c_3*x_0*x_1 + c_4*x_3 + c_5*x_3 - c_6*x_2*x_4
        dx_3 = -c_4*x_3 - c_5*x_3 + c_6*x_2*x_4
        dx_4 = c_4*x_3 - c_6*x_2*x_4 + c_7 - c_8*x_4

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
