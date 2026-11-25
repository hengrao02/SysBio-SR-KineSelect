from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000770'
MODEL_NAME = 'Eftimie2017'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['H1', 'H2', 'M1', 'M2']
CONST_NAMES = ['ah1',
 'ah2',
 'am1',
 'am2',
 'm1',
 'm2',
 'ph1',
 'ph2',
 'pm1',
 'pm2',
 'rm1',
 'rm2',
 'eh1',
 'eh2',
 'em1',
 'em2',
 'tme']

DEFAULT_INIT = np.array([0.0, 0.0, 100.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.008,
 0.001,
 0.001,
 0.001,
 100000000.0,
 1000000000.0,
 0.09,
 0.09,
 0.02,
 0.02,
 0.09,
 0.05,
 0.03,
 0.03,
 0.02,
 0.02,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000770', 'dataset_model_name': 'Eftimie2017', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Eftimie2017.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16 = consts

        # readable state aliases
        H1 = x_0
        H2 = x_1
        M1 = x_2
        M2 = x_3

        # readable parameter aliases
        ah1 = c_0
        ah2 = c_1
        am1 = c_2
        am2 = c_3
        m1 = c_4
        m2 = c_5
        ph1 = c_6
        ph2 = c_7
        pm1 = c_8
        pm2 = c_9
        rm1 = c_10
        rm2 = c_11
        eh1 = c_12
        eh2 = c_13
        em1 = c_14
        em2 = c_15
        tme = c_16

        # explicit ODE right-hand sides
        dx_0 = (1*c_16*c_0*x_2 + 1*c_16*c_6*x_0*x_2*(1 - (x_0 + x_1)/c_4) + (-1)*c_16*c_12*x_0)/c_16
        dx_1 = (1*c_16*c_1*x_3 + 1*c_16*c_7*x_1*x_3*(1 - (x_1 + x_0)/c_4) + (-1)*c_16*c_13*x_1)/c_16
        dx_2 = (1*c_16*c_2*x_0 + 1*c_16*c_8*x_2*(1 - (x_2 + x_3)/c_5) + (-1)*c_16*c_14*x_2 + (-1)*c_16*c_11*x_2 + 1*c_16*c_10*x_3)/c_16
        dx_3 = (1*c_16*c_11*x_2 + (-1)*c_16*c_10*x_3 + 1*c_16*c_3*x_1 + 1*c_16*c_9*x_3*x_1*(1 - (x_3 + x_2)/c_5) + (-1)*c_16*c_15*x_3)/c_16

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
