from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000790'
MODEL_NAME = 'Alvarez2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['T_1', 'T_2', 'E_1_Innate', 'E_2_Adaptive']
CONST_NAMES = ['a',
 'b',
 'mu',
 'beta',
 'nu',
 'c_1',
 'c_2',
 'c_3',
 'c_4',
 'c_5',
 'd_1',
 'd_2',
 'd_3',
 'p',
 'q',
 'r',
 's',
 'compartment']

DEFAULT_INIT = np.array([80000000.0, 20000000.0, 10500000.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.514,
 2e-09,
 1.101e-07,
 1.101e-10,
 1.101e-09,
 13000.0,
 0.0412,
 3.422e-10,
 0.1245,
 20193000.0,
 1.1e-07,
 3.42e-10,
 0.02,
 0.35,
 1.0,
 1.5,
 1.0,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000790', 'dataset_model_name': 'Alvarez2019', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Alvarez2019.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17 = consts

        # readable state aliases
        T_1 = x_0
        T_2 = x_1
        E_1_Innate = x_2
        E_2_Adaptive = x_3

        # readable parameter aliases
        a = c_0
        b = c_1
        mu = c_2
        beta = c_3
        nu = c_4
        c_1_2 = c_5
        c_2_2 = c_6
        c_3_2 = c_7
        c_4_2 = c_8
        c_5_2 = c_9
        d_1 = c_10
        d_2 = c_11
        d_3 = c_12
        p = c_13
        q = c_14
        r = c_15
        s = c_16
        compartment = c_17

        # explicit ODE right-hand sides
        dx_0 = (1*c_17*c_0*x_0*(1 - c_1*x_0) + (-1)*c_17*c_2*x_2*x_0 + (-1)*c_17*c_3*x_3*x_0 + (-1)*c_17*c_4*x_0*x_1)/c_17
        dx_1 = (1*c_17*c_0*c_13*x_1*(1 - c_1*x_1) + (-1)*c_17*c_15*c_4*x_0*x_1 + (-1)*c_17*c_2*c_14*x_2*x_1)/c_17
        dx_2 = (1*c_17*c_5 + (-1)*c_17*c_6*x_2 + (-1)*c_17*c_7*(x_0 + x_1)*x_2 + 1*c_17*c_8*(x_0 + c_16*x_1)*x_2/(c_9 + x_0 + x_1))/c_17
        dx_3 = (1*c_17*c_10*x_0*x_2 + (-1)*c_17*c_11*x_0*x_3 + (-1)*c_17*c_12*x_3)/c_17

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
