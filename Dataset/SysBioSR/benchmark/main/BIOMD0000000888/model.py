from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000888'
MODEL_NAME = 'Unni2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['T', 'N', 'D', 'L']
CONST_NAMES = ['a',
 'b',
 'c_1',
 'c_2',
 'd_1',
 'd_2',
 'd_3',
 'e',
 'f_1',
 'f_2',
 'g',
 'h',
 'i',
 'j',
 'k',
 's_1',
 's_2',
 'u',
 'r_1',
 'g_1',
 'h_1',
 'compartment']

DEFAULT_INIT = np.array([100.0, 1.0, 1.0, 1.0], dtype=float)
DEFAULT_CONSTS = np.array([0.431,
 2.17e-08,
 3.5e-06,
 1e-07,
 1e-06,
 4e-06,
 0.0001,
 0.0412,
 1e-08,
 0.01,
 0.024,
 3.42e-10,
 0.02,
 1e-07,
 1e-07,
 13000.0,
 480.0,
 1.8e-08,
 0.0,
 0.0,
 0.0,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000888', 'dataset_model_name': 'Unni2019', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Unni2019.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19, c_20, c_21 = consts

        # readable state aliases
        T = x_0
        N = x_1
        D = x_2
        L = x_3

        # readable parameter aliases
        a = c_0
        b = c_1
        c_1_2 = c_2
        c_2_2 = c_3
        d_1 = c_4
        d_2 = c_5
        d_3 = c_6
        e = c_7
        f_1 = c_8
        f_2 = c_9
        g = c_10
        h = c_11
        i = c_12
        j = c_13
        k = c_14
        s_1 = c_15
        s_2 = c_16
        u = c_17
        r_1 = c_18
        g_1 = c_19
        h_1 = c_20
        compartment = c_21

        # explicit ODE right-hand sides
        dx_0 = (1*c_21*c_0*x_0*(1 - c_1*x_0) + (-1)*c_21*(c_2*x_1 + c_13*x_2 + c_14*x_3)*x_0)/c_21
        dx_1 = (1*c_21*(c_15 + c_19*x_1*x_0*x_0/(c_20 + x_0*x_0)) + (-1)*c_21*((c_3*x_0 + c_4*x_2)*x_1 + c_7*x_1))/c_21
        dx_2 = (1*c_21*c_16 + (-1)*c_21*((c_8*x_3 + c_5*x_1 - c_6*x_0)*x_2 - c_10*x_2))/c_21
        dx_3 = (1*c_21*(c_9*x_2*x_0 + c_18*x_1*x_0) + (-1)*c_21*(c_11*x_3*x_0 + c_17*x_1*x_3*x_3 + c_12*x_3))/c_21

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
