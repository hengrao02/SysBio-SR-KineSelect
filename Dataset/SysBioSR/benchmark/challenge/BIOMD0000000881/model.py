from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000881'
MODEL_NAME = 'Kogan2013'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['T_1', 'T_2', 'G', 'I']
CONST_NAMES = ['mu_T',
 'r_T',
 'b_1',
 'b_2',
 'mu_G',
 'q_G',
 'p_G',
 'a_G',
 'b_G',
 'mu_I',
 'q_I',
 'p_I',
 'a_I',
 'b_I',
 'c_1',
 'd_1',
 'c_2',
 'c_G',
 'd_G',
 'e_G',
 'f_G',
 'L',
 'compartment']

DEFAULT_INIT = np.array([500000.0, 1000000.0, 1.0, 1.0], dtype=float)
DEFAULT_CONSTS = np.array([0.001,
 1000.0,
 0.35,
 0.18,
 0.6,
 1e-07,
 0.016,
 0.59,
 0.13,
 0.36,
 1e-07,
 0.5,
 0.12,
 0.025,
 30.0,
 0.8,
 0.1,
 12.0,
 0.05,
 5.4,
 0.22,
 0.0,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000881', 'dataset_model_name': 'Kogan2013', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Kogan2013.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19, c_20, c_21, c_22 = consts

        # readable state aliases
        T_1 = x_0
        T_2 = x_1
        G = x_2
        I = x_3

        # readable parameter aliases
        mu_T = c_0
        r_T = c_1
        b_1 = c_2
        b_2 = c_3
        mu_G = c_4
        q_G = c_5
        p_G = c_6
        a_G = c_7
        b_G = c_8
        mu_I = c_9
        q_I = c_10
        p_I = c_11
        a_I = c_12
        b_I = c_13
        c_1_2 = c_14
        d_1 = c_15
        c_2_2 = c_16
        c_G = c_17
        d_G = c_18
        e_G = c_19
        f_G = c_20
        L = c_21
        compartment = c_22

        # explicit ODE right-hand sides
        dx_0 = (1*c_22*(1 + c_14*c_21/(c_21 + c_15))*c_1*c_2/(c_2 + x_3) + (-1)*c_22*c_0*x_0)/c_22
        dx_1 = (1*c_22*(c_16 + (1 - c_16)*c_15/(c_21 + c_15))*c_1*c_3/(c_3 + x_2) + (-1)*c_22*c_0*x_1)/c_22
        dx_2 = (1*c_22*c_6*(1 + c_17*c_21/(c_21 + c_18)) + 1*c_22*c_5*x_0*(c_7 + (1 - c_7)*c_8/(x_3 + c_8))*(1 + c_19*c_21/(c_21 + c_20)) + (-1)*c_22*c_4*x_2)/c_22
        dx_3 = (1*c_22*c_11*(c_12 + (1 - c_12)*c_13/(x_2 + c_13)) + 1*c_22*c_10*x_1 + (-1)*c_22*c_9*x_3)/c_22

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
