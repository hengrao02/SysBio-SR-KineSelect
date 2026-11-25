from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000921'
MODEL_NAME = 'Khajanchi2017'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['G', 'M', 'C_T', 'T_beta', 'I_gamma']
CONST_NAMES = ['r1',
 'G_max',
 'e1',
 'alpha1',
 'alpha2',
 'k1',
 'r2',
 'a1',
 'k4',
 'e2',
 'alpha3',
 'k2',
 'a2',
 'k5',
 'mu1',
 'alpha4',
 'k3',
 's1',
 'b1',
 'mu2',
 'b2',
 'mu3',
 'M_max',
 'compartment']

DEFAULT_INIT = np.array([100000.0, 100000.0, 1.0, 5000.0, 9000.0], dtype=float)
DEFAULT_CONSTS = np.array([0.01,
 882650.0,
 10000.0,
 1.5,
 0.12,
 27000.0,
 0.3307,
 0.1163,
 10500.0,
 10000.0,
 0.0194,
 27000.0,
 0.0,
 2000.0,
 0.007,
 0.1694,
 334450.0,
 63305.0,
 5.75e-06,
 6.93,
 0.000102,
 0.102,
 1.0,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000921', 'dataset_model_name': 'Khajanchi2017', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Khajanchi2017.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19, c_20, c_21, c_22, c_23 = consts

        # readable state aliases
        G = x_0
        M = x_1
        C_T = x_2
        T_beta = x_3
        I_gamma = x_4

        # readable parameter aliases
        r1 = c_0
        G_max = c_1
        e1 = c_2
        alpha1 = c_3
        alpha2 = c_4
        k1 = c_5
        r2 = c_6
        a1 = c_7
        k4 = c_8
        e2 = c_9
        alpha3 = c_10
        k2 = c_11
        a2 = c_12
        k5 = c_13
        mu1 = c_14
        alpha4 = c_15
        k3 = c_16
        s1 = c_17
        b1 = c_18
        mu2 = c_19
        b2 = c_20
        mu3 = c_21
        M_max = c_22
        compartment = c_23

        # explicit ODE right-hand sides
        dx_0 = (1*c_23*c_0*x_0*(1 - x_0/c_1) + (-1)*c_23*1/(x_3 + c_2)*(c_3*x_1 + c_4*x_2)*x_0/(x_0 + c_5))/c_23
        dx_1 = (1*c_23*c_6*x_1*(1 - x_1/c_22) + 1*c_23*c_7*x_4/(c_8 + x_4)*1/(x_3 + c_9) + (-1)*c_23*c_10*x_0/(x_0 + c_11)*x_1)/c_23
        dx_2 = (1*c_23*c_12*x_0/(c_13 + x_3) + (-1)*c_23*c_14*x_2 + (-1)*c_23*c_15*x_0/(x_0 + c_16)*x_2)/c_23
        dx_3 = (1*c_23*c_17 + 1*c_23*c_18*x_0 + (-1)*c_23*c_19*x_3)/c_23
        dx_4 = (1*c_23*c_20*x_2 + (-1)*c_23*c_21*x_4)/c_23

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
