from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000903'
MODEL_NAME = 'perez2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['C', 'T', 'H', 'I', 'E']
CONST_NAMES = ['k1',
 'k2',
 'q',
 'M1',
 'M2',
 'M3',
 'gamma1',
 'gamma2',
 'gamma3',
 'p1',
 'p2',
 'p3',
 'a1',
 'a2',
 'a3',
 'n1',
 'n2',
 'delta',
 's',
 'p',
 'w',
 'u',
 'v',
 'tau',
 'd1',
 'd2',
 'd3',
 'mu',
 'compartment']

DEFAULT_INIT = np.array([737100.0, 7616700.0, 25000000.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.75,
 0.514,
 0.7,
 2270000.0,
 22700000.0,
 25000000.0,
 3e-07,
 3e-06,
 1e-07,
 600.0,
 0.0,
 100.0,
 1135000.0,
 11350000.0,
 1250000.0,
 0.01,
 0.29,
 6e-08,
 13000.0,
 0.2,
 300000.0,
 0.2,
 400.0,
 2700.0,
 0.01,
 0.01,
 0.01,
 0.97,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000903', 'dataset_model_name': 'perez2019', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'perez2019.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19, c_20, c_21, c_22, c_23, c_24, c_25, c_26, c_27, c_28 = consts

        # readable state aliases
        C = x_0
        T = x_1
        H = x_2
        I = x_3
        E = x_4

        # readable parameter aliases
        k1 = c_0
        k2 = c_1
        q = c_2
        M1 = c_3
        M2 = c_4
        M3 = c_5
        gamma1 = c_6
        gamma2 = c_7
        gamma3 = c_8
        p1 = c_9
        p2 = c_10
        p3 = c_11
        a1 = c_12
        a2 = c_13
        a3 = c_14
        n1 = c_15
        n2 = c_16
        delta = c_17
        s = c_18
        p = c_19
        w = c_20
        u = c_21
        v = c_22
        tau = c_23
        d1 = c_24
        d2 = c_25
        d3 = c_26
        mu = c_27
        compartment = c_28

        # explicit ODE right-hand sides
        dx_0 = (1*c_28*(c_0*x_0*(1 - x_0/c_3) + c_9*x_0*x_4/(c_12 + x_0)) + (-1)*c_28*c_6*x_3*x_0)/c_28
        dx_1 = (1*c_28*(c_1*x_0*x_0/c_3*(1 - x_1/c_4) + c_10*x_1*x_4/(c_13 + x_1)) + (-1)*c_28*(c_15*x_1 + c_7*x_3*x_1))/c_28
        dx_2 = (1*c_28*c_2*x_2*(1 - x_2/c_5) + (-1)*c_28*(c_17*x_2*x_1 + c_11*x_2*x_4/(c_14 + x_2)))/c_28
        dx_3 = (1*c_28*(c_18 + c_19*x_3*x_1/(c_20 + x_1)) + (-1)*c_28*(c_8*x_3*x_1 + c_16*x_3 + c_21*x_3*x_4/(c_22 + x_4)))/c_28
        dx_4 = (1*c_28*c_23 + (-1)*c_28*(c_27 + c_24*x_0/(c_12 + x_0) + c_25*x_1/(c_13 + x_1) + c_26*x_2/(c_14 + x_2))*x_4)/c_28

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
