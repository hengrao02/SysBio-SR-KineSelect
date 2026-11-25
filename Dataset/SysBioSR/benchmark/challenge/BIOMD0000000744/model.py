from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000744'
MODEL_NAME = 'Hu2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['x', 'y', 'z', 'w', 'v']
CONST_NAMES = ['r1',
 'b1',
 'beta1',
 'm1',
 'delta1',
 'r2',
 'beta2',
 'k2',
 'b2',
 'u2',
 'beta3',
 'k3',
 'm3',
 'r3',
 'u3',
 'beta4',
 'k4',
 'u4',
 'r4',
 'm4',
 'beta5',
 'k5',
 'u5',
 'Pancreatic_tumor']

DEFAULT_INIT = np.array([1000000000.0, 5600000.0, 190000000.0, 50000.0, 9.4], dtype=float)
DEFAULT_CONSTS = np.array([0.0195,
 1.02e-11,
 3.482115e-12,
 100000000.0,
 0.96,
 0.00195,
 0.125,
 56000000000.0,
 1.7857e-09,
 0.015,
 124.5,
 20000000000.0,
 1000000.0,
 3500.0,
 0.02,
 5.85,
 1000000.0,
 0.034,
 12500.0,
 89000000000.0,
 7.3,
 1000000.0,
 0.034,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000744', 'dataset_model_name': 'Hu2019', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Hu2019.xml', 'warnings': []}


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
        x_5 = x_0
        y = x_1
        z = x_2
        w = x_3
        v = x_4

        # readable parameter aliases
        r1 = c_0
        b1 = c_1
        beta1 = c_2
        m1 = c_3
        delta1 = c_4
        r2 = c_5
        beta2 = c_6
        k2 = c_7
        b2 = c_8
        u2 = c_9
        beta3 = c_10
        k3 = c_11
        m3 = c_12
        r3 = c_13
        u3 = c_14
        beta4 = c_15
        k4 = c_16
        u4 = c_17
        r4 = c_18
        m4 = c_19
        beta5 = c_20
        k5 = c_21
        u5 = c_22
        Pancreatic_tumor = c_23

        # explicit ODE right-hand sides
        dx_0 = (1*c_23*c_0*x_0*(1 - x_0*c_1) + 1*c_23*c_2*x_1*x_0*(1 - x_0*c_1) + (-1)*c_23*c_4*x_0*x_2/(c_3 + x_3))/c_23
        dx_1 = (1*c_23*c_5*x_1*(1 - c_8*x_1) + 1*c_23*c_6*x_3/(c_7 + x_3)*x_1*(1 - c_8*x_1) + (-1)*c_23*c_9*x_1)/c_23
        dx_2 = (1*c_23*c_10*x_2*x_4/((c_11 + x_4)*(c_12 + x_3)) + 1*c_23*c_13 + (-1)*c_23*c_14*x_2)/c_23
        dx_3 = (1*c_23*c_15*x_0*x_2/(c_16 + x_0) + (-1)*c_23*c_17*x_3 + 1*c_23*c_18*x_0*x_1/(c_19 + x_4))/c_23
        dx_4 = (1*c_23*c_20*x_0*x_2/(c_21 + x_0) + (-1)*c_23*c_22*x_4)/c_23

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
