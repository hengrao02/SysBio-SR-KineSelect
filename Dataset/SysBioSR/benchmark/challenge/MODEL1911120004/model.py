from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1911120004'
MODEL_NAME = 'Gupta2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['Ci', 'Ce', 'K', 'P']
CONST_NAMES = ['compartment',
 'v1',
 'k6',
 'k5',
 'v2',
 'n',
 'k2',
 'v3',
 'm',
 'p',
 'gamma2',
 'k4',
 'k3',
 'a2',
 'a0',
 'cm0',
 'h',
 'd0',
 'k0',
 'a1',
 'd3',
 'alpha1',
 'gamma1',
 'alpha2',
 'd2',
 'beta',
 'd1']

DEFAULT_INIT = np.array([0.25, 0.25, 0.375, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0,
 0.04,
 0.06,
 0.1567,
 0.833,
 2.0,
 1.0,
 10.853,
 2.0,
 4.0,
 2.0,
 2.0,
 0.9,
 0.002,
 0.03,
 1.5,
 1.0,
 0.5,
 0.8,
 0.9384,
 0.39,
 0.4,
 0.1,
 0.49542,
 0.74,
 0.648,
 0.39], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1911120004', 'dataset_model_name': 'Gupta2019', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Gupta2019.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19, c_20, c_21, c_22, c_23, c_24, c_25, c_26 = consts

        # readable state aliases
        Ci = x_0
        Ce = x_1
        K = x_2
        P = x_3

        # readable parameter aliases
        compartment = c_0
        v1 = c_1
        k6 = c_2
        k5 = c_3
        v2 = c_4
        n = c_5
        k2 = c_6
        v3 = c_7
        m = c_8
        p = c_9
        gamma2 = c_10
        k4 = c_11
        k3 = c_12
        a2 = c_13
        a0 = c_14
        cm0 = c_15
        h = c_16
        d0 = c_17
        k0 = c_18
        a1 = c_19
        d3 = c_20
        alpha1 = c_21
        gamma1 = c_22
        alpha2 = c_23
        d2 = c_24
        beta = c_25
        d1 = c_26

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1 + c_0*c_2*x_1 - c_0*c_3*x_0 - c_0*c_4*x_0**c_5/(c_6**c_5 + x_0**c_5) + c_0*c_7*x_0**c_9*x_1**c_8/((c_10 + 1)*(c_11**c_8 + x_1**c_8)*(c_12**c_9 + x_0**c_9))
        dx_1 = -c_0*c_2*x_1 + c_0*c_4*x_0**c_5/(c_6**c_5 + x_0**c_5) - c_0*c_7*x_0**c_9*x_1**c_8/((c_10 + 1)*(c_11**c_8 + x_1**c_8)*(c_12**c_9 + x_0**c_9))
        dx_2 = -c_0*c_13*x_2 + c_0*c_14*c_15*x_0**c_16*(c_18 - c_19*x_3/(c_20 + x_3))/(c_17**c_16 + x_0**c_16)
        dx_3 = c_0*c_21*x_2 - c_0*c_22*x_3 + c_0*c_23*x_3/(c_24 + x_3) - c_0*c_25*x_3/(c_26 + x_3)

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
