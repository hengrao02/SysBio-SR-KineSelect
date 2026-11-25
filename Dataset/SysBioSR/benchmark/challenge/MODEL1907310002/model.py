from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1907310002'
MODEL_NAME = 'Arciero2004'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['E', 'T', 'I', 'S']
CONST_NAMES = ['compartment',
 'mu_1',
 'c',
 'gamma',
 'g_1',
 'p_1',
 'q_1',
 'q_2',
 'r',
 'K',
 'p_2',
 'g_3',
 'a',
 'g_2',
 'mu_2',
 'p_3',
 'alpha',
 'g_4',
 'mu_3',
 'p_4',
 'tau_c']

DEFAULT_INIT = np.array([1000.0, 1.0, 1000.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0,
 0.03,
 5e-06,
 1.0,
 20000000.0,
 0.1245,
 0.1121,
 2000000.0,
 0.18,
 1000000000.0,
 0.27,
 20000000.0,
 1.0,
 100000.0,
 10.0,
 5.0,
 0.001,
 1000.0,
 10.0,
 5e-05,
 1000000.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1907310002', 'dataset_model_name': 'Arciero2004', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Arciero2004.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19, c_20 = consts

        # readable state aliases
        E = x_0
        T = x_1
        I = x_2
        S = x_3

        # readable parameter aliases
        compartment = c_0
        mu_1 = c_1
        c = c_2
        gamma = c_3
        g_1 = c_4
        p_1 = c_5
        q_1 = c_6
        q_2 = c_7
        r = c_8
        K = c_9
        p_2 = c_10
        g_3 = c_11
        a = c_12
        g_2 = c_13
        mu_2 = c_14
        p_3 = c_15
        alpha = c_16
        g_4 = c_17
        mu_3 = c_18
        p_4 = c_19
        tau_c = c_20

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*x_0 + c_0*c_2*x_1/(c_3*x_3 + 1) + c_0*x_0*x_2*(c_5 - c_6*x_3/(c_7 + x_3))/(c_4 + x_2)
        dx_1 = c_0*c_10*x_1*x_3/(c_11 + x_3) - c_0*c_12*x_0*x_1/(c_13 + x_1) + c_0*c_8*x_1*(1 - x_1/c_9)
        dx_2 = -c_0*c_14*x_2 + c_0*c_15*x_0*x_1/((c_17 + x_1)*(c_16*x_3 + 1))
        dx_3 = -c_0*c_18*x_3 + c_0*c_19*x_1**2/(c_20**2 + x_1**2)

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
