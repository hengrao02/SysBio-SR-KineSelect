from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1908270001'
MODEL_NAME = 'Nikolopoulou2018'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['C', 'T', 'A']
CONST_NAMES = ['compartment',
 'lambda_C',
 'C_K',
 'eta',
 'd_T',
 'I_12',
 'T_N',
 'lambda_TI_12',
 'ModelValue_11',
 'sigma',
 'ModelValue_10',
 'ModelValue_12',
 'ModelValue_18',
 'ModelValue_13',
 'K_I_12',
 'I_2',
 'lambda_TI_2',
 'K_I_2',
 'd_A',
 'mu_PA']

DEFAULT_INIT = np.array([0.3968, 0.006, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0,
 0.425,
 0.8725,
 57.5,
 0.025,
 1.5e-10,
 0.0006,
 8.81,
 1.1615e-06,
 0.002,
 1.365e-18,
 5.84e-07,
 10.0,
 50.5,
 1.5e-10,
 2.37e-11,
 0.5,
 2.37e-11,
 0.7115,
 105954000.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1908270001', 'dataset_model_name': 'Nikolopoulou2018', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Nikolopoulou2018.xml', 'warnings': []}


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
        x_0, x_1, x_2 = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19 = consts

        # readable state aliases
        C = x_0
        T = x_1
        A = x_2

        # readable parameter aliases
        compartment = c_0
        lambda_C = c_1
        C_K = c_2
        eta = c_3
        d_T = c_4
        I_12 = c_5
        T_N = c_6
        lambda_TI_12 = c_7
        ModelValue_11 = c_8
        sigma = c_9
        ModelValue_10 = c_10
        ModelValue_12 = c_11
        ModelValue_18 = c_12
        ModelValue_13 = c_13
        K_I_12 = c_14
        I_2 = c_15
        lambda_TI_2 = c_16
        K_I_2 = c_17
        d_A = c_18
        mu_PA = c_19

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1*x_0*(1 - x_0/c_2) - c_0*c_3*x_0*x_1
        dx_1 = c_0*c_15*c_16*x_1/((1 + c_8*c_9*x_1*(c_11 - c_12*x_2)*(c_13*x_0 + x_1)/c_10)*(c_15 + c_17)) - c_0*c_4*x_1 + c_0*c_5*c_6*c_7/((1 + c_8*c_9*x_1*(c_11 - c_12*x_2)*(c_13*x_0 + x_1)/c_10)*(c_14 + c_5))
        dx_2 = -c_0*c_18*x_2 - c_0*c_19*x_1*x_2*(c_11 - c_12*x_2) + c_0*x_2

        return np.array([dx_0, dx_1, dx_2], dtype=float)


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
