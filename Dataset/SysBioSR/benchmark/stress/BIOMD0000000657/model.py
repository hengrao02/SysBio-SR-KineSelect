from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000657'
MODEL_NAME = 'Araujo2016'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['Cdk1CycB', 'APC', 'CycB']
CONST_NAMES = ['Ks',
 'Adeg',
 'Bdeg',
 'Acdc',
 'Bcdc',
 'EC50cdc',
 'n_cdc',
 'Awee',
 'Bwee',
 'EC50wee',
 'n_wee',
 'EC50apc',
 'n_apc',
 'AK',
 'BK',
 'EC50K',
 'n_K',
 'AP',
 'BP',
 'EC50P',
 'n_P',
 'Cell']

DEFAULT_INIT = np.array([20.0, 0.0, 59.9999999999999], dtype=float)
DEFAULT_CONSTS = np.array([0.1,
 0.001,
 0.02,
 0.5,
 1.5,
 30.0,
 10.0,
 0.5,
 1.0,
 30.0,
 10.0,
 0.5,
 10.0,
 0.0,
 0.25,
 0.18,
 5.0,
 0.025,
 0.5,
 0.18,
 5.0,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000657', 'dataset_model_name': 'Araujo2016', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Araujo2016.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19, c_20, c_21 = consts

        # readable state aliases
        Cdk1CycB = x_0
        APC = x_1
        CycB = x_2

        # readable parameter aliases
        Ks = c_0
        Adeg = c_1
        Bdeg = c_2
        Acdc = c_3
        Bcdc = c_4
        EC50cdc = c_5
        n_cdc = c_6
        Awee = c_7
        Bwee = c_8
        EC50wee = c_9
        n_wee = c_10
        EC50apc = c_11
        n_apc = c_12
        AK = c_13
        BK = c_14
        EC50K = c_15
        n_K = c_16
        AP = c_17
        BP = c_18
        EC50P = c_19
        n_P = c_20
        Cell = c_21

        # explicit ODE right-hand sides
        dx_0 = 1*c_21*(c_0 - c_1*x_0 - c_2*x_1**c_12*x_0/(c_11**c_12 + x_1**c_12) + (c_3 + c_4*x_0**c_6/(c_5**c_6 + x_0**c_6))*(x_2 - x_0) - (c_7 + c_8*c_9**c_10/(c_9**c_10 + x_0**c_10))*x_0)/c_21
        dx_1 = 1*c_21*((c_13 + c_14*(x_0/200)**c_16/((x_0/200)**c_16 + c_15**c_16))*(1 - x_1) - (c_17 + c_18*c_19**c_20/(x_1**c_20 + c_19**c_20))*x_1)/c_21
        dx_2 = 1*c_21*(c_0 - c_1*x_2 - c_2*x_1**c_12*x_2/(c_11**c_12 + x_1**c_12))/c_21

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
