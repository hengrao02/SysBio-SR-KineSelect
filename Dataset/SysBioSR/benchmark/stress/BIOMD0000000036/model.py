from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000036'
MODEL_NAME = 'Tyson1999'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['M', 'P']
CONST_NAMES = ['CYTOPLASM',
 'TC__Vm',
 'TC__Pcrit',
 'TC__Keq',
 'mRNAD__D',
 'TL__V',
 'ProteinD__D',
 'DBT__J',
 'DBT__k2',
 'DBT__k1',
 'DBT__Keq']

DEFAULT_INIT = np.array([1.0, 1.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 1.0, 0.1, 200.0, 0.1, 0.5, 0.1, 0.05, 0.03, 10.0, 200.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000036', 'dataset_model_name': 'Tyson1999', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Tyson1999.xml', 'warnings': []}


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
        x_0, x_1 = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10 = consts

        # readable state aliases
        M = x_0
        P = x_1

        # readable parameter aliases
        CYTOPLASM = c_0
        TC_Vm = c_1
        TC_Pcrit = c_2
        TC_Keq = c_3
        mRNAD_D = c_4
        TL_V = c_5
        ProteinD_D = c_6
        DBT_J = c_7
        DBT_k2 = c_8
        DBT_k1 = c_9
        DBT_Keq = c_10

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1/(1 + x_1**2*(1 - 2/((8*c_3*x_1 + 1)**0.5 + 1))**2/(4*c_2**2)) - c_0*c_4*x_0
        dx_1 = c_0*c_5*x_0 - c_0*c_6*x_1 - c_0*(c_8*x_1 + 2*c_9*x_1/((8*c_10*x_1 + 1)**0.5 + 1))/(c_7 + x_1)

        return np.array([dx_0, dx_1], dtype=float)


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
