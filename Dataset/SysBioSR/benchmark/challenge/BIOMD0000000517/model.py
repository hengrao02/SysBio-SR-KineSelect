from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000517'
MODEL_NAME = 'Smallbone2013'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['N0', 'N1', 'N2', 'N3']
CONST_NAMES = ['d0',
 'd1',
 'd2',
 'K',
 'r0',
 'p01',
 'r1',
 'p12',
 'q03',
 'K03',
 'd3',
 'K0X',
 'K1X',
 'K2X',
 'compartment']

DEFAULT_INIT = np.array([1.71589842441293, 26.8036712557471, 44.6167755224769, 1.53646349229917], dtype=float)
DEFAULT_CONSTS = np.array([0.02,
 0.5480597115,
 1.888676618,
 106.090440772217,
 1.998624424,
 0.6313780928,
 10.58581849,
 0.8050459589,
 0.9351494412,
 0.783656669828271,
 0.1677359306,
 0.153646265911768,
 15.3645644864404,
 2.70405837954268,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000517', 'dataset_model_name': 'Smallbone2013', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Smallbone2013.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14 = consts

        # readable state aliases
        N0 = x_0
        N1 = x_1
        N2 = x_2
        N3 = x_3

        # readable parameter aliases
        d0 = c_0
        d1 = c_1
        d2 = c_2
        K = c_3
        r0 = c_4
        p01 = c_5
        r1 = c_6
        p12 = c_7
        q03 = c_8
        K03 = c_9
        d3 = c_10
        K0X = c_11
        K1X = c_12
        K2X = c_13
        compartment = c_14

        # explicit ODE right-hand sides
        dx_0 = ((-1)*c_0*x_0*c_11/(x_3 + c_11) + 1*(1 - c_5 - c_8*c_9/(x_3 + c_9))*c_4*x_0*(1 - (x_0 + x_1 + x_2 + x_3)/c_3))/c_14
        dx_1 = (1*c_5*c_4*x_0*(1 - (x_0 + x_1 + x_2 + x_3)/c_3) + (-1)*c_1*x_1*c_12/(x_3 + c_12) + 1*(1 - c_7)*c_6*x_1*(1 - (x_0 + x_1 + x_2 + x_3)/c_3))/c_14
        dx_2 = (1*c_7*c_6*x_1*(1 - (x_0 + x_1 + x_2 + x_3)/c_3) + (-1)*c_2*x_2*c_13/(x_3 + c_13))/c_14
        dx_3 = (1*c_8*c_9/(x_3 + c_9)*c_4*x_0*(1 - (x_0 + x_1 + x_2 + x_3)/c_3) + (-1)*c_10*x_3)/c_14

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
