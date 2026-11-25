from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000573'
MODEL_NAME = 'Aguilera2014'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['V', 'C']
CONST_NAMES = ['compartment',
 '__lp_r2_v',
 '__lp_r3_Km',
 '__lp_r3_Vmax',
 '__lp_r4_k1',
 '__lp_r5_k1',
 '__lp_r6_v',
 '__lp_r7_k1',
 '__lp_r8_k1']

DEFAULT_INIT = np.array([1.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.00134, 380.0, 0.134, 0.0295, 6.85e-05, 0.07, 0.927, 5.01e-05], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000573', 'dataset_model_name': 'Aguilera2014', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Aguilera2014.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8 = consts

        # readable state aliases
        V = x_0
        C = x_1

        # readable parameter aliases
        compartment = c_0
        lp_r2_v = c_1
        lp_r3_Km = c_2
        lp_r3_Vmax = c_3
        lp_r4_k1 = c_4
        lp_r5_k1 = c_5
        lp_r6_v = c_6
        lp_r7_k1 = c_7
        lp_r8_k1 = c_8

        # explicit ODE right-hand sides
        dx_0 = (1*c_0*c_1 + 1*c_0*c_3*x_0/(c_2 + x_0) + (-1)*c_0*c_4*x_0*x_1 + (-1)*c_0*c_5*x_0)/c_0
        dx_1 = (1*c_0*c_6 + (-1)*c_0*c_7*x_0*x_1 + (-1)*c_0*c_8*x_1)/c_0

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
