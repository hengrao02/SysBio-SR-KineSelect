from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1503180005'
MODEL_NAME = 'Smallbone2015'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['X1', 'X2', 'X3']
CONST_NAMES = ['R1__V1', 'R1__K1', 'R0__KI', 'R0__V0', 'X0', 'R0__K0', 'R2__V2', 'R2__K2', 'R3__V3', 'R3__K3']

DEFAULT_INIT = np.array([0.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.1, 0.1, 0.1, 22.0, 1.0, 1.0, 1.1, 0.1, 1.1, 0.1], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1503180005', 'dataset_model_name': 'Smallbone2015', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Smallbone2015.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9 = consts

        # readable state aliases
        X1 = x_0
        X2 = x_1
        X3 = x_2

        # readable parameter aliases
        R1_V1 = c_0
        R1_K1 = c_1
        R0_KI = c_2
        R0_V0 = c_3
        X0 = c_4
        R0_K0 = c_5
        R2_V2 = c_6
        R2_K2 = c_7
        R3_V3 = c_8
        R3_K3 = c_9

        # explicit ODE right-hand sides
        dx_0 = -c_0*x_0/(c_1 + x_0) + c_2*c_3*c_4/((c_2 + x_2)*(c_4 + c_5))
        dx_1 = c_0*x_0/(c_1 + x_0) - c_6*x_1/(c_7 + x_1)
        dx_2 = c_6*x_1/(c_7 + x_1) - c_8*x_2/(c_9 + x_2)

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
