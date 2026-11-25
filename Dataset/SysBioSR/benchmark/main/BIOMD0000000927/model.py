from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000927'
MODEL_NAME = 'Grigolon2018'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['auxin', 'IAA', 'ARF']
CONST_NAMES = ['compartment', 'Sauxin', 'Tauxin', 'SIAA', 'TIAA', 'alpha_no_reg', 'SARF', 'TARF', 'beta']

DEFAULT_INIT = np.array([0.2, 0.0333299969972976, 49.9900030001007], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.02, 10.0, 1.0, 333.0, 0.05, 25.0, 2.0, 0.003], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000927', 'dataset_model_name': 'Grigolon2018', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Grigolon2018.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8 = consts

        # readable state aliases
        auxin = x_0
        IAA = x_1
        ARF = x_2

        # readable parameter aliases
        compartment = c_0
        Sauxin = c_1
        Tauxin = c_2
        SIAA = c_3
        TIAA = c_4
        alpha_no_reg = c_5
        SARF = c_6
        TARF = c_7
        beta = c_8

        # explicit ODE right-hand sides
        dx_0 = c_0*(c_1 - x_0/c_2)
        dx_1 = -c_0*c_5*x_0*x_1 + c_0*(c_3 - x_1/c_4)
        dx_2 = -c_0*c_8*x_1*x_2 + c_0*(c_6 - x_2/c_7)

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
