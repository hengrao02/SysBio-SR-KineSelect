from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL2005130001'
MODEL_NAME = 'Ma2005'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['ATM_D', 'ATM', 'ATM_star']
CONST_NAMES = ['K_dim', 'compartment', 'K_undim', 'K_ar', 'K_af', 'alpha3', 'C', 'alpha1', 'alpha2']

DEFAULT_INIT = np.array([1.0, 1.0, 1.0], dtype=float)
DEFAULT_CONSTS = np.array([10.0, 1.0, 1.0, 3.0, 1.0, 0.1, 1.0, 1.5, 10.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL2005130001', 'dataset_model_name': 'Ma2005', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Ma2005.xml', 'warnings': []}


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
        ATM_D = x_0
        ATM = x_1
        ATM_star = x_2

        # readable parameter aliases
        K_dim = c_0
        compartment = c_1
        K_undim = c_2
        K_ar = c_3
        K_af = c_4
        alpha3 = c_5
        C = c_6
        alpha1 = c_7
        alpha2 = c_8

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1*x_1**2/2 - c_1*c_2*x_0
        dx_1 = -c_1*(c_0*x_1**2 + c_4*x_1*(c_5*x_2 + c_6*c_7 + c_6*c_8*x_2)) + c_1*(2*c_2*x_0 + c_3*x_2)
        dx_2 = -c_1*c_3*x_2 + c_1*c_4*x_1*(c_5*x_2 + c_6*c_7 + c_6*c_8*x_2)

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
