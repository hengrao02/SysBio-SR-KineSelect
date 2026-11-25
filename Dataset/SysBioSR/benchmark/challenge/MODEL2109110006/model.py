from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL2109110006'
MODEL_NAME = 'Mullinax2016'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['T', 'E', 'N', 'R']
CONST_NAMES = ['compartment',
 'v_t',
 'lambda_t',
 'r_t',
 'rho_e',
 'gamma_e',
 'alpha_e',
 'beta_e',
 'mu_e',
 'lambda_e',
 'v_e',
 'rho_n',
 'gamma_n',
 'mu_n',
 'v_n',
 'rho_r',
 'gamma_r',
 'mu_r']

DEFAULT_INIT = np.array([1.0, 0.1, 0.1, 0.1], dtype=float)
DEFAULT_CONSTS = np.array([1.0,
 0.5,
 0.3,
 0.222576299182873,
 0.250680905469374,
 0.1,
 0.31,
 20.0,
 0.01,
 1.0,
 0.19,
 0.0570332388925758,
 0.02,
 0.1,
 0.5,
 0.0820351544329818,
 0.1,
 0.2], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL2109110006', 'dataset_model_name': 'Mullinax2016', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Mullinax2016.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17 = consts

        # readable state aliases
        T = x_0
        E = x_1
        N = x_2
        R = x_3

        # readable parameter aliases
        compartment = c_0
        v_t = c_1
        lambda_t = c_2
        r_t = c_3
        rho_e = c_4
        gamma_e = c_5
        alpha_e = c_6
        beta_e = c_7
        mu_e = c_8
        lambda_e = c_9
        v_e = c_10
        rho_n = c_11
        gamma_n = c_12
        mu_n = c_13
        v_n = c_14
        rho_r = c_15
        gamma_r = c_16
        mu_r = c_17

        # explicit ODE right-hand sides
        dx_0 = c_0*c_3*x_0 - c_0*(c_1*x_0*x_1 + c_2*x_0*x_2)
        dx_1 = c_0*(c_4 + c_5*x_1*x_2 + c_6*x_0*x_1/(c_7 + x_0)) - c_0*(c_10*x_0*x_1 + c_8*x_1 + c_9*x_1*x_3)
        dx_2 = c_0*(c_11 + c_12*x_0*x_2) - c_0*(c_13*x_2 + c_14*x_0*x_2)
        dx_3 = -c_0*c_17*x_3 + c_0*(c_15 + c_16*x_2*x_3)

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
