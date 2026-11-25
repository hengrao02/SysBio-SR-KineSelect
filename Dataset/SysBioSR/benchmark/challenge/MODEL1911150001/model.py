from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1911150001'
MODEL_NAME = 'Tsur2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['A_pc', 'T_il', 'M_el']
CONST_NAMES = ['compartment',
 'mu_APC',
 'alpha_im',
 'b',
 'a_pem',
 'alpha_eff',
 'b_pem',
 'mu_eff',
 'gamma_mel',
 'nu_mel',
 'g']

DEFAULT_INIT = np.array([1.0, 1.0, 50.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.231, 1161.0, 37040.0, 5005.0, 0.8318, 0.9, 0.1777, 0.003269, 0.1245, 20190000.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1911150001', 'dataset_model_name': 'Tsur2019', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Tsur2019.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10 = consts

        # readable state aliases
        A_pc = x_0
        T_il = x_1
        M_el = x_2

        # readable parameter aliases
        compartment = c_0
        mu_APC = c_1
        alpha_im = c_2
        b = c_3
        a_pem = c_4
        alpha_eff = c_5
        b_pem = c_6
        mu_eff = c_7
        gamma_mel = c_8
        nu_mel = c_9
        g = c_10

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*x_0 + c_0*c_2*x_2/(c_3 + x_2)
        dx_1 = c_0*c_4*c_5*x_0 - c_0*c_6*c_7*x_1
        dx_2 = c_0*c_8*x_2 - c_0*c_9*x_1*x_2/(c_10 + x_2)

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
