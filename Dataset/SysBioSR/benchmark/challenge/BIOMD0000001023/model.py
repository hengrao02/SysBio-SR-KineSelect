from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000001023'
MODEL_NAME = 'Alharbi2020'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['Normal_cells', 'Tumor_cells', 'Immune_cells']
CONST_NAMES = ['r',
 'beta_1',
 'eta',
 'gamma',
 'alpha_1',
 'alpha_2',
 'beta_2',
 'alpha_3',
 'sigma',
 'delta',
 'mu',
 'rho',
 'm',
 'mu_1',
 'rho_1',
 'm_1',
 'Whole_organism']

DEFAULT_INIT = np.array([1.0, 1.0, 1.22], dtype=float)
DEFAULT_CONSTS = np.array([0.4312,
 2.99e-06,
 0.1379,
 0.9314,
 0.4426,
 0.4,
 1.189,
 0.1469,
 0.7,
 0.57,
 0.813,
 0.271,
 0.813,
 0.3634,
 0.7829,
 0.862,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000001023', 'dataset_model_name': 'Alharbi2020', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Alharbi2020.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16 = consts

        # readable state aliases
        Normal_cells = x_0
        Tumor_cells = x_1
        Immune_cells = x_2

        # readable parameter aliases
        r = c_0
        beta_1 = c_1
        eta = c_2
        gamma = c_3
        alpha_1 = c_4
        alpha_2 = c_5
        beta_2 = c_6
        alpha_3 = c_7
        sigma = c_8
        delta = c_9
        mu = c_10
        rho = c_11
        m = c_12
        mu_1 = c_13
        rho_1 = c_14
        m_1 = c_15
        Whole_organism = c_16

        # explicit ODE right-hand sides
        dx_0 = (1*c_16*c_0*x_0*(1 - c_1*x_0) + (-1)*c_16*(c_2*x_0*x_2 + c_3*x_0*x_1))/c_16
        dx_1 = (1*c_16*(c_4*x_1*(1 - c_5*x_1) + c_6*x_0*x_1) + (-1)*c_16*c_7*x_1*x_2)/c_16
        dx_2 = (1*c_16*(c_8 + c_11*x_0*x_2/(c_12 + x_0) + c_14*x_1*x_2/(c_15 + x_1)) + (-1)*c_16*(c_9*x_2 + c_10*x_0*x_2 + c_13*x_1*x_2))/c_16

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
