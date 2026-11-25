from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1910240001'
MODEL_NAME = 'Petrov2018'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['y1', 'y2', 'y3', 'y4', 'y5']
CONST_NAMES = ['compartment',
 'k_2',
 'k_m_2b',
 'ModelValue_6_0',
 'ModelValue_7_0',
 'B_0',
 'k_plus_a',
 'k_1',
 'k_rec',
 'E_0',
 'k_4',
 'k_3']

DEFAULT_INIT = np.array([0.0, 0.0, 75.0, 0.0, 75.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.42, 0.04, 1800.0, 33.0, 600.0, 1800.0, 3000.0, 0.05, 150.0, 0.222, 1.19], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1910240001', 'dataset_model_name': 'Petrov2018', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Petrov2018.xml', 'warnings': []}


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
        x_0, x_1, x_2, x_3, x_4 = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11 = consts

        # readable state aliases
        y1 = x_0
        y2 = x_1
        y3 = x_2
        y4 = x_3
        y5 = x_4

        # readable parameter aliases
        compartment = c_0
        k_2 = c_1
        k_m_2b = c_2
        ModelValue_6_0 = c_3
        ModelValue_7_0 = c_4
        B_0 = c_5
        k_plus_a = c_6
        k_1 = c_7
        k_rec = c_8
        E_0 = c_9
        k_4 = c_10
        k_3 = c_11

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1*x_3 - c_0*c_2*x_0 + c_0*c_3*c_4*x_1 - c_0*c_5*c_6*x_0 - c_0*c_7*x_0*x_2
        dx_1 = -c_0*c_3*c_4*x_1 + c_0*c_5*c_6*x_0
        dx_2 = c_0*c_1*x_3 - c_0*c_7*x_0*x_2 + c_0*c_8*(1 - x_2/c_9)
        dx_3 = -c_0*c_1*x_3 + c_0*c_10*x_4 - c_0*c_11*x_3 + c_0*c_7*x_0*x_2
        dx_4 = -c_0*c_10*x_4 + c_0*c_11*x_3 - c_0*c_8*(1 - x_2/c_9)

        return np.array([dx_0, dx_1, dx_2, dx_3, dx_4], dtype=float)


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
