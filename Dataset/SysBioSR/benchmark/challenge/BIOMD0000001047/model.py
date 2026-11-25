from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000001047'
MODEL_NAME = 'Collier1996'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['delta1', 'delta2', 'notch1', 'notch2']
CONST_NAMES = ['cell_1___plasma_membrane',
 'rho',
 'ModelValue_1',
 'ModelValue_3',
 'cell_2___plasma_membrane',
 'ModelValue_4',
 'ModelValue_0',
 'mu']

DEFAULT_INIT = np.array([0.999999999999985, 0.99, 0.999999999999987, 0.99], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 1.0, 100.0, 2.0, 1.0, 2.0, 0.01, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000001047', 'dataset_model_name': 'Collier1996', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Collier1996.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7 = consts

        # readable state aliases
        delta1 = x_0
        delta2 = x_1
        notch1 = x_2
        notch2 = x_3

        # readable parameter aliases
        cell_1_plasma_membrane = c_0
        rho = c_1
        ModelValue_1 = c_2
        ModelValue_3 = c_3
        cell_2_plasma_membrane = c_4
        ModelValue_4 = c_5
        ModelValue_0 = c_6
        mu = c_7

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*x_0 + c_0/(c_1*(c_2*x_2**c_3 + 1))
        dx_1 = -c_1*c_4*x_1 + c_4/(c_1*(c_2*x_3**c_3 + 1))
        dx_2 = -c_0*c_7*x_2 + c_0*x_1**c_5/(c_6 + x_1**c_5)
        dx_3 = -c_4*c_7*x_3 + c_4*x_0**c_5/(c_6 + x_0**c_5)

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
