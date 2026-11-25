from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000868'
MODEL_NAME = 'Simon2019'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['p100', 'p52', 'NIK', 'p100_NIK']
CONST_NAMES = ['compartment',
 'p100_NIK_binding__k2',
 'p100_NIK_binding__k1',
 'p100_mRNA_Expressions__k1',
 'p100t',
 'p100_degradation__k1',
 'p100_processing__k1',
 'p52_degradation__k1']

DEFAULT_INIT = np.array([0.0, 0.0, 10.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.00024, 0.005, 0.2, 2.5, 0.00038, 0.05, 0.00038], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000868', 'dataset_model_name': 'Simon2019', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Simon2019.xml', 'warnings': []}


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
        p100 = x_0
        p52 = x_1
        NIK = x_2
        p100_NIK = x_3

        # readable parameter aliases
        compartment = c_0
        p100_NIK_binding_k2 = c_1
        p100_NIK_binding_k1 = c_2
        p100_mRNA_Expressions_k1 = c_3
        p100t = c_4
        p100_degradation_k1 = c_5
        p100_processing_k1 = c_6
        p52_degradation_k1 = c_7

        # explicit ODE right-hand sides
        dx_0 = c_0*c_3*c_4 - c_0*c_5*x_0 - c_0*(-c_1*x_3 + c_2*x_0*x_2)
        dx_1 = c_0*c_6*x_3 - c_0*c_7*x_1
        dx_2 = c_0*c_6*x_3 - c_0*(-c_1*x_3 + c_2*x_0*x_2)
        dx_3 = -c_0*c_6*x_3 + c_0*(-c_1*x_3 + c_2*x_0*x_2)

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
