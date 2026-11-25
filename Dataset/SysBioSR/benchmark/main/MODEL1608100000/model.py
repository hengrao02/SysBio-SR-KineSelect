from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1608100000'
MODEL_NAME = 'Aguilera2017'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['mRNA', 'Protein']
CONST_NAMES = ['R1__v', 'compartment', 'R2__k1', 'R3__k1', 'R4__k1']

DEFAULT_INIT = np.array([10.0, 100.0], dtype=float)
DEFAULT_CONSTS = np.array([5.0, 1.0, 0.03, 0.1, 0.03], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1608100000', 'dataset_model_name': 'Aguilera2017', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Aguilera2017.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4 = consts

        # readable state aliases
        mRNA = x_0
        Protein = x_1

        # readable parameter aliases
        R1_v = c_0
        compartment = c_1
        R2_k1 = c_2
        R3_k1 = c_3
        R4_k1 = c_4

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1 - c_1*c_2*x_0
        dx_1 = c_1*c_3*x_0 - c_1*c_4*x_1

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
