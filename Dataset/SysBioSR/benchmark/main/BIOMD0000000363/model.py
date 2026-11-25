from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000363'
MODEL_NAME = 'Lee2010'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['II', 'M', 'IIa', 'P2']
CONST_NAMES = ['compartment', 'r1__k1', 'r3__k1', 'r2__k1', 'r4__k1']

DEFAULT_INIT = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.005, 1e-05, 0.01, 2.5e-05], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000363', 'dataset_model_name': 'Lee2010', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Lee2010.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4 = consts

        # readable state aliases
        II = x_0
        M = x_1
        IIa = x_2
        P2 = x_3

        # readable parameter aliases
        compartment = c_0
        r1_k1 = c_1
        r3_k1 = c_2
        r2_k1 = c_3
        r4_k1 = c_4

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*x_0 - c_0*c_2*x_0
        dx_1 = c_0*c_1*x_0 - c_0*c_3*x_1
        dx_2 = c_0*c_3*x_1 + c_0*c_4*x_3
        dx_3 = c_0*c_2*x_0 - c_0*c_4*x_3

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
