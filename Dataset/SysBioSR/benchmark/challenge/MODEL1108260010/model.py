from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1108260010'
MODEL_NAME = 'Jesty1993'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['Z1', 'Z2', 'E1', 'E2']
CONST_NAMES = ['compartment_1', 'kcat1', 'Km1', 'kcat2', 'Km2', 'ki1', 'ki2']

DEFAULT_INIT = np.array([0.0999, 100.0, 0.0001, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 10.0, 10.0, 200.0, 200.0, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1108260010', 'dataset_model_name': 'Jesty1993', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Jesty1993.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6 = consts

        # readable state aliases
        Z1 = x_0
        Z2 = x_1
        E1 = x_2
        E2 = x_3

        # readable parameter aliases
        compartment_1 = c_0
        kcat1 = c_1
        Km1 = c_2
        kcat2 = c_3
        Km2 = c_4
        ki1 = c_5
        ki2 = c_6

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*x_0*x_3/(c_2 + x_0)
        dx_1 = -c_0*c_3*x_1*x_2/(c_4 + x_1)
        dx_2 = c_0*c_1*x_0*x_3/(c_2 + x_0) - c_0*c_5*x_2
        dx_3 = c_0*c_3*x_1*x_2/(c_4 + x_1) - c_0*c_6*x_3

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
