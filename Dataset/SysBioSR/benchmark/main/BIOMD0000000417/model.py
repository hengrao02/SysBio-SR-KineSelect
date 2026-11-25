from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000417'
MODEL_NAME = 'Ratushny2012'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['Target']
CONST_NAMES = ['___r2____RATE__', 'ks', 'Kd', 'ka', 'Ksp', 'P', 's', 'h', 'k0']

DEFAULT_INIT = np.array([10.0], dtype=float)
DEFAULT_CONSTS = np.array([0.1, 10.0, 1e-05, 40.0, 0.001, 40.0, 1000.0, 2.0, 0.1], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000417', 'dataset_model_name': 'Ratushny2012', 'dim': 1, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Ratushny2012.xml', 'warnings': []}


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
        x_0, = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8 = consts

        # readable state aliases
        Target = x_0

        # readable parameter aliases
        r2_RATE = c_0
        ks = c_1
        Kd = c_2
        ka = c_3
        Ksp = c_4
        P = c_5
        s = c_6
        h = c_7
        k0 = c_8

        # explicit ODE right-hand sides
        dx_0 = -c_0*x_0 + c_1*(c_8 + (c_2*(-((1 + c_4*(-((1 + (c_5 + c_6)/c_4)**2 - 4*c_5*c_6/c_4**2)**0.5 + 1 + (c_5 + c_6)/c_4)/c_2)**2 - c_4**2*(-((1 + (c_5 + c_6)/c_4)**2 - 4*c_5*c_6/c_4**2)**0.5 + 1 + (c_5 + c_6)/c_4)**2/c_2**2)**0.5 + 1 + c_4*(-((1 + (c_5 + c_6)/c_4)**2 - 4*c_5*c_6/c_4**2)**0.5 + 1 + (c_5 + c_6)/c_4)/c_2)/(2*c_3))**c_7)/((c_2*(-((1 + c_4*(-((1 + (c_5 + c_6)/c_4)**2 - 4*c_5*c_6/c_4**2)**0.5 + 1 + (c_5 + c_6)/c_4)/c_2)**2 - c_4**2*(-((1 + (c_5 + c_6)/c_4)**2 - 4*c_5*c_6/c_4**2)**0.5 + 1 + (c_5 + c_6)/c_4)**2/c_2**2)**0.5 + 1 + c_4*(-((1 + (c_5 + c_6)/c_4)**2 - 4*c_5*c_6/c_4**2)**0.5 + 1 + (c_5 + c_6)/c_4)/c_2)/(2*c_3))**c_7 + 1)

        return np.array([dx_0], dtype=float)


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
