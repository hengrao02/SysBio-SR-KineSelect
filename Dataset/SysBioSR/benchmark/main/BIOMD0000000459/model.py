from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000459'
MODEL_NAME = 'Liebal2012'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['sigb', 'lacz', 'x']
CONST_NAMES = ['kbd', 'kbs', 'kxd', 'kxs', 'kzd', 'kzs', 'kzx', 'compartment', 'IPTG']

DEFAULT_INIT = np.array([0.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.0164812, 100.0, 1.08559e-09, 9.32517e-08, 1.34615e-07, 9.03538e-07, 0.00317772, 1.0, 100.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000459', 'dataset_model_name': 'Liebal2012', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Liebal2012.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8 = consts

        # readable state aliases
        sigb = x_0
        lacz = x_1
        x_3 = x_2

        # readable parameter aliases
        kbd = c_0
        kbs = c_1
        kxd = c_2
        kxs = c_3
        kzd = c_4
        kzs = c_5
        kzx = c_6
        compartment = c_7
        IPTG = c_8

        # explicit ODE right-hand sides
        dx_0 = (1*(c_8*c_1 - c_0*x_0) + (-1)*(c_5*x_0 - x_1*(c_4 + c_6*x_2)) + (-1)*(c_3*x_0 - c_2*x_2))/c_7
        dx_1 = 1*(c_5*x_0 - x_1*(c_4 + c_6*x_2))/c_7
        dx_2 = 1*(c_3*x_0 - c_2*x_2)/c_7

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
