from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000766'
MODEL_NAME = 'Macnamara2015'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['U', 'I', 'M', 'E', 'V']
CONST_NAMES = ['r', 'K', 'dv', 'du', 'hu', 'he', 'hv', 'delta', 'pm', 'm', 'pe', 'de', 'dt', 'w', 'b', 'tme']

DEFAULT_INIT = np.array([1000000.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
DEFAULT_CONSTS = np.array([0.927,
 181820000.0,
 0.0038,
 2.0,
 1.0,
 1000.0,
 10000.0,
 1.0,
 2.5,
 10000.0,
 0.4,
 0.1,
 5e-09,
 2.042,
 1000.0,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000766', 'dataset_model_name': 'Macnamara2015', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Macnamara2015.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15 = consts

        # readable state aliases
        U = x_0
        I = x_1
        M = x_2
        E = x_3
        V = x_4

        # readable parameter aliases
        r = c_0
        K = c_1
        dv = c_2
        du = c_3
        hu = c_4
        he = c_5
        hv = c_6
        delta = c_7
        pm = c_8
        m = c_9
        pe = c_10
        de = c_11
        dt = c_12
        w = c_13
        b = c_14
        tme = c_15

        # explicit ODE right-hand sides
        dx_0 = (1*c_15*c_0*x_0*(1 - (x_0 + x_1)/c_1) + (-1)*c_15*c_2*x_4*x_0/(c_4 + x_0) + (-1)*c_15*c_3*x_0*x_3/(c_5 + x_3))/c_15
        dx_1 = (1*c_15*c_2*x_4*x_0/(c_4 + x_0) + (-1)*c_15*c_7*x_1 + (-1)*c_15*c_3*x_1*x_3/(c_5 + x_3))/c_15
        dx_2 = 1*c_15*c_8*x_4*x_2*(1 - x_2/c_9)/(c_6 + x_4)/c_15
        dx_3 = (1*c_15*c_10*x_2*(x_0 + x_4)/(x_0 + x_4 + c_6) + (-1)*c_15*c_11*x_3 + (-1)*c_15*c_12*x_0*x_3)/c_15
        dx_4 = (1*c_15*c_7*c_14*x_1 + (-1)*c_15*c_13*x_4)/c_15

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
