from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000912'
MODEL_NAME = 'Caravagna2010'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['T', 'E', 'I']
CONST_NAMES = ['r2', 'b', 'a', 'g2', 'p1', 'g1', 'mu2', 'c', 'p2', 'g3', 'mu3', 'V', 's1', 's2', 'compartment']

DEFAULT_INIT = np.array([1.0, 1.0, 1.0], dtype=float)
DEFAULT_CONSTS = np.array([0.18, 1e-09, 1.0, 100000.0, 0.1245, 20000000.0, 0.03, 0.02, 5.0, 1000.0, 10.0, 3.2, 0.0, 0.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000912', 'dataset_model_name': 'Caravagna2010', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Caravagna2010.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14 = consts

        # readable state aliases
        T = x_0
        E = x_1
        I = x_2

        # readable parameter aliases
        r2 = c_0
        b = c_1
        a = c_2
        g2 = c_3
        p1 = c_4
        g1 = c_5
        mu2 = c_6
        c = c_7
        p2 = c_8
        g3 = c_9
        mu3 = c_10
        V = c_11
        s1 = c_12
        s2 = c_13
        compartment = c_14

        # explicit ODE right-hand sides
        dx_0 = (1*c_14*c_0*x_0*(1 - c_1/c_11*x_0) + (-1)*c_14*c_2*x_0/(c_3*c_11 + x_0)*x_1)/c_14
        dx_1 = (1*c_14*(c_4*x_2/(c_5*x_2)*x_1 + c_7*x_0 + c_11*c_12) + (-1)*c_14*c_6*x_1)/c_14
        dx_2 = (1*c_14*(c_8/c_11*x_0*x_1/(c_9*c_11 + x_0) + c_13) + (-1)*c_14*c_10*x_2)/c_14

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
