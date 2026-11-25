from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000001036'
MODEL_NAME = 'Cappuccio2007'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['Effector_cells', 'Tumor_cells', 'IL_2']
CONST_NAMES = ['c',
 'mu2',
 'p1',
 'g1',
 's1',
 'r2',
 'b',
 'a',
 'g2',
 'p2',
 'g3',
 'mu3',
 's2',
 'Tumor_microenvironment']

DEFAULT_INIT = np.array([1e-05, 1e-05, 1e-05], dtype=float)
DEFAULT_CONSTS = np.array([1.009, 0.0378, 0.044, 0.02, 0.0, 0.123, 1.0, 0.018, 0.0001, 0.9, 1e-05, 1.8, 0.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000001036', 'dataset_model_name': 'Cappuccio2007', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Cappuccio2007.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13 = consts

        # readable state aliases
        Effector_cells = x_0
        Tumor_cells = x_1
        IL_2 = x_2

        # readable parameter aliases
        c = c_0
        mu2 = c_1
        p1 = c_2
        g1 = c_3
        s1 = c_4
        r2 = c_5
        b = c_6
        a = c_7
        g2 = c_8
        p2 = c_9
        g3 = c_10
        mu3 = c_11
        s2 = c_12
        Tumor_microenvironment = c_13

        # explicit ODE right-hand sides
        dx_0 = (1*c_13*(c_0*x_1 + c_2*x_0*x_2/(c_3 + x_2) + c_4) + (-1)*c_13*c_1*x_0)/c_13
        dx_1 = (1*c_13*c_5*x_1*(1 - c_6*x_1) + (-1)*c_13*c_7*x_0*x_1/(c_8 + x_1))/c_13
        dx_2 = (1*c_13*(c_9*x_0*x_1/(c_10 + x_1) + c_12) + (-1)*c_13*c_11*x_2)/c_13

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
