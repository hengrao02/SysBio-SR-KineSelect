from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000922'
MODEL_NAME = 'Turner2015'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['Population_of_Eggs', 'Population_of_Larvae', 'Population_of_Pupae']
CONST_NAMES = ['Ar', 'ep', 'Te', 'Me', 'K0', 'Tl', 'Tp', 'Mp', 'Ml', 'compartment']

DEFAULT_INIT = np.array([10000.0, 5000.0, 3000.0], dtype=float)
DEFAULT_CONSTS = np.array([20.0, 30.0, 0.361, 0.05, 0.0002, 0.134, 0.342, 0.0025, 0.0501, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000922', 'dataset_model_name': 'Turner2015', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Turner2015.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9 = consts

        # readable state aliases
        Population_of_Eggs = x_0
        Population_of_Larvae = x_1
        Population_of_Pupae = x_2

        # readable parameter aliases
        Ar = c_0
        ep = c_1
        Te = c_2
        Me = c_3
        K0 = c_4
        Tl = c_5
        Tp = c_6
        Mp = c_7
        Ml = c_8
        compartment = c_9

        # explicit ODE right-hand sides
        dx_0 = 1*c_9*(c_0*c_1 - x_0*(c_2 + c_3))/c_9
        dx_1 = 1*c_9*(x_0*c_2 - x_1*(c_5 + c_8) - c_4*x_1**2)/c_9
        dx_2 = 1*c_9*(x_1*c_5 - x_2*(c_6 + c_7))/c_9

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
