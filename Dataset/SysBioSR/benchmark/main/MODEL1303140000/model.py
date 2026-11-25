from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1303140000'
MODEL_NAME = 'Mazemondet2012'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['species_1', 'species_2', 'species_3', 'species_4', 'species_6']
CONST_NAMES = ['compartment_1',
 'reaction_1__v',
 'reaction_4__k1',
 'reaction_3__k1',
 'reaction_2__k1',
 'reaction_7__k1',
 'reaction_5__k1',
 'reaction_10__k1',
 'reaction_6__k1',
 'reaction_8__k1',
 'reaction_11__k1',
 'reaction_9__k1',
 'reaction_12__k1']

DEFAULT_INIT = np.array([16365.0, 4.0, 0.0, 1000.0, 6655.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 300.0, 0.135, 0.0549, 0.000113, 0.1, 9.9e-05, 0.0182, 0.001619, 0.0167, 10.0, 0.167, 0.27], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1303140000', 'dataset_model_name': 'Mazemondet2012', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Mazemondet2012.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12 = consts

        # readable state aliases
        species_1 = x_0
        species_2 = x_1
        species_3 = x_2
        species_4 = x_3
        species_6 = x_4

        # readable parameter aliases
        compartment_1 = c_0
        reaction_1_v = c_1
        reaction_4_k1 = c_2
        reaction_3_k1 = c_3
        reaction_2_k1 = c_4
        reaction_7_k1 = c_5
        reaction_5_k1 = c_6
        reaction_10_k1 = c_7
        reaction_6_k1 = c_8
        reaction_8_k1 = c_9
        reaction_11_k1 = c_10
        reaction_9_k1 = c_11
        reaction_12_k1 = c_12

        # explicit ODE right-hand sides
        dx_0 = c_0*c_1 - c_0*c_4*x_0 - c_0*c_5*x_0*x_2 + c_2*x_4 - c_3*x_0
        dx_1 = c_0*c_10*x_2*x_3 + c_0*c_7*x_2 - c_0*c_8*x_1 - c_0*c_9*x_1 + c_6*x_4
        dx_2 = -c_0*c_10*x_2*x_3 - c_0*c_11*x_2 - c_0*c_7*x_2 + c_0*c_8*x_1
        dx_3 = -c_0*c_12*x_3
        dx_4 = -c_2*x_4 + c_3*x_0

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
