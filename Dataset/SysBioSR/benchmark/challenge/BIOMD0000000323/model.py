from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000323'
MODEL_NAME = 'Kim2011'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['species_1', 'species_2', 'species_3']
CONST_NAMES = ['parameter_1',
 'parameter_2',
 'parameter_3',
 'compartment_1',
 '__lp_r2_Shalve',
 '__lp_r3_Shalve',
 '__lp_r4_Shalve']

DEFAULT_INIT = np.array([1.077263224711098, 1.1126122502624984, 0.12041166991529113], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 0.3, 5.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000323', 'dataset_model_name': 'Kim2011', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Kim2011.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6 = consts

        # readable state aliases
        species_1 = x_0
        species_2 = x_1
        species_3 = x_2

        # readable parameter aliases
        parameter_1 = c_0
        parameter_2 = c_1
        parameter_3 = c_2
        compartment_1 = c_3
        lp_r2_Shalve = c_4
        lp_r3_Shalve = c_5
        lp_r4_Shalve = c_6

        # explicit ODE right-hand sides
        dx_0 = (1*c_3*c_0/(c_5**c_2 + x_1**c_2) + (-1)*c_3*x_0/c_1/(1 + x_0/c_1))/c_3
        dx_1 = (1*c_3*c_0/(c_6**c_2 + x_2**c_2) + (-1)*c_3*x_1/c_1/(1 + x_1/c_1))/c_3
        dx_2 = (1*c_3*c_0/(c_4**c_2 + x_0**c_2) + (-1)*c_3*x_2/c_1/(1 + x_2/c_1))/c_3

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
