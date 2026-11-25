from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000178'
MODEL_NAME = 'Lebeda2008'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['bulk', 'free', 'bound', 'translocate', 'lytic']
CONST_NAMES = ['bulk_movement__kS',
 'extracellular',
 'binding__kB',
 'endocytosis__kT',
 'endosome',
 'translocation__kL']

DEFAULT_INIT = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.00015, 1.0, 0.058, 0.141, 1.0, 0.013], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000178', 'dataset_model_name': 'Lebeda2008', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Lebeda2008.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5 = consts

        # readable state aliases
        bulk = x_0
        free = x_1
        bound = x_2
        translocate = x_3
        lytic = x_4

        # readable parameter aliases
        bulk_movement_kS = c_0
        extracellular = c_1
        binding_kB = c_2
        endocytosis_kT = c_3
        endosome = c_4
        translocation_kL = c_5

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*x_0
        dx_1 = c_0*c_1*x_0 - c_1*c_2*x_1
        dx_2 = c_1*c_2*x_1 - c_1*c_3*x_2
        dx_3 = c_1*c_3*x_2 - c_4*c_5*x_3
        dx_4 = c_4*c_5*x_3

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
