from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000916'
MODEL_NAME = 'Kraan199'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['The_FOD_in_the_circulation__X1',
 'The_FOD_in_the_metabolizing_tissues__X4',
 'The_cumulative_FOD_excreted_in_the_non_urinary_pool__X3',
 'The_FOD_in_the_gallbladder___intestinal_lumen__X5',
 'The_cumulative_FOD_excreted_in_the_urine__X2']
CONST_NAMES = ['K3', 'compartment', 'K4', 'K5', 'K1', 'K2']

DEFAULT_INIT = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([26.6, 1.0, 1.2, 1.2, 3.6, 1.2], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000916', 'dataset_model_name': 'Kraan199', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Kraan199.xml', 'warnings': []}


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
        The_FOD_in_the_circulation_X1 = x_0
        The_FOD_in_the_metabolizing_tissues_X4 = x_1
        The_cumulative_FOD_excreted_in_the_non_urinary_pool_X3 = x_2
        The_FOD_in_the_gallbladder_intestinal_lumen_X5 = x_3
        The_cumulative_FOD_excreted_in_the_urine_X2 = x_4

        # readable parameter aliases
        K3 = c_0
        compartment = c_1
        K4 = c_2
        K5 = c_3
        K1 = c_4
        K2 = c_5

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*x_0
        dx_1 = c_0*c_1*x_0 - c_1*c_4*x_1 - c_1*(c_2*x_1 - c_3*x_3)
        dx_2 = c_1*c_5*x_3
        dx_3 = -c_1*c_5*x_3 + c_1*(c_2*x_1 - c_3*x_3)
        dx_4 = c_1*c_4*x_1

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
