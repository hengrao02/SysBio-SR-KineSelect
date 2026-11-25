from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000707'
MODEL_NAME = 'Revilla2003'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['Normal_Th_cells',
 'Pathogen_Virus',
 'Single_Infected_Th_Cells',
 'Recombinant_Virus',
 'Double_Infected_Th_Cells']
CONST_NAMES = ['d', 'lamda', 'beta', 'a', 'alpha', 'b', 'k', 'u', 'c', 'q', 'Plasma']

DEFAULT_INIT = np.array([3.0, 149.0, 6.0, 1.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.01, 2.0, 0.004, 0.33, 0.004, 2.0, 50.0, 2.0, 2000.0, 2.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000707', 'dataset_model_name': 'Revilla2003', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Revilla2003.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10 = consts

        # readable state aliases
        Normal_Th_cells = x_0
        Pathogen_Virus = x_1
        Single_Infected_Th_Cells = x_2
        Recombinant_Virus = x_3
        Double_Infected_Th_Cells = x_4

        # readable parameter aliases
        d = c_0
        lamda = c_1
        beta = c_2
        a = c_3
        alpha = c_4
        b = c_5
        k = c_6
        u = c_7
        c = c_8
        q = c_9
        Plasma = c_10

        # explicit ODE right-hand sides
        dx_0 = (1*c_10*c_1 + (-1)*c_10*c_0*x_0 + (-1)*c_10*c_2*x_0*x_1)/c_10
        dx_1 = (1*c_10*c_6*x_2 + (-1)*c_10*c_7*x_1)/c_10
        dx_2 = (1*c_10*c_2*x_0*x_1 + (-1)*c_10*c_3*x_2 + (-1)*c_10*c_4*x_3*x_2)/c_10
        dx_3 = (1*c_10*c_8*x_4 + (-1)*c_10*c_9*x_3)/c_10
        dx_4 = (1*c_10*c_4*x_3*x_2 + (-1)*c_10*c_5*x_4)/c_10

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
