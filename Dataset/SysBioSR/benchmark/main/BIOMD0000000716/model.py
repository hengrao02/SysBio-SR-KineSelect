from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000716'
MODEL_NAME = 'Lee2018'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['S_b', 'I_b', 'S_h', 'I_a']
CONST_NAMES = ['ModelValue_1', 'compartment', 'Human_Population']

DEFAULT_INIT = np.array([244956.0, 36485.0, 171977.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([171977.0, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000716', 'dataset_model_name': 'Lee2018', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Lee2018.xml', 'warnings': []}


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
        x_0, x_1, x_2, x_3 = x
        c_0, c_1, c_2 = consts

        # readable state aliases
        S_b = x_0
        I_b = x_1
        S_h = x_2
        I_a = x_3

        # readable parameter aliases
        ModelValue_1 = c_0
        compartment = c_1
        Human_Population = c_2

        # explicit ODE right-hand sides
        dx_0 = (1*c_1*2060/365 + (-1)*c_1*1/(2*365)*x_0 + (-1)*c_1*45685/(439197*30)*x_0*x_1)/c_1
        dx_1 = (1*c_1*45685/(439197*30)*x_0*x_1 + (-1)*c_1*45682/(439197*30)*x_1 + (-1)*c_1*1/(2*365)*x_1)/c_1
        dx_2 = (1*c_2*237/10*c_0/365000 + (-1)*c_2*1/(346/5*365)*x_2 + (-1)*16/(108*1000000*1825/2)*x_2*x_1)/c_2
        dx_3 = (1*16/(108*1000000*1825/2)*x_2*x_1 + (-1)*c_2*6/(108*1000000*1825/2)*x_3 + (-1)*c_2*1/(346/5*365)*x_3)/c_2

        return np.array([dx_0, dx_1, dx_2, dx_3], dtype=float)


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
