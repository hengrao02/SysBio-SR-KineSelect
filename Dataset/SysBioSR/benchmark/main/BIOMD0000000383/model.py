from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000383'
MODEL_NAME = 'Arnold2011'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['PGA', 'NADPH']
CONST_NAMES = ['chloroplast', 'CO2', 'E', 'kc', 'Kc', 'O2', 'Ko', 'RuBP', 'Nt', 'Rp', 'I', 'f']

DEFAULT_INIT = np.array([2.4, 0.21], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 230.0, 1.33846153846154, 2.5, 460.0, 210.0, 330.0, 2.0, 0.5, 3.2, 1000.0, 0.23], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000383', 'dataset_model_name': 'Arnold2011', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Arnold2011.xml', 'warnings': []}


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
        x_0, x_1 = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11 = consts

        # readable state aliases
        PGA = x_0
        NADPH = x_1

        # readable parameter aliases
        chloroplast = c_0
        CO2 = c_1
        E = c_2
        kc = c_3
        Kc = c_4
        O2 = c_5
        Ko = c_6
        RuBP = c_7
        Nt = c_8
        Rp = c_9
        I = c_10
        f = c_11

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_2*c_3*x_0*x_1/(c_8*c_9) + 2*c_0*(c_1*c_2*c_3/(2*(c_1 + c_4*(c_5/c_6 + 1))) + c_1*c_3*c_7/(2*(c_1 + c_4*(c_5/c_6 + 1))) - np.abs(c_1*c_2*c_3/(c_1 + c_4*(c_5/c_6 + 1)) - c_1*c_3*c_7/(c_1 + c_4*(c_5/c_6 + 1)))/2) + 0.1575*c_0*c_4*c_5*(c_1*c_2*c_3/(c_1 + c_4*(c_5/c_6 + 1)) + c_1*c_3*c_7/(c_1 + c_4*(c_5/c_6 + 1)) - np.abs(c_1*c_2*c_3/(c_1 + c_4*(c_5/c_6 + 1)) - c_1*c_3*c_7/(c_1 + c_4*(c_5/c_6 + 1))))/(c_1*c_6)
        dx_1 = -2*c_0*(c_1*c_2*c_3/(2*(c_1 + c_4*(c_5/c_6 + 1))) + c_1*c_3*c_7/(2*(c_1 + c_4*(c_5/c_6 + 1))) - np.abs(c_1*c_2*c_3/(c_1 + c_4*(c_5/c_6 + 1)) - c_1*c_3*c_7/(c_1 + c_4*(c_5/c_6 + 1)))/2) + c_0*(c_8 - x_1)*(np.abs(c_10*(c_11 - 1)/2 - 467)/130 - np.abs(c_10*(c_11 - 1)/2 + 467)/130)/(2*c_8) - 0.21*c_0*c_4*c_5*(c_1*c_2*c_3/(c_1 + c_4*(c_5/c_6 + 1)) + c_1*c_3*c_7/(c_1 + c_4*(c_5/c_6 + 1)) - np.abs(c_1*c_2*c_3/(c_1 + c_4*(c_5/c_6 + 1)) - c_1*c_3*c_7/(c_1 + c_4*(c_5/c_6 + 1))))/(c_1*c_6)

        return np.array([dx_0, dx_1], dtype=float)


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
