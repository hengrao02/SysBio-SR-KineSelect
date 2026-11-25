from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000944'
MODEL_NAME = 'Goldbeter2013'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['Cyclin', 'Active_Cdc2_kinase', 'Active_APC']
CONST_NAMES = ['K1', 'K3', 'K4', 'Kc', 'Kd', 'kd', 'V2', 'V4', 'vs', 'vd', 'Vm1', 'Vm3', 'compartment']

DEFAULT_INIT = np.array([0.01, 0.01, 0.01], dtype=float)
DEFAULT_CONSTS = np.array([0.002, 0.01, 0.01, 0.5, 0.001, 0.046, 2.0, 0.7, 0.06, 0.25, 4.0, 1.0, 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000944', 'dataset_model_name': 'Goldbeter2013', 'dim': 3, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Goldbeter2013.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12 = consts

        # readable state aliases
        Cyclin = x_0
        Active_Cdc2_kinase = x_1
        Active_APC = x_2

        # readable parameter aliases
        K1 = c_0
        K3 = c_1
        K4 = c_2
        Kc = c_3
        Kd = c_4
        kd = c_5
        V2 = c_6
        V4 = c_7
        vs = c_8
        vd = c_9
        Vm1 = c_10
        Vm3 = c_11
        compartment = c_12

        # explicit ODE right-hand sides
        dx_0 = (1*c_12*c_8 + (-1)*c_12*(c_9*x_2*x_0/(c_4 + x_0) + c_5*x_0))/c_12
        dx_1 = (1*c_12*x_0/(c_3 + x_0)*c_10*(1 - x_1)/(c_0 + 1 - x_1) + (-1)*c_12*c_6*x_1/(c_0 + x_1))/c_12
        dx_2 = (1*c_12*x_1*c_11*(1 - x_2)/(c_1 + 1 - x_2) + (-1)*c_12*c_7*x_2/(c_2 + x_2))/c_12

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
