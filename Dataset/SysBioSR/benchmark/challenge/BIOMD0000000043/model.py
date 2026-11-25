from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000043'
MODEL_NAME = 'Borghans1997'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['EC', 'Z', 'Rho', 'Y', 'Fraction_Inactive_Channels']
CONST_NAMES = ['cytosol',
 'vin__v0',
 'beta',
 'vin__v1',
 'extracellular',
 'v5__K',
 'v4__Kf',
 'intravesicular',
 'v2__Vm2',
 'v2__K2',
 'a',
 'v3__Vm3',
 'd',
 'v3__Ky',
 'v7__Kr',
 'v6__Kd']

DEFAULT_INIT = np.array([0.0, 0.0, 0.0, 0.36, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 6.5, 0.1, 40000.0, 50.0, 100.0, 0.2, 5.0, 5000.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000043', 'dataset_model_name': 'Borghans1997', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Borghans1997.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15 = consts

        # readable state aliases
        EC = x_0
        Z = x_1
        Rho = x_2
        Y = x_3
        Fraction_Inactive_Channels = x_4

        # readable parameter aliases
        cytosol = c_0
        vin_v0 = c_1
        beta = c_2
        vin_v1 = c_3
        extracellular = c_4
        v5_K = c_5
        v4_Kf = c_6
        intravesicular = c_7
        v2_Vm2 = c_8
        v2_K2 = c_9
        a = c_10
        v3_Vm3 = c_11
        d = c_12
        v3_Ky = c_13
        v7_Kr = c_14
        v6_Kd = c_15

        # explicit ODE right-hand sides
        dx_0 = -c_0*(c_1 + c_2*c_3) + c_4*c_5*x_1
        dx_1 = c_0*c_10*c_11*c_2*x_1**4*x_2*x_3**2/(c_12*(c_13**2 + x_3**2)*(c_10*x_1**4/c_12 + 1)) + c_0*c_6*x_3 + c_0*(c_1 + c_2*c_3) - c_4*c_5*x_1 - c_7*c_8*x_1**2/(c_9**2 + x_1**2)
        dx_2 = c_0*c_14*(-x_2 + 1) - c_0*c_15*x_1**4*x_2
        dx_3 = -c_0*c_10*c_11*c_2*x_1**4*x_2*x_3**2/(c_12*(c_13**2 + x_3**2)*(c_10*x_1**4/c_12 + 1)) - c_0*c_6*x_3 + c_7*c_8*x_1**2/(c_9**2 + x_1**2)
        dx_4 = -c_0*c_14*(-x_2 + 1) + c_0*c_15*x_1**4*x_2

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
