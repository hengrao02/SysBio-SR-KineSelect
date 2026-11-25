from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000062'
MODEL_NAME = 'Bhartiya2003'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['Enz', 'Ts']
CONST_NAMES = ['Enzyme_dilution__mu',
 'compartment',
 'Enzyme_synthesis__Ot',
 'Enzyme_synthesis__k1',
 'Enzyme_synthesis__ki1',
 'Enzyme_synthesis__nH',
 'Tex',
 'Tomax',
 'e_val',
 'f_val',
 'tryptophan_dilution__mu',
 'tryptophan_consumption__g',
 'tryptophan_consumption__Kg',
 'tryptophan_synthesis__Ki2',
 'tryptophan_synthesis__k2']

DEFAULT_INIT = np.array([0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.01, 1.0, 0.0033, 65.0, 3.53, 1.92, 0.14, 100.0, 0.9, 380.0, 0.01, 25.0, 0.2, 810.0, 25.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000062', 'dataset_model_name': 'Bhartiya2003', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Bhartiya2003.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14 = consts

        # readable state aliases
        Enz = x_0
        Ts = x_1

        # readable parameter aliases
        Enzyme_dilution_mu = c_0
        compartment = c_1
        Enzyme_synthesis_Ot = c_2
        Enzyme_synthesis_k1 = c_3
        Enzyme_synthesis_ki1 = c_4
        Enzyme_synthesis_nH = c_5
        Tex = c_6
        Tomax = c_7
        e_val = c_8
        f_val = c_9
        tryptophan_dilution_mu = c_10
        tryptophan_consumption_g = c_11
        tryptophan_consumption_Kg = c_12
        tryptophan_synthesis_Ki2 = c_13
        tryptophan_synthesis_k2 = c_14

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*x_0 + c_1*c_2*c_3*c_4**c_5/(c_4**c_5 + (c_6*c_7/(c_6*(1 + x_1/c_9) + c_8) + x_1)**c_5)
        dx_1 = -c_1*c_10*x_1 - c_1*c_11*x_1/(c_12 + x_1) + c_1*c_13*c_14*x_0/(c_13 + c_6*c_7/(c_6*(1 + x_1/c_9) + c_8) + x_1)

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
