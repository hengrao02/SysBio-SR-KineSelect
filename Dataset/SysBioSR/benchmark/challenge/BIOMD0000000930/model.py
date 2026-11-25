from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000930'
MODEL_NAME = 'Liu2017'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['Tumor_Cell_Population__T',
 'Effector_immune_cell_population__N',
 'Circulating_lymphocyte_population__C',
 'Chemotherapeutic_drug_concentration__M']
CONST_NAMES = ['a',
 'b',
 'c1',
 'mu',
 'g',
 's',
 'Kc',
 'Kn',
 'Kt',
 'kt',
 'p',
 'alpha1',
 'alpha2',
 'beta',
 'gamma',
 'Vm',
 'eta',
 'Tumor_Microenvironment']

DEFAULT_INIT = np.array([10000000.0, 300000.0, 62500000000.0, 0.45], dtype=float)
DEFAULT_CONSTS = np.array([0.431,
 1.02e-14,
 3.41e-10,
 0.0412,
 0.015,
 20.2,
 0.6,
 0.6,
 0.8,
 3.2e-09,
 2e-11,
 12000.0,
 750000000.0,
 0.012,
 0.9,
 0.45,
 0.0,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000930', 'dataset_model_name': 'Liu2017', 'dim': 4, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Liu2017.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17 = consts

        # readable state aliases
        Tumor_Cell_Population_T = x_0
        Effector_immune_cell_population_N = x_1
        Circulating_lymphocyte_population_C = x_2
        Chemotherapeutic_drug_concentration_M = x_3

        # readable parameter aliases
        a = c_0
        b = c_1
        c1 = c_2
        mu = c_3
        g = c_4
        s = c_5
        Kc = c_6
        Kn = c_7
        Kt = c_8
        kt = c_9
        p = c_10
        alpha1 = c_11
        alpha2 = c_12
        beta = c_13
        gamma = c_14
        Vm = c_15
        eta = c_16
        Tumor_Microenvironment = c_17

        # explicit ODE right-hand sides
        dx_0 = (1*c_17*c_0*x_0*(1 - c_1*x_0) + (-1)*c_17*c_2*x_1*x_0 + (-1)*c_17*c_8*x_3*x_0)/c_17
        dx_1 = (1*c_17*c_11 + 1*c_17*c_4*x_0/(c_5 + x_0)*x_1 + (-1)*c_17*c_3*x_1 + (-1)*c_17*c_10*x_0*x_1 + (-1)*c_17*c_7*(1 - c_16)*x_3*x_1)/c_17
        dx_2 = (1*c_17*c_12 + (-1)*c_17*c_13*x_2 + (-1)*c_17*c_6*(1 - c_16)*x_3*x_2)/c_17
        dx_3 = ((-1)*c_17*c_14*x_3 + 1*c_17*c_15 + (-1)*c_17*c_9*x_0*x_3)/c_17

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
