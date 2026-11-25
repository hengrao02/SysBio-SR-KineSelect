from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000076'
MODEL_NAME = 'Cronwright2002'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['G3P']
CONST_NAMES = ['Gpp_p__V2',
 'compartment',
 'Gpp_p__K2g3p',
 'Gpp_p__Phi',
 'Gpp_p__K2phi',
 'Gpd_p__Vf1',
 'Gpd_p__K1dhap',
 'Gpd_p__K1nadh',
 'DHAP',
 'Gpd_p__K1g3p',
 'Gpd_p__NAD',
 'Gpd_p__K1nad',
 'Gpd_p__NADH',
 'Gpd_p__ADP',
 'Gpd_p__K1adp',
 'Gpd_p__ATP',
 'Gpd_p__K1atp',
 'Gpd_p__F16BP',
 'Gpd_p__K1f16bp',
 'Gpd_p__Keq1']

DEFAULT_INIT = np.array([0.0], dtype=float)
DEFAULT_CONSTS = np.array([53.0,
 1.0,
 3.5,
 1.0,
 1.0,
 47.0,
 0.54,
 0.023,
 0.59,
 1.2,
 1.45,
 0.93,
 1.87,
 2.17,
 2.0,
 2.37,
 0.73,
 6.01,
 4.8,
 10000.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000076', 'dataset_model_name': 'Cronwright2002', 'dim': 1, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Cronwright2002.xml', 'warnings': []}


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
        x_0, = x
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19 = consts

        # readable state aliases
        G3P = x_0

        # readable parameter aliases
        Gpp_p_V2 = c_0
        compartment = c_1
        Gpp_p_K2g3p = c_2
        Gpp_p_Phi = c_3
        Gpp_p_K2phi = c_4
        Gpd_p_Vf1 = c_5
        Gpd_p_K1dhap = c_6
        Gpd_p_K1nadh = c_7
        DHAP = c_8
        Gpd_p_K1g3p = c_9
        Gpd_p_NAD = c_10
        Gpd_p_K1nad = c_11
        Gpd_p_NADH = c_12
        Gpd_p_ADP = c_13
        Gpd_p_K1adp = c_14
        Gpd_p_ATP = c_15
        Gpd_p_K1atp = c_16
        Gpd_p_F16BP = c_17
        Gpd_p_K1f16bp = c_18
        Gpd_p_Keq1 = c_19

        # explicit ODE right-hand sides
        dx_0 = -c_0*c_1*x_0/(c_2*(1 + x_0/c_2)*(c_3/c_4 + 1)) + c_1*c_5*(-c_10*x_0/c_19 + c_12*c_8)/(c_6*c_7*(1 + x_0/c_9 + c_8/c_6)*(c_10/c_11 + c_12/c_7 + 1)*(c_13/c_14 + c_15/c_16 + c_17/c_18 + 1))

        return np.array([dx_0], dtype=float)


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
