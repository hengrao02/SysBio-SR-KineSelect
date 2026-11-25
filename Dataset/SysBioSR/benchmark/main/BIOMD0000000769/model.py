from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000769'
MODEL_NAME = 'Eftimie2017'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'ODEBase'

STATE_NAMES = ['T', 'H1', 'H2', 'M1', 'M2']
CONST_NAMES = ['a',
 'b',
 'f',
 'gh1',
 'gh2',
 'gm1',
 'gm2',
 'nh1',
 'nh2',
 'nm1',
 'nm2',
 'ah1',
 'ah2',
 'am1',
 'am2',
 'm1',
 'm2',
 'ph1',
 'ph2',
 'pm1',
 'pm2',
 'rm1',
 'rm2',
 'eh1',
 'eh2',
 'em1',
 'em2',
 'tme']

DEFAULT_INIT = np.array([100000.0, 0.0, 0.0, 100.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.69,
 1000000000.0,
 1e-08,
 4.8e-09,
 1e-09,
 6e-09,
 2.3e-10,
 1e-07,
 1e-07,
 1e-07,
 1e-07,
 0.008,
 0.001,
 0.001,
 0.001,
 100000000.0,
 1000000000.0,
 0.09,
 0.09,
 0.02,
 0.02,
 0.09,
 0.05,
 0.03,
 0.03,
 0.02,
 0.02,
 1.0], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000769', 'dataset_model_name': 'Eftimie2017', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Eftimie2017.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19, c_20, c_21, c_22, c_23, c_24, c_25, c_26, c_27 = consts

        # readable state aliases
        T = x_0
        H1 = x_1
        H2 = x_2
        M1 = x_3
        M2 = x_4

        # readable parameter aliases
        a = c_0
        b = c_1
        f = c_2
        gh1 = c_3
        gh2 = c_4
        gm1 = c_5
        gm2 = c_6
        nh1 = c_7
        nh2 = c_8
        nm1 = c_9
        nm2 = c_10
        ah1 = c_11
        ah2 = c_12
        am1 = c_13
        am2 = c_14
        m1 = c_15
        m2 = c_16
        ph1 = c_17
        ph2 = c_18
        pm1 = c_19
        pm2 = c_20
        rm1 = c_21
        rm2 = c_22
        eh1 = c_23
        eh2 = c_24
        em1 = c_25
        em2 = c_26
        tme = c_27

        # explicit ODE right-hand sides
        dx_0 = (1*c_27*c_0*x_0*(1 - x_0/c_1) + (-1)*c_27*c_2*x_0 + (-1)*c_27*c_3*x_0*x_1 + (-1)*c_27*c_4*x_0*x_2 + (-1)*c_27*c_5*x_0*x_3 + 1*c_27*c_6*x_0*x_4)/c_27
        dx_1 = (1*c_27*c_11*x_3 + 1*c_27*c_17*x_1*x_3*(1 - (x_1 + x_2)/c_15) + (-1)*c_27*c_7*x_1*x_0 + (-1)*c_27*c_23*x_1)/c_27
        dx_2 = (1*c_27*c_12*x_4 + 1*c_27*c_18*x_2*x_4*(1 - (x_2 + x_1)/c_15) + (-1)*c_27*c_8*x_2*x_0 + (-1)*c_27*c_24*x_2)/c_27
        dx_3 = (1*c_27*c_13*x_1 + 1*c_27*c_19*x_3*(1 - (x_3 + x_4)/c_16) + (-1)*c_27*c_9*x_3*x_0 + (-1)*c_27*c_25*x_3 + (-1)*c_27*c_22*x_3 + 1*c_27*c_21*x_4)/c_27
        dx_4 = (1*c_27*c_22*x_3 + (-1)*c_27*c_21*x_4 + 1*c_27*c_14*x_2 + 1*c_27*c_20*x_4*x_2*(1 - (x_4 + x_3)/c_16) + 1*c_27*c_10*x_4*x_0 + (-1)*c_27*c_26*x_4)/c_27

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
