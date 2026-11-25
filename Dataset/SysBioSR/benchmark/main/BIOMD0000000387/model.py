from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'BIOMD0000000387'
MODEL_NAME = 'Arnold2011'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['PGA', 'NADPH']
CONST_NAMES = ['chloroplast',
 'CO2',
 'Vcmax',
 'O2',
 'Ko',
 'Kc',
 'Q',
 'alpha',
 'Jmax',
 'a',
 'starch',
 'TPU',
 'Nt',
 'Rp']

DEFAULT_INIT = np.array([2.4, 0.21], dtype=float)
DEFAULT_CONSTS = np.array([1.0,
 24.5,
 1.91141270310584,
 21000.0,
 187891.032274486,
 26.7125313821435,
 1000.0,
 0.24,
 142.047003854271,
 0.0434,
 0.5,
 0.259027750094988,
 0.5,
 3.2], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'BIOMD0000000387', 'dataset_model_name': 'Arnold2011', 'dim': 2, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Arnold2011.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13 = consts

        # readable state aliases
        PGA = x_0
        NADPH = x_1

        # readable parameter aliases
        chloroplast = c_0
        CO2 = c_1
        Vcmax = c_2
        O2 = c_3
        Ko = c_4
        Kc = c_5
        Q = c_6
        alpha = c_7
        Jmax = c_8
        a = c_9
        starch = c_10
        TPU = c_11
        Nt = c_12
        Rp = c_13

        # explicit ODE right-hand sides
        dx_0 = 2*c_0*(c_1*c_2/(4*(c_1 + c_5)*(c_3/c_4 + 1)) + 0.00173076923076923*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)) + 3*c_11/(2*(1 - 0.00020430517273755*c_3/c_1)) - np.abs(c_1*c_2/((c_1 + c_5)*(c_3/c_4 + 1)) - 0.00692307692307692*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)))/4 - np.abs(c_1*c_2/(2*(c_1 + c_5)*(c_3/c_4 + 1)) + 0.00346153846153846*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)) - 3*c_11/(1 - 0.00020430517273755*c_3/c_1) - np.abs(c_1*c_2/((c_1 + c_5)*(c_3/c_4 + 1)) - 0.00692307692307692*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)))/2)/2) - c_0*c_2*x_0*x_1/(c_12*c_13) + 0.1575*c_0*c_3*c_5*(c_1*c_2/(2*(c_1 + c_5)*(c_3/c_4 + 1)) + 0.00346153846153846*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)) + 3*c_11/(1 - 0.00020430517273755*c_3/c_1) - np.abs(c_1*c_2/((c_1 + c_5)*(c_3/c_4 + 1)) - 0.00692307692307692*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)))/2 - np.abs(c_1*c_2/(2*(c_1 + c_5)*(c_3/c_4 + 1)) + 0.00346153846153846*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)) - 3*c_11/(1 - 0.00020430517273755*c_3/c_1) - np.abs(c_1*c_2/((c_1 + c_5)*(c_3/c_4 + 1)) - 0.00692307692307692*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)))/2))/(c_1*c_4)
        dx_1 = -2*c_0*(c_1*c_2/(4*(c_1 + c_5)*(c_3/c_4 + 1)) + 0.00173076923076923*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)) + 3*c_11/(2*(1 - 0.00020430517273755*c_3/c_1)) - np.abs(c_1*c_2/((c_1 + c_5)*(c_3/c_4 + 1)) - 0.00692307692307692*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)))/4 - np.abs(c_1*c_2/(2*(c_1 + c_5)*(c_3/c_4 + 1)) + 0.00346153846153846*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)) - 3*c_11/(1 - 0.00020430517273755*c_3/c_1) - np.abs(c_1*c_2/((c_1 + c_5)*(c_3/c_4 + 1)) - 0.00692307692307692*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)))/2)/2) + 0.0138461538461538*c_0*c_6*c_7*(c_12 - x_1)*np.exp(-0.0398*c_10*c_9)/(c_12*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)) - 0.21*c_0*c_3*c_5*(c_1*c_2/(2*(c_1 + c_5)*(c_3/c_4 + 1)) + 0.00346153846153846*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)) + 3*c_11/(1 - 0.00020430517273755*c_3/c_1) - np.abs(c_1*c_2/((c_1 + c_5)*(c_3/c_4 + 1)) - 0.00692307692307692*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)))/2 - np.abs(c_1*c_2/(2*(c_1 + c_5)*(c_3/c_4 + 1)) + 0.00346153846153846*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)) - 3*c_11/(1 - 0.00020430517273755*c_3/c_1) - np.abs(c_1*c_2/((c_1 + c_5)*(c_3/c_4 + 1)) - 0.00692307692307692*c_1*c_6*c_7*np.exp(-0.0398*c_10*c_9)/((c_1 + 0.0004086103454751*c_3)*np.sqrt(0.81*c_6**2*c_7**2/c_8**2 + 1)))/2))/(c_1*c_4)

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
