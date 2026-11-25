from __future__ import annotations

"""Portable explicit ODE reference model. Source equation: equation.json.

Model source and attribution are recorded in the accompanying XML and benchmark README.
"""


import numpy as np



from sysbiosr_reference import BaseModel, BaseParameters


MODEL_ID = 'MODEL1705170001'
MODEL_NAME = 'Proctor2017'
DATASET = "SysBio-SR"
SOURCE_GROUP = 'SysBio'

STATE_NAMES = ['IL1', 'miR140', 'miR140_gene', 'ADAMTS5_mRNA', 'ADAMTS5_mRNA_miR140']
CONST_NAMES = ['kdegIL1',
 'kdegADAMTS5mRNA',
 'krelADAMTS5mRNAmiR140',
 'ksynmiR140',
 'kdegmiR140',
 'kbinADAMTS5mRNAmiR140',
 'kdegmiR140byIL1',
 'ksynADAMTS5mRNA']

DEFAULT_INIT = np.array([500.0, 500.0, 2.0, 0.0, 0.0], dtype=float)
DEFAULT_CONSTS = np.array([0.0004, 0.00014, 0.01, 0.0018, 8e-06, 5e-05, 8e-07, 0.001], dtype=float)

MODEL_METADATA = {'dataset_model_id': 'MODEL1705170001', 'dataset_model_name': 'Proctor2017', 'dim': 5, 'equation_json_path': 'equation.json', 'equation_source': 'eq', 'has_delay': False, 'has_piecewise': False, 'has_time_dependency': False, 'source_xml': 'Proctor2017.xml', 'warnings': []}


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
        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7 = consts

        # readable state aliases
        IL1 = x_0
        miR140 = x_1
        miR140_gene = x_2
        ADAMTS5_mRNA = x_3
        ADAMTS5_mRNA_miR140 = x_4

        # readable parameter aliases
        kdegIL1 = c_0
        kdegADAMTS5mRNA = c_1
        krelADAMTS5mRNAmiR140 = c_2
        ksynmiR140 = c_3
        kdegmiR140 = c_4
        kbinADAMTS5mRNAmiR140 = c_5
        kdegmiR140byIL1 = c_6
        ksynADAMTS5mRNA = c_7

        # explicit ODE right-hand sides
        dx_0 = -c_0*x_0
        dx_1 = c_1*x_4 + c_2*x_4 + c_3*x_2 - c_4*x_1 - c_5*x_1*x_3 - c_6*x_0*x_1
        dx_2 = 0
        dx_3 = c_2*x_4 - c_5*x_1*x_3 + c_7*x_0
        dx_4 = -c_1*x_4 - c_2*x_4 + c_5*x_1*x_3

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
