from typing import Tuple

import numpy as np
from aiida import orm


def identify_bravais(structure: orm.StructureData) -> str:
    raise NotImplementedError("Recover file or rewrite function")
    return ''

def get_cell_params(cell: np.ndarray) -> Tuple[float, float, float, float, float,float]:
    raise NotImplementedError("Recover file or rewrite function")
    return 1,2,3,4,5,6
