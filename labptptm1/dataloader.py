import numpy as np
import zarr
from . import store
from typing import Tuple, List


def select(SRC, LP, REP) -> Tuple[List[zarr.Group], List[None]]:
    """Data selection for single-channel LabPtPtm1 dataset.

    Returns two lists:
      - List of data groups (each has 'recv' and 'sent')
      - List of supplementary groups (always None for this dataset)
    """
    droot = store.open_group()

    # Ensure inputs are lists
    SRC = [SRC] if np.isscalar(SRC) else list(SRC)
    LP  = [LP]  if np.isscalar(LP)  else list(LP)
    REP = [REP] if np.isscalar(REP) else list(REP)

    # Validate and collect targets
    targets = []
    for src in SRC:
        for lp in LP:
            for rep in REP:
                _validate_args(src, lp, rep)
                targets.append((src, lp, rep))

    dat_grps = []
    for src, lp, rep in targets:
        # single-channel path under 815 km SSMF
        grp = droot[f"815km_SSMF/src{src}"][f"{lp}dBm_{rep}"]
        dat_grps.append(grp)

    # no supplementary data available in LabPtPtm1
    sup_grps = [None] * len(dat_grps)
    return dat_grps, sup_grps


def _validate_args(src: int, lp: int, rep: int):
    assert src in [1, 2],     f"source index must be 1 or 2, got {src}"
    assert lp in range(-5, 4), f"launched power must be between -5 and 3 dBm, got {lp}"
    assert rep in [1, 2, 3],   f"repeat index must be 1, 2, or 3, got {rep}"


def help():
    print(
        'arguments:\n'
        '  arg#1: int, source index (1 or 2)\n'
        '  arg#2: int, launched power in dBm (from -5 to 3)\n'
        '  arg#3: int, repetition index (1, 2, or 3)\n'
        'returns:\n'
        '  tuple of (data_groups, supplementary_groups)\n'
        '    data_groups: list of Zarr Groups each containing "recv" and "sent"\n'
        '    supplementary_groups: always None for this dataset'
    )
