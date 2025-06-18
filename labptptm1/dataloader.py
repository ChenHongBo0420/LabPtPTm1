import numpy as np
import zarr
from . import store
from typing import Tuple, List, Union

def select(mod, lp_dbm, rep) -> Tuple[List[zarr.Group], List[None]]:
    """
    单通道数据选取：
      mod:  可以是索引（int），也可以是格式名称（str）
      lp_dbm: 发射功率（-20, -35, ...）
      rep: 重复序号（1, 2, 3, ...）
    返回：
      data_groups: list of Zarr groups each containing 'sent' 和 'recv'
      sup_groups:  always [None, None, ...]
    """
    root = store.open_group()
    # 第二层：所有调制格式名称
    mods = list(root['815km_SSMF'].keys())
    
    # 允许用索引或字符串
    if isinstance(mod, int):
        mod_key = mods[mod]
    else:
        mod_key = mod
        assert mod_key in mods, f"{mod_key} not in available mods"
    
    # 第三层：所有 LP-.._.. 名称
    lp_reps = list(root['815km_SSMF'][mod_key].keys())
    target_key = f"LP{lp_dbm:+d}_{rep}"
    assert target_key in lp_reps, f"{target_key} not found under {mod_key}"
    
    grp = root['815km_SSMF'][mod_key][target_key]
    return [grp], [None]


def help():
    print(
        "select(mod, lp_dbm, rep)\n"
        "  mod: int 索引（0-based）或 str 格式名\n"
        "  lp_dbm: 发射功率（dBm），例如 -20、-35\n"
        "  rep: 重复序号，1、2、3……\n"
        "返回：([data_group], [None])"
    )
