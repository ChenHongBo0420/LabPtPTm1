import zarr
from . import store
from typing import Tuple, List, Optional, Union

def select(
    mod: Union[int, str],
    lp_dbm: int,
    rep: Optional[int] = None
) -> Tuple[List[zarr.Group], List[None]]:
    """
    单通道数据选取：
      - mod:   int（索引）或 str（格式名称）
      - lp_dbm: 发射功率（dBm），例如 -20、-35…
      - rep:   重复序号 1、2、3…；或 None（默认）加载该功率下所有 repeats
    返回：
      - data_groups: 包含 'sent' 和 'recv' 的 Zarr group 列表
      - sup_groups:  同长度的 [None, None, …]
    """
    root = store.open_group()
    band = root['815km_SSMF']

    # 1) 选择调制格式
    mods = list(band.keys())
    if isinstance(mod, int):
        try:
            mod_key = mods[mod]
        except IndexError:
            raise IndexError(f"mod 索引 {mod} 越界；可选格式：{mods}")
    else:
        if mod not in mods:
            raise KeyError(f"mod '{mod}' 不存在；可选格式：{mods}")
        mod_key = mod

    # 2) 列出并过滤 LP… 键
    all_lp_keys = list(band[mod_key].keys())
    prefix = f"LP{lp_dbm:d}_"
    matching = [k for k in all_lp_keys if k.startswith(prefix)]
    if not matching:
        raise AssertionError(
            f"在 '{mod_key}' 下找不到功率 {lp_dbm} dBm 的数据。\n"
            f"可用 keys：{all_lp_keys}"
        )

    # 3) 根据 rep 选择
    if rep is None:
        selected = matching
    else:
        target = f"{prefix}{rep}"
        if target not in matching:
            raise AssertionError(
                f"请求的 '{target}' 不存在于 '{mod_key}' 下。\n"
                f"匹配到的 keys：{matching}"
            )
        selected = [target]

    # 4) 返回对应的 Zarr groups
    groups = [band[mod_key][key] for key in selected]
    return groups, [None] * len(groups)


def help():
    print(
        "select(mod, lp_dbm, rep=None)\n"
        "  mod:    int 索引（0-based）或 str 格式名\n"
        "  lp_dbm: 发射功率（dBm），例如 -20、-35\n"
        "  rep:    重复序号，1、2、3…；或 None（默认）加载所有 repeats\n"
        "返回：([data_group…], [None…])"
    )
