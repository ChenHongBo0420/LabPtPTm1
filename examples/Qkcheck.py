# -*- coding: utf-8 -*-
"""QK.ipynb
DP16QAM_RRC0.2_28GBd_1ch' 下的 LP keys：['LP-20_1', 'LP-20_2', 'LP-20_3', 'LP-20_4', 'LP-20_5', 'LP-35_1', 'LP-35_2', 'LP-35_3', 'LP-35_4', 'LP-35_5', 'LP-47_1', 'LP-47_2', 'LP-47_3', 'LP-47_4', 'LP-47_5', 'LP-6_1', 'LP-6_2', 'LP-6_3', 'LP-6_4', 'LP-6_5', 'LP26_1', 'LP26_2', 'LP26_3', 'LP26_4', 'LP26_5', 'LP41_1', 'LP41_2', 'LP41_3', 'LP41_4', 'LP41_5', 'LP9_1', 'LP9_2', 'LP9_3', 'LP9_4', 'LP9_5']
'DP16QAM_RRC0.2_28GBd_1ch_SSNLW' 下的 LP keys：['LP-20_1', 'LP-35_1', 'LP-48_1', 'LP-58_1', 'LP-5_1', 'LP27_1', 'LP34_1', 'LP9_1']
'DP16QAM_RRC0.2_28GBd_5ch_SSNLW' 下的 LP keys：['LP-18_1', 'LP-35_1', 'LP-50_1', 'LP-5_1', 'LP24_1', 'LP37_1', 'LP9_1']
'DP16QAM_RRC0.2_34GBd_5ch_SSNLW' 下的 LP keys：['LP-20_1', 'LP-35_1', 'LP-51_1', 'LP-6_1', 'LP24_1', 'LP36_1', 'LP8_1']
'SP16QAM_RRC0.01_28GBd_1ch_SSNLW' 下的 LP keys：['LP-10_1', 'LP-26_1', 'LP-42_1', 'LP-57_1', 'LP20_1', 'LP34_1', 'LP4_1']
'SP16QAM_RRC0.1_28GBd_1ch_SSNLW' 下的 LP keys：['LP-24_1', 'LP-38_1', 'LP-54_1', 'LP-8_1', 'LP19_1', 'LP36_1', 'LP7_1']
'SP16QAM_RRC0.2_28GBd_1ch' 下的 LP keys：['LP-10_1', 'LP-26_1', 'LP-41_1', 'LP-52_1', 'LP19_1', 'LP35_1', 'LP5_1']
'SP16QAM_RRC0.2_28GBd_1ch_SSNLW' 下的 LP keys：['LP-20_1', 'LP-20_2', 'LP-20_3', 'LP-20_4', 'LP-20_5', 'LP-40_1', 'LP-40_2', 'LP-40_3', 'LP-40_4', 'LP-40_5', 'LP-55_1', 'LP-55_2', 'LP-55_3', 'LP-55_4', 'LP-55_5', 'LP-5_1', 'LP-5_2', 'LP-5_3', 'LP-5_4', 'LP-5_5', 'LP10_1', 'LP10_2', 'LP10_3', 'LP10_4', 'LP10_5', 'LP22_1', 'LP22_2', 'LP22_3', 'LP22_4', 'LP22_5', 'LP35_1', 'LP35_2', 'LP35_3', 'LP35_4', 'LP35_5']
'SPQPSK_RRC0.2_28GBd_1ch_SSNLW' 下的 LP keys：['LP-10_1', 'LP-24_1', 'LP-38_1', 'LP-54_1', 'LP21_1', 'LP35_1', 'LP7_1']

"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
if not hasattr(np, 'PINF'):
    np.PINF = np.inf
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf


try:
  import jax
except ModuleNotFoundError:
#   %pip install --upgrade "jax[cpu]"
# install commplax if not found
try:
  import commplax
except ModuleNotFoundError:
#   %pip install https://github.com/ChenHongBo0420/Comm/archive/master.zip
# install data api if not found
try:
  import labptptm1
except ModuleNotFoundError:
#   %pip install https://github.com/ChenHongBo0420/LabPtPTm1/archive/master.zip


# install GDBP if not found
try:
  import gdbp
except ModuleNotFoundError:
#   %pip install https://github.com/ChenHongBo0420/Q/archive/main.zip
# ===============================================================
# 0) 依赖 & 数据加载（和原脚本一致）
# ===============================================================
import numpy as np, jax, jax.numpy as jnp
from tqdm.auto import tqdm
from functools import partial
from gdbp.Singledata import load as gdat_load
from gdbp import SingleSOTA as gb
from commplax import comm, util

ds_train = gdat_load(1, -20, 1)[0]   # train
ds_test = gdat_load(1, -20, 1)[0]

# ===============================================================
# 1) 只构造 CDC  — 无 GDBP
# ===============================================================
def make_cdc(data, mode='train', steps=3,
             dtaps=271, rtaps=321):
    """返回一个 CDC 模型（ntaps=1, xi=0）"""
    fdbp_init  = partial(gb.fdbp_init, data.a, steps=steps)
    model_init = partial(gb.model_init, data)

    conf = dict(mode=mode, steps=steps,
                dtaps=dtaps, rtaps=rtaps,
                ntaps=1, init_fn=fdbp_init(xi=0.0))

    cdc = model_init(conf,               # 模型结构
                     sparams_flatkeys=[('fdbp_0',)],   # D-滤波固定
                     name='CDC')
    return cdc

model_tr = make_cdc(ds_train, mode='train')
model_te = make_cdc(ds_test,  mode='test')

# ===============================================================
# 2) 训练 CDC（500 iter）
# ===============================================================
buf = [None]*3
params = None
for _, p, _ in gb.train(model_tr, ds_train, n_iter=500):
    buf.append(p); params = buf.pop(0)   # 取第三近参数

# ===============================================================
# 3) 测试 CDC
# ===============================================================
metric, _ = gb.test(model_te, params, ds_test)
print(f"CDC   Q²={metric.QSq.total:6.2f} dB   BER={metric.BER.total:.3e}")

import zarr
import labptptm1.store as store   # 绝对导入，不用“.”
# 或者：
# from labptptm1 import store

root = store.open_group()
mods = list(root['815km_SSMF'].keys())
print("可用的 modulation formats：", mods)

import zarr
import labptptm1.store as store

root = store.open_group()
band = root['815km_SSMF']

for fmt in band.keys():
    lp_keys = list(band[fmt].keys())
    print(f"{fmt!r} 下的 LP keys：{lp_keys}")
