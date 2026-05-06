"""
防御模型的 label：相对指数的前向超额收益，叠加对个股下跌的惩罚。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_defensive_label(
    df: pd.DataFrame,
    index_returns: pd.Series,
    horizon: int = 5,
    lambda_penalty: float = 1.0,
) -> pd.Series:
    """
    label = (个股未来 horizon 收益 - 指数可比前向收益) - lambda_penalty * max(0, -个股未来收益)

    * 第一项：相对收益，偏好跑赢指数；
    * 第二项：绝对下跌惩罚（仅对个股负收益）。

    Parameters
    ----------
    df
        须含 ``close``、``stock_code``、``date``（与面板一致）。
    index_returns
        指数 **日收益率** ，以交易日 ``date`` 为索引（与 ``train_defensive`` 中自 ``index.parquet`` 算出的一致）。
    horizon
        与 ``features.FORWARD_HORIZON`` 一致的前瞻交易日数。
    lambda_penalty
        个股下跌惩罚强度（等价于原先方案里的 penalty 缩放）。

    Notes
    -----
    指数前向收益由日收益序列 ``rolling(horizon).sum().shift(-horizon)`` 按决策日对齐，
    与个股 ``close.shift(-horizon)/close - 1`` 在同一交易日截面上可对齐；
    严格复利可用指数收盘价自行替换，此处遵循既定滚动求和定义。
    """
    if not {"close", "stock_code", "date"}.issubset(df.columns):
        raise ValueError("make_defensive_label requires columns: close, stock_code, date")

    future_stock = df.groupby("stock_code", sort=False)["close"].transform(
        lambda x: x.shift(-horizon) / x - 1.0
    )
    fs = pd.to_numeric(future_stock, errors="coerce")

    ix = pd.to_numeric(index_returns, errors="coerce").sort_index()
    idx_forward = ix.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    dt = pd.to_datetime(df["date"], errors="coerce")
    index_future = pd.to_numeric(dt.map(idx_forward), errors="coerce")

    relative = fs - index_future
    fs_a = fs.to_numpy(dtype=float, copy=False)
    rel_a = relative.to_numpy(dtype=float, copy=False)
    penalty = float(lambda_penalty) * np.maximum(0.0, -fs_a)
    label_vals = rel_a - penalty

    return pd.Series(label_vals, index=df.index, dtype=np.float64)
