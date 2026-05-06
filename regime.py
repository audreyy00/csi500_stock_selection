"""
Regime：截面「可预测性」打分与分桶。

* S：典型价 Typical Price \(=(H+L+2C)/4\) 上 EMA20 相对 20 日前变化幅度；
* V、I：仍基于 **收盘价**日收益序列（与原逻辑一致）。
* 扩展：`regime_corr_vp`（5 量价相关）、量价发散截面秩差 `regime_vp_div`、其滚动不稳
  `regime_vp_div_vol`、`regime_compact_stable`（收盘价在 HL 带内位置的短时波动）。
* `regime_score`：在 Rank(S)−Rank(V)−Rank(I) 上叠加加权秩项；`regime_bucket` 仍按当日三分位。
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

N: Final[int] = 20
EMA_SPAN: Final[int] = 20
# regime_score 中扩展项权重（可先固定；树模型仍可再分配特征重要性）
WT_CORR: Final[float] = 0.5
WT_COMPACT_STABLE: Final[float] = 0.3
WT_VP_DIV_VOL: Final[float] = 0.3


def _pct_rank_fill0(g: pd.Series) -> pd.Series:
    """按日截面 percent rank；NaN→0 再 rank（与给定公式对齐）。"""
    return pd.to_numeric(g, errors="coerce").fillna(0.0).rank(method="average", pct=True)


def _per_stock_series(df: pd.DataFrame) -> pd.DataFrame:
    """按股时间序列：S/V/I 骨架 + corr / ret&turnover for div / compact stability。"""
    df = df.sort_values("date").copy()
    close = pd.to_numeric(df["close"], errors="coerce").astype(np.float64)
    high = pd.to_numeric(df["high"], errors="coerce").astype(np.float64)
    low = pd.to_numeric(df["low"], errors="coerce").astype(np.float64)
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(np.float64)

    # 模块 1：典型价上 EMA（趋势项 S）
    typical = (high + low + 2.0 * close) / 4.0
    df["ema20"] = typical.ewm(span=EMA_SPAN, adjust=False, min_periods=EMA_SPAN).mean()

    ema_prev = df["ema20"].shift(N)
    df["regime_S_raw"] = np.where(
        np.isfinite(ema_prev.values) & (np.abs(np.asarray(ema_prev)) >= 1e-12),
        np.abs(df["ema20"] / ema_prev - 1.0),
        np.nan,
    )

    r = close.pct_change(1)
    df["regime_V_raw"] = r.rolling(N, min_periods=N).std(ddof=1)
    sigma_prior = r.shift(1).rolling(N, min_periods=N).std(ddof=1).replace(0.0, np.nan)
    df["regime_I_raw"] = np.abs(r) / sigma_prior

    # 模块 2：5 日滚量价相关（数值边界可能出 ±inf，XGBoost 不接受 inf→转 NaN）
    _corr = close.rolling(window=5, min_periods=5).corr(volume)
    df["regime_corr_vp"] = pd.to_numeric(_corr, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # 模块 3：量价发散截面用（先做 time-series）
    hl_spread = np.maximum(high - low, 1e-8)
    compact = (close - low) / hl_spread
    df["regime_compact_stable"] = compact.rolling(3, min_periods=2).std()

    if "turnover" in df.columns:
        turnover = pd.to_numeric(df["turnover"], errors="coerce").fillna(0.0).astype(np.float64)
    else:
        turnover = pd.Series(0.0, index=df.index, dtype=np.float64)

    df["regime_ret_5d"] = close.pct_change(5)
    df["regime_turnover_5d"] = turnover.pct_change(5).replace([np.inf, -np.inf], np.nan)

    return df[
        [
            "date",
            "stock_code",
            "regime_S_raw",
            "regime_V_raw",
            "regime_I_raw",
            "regime_corr_vp",
            "regime_compact_stable",
            "regime_ret_5d",
            "regime_turnover_5d",
        ]
    ]


def _daily_bucket(series: pd.Series) -> pd.Series:
    """Tertiles of regime_score within one day (~equal-count bins). Same index."""
    result = pd.Series(np.nan, index=series.index, dtype=np.float64)
    mask = series.notna()
    if mask.sum() < 10:
        return result
    ssub = series[mask].astype(np.float64)
    rnk = ssub.rank(method="first").astype(np.float64)
    try:
        b = pd.qcut(rnk, q=3, labels=np.array([0.0, 1.0, 2.0]), duplicates="drop")
    except (ValueError, TypeError):
        return result
    result.loc[ssub.index] = b.astype(float).values
    return result


def compute_regime_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """每股每日：raw → 截面 rank/score、bucket。"""
    need = {"date", "stock_code", "close", "high", "low", "volume"}
    miss = need - set(prices.columns)
    if miss:
        raise ValueError(f"prices missing columns: {sorted(miss)}")

    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["stock_code"] = p["stock_code"].astype(str).str.zfill(6)

    pieces: list[pd.DataFrame] = []
    for _, g in p.groupby("stock_code", sort=False):
        pieces.append(_per_stock_series(g))
    stacked = pd.concat(pieces, ignore_index=True)

    for letter, raw in (("S", "regime_S_raw"), ("V", "regime_V_raw"), ("I", "regime_I_raw")):
        stacked[f"regime_rank_{letter}"] = stacked.groupby("date")[raw].rank(method="average", pct=True)

    # 模块 3 + 5：截面 rank → 发散度 → 每股 rolling std
    stacked["regime_ret_rank_xs"] = stacked.groupby("date", group_keys=False)["regime_ret_5d"].transform(
        _pct_rank_fill0
    )
    stacked["regime_turnover_rank_xs"] = stacked.groupby("date", group_keys=False)[
        "regime_turnover_5d"
    ].transform(_pct_rank_fill0)
    stacked["regime_vp_div"] = stacked["regime_ret_rank_xs"] - stacked["regime_turnover_rank_xs"]
    stacked = stacked.sort_values(["stock_code", "date"]).reset_index(drop=True)
    stacked["regime_vp_div_vol"] = stacked.groupby("stock_code", sort=False)["regime_vp_div"].transform(
        lambda x: x.rolling(5, min_periods=3).std()
    )

    rk_corr = stacked.groupby("date", group_keys=False)["regime_corr_vp"].transform(_pct_rank_fill0)
    rk_compat = stacked.groupby("date", group_keys=False)["regime_compact_stable"].transform(_pct_rank_fill0)
    rk_vpvol = stacked.groupby("date", group_keys=False)["regime_vp_div_vol"].transform(_pct_rank_fill0)

    stacked["regime_score"] = (
        stacked["regime_rank_S"]
        - stacked["regime_rank_V"]
        - stacked["regime_rank_I"]
        + WT_CORR * rk_corr
        - WT_COMPACT_STABLE * rk_compat
        - WT_VP_DIV_VOL * rk_vpvol
    )

    stacked["regime_bucket"] = stacked.groupby("date", group_keys=False)["regime_score"].transform(
        _daily_bucket
    )
    stacked["regime_bucket"] = pd.to_numeric(stacked["regime_bucket"], errors="coerce").astype(np.float64)

    drop_tmp = ["regime_ret_rank_xs", "regime_turnover_rank_xs"]
    stacked = stacked.drop(columns=[c for c in drop_tmp if c in stacked.columns])

    return stacked


# 并入面板供排查 / XGBoost 选用的列。
_MERGE_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "regime_S_raw",
    "regime_V_raw",
    "regime_I_raw",
    "regime_rank_S",
    "regime_rank_V",
    "regime_rank_I",
    "regime_score",
)

_REGIME_MERGED_FEATURES: tuple[str, ...] = (
    "regime_corr_vp",
    "regime_compact_stable",
    "regime_ret_5d",
    "regime_turnover_5d",
    "regime_vp_div",
    "regime_vp_div_vol",
)

REGIME_FEATURE_COLUMNS: tuple[str, ...] = (
    "regime_bucket",
    "regime_corr_vp",
    "regime_compact_stable",
    "regime_vp_div",
    "regime_vp_div_vol",
)


def merge_regime_features(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """将 regime bucket、诊断与新特征从 ``prices`` 对齐合并到面板。"""
    reg = compute_regime_panel(prices)
    merge_cols = [
        "date",
        "stock_code",
        "regime_bucket",
        *_MERGE_DIAGNOSTIC_COLUMNS,
        *_REGIME_MERGED_FEATURES,
    ]
    cols = [c for c in merge_cols if c in reg.columns]
    out = panel.merge(reg[cols], on=["date", "stock_code"], how="left")

    rb = pd.to_numeric(out["regime_bucket"], errors="coerce").clip(lower=0.0, upper=2.0)
    out["regime_bucket"] = rb

    return out
