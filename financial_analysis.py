"""
财务（Baostock 季频）数据处理：读 ``data/financial_quarter_bs.parquet`` 等。

对齐约定（防前视）::

* 面板行 ``date=T`` 视作 **T 日决策**；三连季 G 仅使用 **NOTICE_DATE <= T−1** 的披露（**不用** REPORT 顶替公告）。
* ``financial_ret_5d_rank`` 使用 **T−1** 行上的 ``ret_5d``（同股内 `shift(1)`），即截止 **T−1 收盘** 的 5 日收益再截面 rank；**不覆盖** 面板原列 ``ret_5d``（仍含当日 close，供其他技术特征）。
* 截面 Rank(D)、Rank(A)、Rank(F)、Rank(价) 均在 **各自 ``date``** 内重置（非全历史一统排名）。

列由 ``merge_financial_features`` 并入；通常通过 ``features.build_features`` 调用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# 默认值：半衰期落在 60–90 日区间内，使季内较早公告仍保留一定 freshness。
DEFAULT_FRESH_HALF_LIFE_DAYS: Final[float] = 75.0

FINANCIAL_BS_DROP_COLUMNS: Final[tuple[str, ...]] = ("XSMLL_TB", "TOTALOPERATEREVETZ")

GROWTH_COL: Final[str] = "PARENTNETPROFITTZ"

FINANCIAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "financial_has_triple",
    "financial_delta",
    "financial_accel",
    "financial_r_delta",
    "financial_r_accel",
    "financial_inner",
    "financial_inner_rank",
    "financial_ret_5d_rank",
    "financial_gap",
    "financial_fresh",
    "financial_signal",
)


def load_financial_quarter_bs(
    path: Path | None = None,
    *,
    drop_sparse_cols: bool = True,
) -> pd.DataFrame:
    """读取 ``financial_quarter_bs.parquet``。"""
    p = path or DATA_DIR / "financial_quarter_bs.parquet"
    df = pd.read_parquet(p)
    if drop_sparse_cols:
        drop = [c for c in FINANCIAL_BS_DROP_COLUMNS if c in df.columns]
        if drop:
            df = df.drop(columns=drop)
    return df


def _prepare_financial_quarter(fin: pd.DataFrame) -> pd.DataFrame:
    """可见性仅以 **NOTICE_DATE**。无公告日 → 当季视为未进入市场，不可用 REPORT_DATE 顶替（防偷看）。"""
    df = fin.copy()
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
    if "NOTICE_DATE" in df.columns:
        df["_vis"] = pd.to_datetime(df["NOTICE_DATE"], errors="coerce")
    else:
        df["_vis"] = pd.NaT
    if GROWTH_COL not in df.columns:
        raise KeyError(f"财务表需含列 {GROWTH_COL}")
    df["_g"] = pd.to_numeric(df[GROWTH_COL], errors="coerce")
    return df.dropna(subset=["REPORT_DATE"]).sort_values(["stock_code", "REPORT_DATE"])


def _triple_delta_accel_notice(
    sub: pd.DataFrame,
    cutoff_notice_on_or_before: pd.Timestamp | float | np.floating | None,
) -> tuple[float, float, pd.Timestamp | None, bool]:
    """仅采纳 **NOTICE_DATE <= cutoff** 的披露行构造三连季 G；新鲜度仅用公告日。"""
    if cutoff_notice_on_or_before is None or pd.isna(cutoff_notice_on_or_before):
        return np.nan, np.nan, None, False
    cutoff = pd.Timestamp(cutoff_notice_on_or_before).normalize()
    vis = pd.to_datetime(sub["_vis"], errors="coerce").dt.normalize()
    ok = vis.notna() & (vis <= cutoff) & sub["_g"].notna()
    fe = sub.loc[ok].sort_values("REPORT_DATE")
    if len(fe) < 3:
        return np.nan, np.nan, None, False
    tail = fe.tail(3)
    g2 = float(tail.iloc[0]["_g"])
    g1 = float(tail.iloc[1]["_g"])
    gt = float(tail.iloc[2]["_g"])
    if any(np.isnan([g2, g1, gt])):
        return np.nan, np.nan, None, False
    nt_notice = pd.to_datetime(tail.iloc[2]["_vis"], errors="coerce")
    notice_ts = pd.Timestamp(nt_notice).normalize() if pd.notna(nt_notice) else None

    delta = gt - g1
    accel = (gt - g1) - (g1 - g2)
    return delta, accel, notice_ts, True


def _ensure_financial_nan_frame(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    n = len(out)
    out["financial_has_triple"] = np.zeros(n, dtype=np.int8)
    fill_nan = (
        col
        for col in FINANCIAL_FEATURE_COLUMNS
        if col != "financial_has_triple"
    )
    for c in fill_nan:
        out[c] = np.nan
    out["financial_notice_date"] = pd.NaT
    return out


def merge_financial_features(
    panel: pd.DataFrame,
    *,
    financial_df: pd.DataFrame | None = None,
    fin_path: Path | None = None,
    data_dir: Path | None = None,
    ret_column: str = "ret_5d",
    fresh_half_life_days: float = DEFAULT_FRESH_HALF_LIFE_DAYS,
    drop_sparse_financial_cols: bool = True,
) -> pd.DataFrame:
    """并入财务量价列（``FINANCIAL_FEATURE_COLUMNS``）。

    与三道「防火墙」对齐：仅 **公告日** 决定财报是否可见；**T 日**行只用 **T−1 及以前** 的公告与 **T−1 收盘**意义下的 5 日收益再做错配秩；截面排名按 **日** 切分。
    """
    base = data_dir if data_dir is not None else DATA_DIR
    path_default = base / "financial_quarter_bs.parquet"
    resolved = fin_path if fin_path is not None else path_default

    need = {"date", "stock_code", ret_column}
    miss = need - set(panel.columns)
    if miss:
        raise ValueError(f"panel 缺少列: {miss}")

    if financial_df is None:
        if not resolved.exists():
            out = panel.copy()
            out["date"] = pd.to_datetime(out["date"])
            return _ensure_financial_nan_frame(out)
        financial_df = load_financial_quarter_bs(path=resolved, drop_sparse_cols=drop_sparse_financial_cols)

    fin = _prepare_financial_quarter(financial_df)

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])

    fin_by_code: dict[str, pd.DataFrame] = {}
    for code, grp in fin.groupby("stock_code", sort=False):
        fin_by_code[str(code)] = grp

    trad = sorted(pd.DatetimeIndex(pd.to_datetime(out["date"]).unique()))
    prev_trade: dict[pd.Timestamp, pd.Timestamp] = {}
    for j in range(1, len(trad)):
        prev_trade[pd.Timestamp(trad[j]).normalize()] = pd.Timestamp(trad[j - 1]).normalize()

    n = len(out)
    delta_arr = np.full(n, np.nan, dtype=np.float64)
    accel_arr = np.full(n, np.nan, dtype=np.float64)
    notice_objs: list = []
    has_arr = np.zeros(n, dtype=bool)

    for i in range(n):
        t_curr = pd.Timestamp(out.iloc[i]["date"]).normalize()
        code = str(out.iloc[i]["stock_code"]).zfill(6)
        grp = fin_by_code.get(code)
        cutoff_t_minus_1 = prev_trade.get(t_curr)
        if grp is None or grp.empty:
            notice_objs.append(pd.NaT)
            continue
        if cutoff_t_minus_1 is None:
            notice_objs.append(pd.NaT)
            continue
        delta, accel, nt, ok = _triple_delta_accel_notice(grp, cutoff_t_minus_1)
        delta_arr[i] = delta
        accel_arr[i] = accel
        has_arr[i] = ok
        notice_objs.append(pd.Timestamp(nt) if nt is not None else pd.NaT)

    out["financial_delta"] = delta_arr
    out["financial_accel"] = accel_arr
    out["financial_has_triple"] = has_arr.astype(np.int8)
    out["_notice_for_latest"] = notice_objs

    out["financial_r_delta"] = out.groupby("date")["financial_delta"].rank(method="average", pct=True)
    out["financial_r_accel"] = out.groupby("date")["financial_accel"].rank(method="average", pct=True)
    out["financial_inner"] = out["financial_r_delta"] + out["financial_r_accel"]

    out[ret_column] = pd.to_numeric(out[ret_column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    keyed = out.assign(__ROW=np.arange(len(out))).sort_values(["stock_code", "date"])
    keyed["__rp"] = keyed.groupby("stock_code", sort=False)[ret_column].shift(1)
    keyed = keyed.sort_values("__ROW")
    out["__ret_t_minus_1"] = keyed["__rp"].to_numpy(dtype=float, copy=False)

    out["financial_inner_rank"] = out.groupby("date")["financial_inner"].rank(method="average", pct=True)
    out["financial_ret_5d_rank"] = out.groupby("date")["__ret_t_minus_1"].rank(method="average", pct=True)
    out["financial_gap"] = out["financial_inner_rank"] - out["financial_ret_5d_rank"]

    asof_dates = pd.to_datetime(out["date"]).dt.normalize()
    notices_ser = pd.to_datetime(out["_notice_for_latest"], errors="coerce")
    delta_days = (asof_dates - notices_ser.dt.normalize()).dt.days.astype(float)

    hl = float(fresh_half_life_days)
    if hl <= 0:
        raise ValueError("fresh_half_life_days must be positive")
    out["financial_fresh"] = np.exp(-np.maximum(delta_days, 0.0) / hl)
    out.loc[notices_ser.isna(), "financial_fresh"] = np.nan
    out["financial_signal"] = out["financial_gap"] * out["financial_fresh"]

    bad_sig = (~out["financial_has_triple"].astype(bool)) | out["financial_gap"].isna()
    out.loc[bad_sig, "financial_signal"] = np.nan

    out["financial_notice_date"] = pd.to_datetime(out["_notice_for_latest"], errors="coerce")
    out.drop(columns=["_notice_for_latest", "__ret_t_minus_1"], inplace=True, errors="ignore")

    return out


def strip_financial_quarter_bs_parquet(
    path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """读入 parquet，去掉 ``FINANCIAL_BS_DROP_COLUMNS`` 后写回。"""
    p = path or DATA_DIR / "financial_quarter_bs.parquet"
    dst = out_path or p
    df = pd.read_parquet(p)
    drop = [c for c in FINANCIAL_BS_DROP_COLUMNS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    df.to_parquet(dst, index=False)
    return dst
