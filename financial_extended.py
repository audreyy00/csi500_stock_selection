"""
财务外延：共识对撞 / 行业内 Z，以及「事件驱动」财报冲击（sparse trigger + 量价错定价）。

须在 ``merge_financial_features`` **之后**；面板需含 ``ret_3d``、``ret_3d_rank``（三日收益及截面秩）。

事件列语义（短周期对齐）::
* ``financial_evt_shock``: \( \Delta Q \cdot Active \)，\(\Delta Q\) 即 ``financial_delta``（季间变化）。
* ``ret_3d`` / ``ret_3d_rank``：三日收益对齐 **截止 T−1 收盘** 的已实现收益（每股 ``pct_change(3).shift(1)``），与财报侧的 ``ret_5d`` 滞后一致，避免用 T 当日收盘造成前视。
* ``financial_evt_mispricing_rankdiff``：先做 \(\mathrm{rank}^{pct}(\text{Shock})-\mathrm{rank}^{pct}(R^{(3)})\)，
  再 \(S=\mathrm{sign}(d)\,\max(|d|-\delta,\,0)^2\)（\(\delta\) = ``DEFAULT_RANKDIFF_DEADBAND``），最后乘 ``Active``。
* ``financial_evt_mispricing_aggr``：激进式 \(\text{Shock}\cdot(1-R^{(3)}/\max|R^{(3)}|)\) 于 Active 内，再显式乘门控。
* 另见 ``financial_days_since_notice``、``financial_fresh_burst_3d``、``financial_underreaction``。
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

DEFAULT_INDUSTRY_FILE = "stock_industry_bs.parquet"
# 公告滞后反应：略长于 3 日历日；行业内样本不足则 industry_z 不可靠。
DEFAULT_EVENT_GATE_CALENDAR_DAYS: Final[int] = 5
DEFAULT_MIN_NAMES_PER_INDUSTRY_DAY: Final[int] = 17
# 与用户提出的「发布后极短窗口」对齐：Burst 阈值（日历日）；可与 evt 窗分拆调参。
DEFAULT_FRESH_BURST_CALENDAR_DAYS: Final[int] = 3


def _mispricing_rankdiff_active_universe(
    panel: pd.DataFrame,
    *,
    shock_col: str,
    ret_col: str,
    gate_col: str,
) -> pd.Series:
    """Within each calendar date and among rows with ``gate_col > 0.5``: rank pct shock − rank pct ret."""
    out = pd.Series(np.nan, index=panel.index, dtype=np.float64)
    for _, sub in panel.groupby("date", sort=False):
        act = pd.to_numeric(sub[gate_col], errors="coerce").fillna(0.0).to_numpy() > 0.5
        if not act.any():
            continue
        g = sub.loc[act]
        shock = pd.to_numeric(g[shock_col], errors="coerce")
        ret = pd.to_numeric(g[ret_col], errors="coerce")
        if len(g) == 1:
            out.loc[g.index] = 0.0
            continue
        rk_s = shock.rank(method="average", pct=True)
        rk_rr = ret.rank(method="average", pct=True)
        out.loc[g.index] = (rk_s - rk_rr).to_numpy(dtype=float, copy=False)
    return out


def _mispricing_aggressive_active_universe(
    panel: pd.DataFrame,
    *,
    shock_col: str,
    ret_col: str,
    gate_col: str,
) -> pd.Series:
    """ShockSignal × (1 − R^{(3)} / max|R^{(3)}|) on active subset; max over same day active rows."""
    out = pd.Series(np.nan, index=panel.index, dtype=np.float64)
    for _, sub in panel.groupby("date", sort=False):
        act = pd.to_numeric(sub[gate_col], errors="coerce").fillna(0.0).to_numpy() > 0.5
        if not act.any():
            continue
        g = sub.loc[act]
        sh = pd.to_numeric(g[shock_col], errors="coerce").to_numpy(dtype=float, copy=False)
        r3 = pd.to_numeric(g[ret_col], errors="coerce").to_numpy(dtype=float, copy=False)
        mx = np.nanmax(np.abs(r3))
        if not np.isfinite(mx) or mx < 1e-12:
            scale = np.ones(len(g), dtype=float)
        else:
            scale = 1.0 - (r3 / mx)
        out.loc[g.index] = sh * scale
    return out


def _mispricing_signed_deadband_square(
    diff: np.ndarray | pd.Series,
    *,
    deadband: float,
) -> np.ndarray:
    """
    \\(S = \\mathrm{sign}(\\mathrm{diff}) \\cdot \\max(|\\mathrm{diff}|-\\mathrm{deadband},\\,0)^2\\)。
    percentile-rank 差上的小噪声压低为 0，强分歧平方放大并保持方向；NaN 保留。
    """
    d = np.asarray(diff, dtype=float)
    out = np.empty_like(d, dtype=float)
    nan_m = np.isnan(d)
    excess = np.maximum(np.abs(d) - deadband, 0.0)
    s = np.sign(d) * (excess**2)
    out[:] = np.where(nan_m, np.nan, s)
    return out


DEFAULT_RANKDIFF_DEADBAND: Final[float] = 0.2


FINANCIAL_EXT_FEATURE_COLUMNS: tuple[str, ...] = (
    "financial_surprise_vs_analyst",
    "financial_evt_gate",
    "financial_accel_industry_z",
    "financial_evt_shock",
    "financial_evt_mispricing_rankdiff",
    "financial_evt_mispricing_aggr",
    "financial_days_since_notice",
    "financial_fresh_burst_3d",
    "financial_underreaction",
)


def merge_financial_extended_features(
    panel: pd.DataFrame,
    *,
    data_dir: Path | None = None,
    industry_parquet_name: str = DEFAULT_INDUSTRY_FILE,
    event_gate_calendar_days: int = DEFAULT_EVENT_GATE_CALENDAR_DAYS,
    min_names_per_industry_day: int = DEFAULT_MIN_NAMES_PER_INDUSTRY_DAY,
    fresh_burst_calendar_days: int = DEFAULT_FRESH_BURST_CALENDAR_DAYS,
) -> pd.DataFrame:
    """并入对撞 / 行业 Z / **事件财报**（冲击×窗×价秩）。"""
    base = data_dir if data_dir is not None else Path(__file__).parent / "data"
    out = panel.copy()

    need_f = {
        "financial_r_delta",
        "financial_accel",
        "financial_delta",
        "financial_notice_date",
        "ret_3d",
        "ret_3d_rank",
    }
    need_a = {"analyst_growth"}
    miss = need_f.union(need_a) - set(out.columns)
    if miss:
        raise ValueError(f"merge_financial_extended_features: missing columns {sorted(miss)}")

    rk_a = out.groupby("date")["analyst_growth"].rank(method="average", pct=True)
    out["financial_surprise_vs_analyst"] = pd.to_numeric(out["financial_r_delta"], errors="coerce").sub(
        rk_a
    )

    dt = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    nt = pd.to_datetime(out["financial_notice_date"], errors="coerce").dt.normalize()
    delta_cd = (dt - nt).dt.days.astype(float)
    ok_win = (~nt.isna()) & (delta_cd >= 0) & (delta_cd <= float(event_gate_calendar_days))
    evt_g = np.where(nt.isna(), 0.0, ok_win.astype(np.float64))
    out["financial_evt_gate"] = evt_g

    # --- 事件驱动财报（sparse / trigger），与三日价绑定的错定价 ---
    dsc = pd.Series(delta_cd, index=out.index, dtype=float)
    out["financial_days_since_notice"] = dsc.where(nt.notna() & (dsc >= 0))
    bk = float(fresh_burst_calendar_days)
    out["financial_fresh_burst_3d"] = (
        (~nt.isna())
        & (delta_cd >= 0.0)
        & (delta_cd <= bk)
    ).astype(np.float64)
    fd = pd.to_numeric(out["financial_delta"], errors="coerce").to_numpy(dtype=float, copy=False)
    fd[np.isnan(fd)] = 0.0
    out["financial_evt_shock"] = fd * evt_g
    rk_r = pd.to_numeric(out["ret_3d_rank"], errors="coerce").to_numpy(dtype=float, copy=False)
    rk_f = pd.to_numeric(out["financial_r_delta"], errors="coerce").to_numpy(dtype=float, copy=False)
    out["financial_underreaction"] = np.where(
        np.isnan(rk_r),
        np.nan,
        rk_f * (1.0 - rk_r) * evt_g,
    )

    rk_raw = _mispricing_rankdiff_active_universe(
        out, shock_col="financial_evt_shock", ret_col="ret_3d", gate_col="financial_evt_gate"
    ).to_numpy(dtype=float, copy=False)
    rk_s = _mispricing_signed_deadband_square(rk_raw, deadband=float(DEFAULT_RANKDIFF_DEADBAND))
    ag_raw = _mispricing_aggressive_active_universe(
        out, shock_col="financial_evt_shock", ret_col="ret_3d", gate_col="financial_evt_gate"
    ).to_numpy(dtype=float, copy=False)
    # Alpha = (...) × Active；门控外强制 0（与公式一致）。
    evt_mask = evt_g.astype(float) > 0.5
    out["financial_evt_mispricing_rankdiff"] = np.where(
        evt_mask, np.nan_to_num(rk_s, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    out["financial_evt_mispricing_aggr"] = np.where(
        evt_mask, np.nan_to_num(ag_raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )

    ind_path = base / industry_parquet_name
    if ind_path.exists():
        ind = pd.read_parquet(ind_path)
        ind["stock_code"] = ind["stock_code"].astype(str).str.zfill(6)
        if "industry_name" not in ind.columns:
            raise KeyError(f"{ind_path} lacks industry_name column")
        imap = (
            ind.sort_values("stock_code")
            .drop_duplicates(subset=["stock_code"], keep="last")[["stock_code", "industry_name"]]
            .rename(columns={"industry_name": "_fin_industry"})
        )
        m = out.merge(imap, on="stock_code", how="left")
        lab = m["_fin_industry"].where(m["_fin_industry"].notna(), np.nan)
        gx = pd.to_numeric(m["financial_accel"], errors="coerce")
        dnorm = pd.to_datetime(m["date"], errors="coerce").dt.normalize()
        keys = [dnorm, lab]

        def _mad(series: pd.Series) -> float:
            v = np.asarray(series.dropna(), dtype=float)
            if v.size == 0:
                return float("nan")
            med_loc = np.median(v)
            return float(np.median(np.abs(v - med_loc)))

        med = gx.groupby(keys, dropna=False).transform("median")
        mad = gx.groupby(keys, dropna=False).transform(_mad)
        cnt = gx.groupby(keys, dropna=False).transform("count")
        denom = pd.to_numeric(mad, errors="coerce").replace(0.0, np.nan) + 1e-9
        raw_z = (gx - med) / denom
        bad = (cnt < float(min_names_per_industry_day)) | lab.isna()
        out["financial_accel_industry_z"] = raw_z.mask(bad).to_numpy(dtype=float)
    else:
        out["financial_accel_industry_z"] = np.nan

    if "financial_notice_date" in out.columns:
        out = out.drop(columns=["financial_notice_date"])

    return out
