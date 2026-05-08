"""
Feature engineering for the CSI500 stock-selection baseline.

A small set of classic technical features + cross-sectional ranks.  Students are
encouraged to extend this (add fundamentals, industry dummies, alternative data,
better cross-sectional normalization, etc.).

The target is the 5-trading-day forward return on the forward-adjusted close,
i.e. what the portfolio earns if you hold a $1 position from close(t) to close(t+5).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from consensus_logic import ANALYST_FEATURE_COLUMNS, merge_consensus_logic
from financial_analysis import (
    DEFAULT_FRESH_HALF_LIFE_DAYS,
    FINANCIAL_FEATURE_COLUMNS,
    merge_financial_features,
)
from financial_extended import (
    DEFAULT_EVENT_GATE_CALENDAR_DAYS,
    DEFAULT_FRESH_BURST_CALENDAR_DAYS,
    DEFAULT_MIN_NAMES_PER_INDUSTRY_DAY,
    FINANCIAL_EXT_FEATURE_COLUMNS,
    merge_financial_extended_features,
)
from regime import REGIME_FEATURE_COLUMNS, merge_regime_features

DATA_DIR = Path(__file__).parent / "data"

# 干净 CSI500 数据（由 ``download_data.py`` 写入）；训练脚本应只读这些路径并做成分过滤。
PRICES_CSI500_PARQUET = DATA_DIR / "prices_csi500.parquet"
INDEX_CSI500_PARQUET = DATA_DIR / "index_csi500.parquet"
CONSTITUENTS_CSI500_CSV = DATA_DIR / "constituents_csi500.csv"

# ``alpha_searchlight.py`` 产出的斜率 + 加速度 + 斜率截面分位秩（见 ``panel_alpha_searchlight.parquet``）。
# 与 ``prepare_atoms`` 中机制原子一一对应（未过挖掘阈值的列合并后为 NaN）。
SEARCHLIGHT_SLOPE_COLUMNS: tuple[str, ...] = (
    # L1
    "alpha_slope_atom_ret_5d_accel",
    "alpha_slope_atom_speed_bias",
    "alpha_slope_atom_pv_corr_xsr",
    # L2
    "alpha_slope_atom_vol_squeeze",
    "alpha_slope_atom_avg_trade_amt",
    "alpha_slope_atom_pos",
    # L3
    "alpha_slope_atom_mech_breakout",
    "alpha_slope_atom_pullback_dry",
    "alpha_slope_atom_ret_vol_align",
    "alpha_slope_atom_rank_div_ret_vol",
    # L4
    "alpha_slope_atom_timing_trigger",
    "alpha_slope_atom_convex_vol",
    "alpha_slope_atom_cs_pressure2",
    "alpha_slope_atom_cs_pressure3",
)

SEARCHLIGHT_ACCEL_COLUMNS: tuple[str, ...] = tuple(
    f"alpha_accel_{c.removeprefix('alpha_slope_')}" for c in SEARCHLIGHT_SLOPE_COLUMNS
)
SEARCHLIGHT_SLOPE_RANK_COLUMNS: tuple[str, ...] = tuple(
    f"{c}_rank" for c in SEARCHLIGHT_SLOPE_COLUMNS
)
SEARCHLIGHT_COLUMNS: tuple[str, ...] = (
    SEARCHLIGHT_SLOPE_COLUMNS + SEARCHLIGHT_ACCEL_COLUMNS + SEARCHLIGHT_SLOPE_RANK_COLUMNS
)

# Analyst 分量 / 结构上允许 NaN；has_coverage 恒为 0/1。
# 财务块在 ``_neutral_fill_financial_panel`` 后训练路径上通常为有限值；
# ``financial_evt_gate`` 仍为 0/1 必选列。（历史：遗漏 EXT 会把 surprise / industry_z
# 并进 ``_required_training_columns``，配合 dropna 误杀绝大多数交易日——已修复）。
FEATURES_NA_OK = (
    frozenset(ANALYST_FEATURE_COLUMNS) - frozenset({"has_coverage"})
) | (
    frozenset(FINANCIAL_FEATURE_COLUMNS) - frozenset({"financial_has_triple"})
) | (
    frozenset(FINANCIAL_EXT_FEATURE_COLUMNS) - frozenset({"financial_evt_gate"})
) | frozenset({"financial_days_since_notice"}) | (
    frozenset(REGIME_FEATURE_COLUMNS) - frozenset({"regime_bucket"})
) | frozenset(SEARCHLIGHT_COLUMNS)

# columns passed to XGBoost（技术 + consensus + regime + …）

FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "vol_20d", "volume_z_20d", "turnover_ma_20d",
    "close_over_ma20", "close_over_ma60", "rsi_14",
    "ret_3d_rank", "ret_5d_rank", "ret_20d_rank", "vol_20d_rank",
    # *ANALYST_FEATURE_COLUMNS,
    *REGIME_FEATURE_COLUMNS,
    *SEARCHLIGHT_COLUMNS,
    # *FINANCIAL_FEATURE_COLUMNS,
    # *FINANCIAL_EXT_FEATURE_COLUMNS,
]
TARGET_COLUMN = "target_5d"
FORWARD_HORIZON = 5

# 截面分位与 Top-K 二分类（相对强弱；非特征列）
LABEL_CS_RANK_PCT = "target_rank_pct"
LABEL_TOPK_BINARY = "target_topk_bin"


def attach_cross_sectional_rank_labels(
    panel: pd.DataFrame,
    *,
    top_pct_threshold: float = 0.9,
    ret_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """按交易日截面 rank(pct)；``target_topk_bin``：分位 ≥ threshold 为 1。"""
    out = panel.copy()
    rk = out.groupby("date", sort=False)[ret_col].rank(method="average", pct=True)
    out[LABEL_CS_RANK_PCT] = rk
    tb = (rk >= float(top_pct_threshold)).astype(np.float64)
    out[LABEL_TOPK_BINARY] = tb.where(rk.notna(), np.nan)
    return out


def filter_prices_to_csi500_constituents(
    prices: pd.DataFrame,
    *,
    constituents_csv: Path | None = None,
) -> pd.DataFrame:
    """只保留 ``constituents_csi500.csv`` 中的代码（剔除合并进来的非 CSI500 等）。"""
    csv_path = Path(constituents_csv) if constituents_csv is not None else CONSTITUENTS_CSI500_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"限定 CSI500 训练需要成分表 {csv_path}。请先运行: python download_data.py"
        )
    cons = pd.read_csv(csv_path, dtype={"stock_code": str})
    allow = set(cons["stock_code"].astype(str).str.zfill(6))
    out = prices.copy()
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
    n0, r0 = out["stock_code"].nunique(), len(out)
    out = out[out["stock_code"].isin(allow)].copy()
    n1, r1 = out["stock_code"].nunique(), len(out)
    if n0 != n1 or r0 != r1:
        print(
            f">> CSI500 成分过滤（{csv_path.name}）: "
            f"股票 {n0}→{n1}，行 {r0:,}→{r1:,}"
        )
    return out


def _replace_inf_nan_features(df: pd.DataFrame, cols: list[str]) -> None:
    """XGBoost 不允许特征含 ±inf（NaN 可作缺失）；原地写回。"""
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        df[c] = s.mask(np.isinf(s))


def _neutral_fill_financial_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """无财报可用的位置：**中性**补缺（截面 rank→0.5；增速差、外延对撞/Z 等→0）。

    语义是「无信息 = 中立」，避免财报缺数连累技术面 / 共识行被 ``dropna`` 误删。
    ``financial_has_triple`` **不在此列改**：仍标示是否已形成可用三连季；与补齐搭配后，
    平时可走技术面与分析师主导，有待披露三连季时再走真实分项。

    重算 ``financial_inner``、``financial_gap``、``financial_signal``；``financial_fresh`` 缺失视为 0。
    「事件」外延列（``financial_evt_shock`` / ``financial_fresh_burst_3d``）无窗时为 0；
    ``financial_days_since_notice`` 保持 NaN=无对齐公告语义；其余事件列补缺为 0。
    """
    out = panel.copy()

    pct_rank_cols = (
        "financial_r_delta",
        "financial_r_accel",
        "financial_inner_rank",
        "financial_ret_5d_rank",
    )
    for c in pct_rank_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.5)

    for c in ("financial_delta", "financial_accel"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    if {"financial_r_delta", "financial_r_accel"}.issubset(out.columns):
        out["financial_inner"] = out["financial_r_delta"] + out["financial_r_accel"]

    if {"financial_inner_rank", "financial_ret_5d_rank"}.issubset(out.columns):
        out["financial_gap"] = out["financial_inner_rank"] - out["financial_ret_5d_rank"]
    elif "financial_gap" in out.columns:
        out["financial_gap"] = pd.to_numeric(out["financial_gap"], errors="coerce").fillna(0.0)

    if "financial_fresh" in out.columns:
        out["financial_fresh"] = pd.to_numeric(out["financial_fresh"], errors="coerce").fillna(0.0)

    if {"financial_gap", "financial_fresh"}.issubset(out.columns):
        out["financial_signal"] = out["financial_gap"] * out["financial_fresh"]
    elif "financial_signal" in out.columns:
        out["financial_signal"] = pd.to_numeric(out["financial_signal"], errors="coerce").fillna(0.0)

    if "financial_surprise_vs_analyst" in out.columns:
        out["financial_surprise_vs_analyst"] = pd.to_numeric(
            out["financial_surprise_vs_analyst"], errors="coerce"
        ).fillna(0.0)
    if "financial_evt_shock" in out.columns:
        out["financial_evt_shock"] = pd.to_numeric(out["financial_evt_shock"], errors="coerce").fillna(0.0)
    if "financial_fresh_burst_3d" in out.columns:
        out["financial_fresh_burst_3d"] = pd.to_numeric(
            out["financial_fresh_burst_3d"], errors="coerce"
        ).fillna(0.0)
    if "financial_underreaction" in out.columns:
        out["financial_underreaction"] = pd.to_numeric(
            out["financial_underreaction"], errors="coerce"
        ).fillna(0.0)
    if "financial_evt_mispricing_rankdiff" in out.columns:
        out["financial_evt_mispricing_rankdiff"] = pd.to_numeric(
            out["financial_evt_mispricing_rankdiff"], errors="coerce"
        ).fillna(0.0)
    if "financial_evt_mispricing_aggr" in out.columns:
        out["financial_evt_mispricing_aggr"] = pd.to_numeric(
            out["financial_evt_mispricing_aggr"], errors="coerce"
        ).fillna(0.0)

    return out


def _required_training_columns() -> list[str]:
    return [c for c in FEATURE_COLUMNS if c not in FEATURES_NA_OK]


def _rolling_max_dd_over_window(close_window: np.ndarray) -> float:
    """给定按时间排序的收盘价窗口，返回 (min_i (P/min_{j<=i} P_j)-1) ≤ 0 的最大回撤（负值）."""
    if close_window.ndim != 1 or close_window.size < 3:
        return float("nan")
    a = close_window.astype(float, copy=False)
    if np.any(~np.isfinite(a)):
        return float("nan")
    mx = np.maximum.accumulate(a)
    dd = (a - mx) / np.clip(mx, 1e-15, np.inf)
    return float(dd.min())


def _append_defensive_risk_stock_features(df: pd.DataFrame) -> None:
    """原地增加风险扩展列（``ret_1d`` 已存在）；仅写入面板，不进 ``FEATURE_COLUMNS``。"""
    r = pd.to_numeric(df["ret_1d"], errors="coerce")
    down_clip = r.clip(upper=0.0)
    df["down_vol_20d"] = down_clip.rolling(20, min_periods=15).std()
    cc = pd.to_numeric(df["close"], errors="coerce")
    df["max_dd_20d"] = cc.rolling(20, min_periods=15).apply(_rolling_max_dd_over_window, raw=True)
    df["vol_60d"] = r.rolling(60, min_periods=48).std()
    cl = cc.clip(lower=np.finfo(float).tiny)
    if {"high", "low"}.issubset(df.columns):
        hi = pd.to_numeric(df["high"], errors="coerce")
        lo = pd.to_numeric(df["low"], errors="coerce")
        df["amp_20d"] = ((hi - lo) / cl).rolling(20, min_periods=15).mean()
    else:
        df["amp_20d"] = np.nan
    if "turnover" in df.columns:
        df["turnover_std_20d"] = (
            pd.to_numeric(df["turnover"], errors="coerce").rolling(20, min_periods=15).std()
        )
    else:
        df["turnover_std_20d"] = np.nan


def _attach_index_beta_to_panel(panel: pd.DataFrame, *, data_dir: Path) -> pd.DataFrame:
    """按日 merge 指数收益，再在每只股票上 rolling ``beta_20d``（cov/var）。"""
    ix_path = data_dir / "index_csi500.parquet"
    if not ix_path.is_file():
        out = panel.copy()
        out["beta_20d"] = np.nan
        return out
    ix = pd.read_parquet(ix_path).copy()
    ix["date"] = pd.to_datetime(ix["date"])
    ix["idx_ret"] = pd.to_numeric(ix["close"], errors="coerce").pct_change()
    m = panel.merge(ix[["date", "idx_ret"]], on="date", how="left")

    def _beta_one(g: pd.DataFrame) -> pd.DataFrame:
        out_g = g.copy()
        rr = pd.to_numeric(out_g["ret_1d"], errors="coerce")
        ir = pd.to_numeric(out_g["idx_ret"], errors="coerce")
        cov_ri = rr.rolling(20, min_periods=15).cov(ir)
        var_i = ir.rolling(20, min_periods=15).var()
        out_g["beta_20d"] = cov_ri / var_i.replace(0, np.nan)
        return out_g

    gb = m.groupby("stock_code", group_keys=False, sort=False)
    out = gb.apply(_beta_one)
    return out.drop(columns=["idx_ret"], errors="ignore")


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that only depend on a single stock's time series."""
    df = df.sort_values("date").copy()
    close = df["close"]

    df["ret_1d"] = close.pct_change(1)
    # 对齐 T 日决策：\(R^{(3)}\) 为截至 **T−1 收盘** 的过去 3 个交易日收益（不经由 T 日 close）。
    df["ret_3d"] = close.pct_change(3).shift(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["ret_60d"] = close.pct_change(60)

    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    vol = df["volume"].astype(float)
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (vol - vol_mean) / vol_std

    if "turnover" in df.columns:
        df["turnover_ma_20d"] = df["turnover"].astype(float).rolling(20).mean()
    else:
        df["turnover_ma_20d"] = np.nan

    df["close_over_ma20"] = close / close.rolling(20).mean() - 1.0
    df["close_over_ma60"] = close / close.rolling(60).mean() - 1.0

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = up / down
    df["rsi_14"] = 100 - 100 / (1 + rs)

    _append_defensive_risk_stock_features(df)

    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return df


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional rank of selected features (values in [0, 1])."""
    # ret_3d 已 shift(1)；此处 rank 与事件错配中公式的 R^{(3)} 截面秩一致。
    for base in ["ret_3d", "ret_5d", "ret_20d", "vol_20d"]:
        panel[f"{base}_rank"] = (
            panel.groupby("date")[base].rank(method="average", pct=True)
        )
    return panel


def _merge_alpha_searchlight(panel: pd.DataFrame, *, data_dir: Path) -> pd.DataFrame:
    """从 ``panel_alpha_searchlight.parquet`` 并入 Searchlight 斜率、加速度与斜率截面分位秩列（若缺文件则告警并填空）。"""
    path = data_dir / "panel_alpha_searchlight.parquet"
    out = panel.copy()
    cols = list(SEARCHLIGHT_COLUMNS)
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"])

    if not path.is_file():
        warnings.warn(
            f"未找到 {path}，Searchlight 斜率/加速度/斜率秩列将为 NaN。生成方式: "
            f"`python alpha_searchlight.py --prices data/prices_csi500.parquet "
            f"--out-parquet {path}`",
            stacklevel=2,
        )
        for c in cols:
            out[c] = np.nan
        return out

    slab = pd.read_parquet(path)
    need = ["date", "stock_code", *cols]
    miss = set(need) - set(slab.columns)
    if miss:
        warnings.warn(f"{path} 缺少列 {sorted(miss)}，对应特征以 NaN 填充。", stacklevel=2)

    slab = slab.copy()
    slab["stock_code"] = slab["stock_code"].astype(str).str.zfill(6)
    slab["date"] = pd.to_datetime(slab["date"])
    for c in cols:
        if c not in slab.columns:
            slab[c] = np.nan
    keep = ["date", "stock_code"] + [c for c in cols if c in slab.columns]
    slab = slab[keep].drop_duplicates(subset=["date", "stock_code"], keep="last")

    out = out.merge(slab, on=["date", "stock_code"], how="left", validate="many_to_one")

    dup_suffix = [c for c in out.columns if c.endswith("_sx") or "_sx_" in c]
    if dup_suffix:
        out = out.drop(columns=dup_suffix, errors="ignore")
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def build_features(
    prices: pd.DataFrame,
    *,
    fresh_half_life_days: float = DEFAULT_FRESH_HALF_LIFE_DAYS,
    event_gate_calendar_days: int = DEFAULT_EVENT_GATE_CALENDAR_DAYS,
    min_names_per_industry_day: int = DEFAULT_MIN_NAMES_PER_INDUSTRY_DAY,
    fresh_burst_calendar_days: int = DEFAULT_FRESH_BURST_CALENDAR_DAYS,
) -> pd.DataFrame:
    """Build a (date, stock_code) panel of features + target.

    Parameters
    ----------
    prices : DataFrame with columns [date, stock_code, open, close, high, low,
             volume, amount, turnover?]
    fresh_half_life_days
        Passed to ``merge_financial_features``: 财报「新鲜」指数衰减尺度（交易日口径上的半衰）。
    event_gate_calendar_days, min_names_per_industry_day, fresh_burst_calendar_days
        Passed to ``merge_financial_extended_features``。

    Returns
    -------
    DataFrame with ``FEATURE_COLUMNS`` and ``TARGET_COLUMN`` populated；另含
    ``down_vol_20d`` / ``max_dd_20d`` / ``beta_20d`` / ``vol_60d`` / ``amp_20d`` /
    ``turnover_std_20d``（仅写入面板；由 ``train_defensive`` 自主选择是否使用）。
    Searchlight：``SEARCHLIGHT_COLUMNS`` 自 ``panel_alpha_searchlight.parquet`` 左并入。
    """
    required = {"date", "stock_code", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    chunks: list[pd.DataFrame] = []
    for _, sub in prices.groupby("stock_code", sort=False):
        chunks.append(_per_stock_features(sub))
    panel = pd.concat(chunks, ignore_index=True)
    panel = _attach_index_beta_to_panel(panel, data_dir=DATA_DIR)
    panel = _cross_sectional_ranks(panel)
    panel = merge_consensus_logic(panel, data_dir=DATA_DIR)
    panel = merge_regime_features(panel, prices)
    panel = merge_financial_features(
        panel,
        data_dir=DATA_DIR,
        fresh_half_life_days=fresh_half_life_days,
    )
    panel = merge_financial_extended_features(
        panel,
        data_dir=DATA_DIR,
        event_gate_calendar_days=event_gate_calendar_days,
        min_names_per_industry_day=min_names_per_industry_day,
        fresh_burst_calendar_days=fresh_burst_calendar_days,
    )
    panel = _neutral_fill_financial_panel(panel)
    panel = _merge_alpha_searchlight(panel, data_dir=DATA_DIR)
    return panel


def training_frame(
    panel: pd.DataFrame,
    min_date=None,
    max_date=None,
    *,
    label_column: str | None = None,
) -> pd.DataFrame:
    """监督行：特征齐全且标签列有限；``label_column`` 默认 ``TARGET_COLUMN``。"""
    target_col = label_column if label_column is not None else TARGET_COLUMN
    req = _required_training_columns() + [target_col]
    df = panel.dropna(subset=req).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    _replace_inf_nan_features(df, FEATURE_COLUMNS)
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Single-day inference rows aligned with ``training_frame`` feature availability."""
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    req = _required_training_columns()
    df = panel[panel["date"] == as_of].dropna(subset=req).copy()
    _replace_inf_nan_features(df, FEATURE_COLUMNS)
    return df
