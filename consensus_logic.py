"""
Analyst 「预期结构」特征：源于 ``data/profit_forecast_csi500.parquet`` 截面快照。

设计原则（与课程设计规范一致）：

* **不搞单一合成强绑定**：不再强行 ``growth × consensus × sqrt(coverage)`` 作为主要信号。
* **正交基底**：growth / consensus / coverage（log 研报）分开进模型。
* **结构层**：横截面 ``mispricing``（盈利预期 vs 一致预期的秩差）；``dispersion``
  （买/增持/中性分布的 Shannon 熵，分歧）；``imbalance``（买−中性 相对总量的偏置）。
* **弱交互**：仅提供乘积列，交给树模型拆分，不设唯一打分函数。
* **coverage**：作信息密度闸门（``has_coverage`` + ``analyst_coverage``），不作强度放大因子。
* **legacy**：旧的乘积记分保留 ``analyst_legacy_score``（**不进 FEATURE_COLUMNS**，仅排查用）。

快照无公告日时，按股在日频面板上常量广播。
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

COL_BUY: Final = "机构投资评级(近六个月)-买入"
COL_ADD: Final = "机构投资评级(近六个月)-增持"
COL_NEU: Final = "机构投资评级(近六个月)-中性"
COL_RED: Final = "机构投资评级(近六个月)-减持"
COL_SELL: Final = "机构投资评级(近六个月)-卖出"
COL_EPS_2025: Final = "2025预测每股收益"
COL_EPS_2026: Final = "2026预测每股收益"
COL_N_REPORT: Final = "研报数"
COL_CODE: Final = "stock_code"

# 送进 XGBoost 的 analyst 块（不把 legacy 算进去）
ANALYST_FEATURE_COLUMNS: tuple[str, ...] = (
    "has_coverage",
    "analyst_growth",
    "analyst_consensus",
    "analyst_coverage",
    "analyst_dispersion",
    "analyst_imbalance",
    "analyst_mispricing",
    "analyst_ix_growth_consensus",
    "analyst_ix_growth_coverage",
    "analyst_ix_consensus_coverage",
)

_RATING_COLS_CHECK: tuple[str, ...] = (
    COL_BUY,
    COL_ADD,
    COL_NEU,
    COL_RED,
    COL_SELL,
)


def _drop_all_zero_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """整列恒为 0（NaN 先当 0）则物理删除."""
    out = df
    to_drop: list[str] = []
    for c in columns:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        if (s == 0).all():
            to_drop.append(c)
    return out.drop(columns=to_drop) if to_drop else out


def _entropy_3_probs(p_buy: np.ndarray, p_add: np.ndarray, p_neu: np.ndarray) -> np.ndarray:
    """三分类分布熵（买/增持/中性归一）；分母≤0→NaN。"""
    s = p_buy + p_add + p_neu
    out = np.full(len(p_buy), np.nan, dtype=float)
    ok = s > 1e-12
    bb = np.zeros_like(p_buy)
    aa = bb.copy()
    nn = bb.copy()
    bb[ok] = p_buy[ok] / s[ok]
    aa[ok] = p_add[ok] / s[ok]
    nn[ok] = p_neu[ok] / s[ok]
    probs = np.stack([np.clip(bb, 1e-12, 1.0), np.clip(aa, 1e-12, 1.0), np.clip(nn, 1e-12, 1.0)])
    ln = np.log(probs)
    ent = -(probs * ln).sum(axis=0)
    out[ok] = ent[ok]
    return out


def _build_stock_table(fc: pd.DataFrame) -> pd.DataFrame:
    df = _drop_all_zero_columns(fc.copy(), _RATING_COLS_CHECK)
    df[COL_CODE] = df[COL_CODE].astype(str).str.zfill(6)

    buy = pd.to_numeric(df.get(COL_BUY, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    add_ = pd.to_numeric(df.get(COL_ADD, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    neu = pd.to_numeric(df.get(COL_NEU, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    denom = buy + add_ + neu
    consensus = np.where(denom > 1e-12, (buy + 0.5 * add_) / denom, np.nan)

    eps25 = pd.to_numeric(df.get(COL_EPS_2025), errors="coerce").to_numpy(dtype=float)
    eps26 = pd.to_numeric(df.get(COL_EPS_2026), errors="coerce").to_numpy(dtype=float)
    growth = (eps26 - eps25) / (np.abs(eps25) + 0.1)

    n_rep = pd.to_numeric(df.get(COL_N_REPORT, 0), errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    cov_log = np.log1p(n_rep)

    dispersion = _entropy_3_probs(buy, add_, neu)
    imb = np.where(denom > 1e-12, (buy - neu) / denom, np.nan)

    has_cov = (n_rep > 0.0).astype(np.int8)
    coverage_s = np.where(has_cov.astype(bool), cov_log, np.nan)

    ok_ls = (
        has_cov.astype(bool)
        & np.isfinite(growth)
        & np.isfinite(consensus)
        & np.isfinite(cov_log)
        & (cov_log > 0.0)
    )
    legacy = np.full(len(df), np.nan, dtype=float)
    legacy[ok_ls] = growth[ok_ls] * consensus[ok_ls] * np.sqrt(cov_log[ok_ls])

    return pd.DataFrame(
        {
            COL_CODE: df[COL_CODE].values,
            "has_coverage": has_cov,
            "analyst_growth": growth,
            "analyst_consensus": consensus,
            "analyst_coverage": coverage_s,
            "analyst_dispersion": dispersion,
            "analyst_imbalance": imb,
            "analyst_legacy_score": legacy,
        }
    ).drop_duplicates(subset=[COL_CODE], keep="last")


def _ensure_analyst_nan_frame(out: pd.DataFrame, sc: pd.Series) -> pd.DataFrame:
    """对齐列：missing file / empty merger."""
    n = len(sc)
    out["has_coverage"] = np.zeros(n, dtype=np.int8)
    for c in (
        "analyst_growth",
        "analyst_consensus",
        "analyst_coverage",
        "analyst_dispersion",
        "analyst_imbalance",
        "analyst_mispricing",
        "analyst_ix_growth_consensus",
        "analyst_ix_growth_coverage",
        "analyst_ix_consensus_coverage",
        "analyst_legacy_score",
    ):
        out[c] = np.nan
    return out


def _interaction_block(growth: pd.Series, consensus: pd.Series, coverage: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    return growth * consensus, growth * coverage, consensus * coverage


def merge_consensus_logic(
    panel: pd.DataFrame,
    forecast_path: Path | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    base = data_dir if data_dir is not None else Path(__file__).parent / "data"
    path = forecast_path if forecast_path is not None else base / "profit_forecast_csi500.parquet"

    out = panel.copy()
    sc = out["stock_code"].astype(str).str.zfill(6)

    def _finalize_frame(df: pd.DataFrame) -> pd.DataFrame:
        # 截面 mispricing ≈ Δ rank(growth) − rank(consensus)
        rk_g = df.groupby("date")["analyst_growth"].rank(method="average", pct=True)
        rk_c = df.groupby("date")["analyst_consensus"].rank(method="average", pct=True)
        df["analyst_mispricing"] = rk_g - rk_c

        g = pd.to_numeric(df["analyst_growth"], errors="coerce")
        co = pd.to_numeric(df["analyst_consensus"], errors="coerce")
        cv = pd.to_numeric(df["analyst_coverage"], errors="coerce")
        gx, gx2, gx3 = _interaction_block(g, co, cv)
        df["analyst_ix_growth_consensus"] = gx
        df["analyst_ix_growth_coverage"] = gx2
        df["analyst_ix_consensus_coverage"] = gx3
        return df

    if not path.exists():
        _ensure_analyst_nan_frame(out, sc)
        return _finalize_frame(out)

    fc = pd.read_parquet(path)
    if fc.empty:
        _ensure_analyst_nan_frame(out, sc)
        return _finalize_frame(out)

    if COL_CODE not in fc.columns and "代码" in fc.columns:
        fc = fc.rename(columns={"代码": COL_CODE})
    if COL_CODE not in fc.columns:
        _ensure_analyst_nan_frame(out, sc)
        return _finalize_frame(out)

    tab = _build_stock_table(fc)
    idx = tab.set_index(COL_CODE)

    out["has_coverage"] = sc.map(idx["has_coverage"]).fillna(0).astype(np.int8)

    merge_cols_rest = (
        "analyst_growth",
        "analyst_consensus",
        "analyst_coverage",
        "analyst_dispersion",
        "analyst_imbalance",
        "analyst_legacy_score",
    )
    for c in merge_cols_rest:
        out[c] = sc.map(idx[c])

    return _finalize_frame(out)


# backwards-compatible alias
LOGIC_FEATURE_COLUMNS = ANALYST_FEATURE_COLUMNS
