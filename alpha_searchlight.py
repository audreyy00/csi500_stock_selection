"""
AlphaSearchlight：在给定牛股池上做「原子 → 爆发前轨迹 → 斜率因子」的离线挖掘脚手架。

与设计约定
-----------
* **原子 (atoms)**：`prepare_atoms()` 在每只股票的时间序列上计算基础量。
* **爆发日 t₀**：在每只股票历史中取某列的最大值对应的交易日（默认 ``past_ret_5d``，无未来函数）。
  若改用 ``target_5d``（远期收益峰值）会做**窥探未来**，只适合探索性对照，不推荐用于严谨的因果叙事。
* **轨迹**：爆发日前 ``window`` 日原子序列的 OLS 斜率；可汇总 **consistency** 或 **discriminative**
  （牛股轨迹斜率相对全市场同窗口滚动斜率分布）。
* **因子**：``alpha_slope_<atom>`` 与 **加速度** ``alpha_accel_<atom>``（斜率沿时间的一阶差分）。

依赖：pandas、numpy；默认读 ``data/prices_csi500.parquet``（与 baseline 一致）。

用法（项目根目录）
-----------------

.. code-block:: bash

    python alpha_searchlight.py \\
        --prices data/prices_csi500.parquet \\
        --mining discriminative --discriminative-min 0.5 \\
        --window 10 \\
        --out-parquet data/panel_alpha_searchlight.parquet

牛股池若在 CSI500 主库中不全：可用 ``download_extra_stocks.py`` 写入 ``prices_extended.parquet``，
再 ``--prices`` 指向该扩展库跑一次 Searchlight。

与 XGBoost 衔接：``features.SEARCHLIGHT_COLUMNS``（``alpha_slope_*`` / ``alpha_accel_*``）与 ``features.build_features`` merge 的列名对齐。
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from features import PRICES_CSI500_PARQUET

# ---------------------------------------------------------------------------
# 牛股池：首批 63 只（代码 6 位，读盘时 zfill）
# ---------------------------------------------------------------------------
SUPER_STOCKS_LIST: list[str] = [
    "603228",
    "301611",
    "600388",
    "600545",
    "301358",
    "600176",
    "002709",
    "600522",
    "600487",
    "300769",
    "002240",
    "002497",
    "301308",
    "601138",
    "003036",
    "688206",
    "603259",
    "603799",
    "300953",
    "300750",
    "300671",
    "300331",
    "688630",
    "300885",
    "002361",
    "301387",
    "688226",
    "300196",
    "301526",
    "603601",
    "002407",
    "300014",
    "300763",
    "600030",
    "600499",
    "002812",
    "300390",
    "002466",
    "301292",
    "300274",
    "688063",
    "300475",
    "688525",
    "001309",
    "002460",
    "000792",
    "002738",
    "002080",
    "002281",
    "600089",
    "600989",
    "002465",
    "300624",
    "002050",
    "002865",
    "300316",
    "603993",
    "605117",
    "300395",
    "600150",
    "002202",
    "603308",
    "002130",
]


def _zcode(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.zfill(6)


def load_prices_panel(prices_parquet: str | Path) -> pd.DataFrame:
    """读 parquet；统一 ``stock_code``、``date``。"""
    df = pd.read_parquet(Path(prices_parquet)).copy()
    if "stock_code" not in df.columns or "date" not in df.columns or "close" not in df.columns:
        raise ValueError("prices 至少需要列: stock_code, date, close")
    df["stock_code"] = _zcode(df["stock_code"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["stock_code", "date"]).reset_index(drop=True)


def _ols_slope_window(y: np.ndarray) -> float:
    """y 长度 n，自变量 x = 0..n-1 的最小二乘斜率；任一无效应返回 nan。"""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n < 2 or np.any(~np.isfinite(y)):
        return float("nan")
    x = np.arange(n, dtype=np.float64)
    x_demean = x - x.mean()
    y_demean = y - y.mean()
    denom = float(np.dot(x_demean, x_demean))
    if denom <= 1e-18:
        return float("nan")
    return float(np.dot(x_demean, y_demean) / denom)


def rolling_ols_slope(series: pd.Series, window: int) -> pd.Series:
    """单列上的滚动斜率（与 trajectory 中提取方式一致）。"""
    return series.rolling(window, min_periods=window).apply(
        lambda a: _ols_slope_window(a), raw=True
    )


class AlphaSearchlight:
    def __init__(
        self,
        panel_df: pd.DataFrame,
        winners_list: list[str],
        *,
        slope_window: int = 10,
    ):
        """
        winners_list
            牛股代码列表（可读入后自行 ``zfill``）。
        slope_window
            轨迹长度与滚动斜率窗口（交易日）。
        """
        self.df = panel_df.sort_values(["stock_code", "date"]).reset_index(drop=True).copy()
        self.df["stock_code"] = _zcode(self.df["stock_code"])
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.winners = sorted({str(w).strip().zfill(6) for w in winners_list})
        self.slope_window = int(slope_window)
        self.atom_columns: list[str] = []
        self.last_valid_alphas: list[str] = []

    def prepare_atoms(self) -> list[str]:
        """写入原子列；返回可供挖掘 slope 一致性的原子名。"""
        df = self.df

        gid = df["stock_code"]
        close_num = pd.to_numeric(df["close"], errors="coerce").astype(np.float64)
        vol = pd.to_numeric(df["volume"], errors="coerce").astype(np.float64)

        ret_1d = close_num.groupby(gid, sort=False).transform(lambda x: x.pct_change())
        df["past_ret_5d"] = close_num.groupby(gid, sort=False).transform(lambda x: x.pct_change(5))

        df["atom_vol_ratio"] = vol / vol.groupby(gid).transform(
            lambda x: x.rolling(20, min_periods=15).mean()
        ).replace(0, np.nan)

        r10 = ret_1d.groupby(gid).transform(lambda x: x.rolling(10, min_periods=8).std())
        r60 = ret_1d.groupby(gid).transform(lambda x: x.rolling(60, min_periods=40).std())
        df["atom_ret_std_ratio"] = r10 / r60.replace(0, np.nan)

        if {"high", "low"}.issubset(df.columns):
            hi = pd.to_numeric(df["high"], errors="coerce").astype(np.float64)
            lo = pd.to_numeric(df["low"], errors="coerce").astype(np.float64)
            rng = (hi - lo).replace(0, np.nan)
            df["atom_pos"] = (close_num - lo) / (rng + 1e-9)
        else:
            df["atom_pos"] = np.nan

        if "amount" in df.columns:
            amt = pd.to_numeric(df["amount"], errors="coerce").astype(np.float64)
            df["atom_avg_trade_amt"] = amt / vol.replace(0, np.nan)
        else:
            df["atom_avg_trade_amt"] = np.nan

        r5 = ret_1d.groupby(gid, sort=False).transform(lambda x: x.rolling(5, min_periods=4).std())
        r20 = ret_1d.groupby(gid, sort=False).transform(lambda x: x.rolling(20, min_periods=15).std())
        ratio = r5 / r20.replace(0, np.nan)
        df["atom_vol_shortlong_ratio_log"] = np.log(np.clip(ratio, 1e-12, np.inf))

        std_s = ret_1d.groupby(gid, sort=False).transform(
            lambda x: x.rolling(5, min_periods=4).std()
        )
        std_l = ret_1d.groupby(gid, sort=False).transform(
            lambda x: x.rolling(20, min_periods=15).std()
        )
        df["atom_vol_squeeze"] = std_s / (std_l.replace(0, np.nan) + 1e-9)

        df["_r1d"] = ret_1d
        pv_parts: list[pd.Series] = []
        for _, g in df.groupby("stock_code", sort=False):
            v = pd.to_numeric(g["volume"], errors="coerce")
            s = g["_r1d"].rolling(10, min_periods=5).corr(v)
            s = s.copy()
            s.index = g.index
            pv_parts.append(s)
        df["atom_pv_corr"] = pd.concat(pv_parts).sort_index()
        df = df.drop(columns=["_r1d"])

        self.df = df

        slope_atoms = [
            "atom_vol_ratio",
            "atom_ret_std_ratio",
            "atom_pos",
            "atom_avg_trade_amt",
            "atom_vol_shortlong_ratio_log",
            "atom_vol_squeeze",
            "atom_pv_corr",
        ]
        self.atom_columns = [c for c in slope_atoms if c in df.columns]
        return self.atom_columns

    def _burst_trajectory_slopes(
        self,
        atom_name: str,
        *,
        burst_col: str,
        window: int | None = None,
    ) -> list[float]:
        """爆发日前窗口内、该原子轨迹的 OLS 斜率（牛股样本）。"""
        w = int(window or self.slope_window)
        slopes: list[float] = []

        sub = self.df[["date", "stock_code", burst_col, atom_name]].dropna(
            subset=[burst_col, atom_name], how="any"
        )
        for sid in self.winners:
            g = sub[sub["stock_code"] == sid].sort_values("date")
            if len(g) < w + 1:
                continue
            bc = pd.to_numeric(g[burst_col], errors="coerce")
            bi = int(bc.to_numpy().argmax())
            if bi < w:
                continue
            traj = pd.to_numeric(
                g.iloc[bi - w : bi][atom_name], errors="coerce"
            ).to_numpy(dtype=np.float64, copy=False)
            if traj.size != w or not np.all(np.isfinite(traj)):
                continue
            slopes.append(_ols_slope_window(traj))
        return slopes

    def extract_trajectory_consistency(
        self,
        atom_name: str,
        *,
        window: int | None = None,
        burst_col: str = "past_ret_5d",
    ) -> tuple[float, float, int]:
        """爆发前窗口斜率样本 → mean_slope, consistency, n_winners_used。"""
        slopes = self._burst_trajectory_slopes(atom_name, burst_col=burst_col, window=window)
        if not slopes:
            return float("nan"), float("nan"), 0

        arr = np.asarray(slopes, dtype=np.float64)
        mean_slope = float(np.mean(arr))
        consistency = abs(mean_slope) / (float(np.std(arr, ddof=0)) + 1e-9)
        return mean_slope, consistency, len(slopes)

    def extract_discriminative_power(
        self,
        atom_name: str,
        *,
        burst_col: str,
        window: int | None = None,
    ) -> tuple[float, float, float, int]:
        """(牛股轨迹斜率均值 − 全截面原子滚动斜率均值) / 全截面标准差；T-Stat 风格。"""
        win_slopes = self._burst_trajectory_slopes(atom_name, burst_col=burst_col, window=window)
        nw = len(win_slopes)
        if nw == 0:
            return float("nan"), float("nan"), float("nan"), 0
        super_mean = float(np.mean(win_slopes))

        def _grp_slope(s: pd.Series) -> pd.Series:
            return rolling_ols_slope(pd.to_numeric(s, errors="coerce"), self.slope_window)

        temp = self.df.groupby("stock_code", sort=False)[atom_name].transform(_grp_slope)
        v = pd.to_numeric(temp, errors="coerce").to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan"), super_mean, float("nan"), nw
        all_mean = float(np.mean(v))
        all_std = float(np.std(v, ddof=0))
        disc = (super_mean - all_mean) / (all_std + 1e-9)
        return disc, super_mean, all_mean, nw

    def build_alpha_from_pattern(self, atom_name: str, mean_slope: float) -> list[str]:
        """斜率因子 + 沿时间的加速度（对已定方向的 ``alpha_slope`` 做 diff）。"""
        direction = -1.0 if mean_slope < 0 else 1.0
        alpha_slope = f"alpha_slope_{atom_name}"
        alpha_accel = f"alpha_accel_{atom_name}"

        def _grp_slope(s: pd.Series) -> pd.Series:
            return rolling_ols_slope(pd.to_numeric(s, errors="coerce"), self.slope_window)

        slopes = self.df.groupby("stock_code", sort=False)[atom_name].transform(_grp_slope)
        self.df[alpha_slope] = slopes * direction
        self.df[alpha_accel] = self.df.groupby("stock_code", sort=False)[alpha_slope].diff()

        cols_out = [alpha_slope, alpha_accel]
        self.last_valid_alphas.extend(cols_out)
        return cols_out

    def run_mining_pipeline(
        self,
        atoms: list[str] | None = None,
        *,
        mining: str = "discriminative",
        consistency_min: float = 0.5,
        discriminative_min: float = 0.5,
        burst_col: str = "past_ret_5d",
    ) -> list[str]:
        """``discriminative``：牛股 vs 全截面滚动斜率；``consistency``：牛股轨迹内稳定性。"""
        if atoms is None:
            atoms = list(self.atom_columns)
        if not atoms:
            raise ValueError("无可用 atoms，请先调用 prepare_atoms()")

        if burst_col == "target_5d":
            warnings.warn(
                "burst_col='target_5d'：爆发日依赖未来标签，只适合对照实验，不参与实盘因果解释。",
                stacklevel=2,
            )

        mining = (mining or "discriminative").strip().lower()
        if mining not in {"discriminative", "consistency"}:
            raise ValueError("mining must be 'discriminative' or 'consistency'")

        valid: list[str] = []
        self.last_valid_alphas = []

        print(
            f">> burst 列 = {burst_col}, window={self.slope_window}, 牛股 {len(self.winners)}, "
            f"mining={mining!r}"
        )
        for atom in atoms:
            if atom not in self.df.columns:
                warnings.warn(f"跳过缺少列: {atom}", stacklevel=2)
                continue
            ms_trad, cs, nw_cs = self.extract_trajectory_consistency(atom, burst_col=burst_col)
            disc, sm, all_m, nw_d = self.extract_discriminative_power(
                atom, burst_col=burst_col
            )

            if mining == "discriminative":
                ok = nw_d > 0 and np.isfinite(disc) and disc >= float(discriminative_min)
                mean_for_sign = sm if np.isfinite(sm) else 0.0
                nw = nw_d
            else:
                ok = nw_cs > 0 and np.isfinite(cs) and cs >= float(consistency_min)
                mean_for_sign = ms_trad if np.isfinite(ms_trad) else 0.0
                nw = nw_cs

            print(
                f"   atom={atom!r}: mining_ok={ok} consistency={cs:.4g} (n={nw_cs}) "
                f"discriminative={disc:.4g} super={sm:.6g} all_mean={all_m:.6g} (n={nw_d})"
            )
            if not ok:
                continue
            alpha_cols = self.build_alpha_from_pattern(atom, mean_for_sign)
            valid.extend(alpha_cols)

        print(f">> 写入 alpha 列 {len(valid)} 个: {valid}")
        return valid


def _main_argv() -> None:
    p = argparse.ArgumentParser(description="AlphaSearchlight 离线轨迹→斜率因子")
    p.add_argument("--prices", type=Path, default=PRICES_CSI500_PARQUET)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--consistency-min", type=float, default=0.5)
    p.add_argument(
        "--discriminative-min",
        type=float,
        default=0.5,
        help="mining=discriminative 时的 T-Stat 风格阈值",
    )
    p.add_argument(
        "--mining",
        choices=("discriminative", "consistency"),
        default="discriminative",
        help="discriminative：牛股 vs 全市场滚动斜率；consistency：牛股轨迹内稳定",
    )
    p.add_argument(
        "--burst-on",
        choices=("past_ret_5d", "target_5d"),
        default="past_ret_5d",
        help="past_ret_5d=仅用过去可得；target_5d=有前视渗漏",
    )
    p.add_argument("--out-parquet", type=Path, default=None, help="导出含原子与 alpha 的宽表 parquet")
    args = p.parse_args()

    px = load_prices_panel(args.prices)
    if args.burst_on == "target_5d":
        cid = px["stock_code"]
        cn = pd.to_numeric(px["close"], errors="coerce")
        fwd = cn.groupby(cid, sort=False).transform(lambda s: s.shift(-5))
        px["target_5d"] = fwd / cn - 1.0
    burst_col = "target_5d" if args.burst_on == "target_5d" else "past_ret_5d"

    have = set(px["stock_code"].astype(str).str.zfill(6))
    want = {str(w).strip().zfill(6) for w in SUPER_STOCKS_LIST}
    missing_w = sorted(want - have)
    print(f">> 牛股池 SUPER_STOCKS_LIST 在 parquet 中命中 {len(want) - len(missing_w)}/{len(want)}")
    if missing_w:
        print(f"   未命中（轨迹统计会跳过）示例: {missing_w[:20]}{'...' if len(missing_w) > 20 else ''}")

    eng = AlphaSearchlight(px, SUPER_STOCKS_LIST, slope_window=args.window)
    eng.prepare_atoms()
    eng.run_mining_pipeline(
        atoms=None,
        mining=args.mining,
        consistency_min=args.consistency_min,
        discriminative_min=args.discriminative_min,
        burst_col=burst_col,
    )

    if args.out_parquet is not None:
        out_path = Path(args.out_parquet)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["date", "stock_code"]
        cols += list(eng.atom_columns)
        cols += [
            c
            for c in eng.df.columns
            if c.startswith("alpha_slope_") or c.startswith("alpha_accel_")
        ]
        cols_use = list(dict.fromkeys([c for c in cols if c in eng.df.columns]))
        out_tbl = eng.df[cols_use]
        for name in cols_use:
            if name in ("date", "stock_code"):
                continue
            nn = pd.to_numeric(out_tbl[name], errors="coerce").notna().sum()
            pct = 100.0 * float(nn) / max(len(out_tbl), 1)
            print(f">> 导出列预览 {name}: 非 NaN {int(nn)} / {len(out_tbl)} ({pct:.2f}%)")
        out_tbl.to_parquet(out_path, index=False)
        print(f">> 已写入 {out_path}")


if __name__ == "__main__":
    _main_argv()
