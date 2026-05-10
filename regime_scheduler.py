"""
Regime scheduler: 二分类「未来 H 日内进攻组合是否跑赢防御组合」。
用指数+截面统计作市场状态特征，逻辑回归 + 标准化；推理端带缓冲状态机。

用法（标签阶段会多次子进程训练 baseline / defensive，较慢）::

    python regime_scheduler.py train --label-start 2025-06-01 --label-end 2026-04-15 \\
        --label-stride 10 --bundle-out models/regime_scheduler.joblib

    python regime_scheduler.py decide --as-of 20260424 --bundle models/regime_scheduler.joblib \\
        --out submissions/scheduled.csv --state submissions/regime_scheduler_state.json

说明：``train`` 默认按 ``--label-stride`` 下采样标签日；每个标签日会各跑一次
``baseline_xgboost.py`` 与 ``train_defensive.py``（与各自 ``--as-of`` 截断一致），
再以 H=5 日前向组合收益差打 0/1 标签（中间带 ``|ΔR|≤ε`` 丢弃）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from features import (
    FORWARD_HORIZON,
    INDEX_CSI500_PARQUET,
    PRICES_CSI500_PARQUET,
    filter_prices_to_csi500_constituents,
)

ROOT = Path(__file__).resolve().parent


def _calendar(d: pd.Timestamp) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def build_regime_features(prices: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """日频市场状态特征：指数动量/波动 + 截面离散度/赚钱效应。"""
    idx = index_df.sort_values("date").copy()
    idx["date"] = pd.to_datetime(idx["date"])
    c = pd.to_numeric(idx["close"], errors="coerce")
    idx["ret_1d"] = c.pct_change()
    idx["idx_mom_5d"] = c / c.shift(5) - 1.0
    idx["idx_mom_20d"] = c / c.shift(20) - 1.0
    idx["idx_vol_20d"] = idx["ret_1d"].rolling(20, min_periods=15).std(ddof=0)

    px = prices.sort_values(["stock_code", "date"]).copy()
    px["date"] = pd.to_datetime(px["date"])
    px["ret_1d"] = px.groupby("stock_code", sort=False)["close"].pct_change()

    def _disp(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        return float(s.std(ddof=0)) if len(s) > 5 else float("nan")

    def _breadth(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            return float("nan")
        return float((s > 0).mean())

    cs = px.groupby("date", sort=False).agg(
        xs_dispersion_1d=("ret_1d", _disp),
        xs_breadth_1d=("ret_1d", _breadth),
    )
    # 指数10日动量
    idx["idx_mom_10d"] = c / c.shift(10) - 1.0

    # 成交量相对强度：当日成交量 / 20日均量
    vol_idx = pd.to_numeric(idx["volume"], errors="coerce") if "volume" in idx.columns else None
    if vol_idx is not None:
        idx["idx_vol_ratio"] = vol_idx / vol_idx.rolling(20, min_periods=10).mean()

    # 涨跌宽度5日均值（更稳定）
    cs["xs_breadth_5d"] = cs["xs_breadth_1d"].rolling(5, min_periods=3).mean()
    out = idx.set_index("date")[["idx_mom_5d", "idx_mom_10d", "idx_mom_20d", "idx_vol_20d"]].join(cs, how="inner")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def forward_holding_return_per_stock(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """每行 (date, stock_code) 上对应用 close 持有 horizon 个**交易日**的收益。"""
    df = prices.sort_values(["stock_code", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    fut = df.groupby("stock_code", sort=False)["close"].shift(-horizon)
    cur = pd.to_numeric(df["close"], errors="coerce")
    fut = pd.to_numeric(fut, errors="coerce")
    df["fwd_ret"] = np.where((cur > 0) & fut.notna(), fut / cur - 1.0, np.nan)
    return df[["date", "stock_code", "fwd_ret"]]


def portfolio_dot_forward(
    weights: pd.Series,
    fwd_slice: pd.Series,
) -> float:
    """组合权重与当日截面对齐后的点积；缺失前向收益权重按 0 处理。"""
    w = weights.reindex(fwd_slice.index).fillna(0.0).astype(float)
    r = pd.to_numeric(fwd_slice, errors="coerce").fillna(0.0)
    return float((w * r).sum())


def load_weights_csv(path: Path) -> pd.Series:
    sub = pd.read_csv(path, dtype={"stock_code": str})
    sub["stock_code"] = sub["stock_code"].str.zfill(6)
    return sub.set_index("stock_code")["weight"].astype(float)


def run_model_csv(
    *,
    script: str,
    as_of: str,
    out_csv: Path,
    python_exe: str,
    top_k: int,
    prices: Path,
    index_p: Path | None,
    quiet: bool,
) -> None:
    cmd = [
        python_exe,
        str(ROOT / script),
        "--as-of",
        as_of,
        "--out",
        str(out_csv),
        "--top-k",
        str(top_k),
        "--prices",
        str(prices),
    ]
    if script == "train_defensive.py":
        cmd.extend(["--index", str(index_p)])
    stderr = subprocess.DEVNULL if quiet else None
    stdout = subprocess.DEVNULL if quiet else None
    r = subprocess.run(cmd, cwd=str(ROOT), stdout=stdout, stderr=stderr)
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed (--as-of {as_of})")


def cmd_train(args: argparse.Namespace) -> None:
    prices_path = Path(args.prices)
    index_path = Path(args.index)
    prices = pd.read_parquet(prices_path)
    prices = filter_prices_to_csi500_constituents(prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(index_path)
    index_df["date"] = pd.to_datetime(index_df["date"])

    regime = build_regime_features(prices, index_df)
    fwd_tbl = forward_holding_return_per_stock(prices, int(args.horizon))

    trading = np.sort(prices["date"].unique())
    d0 = pd.Timestamp(args.label_start)
    d1 = pd.Timestamp(args.label_end)
    stride = max(1, int(args.label_stride))

    eps_floor = float(args.epsilon)

    label_rows: list[dict] = []
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # 需要 t 与 t+horizon 均为交易日且在 regime 表里
    cand: list[pd.Timestamp] = []
    for i, d64 in enumerate(trading):
        d = pd.Timestamp(d64)
        if d < d0 or d > d1:
            continue
        if i + args.horizon >= len(trading):
            continue
        if pd.Timestamp(trading[i + args.horizon]) > prices["date"].max():
            continue
        if d not in regime.index:
            continue
        cand.append(d)
    sampled = cand[::stride]

    print(f">> 候选标签日 {len(cand)}，stride={stride} → 拟合 {len(sampled)} 天（各日全量重训 baseline+defensive）")

    for d in tqdm(sampled, desc="labels"):
        cal = _calendar(d)
        pb = work / f"sched_b_{cal}.csv"
        pd_ = work / f"sched_d_{cal}.csv"
        try:
            if not pb.is_file():
                run_model_csv(
                    script="baseline_xgboost.py",
                    as_of=cal,
                    out_csv=pb,
                    python_exe=args.python,
                    top_k=args.top_k,
                    prices=prices_path,
                    index_p=None,
                    quiet=args.quiet,
                )
            if not pd_.is_file():
                run_model_csv(
                    script="train_defensive.py",
                    as_of=cal,
                    out_csv=pd_,
                    python_exe=args.python,
                    top_k=args.top_k,
                    prices=prices_path,
                    index_p=index_path,
                    quiet=args.quiet,
                )
        except Exception:
            tqdm.write(f"[skip] train failed at {cal}")
            continue

        w_a = load_weights_csv(pb)
        w_d = load_weights_csv(pd_)
        fsub = fwd_tbl[fwd_tbl["date"] == d].set_index("stock_code")["fwd_ret"]
        ra = portfolio_dot_forward(w_a, fsub)
        rd = portfolio_dot_forward(w_d, fsub)
        delta = ra - rd
        if delta > eps_floor:
            y = 1
        elif delta < -eps_floor:
            y = 0
        else:
            continue
        feats = regime.loc[d]
        row = feats.to_dict()
        row["date"] = d
        row["y_attack_wins"] = y
        row["delta_ra_minus_rd"] = delta
        label_rows.append(row)

    if len(label_rows) < 15:
        raise RuntimeError(f"标签样本过少 ({len(label_rows)}）—放宽日期、e 或减小 stride。")

    lab = pd.DataFrame(label_rows).set_index("date").sort_index()
    feat_cols = [c for c in lab.columns if c not in ("y_attack_wins", "delta_ra_minus_rd")]
    X = lab[feat_cols].to_numpy(dtype=float)
    y = lab["y_attack_wins"].to_numpy(dtype=int)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    logit = LogisticRegression(
        penalty="l2",
        C=float(args.C),
        max_iter=int(args.max_iter),
        random_state=int(args.seed),
        solver="lbfgs",
    )
    logit.fit(Xs, y)

    bundle = {
        "logit": logit,
        "scaler": scaler,
        "feature_columns": feat_cols,
        "epsilon": eps_floor,
        "horizon": int(args.horizon),
        "p_attack_high": float(args.p_attack_high),
        "p_attack_low": float(args.p_attack_low),
        "n_labels": len(lab),
        "label_stride": stride,
        "scores": {"train_accuracy": float(logit.score(Xs, y))},
    }
    out_p = Path(args.bundle_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_p)
    coef = dict(zip(feat_cols, logit.coef_.ravel()))
    print(f">> 已写入 {out_p}  （{len(lab)} 条标签，特征维 {len(feat_cols)}）")
    print("   近似系数:", {k: float(f"{v:.4g}") for k, v in sorted(coef.items(), key=lambda x: -abs(x[1]))})


def _load_bundle(path: Path) -> dict:
    return joblib.load(path)


def _state_read(path: Path) -> tuple[str, str | None]:
    if not path.exists():
        return "attack", None
    js = json.loads(path.read_text(encoding="utf-8"))
    choice = js.get("choice", "attack")
    if choice not in ("attack", "defense"):
        choice = "attack"
    prev = js.get("date")
    return choice, prev


def _state_write(path: Path, choice: str, date_s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"choice": choice, "date": date_s}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def cmd_decide(args: argparse.Namespace) -> None:
    bundle = _load_bundle(Path(args.bundle))
    logit = bundle["logit"]
    scaler = bundle["scaler"]
    feat_cols: list[str] = bundle["feature_columns"]
    p_hi = float(args.p_attack_high)
    p_lo = float(args.p_attack_low)

    prices = pd.read_parquet(args.prices)
    prices = filter_prices_to_csi500_constituents(prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    regime = build_regime_features(prices, index_df)
    as_ts = pd.Timestamp(args.as_of)
    if as_ts not in regime.index:
        # 退化：用最晚可用行近似
        sub = regime[regime.index <= as_ts]
        if sub.empty:
            raise RuntimeError("regime features empty for dates ≤ as-of")
        xrow = sub.iloc[-1]
        print(f">> 警告：{as_ts.date()} 无完整 regime 行，使用 {sub.index[-1].date()} 截面特征近似")
    else:
        xrow = regime.loc[as_ts]

    x = pd.DataFrame([xrow[feat_cols].to_numpy()], columns=feat_cols, dtype=float)
    Xs = scaler.transform(x.to_numpy())
    pa = float(logit.predict_proba(Xs)[0, 1])
    print(f">> P(attack wins)≈{pa:.4f}")

    prev_choice, prev_date = _state_read(Path(args.state))

    if pa > p_hi:
        chosen = "attack"
        reason = f"P_attack={pa:.3f}>{p_hi}"
    elif pa < p_lo:
        chosen = "defense"
        reason = f"P_attack={pa:.3f}<{p_lo}"
    else:
        chosen = prev_choice
        reason = f"P_attack={pa:.3f}∈[{p_lo},{p_hi}] → hold {chosen}"

    cal = _calendar(as_ts)
    out_csv = Path(args.out)
    if chosen == "attack":
        run_model_csv(
            script="baseline_xgboost.py",
            as_of=cal,
            out_csv=out_csv,
            python_exe=args.python,
            top_k=args.top_k,
            prices=Path(args.prices),
            index_p=None,
            quiet=False,
        )
    else:
        run_model_csv(
            script="train_defensive.py",
            as_of=cal,
            out_csv=out_csv,
            python_exe=args.python,
            top_k=args.top_k,
            prices=Path(args.prices),
            index_p=Path(args.index),
            quiet=False,
        )

    _state_write(Path(args.state), chosen, cal)
    print(f">> Scheduler: {reason} → 选用 {chosen} → {out_csv}")


def main() -> None:
    root_p = ROOT
    ap = argparse.ArgumentParser(description="Regime classifier: attack vs defense portfolio.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="离线打标签并训练 logistic + scaler")
    pt.add_argument("--prices", type=Path, default=PRICES_CSI500_PARQUET)
    pt.add_argument("--index", type=Path, default=INDEX_CSI500_PARQUET)
    pt.add_argument("--label-start", required=True)
    pt.add_argument("--label-end", required=True)
    pt.add_argument("--label-stride", type=int, default=10)
    pt.add_argument("--horizon", type=int, default=FORWARD_HORIZON)
    pt.add_argument("--epsilon", type=float, default=1e-4, help="|ΔR|≤ε 的样本丢弃（收益差死区）")
    pt.add_argument("--C", type=float, default=1.0, dest="C", help="L2 逆正则强度（sklearn LogisticRegression）")
    pt.add_argument("--max-iter", type=int, default=500)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--p-attack-high", type=float, default=0.6)
    pt.add_argument("--p-attack-low", type=float, default=0.4)
    pt.add_argument("--top-k", type=int, default=50)
    pt.add_argument("--work-dir", type=Path, default=root_p / "submissions" / "_sched_train_work")
    pt.add_argument("--bundle-out", type=Path, default=root_p / "models" / "regime_scheduler.joblib")
    pt.add_argument("--python", default=sys.executable)
    pt.add_argument("--quiet", action="store_true", help="隐藏子进程训练日志")
    pt.set_defaults(_fn=cmd_train)

    pd_ = sub.add_parser("decide", help="读模型+状态机缓冲，跑一次被选脚本写出 CSV")
    pd_.add_argument("--bundle", type=Path, required=True)
    pd_.add_argument("--as-of", required=True)
    pd_.add_argument("--out", type=Path, required=True)
    pd_.add_argument("--state", type=Path, default=root_p / "submissions" / "regime_scheduler_state.json")
    pd_.add_argument("--p-attack-high", type=float, default=None)
    pd_.add_argument("--p-attack-low", type=float, default=None)
    pd_.add_argument("--top-k", type=int, default=50)
    pd_.add_argument("--prices", type=Path, default=PRICES_CSI500_PARQUET)
    pd_.add_argument("--index", type=Path, default=INDEX_CSI500_PARQUET)
    pd_.add_argument("--python", default=sys.executable)
    pd_.set_defaults(_fn=cmd_decide)

    args = ap.parse_args() 
    if args.cmd == "decide":
        bundle = joblib.load(Path(args.bundle))
        if args.p_attack_high is None:
            args.p_attack_high = float(bundle.get("p_attack_high", 0.6))
        if args.p_attack_low is None:
            args.p_attack_low = float(bundle.get("p_attack_low", 0.4))
    args._fn(args)


if __name__ == "__main__":
    main()
