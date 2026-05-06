"""
规则版「公告窗内错配」选股：与数学定义一致。

在 ``financial_evt_gate=1``（公告后 0–N 日历日，N 由 ``merge_financial_extended`` 默认）的
子集内，按 ``financial_evt_mispricing_rankdiff`` 或 ``financial_evt_mispricing_aggr`` 降序
取 Top-K，等权写出 submission CSV。

若当日无 Active 股票，则退出并提示（不强行用全市场冒充事件宇宙）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features import (
    PRICES_CSI500_PARQUET,
    build_features,
    filter_prices_to_csi500_constituents,
    prediction_frame,
)  # noqa: E402

DATA_DIR = ROOT / "data"
MIN_STOCKS = 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default=str(PRICES_CSI500_PARQUET))
    ap.add_argument("--as-of", default=None, help="YYYYMMDD; latest if omitted")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument(
        "--score",
        choices=("rankdiff", "aggr"),
        default="rankdiff",
        help="rankdiff=M (rank pct); aggr=ShockSignal×(1−|R^{(3)}|/max|R|) on Active",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.top_k < MIN_STOCKS:
        raise SystemExit(f"top-k must be >= {MIN_STOCKS} per competition rule")

    score_col = (
        "financial_evt_mispricing_rankdiff"
        if args.score == "rankdiff"
        else "financial_evt_mispricing_aggr"
    )

    prices = pd.read_parquet(args.prices)
    prices = filter_prices_to_csi500_constituents(prices)
    panel = build_features(prices)
    df = prediction_frame(panel, as_of=args.as_of)
    if df.empty:
        raise SystemExit("No prediction rows after dropna(required features).")

    gate = pd.to_numeric(df["financial_evt_gate"], errors="coerce").fillna(0) > 0.5
    pool = df.loc[gate].copy()
    pool[score_col] = pd.to_numeric(pool[score_col], errors="coerce")
    pool = pool.dropna(subset=[score_col])

    if pool.empty:
        raise SystemExit(
            "No stocks with Active=1 on as-of date; cannot form event-universe portfolio."
        )

    chosen = pool.nlargest(args.top_k, score_col)
    w = 1.0 / len(chosen)
    out = pd.DataFrame({"stock_code": chosen["stock_code"].astype(str).str.zfill(6), "weight": w})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} names (equal weight) to {args.out}")
    print(f"  score={score_col}, as_of row date={chosen['date'].iloc[0].date()}, active_pool={len(pool)}")


if __name__ == "__main__":
    main()
