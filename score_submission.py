"""
Score a submission against realized CSI500 returns over an evaluation window.

Convention
----------
Weights in the submission apply at the open of `--start` and are held through
the close of `--end`.  Per-stock return uses forward-adjusted close:

    r_i = close_adj(end) / close_adj(day_before_start) - 1

If `day_before_start` is missing (e.g. suspended), we fall back to open(start).
If close(end) is missing for a stock, we use the last available close in the
window and print a warning.

Primary metric: cumulative excess return (portfolio minus CSI500 index),
compounded across the two evaluation windows of the competition.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import INDEX_CSI500_PARQUET, PRICES_CSI500_PARQUET, filter_prices_to_csi500_constituents

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def resolve_submission_path(raw: str) -> Path:
    """支持从项目根或纯文件名查找（常见于 CSV 放在 ``submissions/``）。"""
    path = Path(raw).expanduser()
    if path.is_file():
        return path.resolve()
    name = path.name if path.parts else raw
    for candidate in (ROOT / path, ROOT / "submissions" / name):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"找不到提交文件 {raw!r}。若在仓库里，试试: submissions/{name}"
    )


def _stock_return(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, str]:
    """Realized return for one stock over [start, end].

    Returns (return, note).  note is '' if clean, otherwise describes the fallback.
    """
    df = df.sort_values("date")
    in_window = df[(df["date"] >= start) & (df["date"] <= end)]
    if in_window.empty:
        return 0.0, "no data in window -> return=0"

    before = df[df["date"] < start]
    if not before.empty:
        entry = before["close"].iloc[-1]
        entry_src = "prev_close"
    else:
        entry = in_window["open"].iloc[0]
        entry_src = "first_open"

    exit_ = in_window["close"].iloc[-1]
    exit_date = in_window["date"].iloc[-1]
    note = ""
    if exit_date < end:
        note = f"last close on {exit_date.date()} (halted?); entry={entry_src}"
    elif entry_src != "prev_close":
        note = f"no prior close; entry={entry_src}"

    if entry <= 0 or pd.isna(entry) or pd.isna(exit_):
        return 0.0, f"bad prices -> return=0 ({note})"

    return float(exit_ / entry - 1.0), note


def score_window(
    weights: pd.Series, prices: pd.DataFrame, index_df: pd.DataFrame,
    start: pd.Timestamp, end: pd.Timestamp,
) -> dict:
    """Score a single (start, end) window."""
    rets = {}
    notes = {}
    for code, w in weights.items():
        sub = prices[prices["stock_code"] == code]
        r, note = _stock_return(sub, start, end)
        rets[code] = r
        if note:
            notes[code] = note

    rets = pd.Series(rets)
    portfolio_return = float((weights * rets).sum())

    idx_window = index_df[(index_df["date"] >= start) & (index_df["date"] <= end)].sort_values("date")
    if idx_window.empty:
        raise RuntimeError(f"No CSI500 index data in [{start.date()}, {end.date()}]")
    idx_before = index_df[index_df["date"] < start]
    idx_entry = idx_before["close"].iloc[-1] if not idx_before.empty else idx_window["open"].iloc[0]
    idx_exit = idx_window["close"].iloc[-1]
    bench_return = float(idx_exit / idx_entry - 1.0)

    return {
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "trading_days": len(idx_window),
        "portfolio_return": portfolio_return,
        "benchmark_return": bench_return,
        "excess_return": portfolio_return - bench_return,
        "n_with_notes": len(notes),
        "notes_sample": dict(list(notes.items())[:5]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("submission", help="path to submission CSV")
    p.add_argument("--start", required=True, help="eval window start, YYYYMMDD")
    p.add_argument("--end", required=True, help="eval window end, YYYYMMDD")
    p.add_argument("--prices", default=str(PRICES_CSI500_PARQUET))
    p.add_argument("--index", default=str(INDEX_CSI500_PARQUET))
    args = p.parse_args()

    sub_path = resolve_submission_path(args.submission)
    sub = pd.read_csv(sub_path, dtype={"stock_code": str})
    sub["stock_code"] = sub["stock_code"].str.zfill(6)
    weights = sub.set_index("stock_code")["weight"].astype(float)

    prices = pd.read_parquet(args.prices)
    prices = filter_prices_to_csi500_constituents(prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    result = score_window(weights, prices, index_df, start, end)

    print(f"Window: {result['start']} to {result['end']}  ({result['trading_days']} trading days)")
    print(f"  portfolio return : {result['portfolio_return']*100:+.3f}%")
    print(f"  benchmark return : {result['benchmark_return']*100:+.3f}%")
    print(f"  excess return    : {result['excess_return']*100:+.3f}%")
    if result["n_with_notes"]:
        print(f"  stocks with notes: {result['n_with_notes']} (e.g. {result['notes_sample']})")


if __name__ == "__main__":
    main()
