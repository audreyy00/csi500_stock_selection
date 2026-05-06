"""
Download CSI500 data via akshare.

Produces **干净 CSI500** 三件套（文件名显式带 ``csi500``，供 baseline / train_defensive 专用）::

  - data/constituents_csi500.csv   CSI500 constituents (as of download time)
  - data/prices_csi500.parquet     daily OHLCV, forward-adjusted
  - data/index_csi500.parquet      CSI500 index sh000905（基准）

Usage
-----
  # initial download (slow: ~500 API calls, expect 10-30 min)
  python download_data.py --start 20250101 --end 20260421

  # incremental: resume from max date in prices_csi500.parquet
  python download_data.py --update --end 20260430
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import akshare as ak
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
CONSTITUENTS_CSI500 = DATA_DIR / "constituents_csi500.csv"
PRICES_CSI500 = DATA_DIR / "prices_csi500.parquet"
INDEX_CSI500 = DATA_DIR / "index_csi500.parquet"
PRICES_CSI500_PARTIAL = DATA_DIR / "prices_csi500.partial.parquet"

CSI500_SYMBOL = "000905"


def fetch_constituents() -> pd.DataFrame:
    df = ak.index_stock_cons_csindex(symbol=CSI500_SYMBOL)
    rename = {
        "成分券代码": "stock_code",
        "成分券名称": "stock_name",
        "日期": "as_of_date",
    }
    df = df.rename(columns=rename)
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    return df[["stock_code", "stock_name", "as_of_date"]]


def _exchange_prefix(code: str) -> str:
    return "sh" if code.startswith("6") else "sz"


def fetch_stock_hist(code: str, start: str, end: str, retries: int = 3) -> pd.DataFrame | None:
    symbol = f"{_exchange_prefix(code)}{code}"
    df = None
    last_err = None
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start,
                end_date=end,
                adjust="qfq",
            )
            break
        except Exception as e:
            last_err = e
            time.sleep(1.0 + attempt)
    if df is None:
        print(f"  [warn] {symbol} failed after {retries} tries: {last_err}")
        return None
    if df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["stock_code"] = code
    df["pct_change"] = df["close"].pct_change() * 100.0
    keep = ["date", "stock_code", "open", "close", "high", "low", "volume", "amount", "turnover", "pct_change"]
    return df[[c for c in keep if c in df.columns]]


def fetch_index_hist(start: str, end: str) -> pd.DataFrame:
    df = ak.stock_zh_index_daily(symbol="sh000905")
    df["date"] = pd.to_datetime(df["date"])
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="20250101", help="YYYYMMDD (ignored if --update)")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"))
    p.add_argument(
        "--update",
        action="store_true",
        help="incremental from max date in data/prices_csi500.parquet",
    )
    p.add_argument("--sleep", type=float, default=0.1, help="seconds between stock requests")
    args = p.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    print(">> Fetching CSI500 constituents...")
    cons = fetch_constituents()
    cons_codes = set(cons["stock_code"].astype(str).str.zfill(6))
    print(f"   {len(cons)} constituents from CSI500 index.")

    existing = None
    if args.update and PRICES_CSI500.is_file():
        existing = pd.read_parquet(PRICES_CSI500)
        max_date = existing["date"].max()
        start = (max_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        print(f">> Incremental update from {start} to {args.end}")
    else:
        start = args.start
        print(f">> Full download from {start} to {args.end}")

    start_ts = pd.to_datetime(str(start), format="%Y%m%d")
    end_ts = pd.to_datetime(str(args.end), format="%Y%m%d")
    if start_ts > end_ts:
        if args.update and existing is not None:
            md = pd.to_datetime(existing["date"]).max()
            print(
                f">> 跳过增量：库里最新交易日为 {md.date()}，请求的增量起点 {start_ts.date()} 已晚于 "
                f"--end {end_ts.date()}，区间为空（请把 --end 设到 {start_ts.date()} 之后，或无需再跑 --update）。"
            )
        else:
            print(f">> 跳过：起始日 {start_ts.date()} 晚于结束日 {end_ts.date()}，请检查参数。")
        return

    print(">> Fetching CSI500 index benchmark (sh000905)...")
    idx = fetch_index_hist(start, args.end)
    if INDEX_CSI500.is_file() and args.update:
        idx_old = pd.read_parquet(INDEX_CSI500)
        idx = pd.concat([idx_old, idx]).drop_duplicates(subset=["date"]).sort_values("date")
    idx.to_parquet(INDEX_CSI500, index=False)
    print(f"   {len(idx)} index rows → {INDEX_CSI500}")

    print(">> Fetching constituent OHLCV...")
    frames = []
    ok_codes: list[str] = []
    n_fail = 0
    codes = cons["stock_code"].tolist()
    checkpoint_every = 100
    for i, code in enumerate(tqdm(codes)):
        df = fetch_stock_hist(code, start, args.end)
        if df is not None and not df.empty:
            frames.append(df)
            ok_codes.append(code)
        else:
            n_fail += 1
        if frames and (i + 1) % checkpoint_every == 0:
            tmp = pd.concat(frames, ignore_index=True)
            tmp.to_parquet(PRICES_CSI500_PARTIAL, index=False)
        time.sleep(args.sleep)

    if not frames:
        print(">> ERROR: no stocks downloaded successfully. Nothing to save.")
        return

    new_prices = pd.concat(frames, ignore_index=True)
    if existing is not None:
        prices = pd.concat([existing, new_prices], ignore_index=True)
        prices = prices.drop_duplicates(subset=["date", "stock_code"]).sort_values(["stock_code", "date"])
    else:
        prices = new_prices

    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    # 只保留**当前**官方 CSI500 名单（剔除已调出、或历史残留）
    prices = prices[prices["stock_code"].isin(cons_codes)].copy()

    prices.to_parquet(PRICES_CSI500, index=False)
    if PRICES_CSI500_PARTIAL.exists():
        PRICES_CSI500_PARTIAL.unlink()

    final_codes = set(prices["stock_code"].unique())
    filtered = cons[cons["stock_code"].isin(final_codes)].reset_index(drop=True)
    filtered.to_csv(CONSTITUENTS_CSI500, index=False)

    print(f">> Saved {len(prices):,} rows across {len(final_codes)} stocks → {PRICES_CSI500}")
    print(f"   success: {len(ok_codes)}, failed: {n_fail}")
    print(f">> Saved {len(filtered)} constituents → {CONSTITUENTS_CSI500}")
    if n_fail:
        missing = sorted(set(codes) - final_codes)[:10]
        print(f"   dropped from universe (no price data): {missing}{'...' if n_fail > 10 else ''}")


if __name__ == "__main__":
    main()
