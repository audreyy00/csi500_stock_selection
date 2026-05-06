"""
用 akshare 为「中证 500 成分之外」的股票补日线，并合并进 **扩展行情库**（默认 **不** 污染干净 CSI500）。

* **训练用干净库**：``data/prices_csi500.parquet``（仅 ``download_data.py`` 写入；baseline / train_defensive 专用）
* **本脚本默认写入**：``data/prices_extended.parquet``（若不存在则从 ``prices_csi500`` 复制一份再合并）

典型用法：``alpha_searchlight.SUPER_STOCKS_LIST`` 中非 CSI500 的代码可由此补数。

用法（项目根）::

    python download_extra_stocks.py --super-winners --dry-run

    python download_extra_stocks.py --super-winners --start 20250101 --end 20260430

说明：基准指数仍为 ``data/index_csi500.parquet``（不必为扩展股票单独拉指数）。
本批下载到的、且尚未出现在 ``data/prices.parquet`` 中的股票，会在写入主目标（默认
``prices_extended``）后**同步并入** ``prices.parquet``；若 ``prices.parquet`` 不存在则
在存在 ``prices_csi500.parquet`` 时先复制再追加。
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from alpha_searchlight import SUPER_STOCKS_LIST
from download_data import DATA_DIR, PRICES_CSI500, fetch_stock_hist

PRICES_EXTENDED_DEFAULT = DATA_DIR / "prices_extended.parquet"
# 历史/宽泛行情库：补股时若新股不在此文件中则一并写入
PRICES_LEGACY_PANEL = DATA_DIR / "prices.parquet"


def _normalize_codes(codes: list[str]) -> list[str]:
    out: list[str] = []
    for c in codes:
        s = str(c).strip().zfill(6)
        if s and s not in out:
            out.append(s)
    return out


def _codes_already_in_parquet(parquet_path: Path) -> set[str]:
    pr = pd.read_parquet(parquet_path, columns=["stock_code"])
    return set(pr["stock_code"].astype(str).str.zfill(6).unique())


def _merge_new_rows_into_prices_parquet(new_rows: pd.DataFrame, *, primary_target: Path) -> None:
    """将本批下载中、尚未出现在 ``data/prices.parquet`` 的股票行情并入该文件。"""
    path = PRICES_LEGACY_PANEL
    if path.resolve() == Path(primary_target).resolve():
        return

    nr = new_rows.copy()
    nr["stock_code"] = nr["stock_code"].astype(str).str.zfill(6)
    batch_codes = set(nr["stock_code"].astype(str).str.zfill(6).unique())

    if not path.is_file():
        if PRICES_CSI500.is_file():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(PRICES_CSI500, path)
            print(f">> 已初始化 {path.name}（从 {PRICES_CSI500.name} 复制），用于并入补股数据")
        else:
            print(f">> 跳过 {path.name}：文件不存在且无法从 {PRICES_CSI500.name} 初始化")
            return

    have = _codes_already_in_parquet(path)
    need_codes = sorted(batch_codes - have)
    if not need_codes:
        print(f">> {path.name} 已含本批全部股票代码，无需追加")
        return

    sub = nr[nr["stock_code"].isin(need_codes)]
    old = pd.read_parquet(path)
    old["stock_code"] = old["stock_code"].astype(str).str.zfill(6)

    all_cols = sorted(set(old.columns) | set(sub.columns))
    old = old.reindex(columns=all_cols)
    sub = sub.reindex(columns=all_cols)

    merged = pd.concat([old, sub], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "stock_code"], keep="last").sort_values(
        ["stock_code", "date"]
    )
    merged.to_parquet(path, index=False)
    print(
        f">> 已更新 {path.name}：追加 {len(need_codes)} 只新代码，"
        f"{len(merged):,} 行，{merged['stock_code'].nunique()} 只股票"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="补股合并到扩展 parquet（默认 prices_extended.Parquet；勿覆盖 prices_csi500）"
    )
    p.add_argument(
        "--prices",
        type=Path,
        default=PRICES_EXTENDED_DEFAULT,
        help="并入目标 parquet（默认 data/prices_extended.parquet）；不存在则从 prices_csi500 复制初始化",
    )
    p.add_argument(
        "--super-winners",
        action="store_true",
        help="包含 alpha_searchlight.SUPER_STOCKS_LIST 中的全部代码",
    )
    p.add_argument(
        "--codes-file",
        type=Path,
        default=None,
        help="自定义代码列表，一行一只",
    )
    p.add_argument("--start", default="20250101", help="YYYYMMDD")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"), help="YYYYMMDD")
    p.add_argument("--sleep", type=float, default=0.12)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    targets: list[str] = []
    if args.super_winners:
        targets.extend(SUPER_STOCKS_LIST)
    if args.codes_file is not None:
        text = Path(args.codes_file).read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            targets.append(line.split()[0])
    targets = _normalize_codes(targets)
    if not targets:
        raise SystemExit("请使用 --super-winners 和/或 --codes-file")

    pq = Path(args.prices)
    if not pq.is_file():
        if PRICES_CSI500.is_file():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(PRICES_CSI500, pq)
            print(f">> 已初始化 {pq}（从 {PRICES_CSI500} 复制）")
        else:
            raise SystemExit(
                f"找不到 {pq} 且无权复制：请先运行 python download_data.py 生成 {PRICES_CSI500}"
            )

    have = _codes_already_in_parquet(pq)
    missing = [c for c in targets if c not in have]
    print(f">> 目标 {len(targets)} 只，已在 {pq.name} 中 {len(targets) - len(missing)}，待补 {len(missing)}")
    if missing:
        print(f"   待补示例: {missing[:15]}{'...' if len(missing) > 15 else ''}")

    if args.dry_run:
        return

    if not missing:
        print(">> 无需下载")
        return

    frames: list[pd.DataFrame] = []
    for code in tqdm(missing, desc="akshare"):
        df = fetch_stock_hist(code, args.start, args.end)
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(args.sleep)

    if not frames:
        print(">> 下载全部失败")
        return

    new_rows = pd.concat(frames, ignore_index=True)
    old = pd.read_parquet(pq)
    old["stock_code"] = old["stock_code"].astype(str).str.zfill(6)
    new_rows["stock_code"] = new_rows["stock_code"].astype(str).str.zfill(6)

    all_cols = sorted(set(old.columns) | set(new_rows.columns))
    old = old.reindex(columns=all_cols)
    new_rows = new_rows.reindex(columns=all_cols)

    out = pd.concat([old, new_rows], ignore_index=True)
    out = out.drop_duplicates(subset=["date", "stock_code"], keep="last").sort_values(
        ["stock_code", "date"]
    )
    out.to_parquet(pq, index=False)
    print(f">> 已写入 {pq}：{len(out):,} 行，{out['stock_code'].nunique()} 只股票")

    _merge_new_rows_into_prices_parquet(new_rows, primary_target=pq)


if __name__ == "__main__":
    main()
