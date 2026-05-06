"""
Pull optional fundamentals tables into ./data/ (AkShare and/or baostock).

* **盈利预测**（AkShare 东财）：``stock_profit_forecast_em``

* **行业**、**季频财务**：二选一  
  - **``--provider akshare``（默认）**：东财网页接口，行业逐股慢、财务 ``stock_financial_analysis_indicator_em``。  
  - **``--provider baostock``**：``query_stock_industry`` 一次筛 CSI500；财务 ``query_profit/growth/balance_data`` 按季合并，写入  
    ``stock_industry_bs.parquet``、``financial_quarter_bs.parquet``。

Requires ``pip install baostock`` when using ``--provider baostock``.

Usage examples
--------------
  python download_ak_facts.py --only-profit
  python download_ak_facts.py --only-industry --provider baostock --sleep 0.03
  python download_ak_facts.py --only-financial --provider baostock --resume --financial-quarters 5
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import akshare as ak
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"

_BS_FINANCIAL_MODE = "baostock"

# 写入磁盘前裁剪：每只票只保留最近 N 季度 + 景气度少量列（净利/营收同比、ROE、毛利率及其变动、杠杆率变动）
_FINANCIAL_MAX_QUARTERS_DEFAULT = 5
_FINANCIAL_METRIC_COLUMNS = (
    "PARENTNETPROFITTZ",  # 净利润同比增长率（%）
    "ROEJQ",  # ROE——特征侧可做连续两季抬升
    "XSMLL",  # 销售毛利率；配合 XSMLL_TB 表征边际变化
    "XSMLL_TB",
    "TOTALOPERATEREVETZ",  # 营收同比增速
    "ZCFZL",  # 资产负债率
    "ZCFZLTZ",  # 资产负债率同比变动
)


def slim_financial_for_disk(df: pd.DataFrame, max_quarters: int) -> pd.DataFrame:
    """Keep last ``max_quarters`` rows per ``stock_code`` by ``REPORT_DATE`` + metric/meta columns."""
    if df.empty:
        return df
    out = df.copy()
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
    if "REPORT_DATE" not in out.columns:
        return out
    out["REPORT_DATE"] = pd.to_datetime(out["REPORT_DATE"], errors="coerce")
    if "NOTICE_DATE" in out.columns:
        out["NOTICE_DATE"] = pd.to_datetime(out["NOTICE_DATE"], errors="coerce")
    out = out.sort_values(["stock_code", "REPORT_DATE"])
    out = out.groupby("stock_code", group_keys=False).tail(max_quarters)
    meta = [c for c in ("stock_code", "REPORT_DATE", "NOTICE_DATE") if c in out.columns]
    meta += [c for c in ("indicator_mode", "fetched_at") if c in out.columns]
    metrics = [c for c in _FINANCIAL_METRIC_COLUMNS if c in out.columns]
    missing_m = set(_FINANCIAL_METRIC_COLUMNS) - set(metrics)
    if missing_m:
        print(f"   [warn] financial slim: missing metric columns from API: {sorted(missing_m)}")
    cols = meta + metrics
    return out[cols].reset_index(drop=True)


def ak_em_symbol(code: str) -> str:
    """East Money A-share symbol like 600519.SH / 301389.SZ / 834021.BJ."""
    c = str(code).strip().zfill(6)
    first = c[0]
    if first == "6":
        return f"{c}.SH"
    if first in ("0", "3"):
        return f"{c}.SZ"
    if first in ("4", "8"):
        return f"{c}.BJ"
    return f"{c}.SZ"


def load_codes(path: Path) -> list[str]:
    cons = pd.read_csv(path, dtype={"stock_code": str})
    return cons["stock_code"].astype(str).str.zfill(6).tolist()


def fetch_profit_forecast(retries: int = 4) -> pd.DataFrame:
    last_err = None
    for attempt in range(retries):
        try:
            df = ak.stock_profit_forecast_em(symbol="")
            break
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    else:
        raise RuntimeError(f"stock_profit_forecast_em failed: {last_err}")

    df = df.copy()
    if "代码" in df.columns:
        df.rename(columns={"代码": "stock_code"}, inplace=True)
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df["fetched_at"] = pd.Timestamp.now(tz=timezone.utc)
    return df


def fetch_industry_one(code: str, retries: int = 5) -> dict:
    """Always return one row per code; ``industry_name`` may be missing on API failure."""
    sc = str(code).strip().zfill(6)
    out: dict = {"stock_code": sc, "industry_name": pd.NA}
    for attempt in range(retries):
        try:
            info = ak.stock_individual_info_em(symbol=sc)
            if info.empty or "item" not in info.columns:
                time.sleep(0.3 * (attempt + 1))
                continue
            items = info["item"].astype(str).str.strip()
            m = dict(zip(items, info["value"]))
            ind = None
            for k, v in m.items():
                if k in ("行业", "所处行业"):
                    ind = v
                    break
            if ind is None and "行业" in m:
                ind = m["行业"]
            out["industry_name"] = ind if ind is not None and str(ind).strip() != "" else pd.NA
            return out
        except Exception:
            time.sleep(0.5 + attempt * 0.4)
    return out


def fetch_financial_one(code: str, indicator: str, retries: int = 3) -> pd.DataFrame | None:
    sym = ak_em_symbol(code)
    last_err = None
    for attempt in range(retries):
        try:
            df = ak.stock_financial_analysis_indicator_em(symbol=sym, indicator=indicator)
            if df is None or df.empty:
                return None
            out = df.copy()
            out["stock_code"] = str(code).zfill(6)
            return out
        except Exception as e:
            last_err = e
            time.sleep(0.5 + attempt)
    print(f"  [warn] financial {sym}: {last_err}")
    return None


def _financial_done_codes(path: Path, indicator: str, universe: set[str]) -> set[str]:
    """Stock codes that already have rows for this ``indicator`` (for --resume)."""
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    if "indicator_mode" in df.columns:
        df = df[df["indicator_mode"].fillna("") == indicator]
    return set(df["stock_code"].unique()) & universe


def _require_baostock():
    try:
        import baostock as bs

        return bs
    except ImportError as exc:
        raise SystemExit("使用 --provider baostock 时需要已安装: pip install baostock（你若在别的 env 装了，请先激活对应环境）") from exc


def stock_code_to_baostock(code: str) -> str | None:
    """sh./sz./bj.+六位数，符合 baostock code 格式。"""
    c = str(code).strip().zfill(6)
    if len(c) != 6 or not c.isdigit():
        return None
    if c.startswith("6"):
        return f"sh.{c}"
    if c[0] in ("0", "3"):
        return f"sz.{c}"
    if c[0] in ("4", "8"):
        return f"bj.{c}"
    return None


def _bs_collect_result(rs) -> pd.DataFrame:
    fields = list(getattr(rs, "fields", []) or [])
    rows: list[list] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=fields)
    return pd.DataFrame(rows, columns=fields)


def _parse_bs_exchange_code(bs_code: str) -> str | None:
    s = str(bs_code).strip().lower()
    if "." not in s:
        return None
    _ex, digits = s.split(".", 1)
    digits_only = "".join(ch for ch in digits if ch.isdigit())
    if len(digits_only) >= 6:
        return digits_only[-6:].zfill(6)
    return None


def _float_or_nan(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        v = float(x)
        return v if not np.isnan(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _yoy_frac_to_pct(x) -> float:
    """Growth 表中 YOY* 常为「同比比例」小数（如 ~0.05）或已为百分数口径，统一成「百分点」。"""
    v = _float_or_nan(x)
    if np.isnan(v):
        return np.nan
    if abs(v) <= 2.5:
        return v * 100.0
    return v


def baostock_pull_industry_for_universe(universe: set[str], fetched_at: str) -> pd.DataFrame:
    """申万一级行业：`query_stock_industry` 全市场一次再在内存里筛 CSI500。"""
    bs = _require_baostock()
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    rs = bs.query_stock_industry()
    full = _bs_collect_result(rs)
    bs.logout()

    if full.empty:
        return pd.DataFrame(
            columns=["stock_code", "industry_name", "update_date", "fetched_at", "provider"],
        )

    if "code" not in full.columns:
        raise RuntimeError(f"unexpected query_stock_industry columns: {list(full.columns)}")
    sc = full["code"].map(_parse_bs_exchange_code).astype(str).str.zfill(6)
    full = full.copy()
    full.insert(0, "stock_code", sc)
    sub = full[full["stock_code"].isin(universe)].copy()
    out = pd.DataFrame(
        {
            "stock_code": sub["stock_code"].astype(str).str.zfill(6),
            "industry_name": sub["industry"] if "industry" in sub.columns else pd.NA,
            "industry_classification": sub["industryClassification"]
            if "industryClassification" in sub.columns
            else pd.NA,
            "update_date": sub["updateDate"] if "updateDate" in sub.columns else pd.NA,
        }
    )
    out["fetched_at"] = fetched_at
    out["provider"] = "baostock"
    return out.sort_values("stock_code").reset_index(drop=True)


def recent_year_quarters(n: int) -> list[tuple[int, int]]:
    """从今天所在报告期往回取 n 个已结束季度 (year, quarter)。"""
    today = datetime.now(timezone.utc).date()
    y, mq = today.year, (today.month - 1) // 3 + 1
    out: list[tuple[int, int]] = []
    for _ in range(n):
        out.append((y, mq))
        mq -= 1
        if mq == 0:
            mq = 4
            y -= 1
    return list(reversed(out))


def first_non_null(*vals):
    for v in vals:
        if v is None or v is pd.NA:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none"}:
            return v
    return None


def _merge_one_quarter_row(bs, bs_code: str, year: int, quarter: int) -> dict | None:
    """合并单季 profitability / growth / balance 首行。"""
    dfp = _bs_collect_result(bs.query_profit_data(bs_code, year=year, quarter=quarter))
    dfg = _bs_collect_result(bs.query_growth_data(bs_code, year=year, quarter=quarter))
    dfb = _bs_collect_result(bs.query_balance_data(bs_code, year=year, quarter=quarter))

    def row0(df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return {}
        ser = df.iloc[0]
        return {str(k): ser[k] for k in ser.index}

    P, G, B = row0(dfp), row0(dfg), row0(dfb)

    stat = first_non_null(P.get("statDate"), G.get("statDate"), B.get("statDate"))
    if stat is None and not P and not G and not B:
        return None

    pub = first_non_null(P.get("pubDate"), G.get("pubDate"), B.get("pubDate"))

    if stat is None:
        return None

    rep = pd.to_datetime(stat, errors="coerce")
    if pd.isna(rep):
        return None

    roe = _float_or_nan(P.get("roeAvg"))
    gp_margin = _float_or_nan(P.get("gpMargin"))

    row_out: dict = {
        "REPORT_DATE": rep,
        "NOTICE_DATE": pd.to_datetime(pub, errors="coerce") if pub is not None else pd.NaT,
        "MBRevenue_raw": _float_or_nan(P.get("MBRevenue")),
        "year": year,
        "quarter": quarter,
        "ROEJQ": roe,
        "XSMLL": gp_margin,
    }

    row_out["PARENTNETPROFITTZ"] = _yoy_frac_to_pct(G.get("YOYPNI"))

    lia = _float_or_nan(B.get("liabilityToAsset"))
    if np.isnan(lia):
        row_out["ZCFZL"] = np.nan
    else:
        row_out["ZCFZL"] = lia * (100.0 if abs(lia) <= 1.5 else 1.0)

    row_out["ZCFZLTZ"] = _yoy_frac_to_pct(B.get("YOYLiability"))

    row_out["XSMLL_TB"] = np.nan
    row_out["TOTALOPERATEREVETZ"] = np.nan
    return row_out


def _baostock_fetch_financial_for_stock(bs, stock_code_6: str, n_quarters: int) -> pd.DataFrame | None:
    """单只股票：拉最近 n_quarters 个季度维度，并按 MBRevenue 计算营收同比近似列。"""
    bs_code = stock_code_to_baostock(stock_code_6)
    if bs_code is None:
        return None

    tuples = recent_year_quarters(n_quarters)
    rows_out: list[dict] = []
    for yr, mq in tuples:
        r = _merge_one_quarter_row(bs, bs_code, yr, mq)
        if not r:
            continue
        r["stock_code"] = str(stock_code_6).zfill(6)
        rows_out.append(r)

    if not rows_out:
        return None

    df = pd.DataFrame(rows_out)
    df = df.sort_values("REPORT_DATE").reset_index(drop=True)
    yr = df["REPORT_DATE"].dt.year.astype("Int64")
    qtr = df["REPORT_DATE"].dt.quarter.astype("Int64")
    rev_cur = pd.to_numeric(df["MBRevenue_raw"], errors="coerce")
    prior_rev_list: list[float] = []
    for i in range(len(df)):
        y_i, qi = int(yr.iloc[i]), int(qtr.iloc[i])
        mask = (yr == y_i - 1) & (qtr == qi)
        pr = rev_cur[mask]
        if len(pr):
            prior_rev_list.append(float(pr.iloc[0]))
        else:
            prior_rev_list.append(np.nan)
    prior_series = pd.Series(prior_rev_list, index=df.index)
    denom = prior_series.replace(0, np.nan)
    df["TOTALOPERATEREVETZ"] = np.where(denom.notna(), (rev_cur - prior_series) / denom * 100.0, np.nan)

    drop_cols = [c for c in ("MBRevenue_raw", "year", "quarter", "gpMargin_raw") if c in df.columns]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    prev_gp: list[float] = []
    gp = pd.to_numeric(df["XSMLL"], errors="coerce")
    for i in range(len(df)):
        y_i, qi = int(yr.iloc[i]), int(qtr.iloc[i])
        mask = (yr == y_i - 1) & (qtr == qi)
        pg = gp[mask]
        if len(pg):
            prev_gp.append(float(pg.iloc[0]))
        else:
            prev_gp.append(np.nan)
    prev_gps = pd.Series(prev_gp, index=df.index)
    df["XSMLL_TB"] = np.where((gp.notna()) & prev_gps.notna(), gp - prev_gps, df["XSMLL_TB"])

    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--constituents",
        type=Path,
        default=DATA_DIR / "constituents_csi500.csv",
        help="universe (default: data/constituents_csi500.csv)",
    )
    p.add_argument("--out-dir", type=Path, default=DATA_DIR)
    p.add_argument(
        "--provider",
        default="akshare",
        choices=["akshare", "baostock"],
        help="行业/季频财务数据源（盈利预测仍用 AkShare）",
    )
    p.add_argument(
        "--indicator",
        default="按报告期",
        choices=["按报告期", "按单季度"],
        help="仅 --provider akshare 时有效：stock_financial_analysis_indicator_em 的 indicator",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="AkShare 每股间隔；baostock 在每只股票多条查询之间也会 pause",
    )
    p.add_argument("--max-stocks", type=int, default=0, help="limit concurrent universe (0=all)")
    p.add_argument("--only-profit", action="store_true", help="only profit forecast tables")
    p.add_argument("--only-industry", action="store_true")
    p.add_argument("--only-financial", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing parquet: only fetch missing stocks (financial: no rows yet; "
        "industry: industry_name still null). Merge with old file at the end.",
    )
    p.add_argument(
        "--financial-quarters",
        type=int,
        default=_FINANCIAL_MAX_QUARTERS_DEFAULT,
        metavar="N",
        help=(
            "after each financial pull write only the latest N report rows per stock "
            f"(default {_FINANCIAL_MAX_QUARTERS_DEFAULT}) plus YoY profit/ROE/margin/revenue/leverage cols"
        ),
    )
    args = p.parse_args()
    if args.financial_quarters < 2:
        p.error("--financial-quarters must be >= 2 (need history for QoQ/YoY style features)")
    exclusive = int(args.only_profit) + int(args.only_industry) + int(args.only_financial)
    if exclusive > 1:
        p.error("use at most one of --only-profit/--only-industry/--only-financial")

    if exclusive == 0:
        do_profit = do_industry = do_financial = True
    else:
        do_profit = args.only_profit
        do_industry = args.only_industry
        do_financial = args.only_financial

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.constituents.exists():
        raise SystemExit(f"Missing {args.constituents}; run python download_data.py first.")

    codes = load_codes(args.constituents)
    if args.max_stocks and args.max_stocks > 0:
        codes = codes[: args.max_stocks]

    fetched_at = datetime.now(timezone.utc).isoformat()

    if do_profit:
        print(">> stock_profit_forecast_em(symbol='') …")
        full = fetch_profit_forecast()
        full_path = args.out_dir / "profit_forecast_em.parquet"
        full.to_parquet(full_path, index=False)
        print(f"   wrote {full_path} ({len(full)} rows)")
        univ = set(codes)
        sub = full[full["stock_code"].isin(univ)].reset_index(drop=True)
        sub_path = args.out_dir / "profit_forecast_csi500.parquet"
        sub.to_parquet(sub_path, index=False)
        print(f"   wrote {sub_path} ({len(sub)} CSI500-filtered rows)")

    if do_industry:
        out_name = (
            "stock_industry_bs.parquet" if args.provider == "baostock" else "stock_industry_em.parquet"
        )
        out = args.out_dir / out_name
        old_ind = pd.DataFrame()
        codes_to_fetch = list(codes)
        if args.resume and out.exists():
            old_ind = pd.read_parquet(out)
            old_ind["stock_code"] = old_ind["stock_code"].astype(str).str.zfill(6)
            ok = old_ind.loc[old_ind["industry_name"].notna(), "stock_code"].astype(str).str.zfill(6)
            complete_set = set(ok.tolist())
            codes_to_fetch = [c for c in codes if c not in complete_set]
            lbl = "baostock query_stock_industry" if args.provider == "baostock" else "stock_individual_info_em"
            if codes_to_fetch:
                print(f">> {lbl} (resume): universe={len(codes)}, "
                      f"{len(complete_set)} already have industry, {len(codes_to_fetch)} to fetch …")
            else:
                print(f">> resume: all {len(codes)} stocks already have industry_name; nothing to do.")
        else:
            print(f">> industry ({args.provider}): {len(codes)} stocks …")

        if codes_to_fetch:
            fetch_set_cod = sorted(set(codes_to_fetch))
            if args.provider == "baostock":
                df_new = baostock_pull_industry_for_universe(set(fetch_set_cod), fetched_at)
            else:
                rows_new = []
                for code in tqdm(fetch_set_cod):
                    rows_new.append(fetch_industry_one(code))
                    time.sleep(args.sleep)
                df_new = pd.DataFrame(rows_new)
                df_new["fetched_at"] = fetched_at

            if df_new.empty and args.provider == "baostock":
                print("   WARNING: baostock industry returned empty for requested codes; parquet unchanged.")

            merged_fetch = (
                set(df_new["stock_code"].astype(str).str.zfill(6)) if not df_new.empty else set()
            )
            if df_new.empty:
                if not old_ind.empty:
                    print("   WARNING: no industry rows returned; existing parquet unchanged.")
            elif not old_ind.empty:
                mask = ~old_ind["stock_code"].isin(merged_fetch)
                df_ind = pd.concat([old_ind.loc[mask], df_new], ignore_index=True)
                df_ind = df_ind.sort_values("stock_code").reset_index(drop=True)
                df_ind.to_parquet(out, index=False)
                print(f"   wrote {out} ({len(df_ind)} rows)")
            else:
                df_ind = df_new.sort_values("stock_code").reset_index(drop=True)
                df_ind.to_parquet(out, index=False)
                print(f"   wrote {out} ({len(df_ind)} rows)")
        elif args.resume and old_ind.shape[0] > 0 and not codes_to_fetch:
            print(f"   kept {out} ({len(old_ind)} rows, unchanged)")

    if do_financial:
        out_fin = (
            args.out_dir / "financial_quarter_bs.parquet"
            if args.provider == "baostock"
            else args.out_dir / "financial_analysis_indicator_em.parquet"
        )

        bs_mode = args.provider == "baostock"
        fin_indicator = _BS_FINANCIAL_MODE if bs_mode else args.indicator

        universe_set = set(codes)
        old_fin = pd.DataFrame()
        codes_to_fetch = list(codes)
        if args.resume and out_fin.exists():
            old_fin = pd.read_parquet(out_fin)
            done = _financial_done_codes(out_fin, fin_indicator, universe_set)
            codes_to_fetch = [c for c in codes if c not in done]
            if codes_to_fetch:
                src = "baostock profit/growth/balance_by_quarter" if bs_mode else "stock_financial_analysis_indicator_em"
                print(f">> financial (resume) {src}: indicator={fin_indicator}, "
                      f"universe={len(codes)}, {len(done)} already in parquet, "
                      f"{len(codes_to_fetch)} to fetch …")
            else:
                print(
                    f">> resume: all {len(codes)} stocks already have financial rows "
                    f"for indicator_mode={fin_indicator}; nothing to do."
                )
        else:
            if bs_mode:
                print(f">> baostock quarter financial: {len(codes)} stocks …")
            else:
                print(f">> stock_financial_analysis_indicator_em(.., indicator={args.indicator}): {len(codes)} stocks …")

        chunks: list = []
        success_codes: set[str] = set()
        if bs_mode:
            bs = _require_baostock()
            lg = bs.login()
            if lg.error_code != "0":
                raise RuntimeError(f"baostock login failed: {lg.error_msg}")
            # 与 --financial-quarters 一致，只拉 N 季；同比营收/毛利率差若缺上年同季会在列里为 NaN。
            n_fetch_quarters = args.financial_quarters
            try:
                for code in tqdm(codes_to_fetch):
                    block = _baostock_fetch_financial_for_stock(bs, code, n_fetch_quarters)
                    if block is not None and not block.empty:
                        block["indicator_mode"] = fin_indicator
                        block["fetched_at"] = fetched_at
                        chunks.append(block)
                        success_codes.add(str(code).zfill(6))
                    time.sleep(max(0.0, args.sleep))
            finally:
                bs.logout()
        else:
            for code in tqdm(codes_to_fetch):
                block = fetch_financial_one(code, args.indicator)
                if block is not None and not block.empty:
                    block["indicator_mode"] = args.indicator
                    block["fetched_at"] = fetched_at
                    chunks.append(block)
                    success_codes.add(str(code).zfill(6))
                time.sleep(args.sleep)

        if chunks:
            new_fin = pd.concat(chunks, ignore_index=True)
            if args.resume and not old_fin.empty:
                if "indicator_mode" in old_fin.columns:
                    im = old_fin["indicator_mode"].fillna("")
                    old_other = old_fin.loc[im != fin_indicator].copy()
                    old_same = old_fin.loc[im == fin_indicator].copy()
                else:
                    old_other = pd.DataFrame()
                    old_same = old_fin.copy()
                old_same["stock_code"] = old_same["stock_code"].astype(str).str.zfill(6)
                old_keep = old_same[~old_same["stock_code"].isin(success_codes)]
                fin = pd.concat([old_other, old_keep, new_fin], ignore_index=True)
            else:
                fin = new_fin
            fin = slim_financial_for_disk(fin, max_quarters=args.financial_quarters)
            fin.to_parquet(out_fin, index=False)
            print(f"   wrote {out_fin} ({len(fin)} rows, slim ≤{args.financial_quarters} quarters / key cols)")
            cols_show = [
                c
                for c in (
                    "stock_code",
                    "REPORT_DATE",
                    "PARENTNETPROFITTZ",
                    "ROEJQ",
                    "XSMLL",
                    "TOTALOPERATEREVETZ",
                    "ZCFZL",
                )
                if c in fin.columns
            ]
            if cols_show:
                print(f"   sample columns: {cols_show}")
        elif args.resume and not old_fin.empty and not codes_to_fetch:
            print(f"   kept {out_fin} ({len(old_fin)} rows, unchanged)")
        elif args.resume and not old_fin.empty and codes_to_fetch and not chunks:
            print(
                "   WARNING: no new financial blocks this run; existing parquet unchanged "
                "(all fetches empty or failed)."
            )
        else:
            print("   WARNING: no financial blocks downloaded.")

    print(">> Done.")


if __name__ == "__main__":
    main()
