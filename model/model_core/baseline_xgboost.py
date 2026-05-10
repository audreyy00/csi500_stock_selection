"""
XGBoost baseline for the CSI500 stock-selection competition.

Pipeline
--------
1. Load CSI500 prices → ``build_features``（含 ``target_5d``）
2. 可选 ``attach_cross_sectional_rank_labels`` → ``target_topk_bin`` + ``target_rank_pct``
3. Train / validate；``topk_binary`` → ``XGBClassifier(logistic)``，推断 ``predict_proba`` 排序
4. ``prediction_frame`` → 组合权重

Usage
-----
  python baseline_xgboost.py
  python baseline_xgboost.py --label-mode topk_binary --top-pct-threshold 0.9
  python baseline_xgboost.py --label-mode forward_return
  python baseline_xgboost.py --as-of 20260503 --top-k 50 --out submissions/week1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from financial_analysis import DEFAULT_FRESH_HALF_LIFE_DAYS
from financial_extended import (
    DEFAULT_EVENT_GATE_CALENDAR_DAYS,
    DEFAULT_MIN_NAMES_PER_INDUSTRY_DAY,
)
from features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FORWARD_HORIZON,
    LABEL_TOPK_BINARY,
    PRICES_CSI500_PARQUET,
    attach_cross_sectional_rank_labels,
    build_features,
    filter_prices_to_csi500_constituents,
    prediction_frame,
    training_frame,
)

EMBARGO_DAYS = 5
VAL_DAYS = 40
MIN_TRAINING_DATES_EXTRA = 20
DEFAULT_SAMPLE_WEIGHT_HAS_TRIPLE = 1.5
MIN_STOCKS = 30
MAX_WEIGHT = 0.10
DEFAULT_TOP_K = 50


def _interaction_constraint_groups(
    feature_names: list[str], groups_csv: str | None
) -> list[list[int]] | None:
    if not groups_csv or not groups_csv.strip():
        return None
    out: list[list[int]] = []
    for grp in groups_csv.strip().split(";"):
        grp = grp.strip()
        if not grp:
            continue
        cols = [c.strip() for c in grp.split("|") if c.strip()]
        idx = []
        for c in cols:
            if c not in feature_names:
                raise ValueError(f"interaction column {c!r} not in FEATURE_COLUMNS")
            idx.append(feature_names.index(c))
        if idx:
            out.append(idx)
    return out if out else None


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    label_column: str,
    classifier: bool,
    interaction_constraints: list[list[int]] | None = None,
    sample_weight_has_triple: float | None = None,
) -> Union[xgb.XGBRegressor, xgb.XGBClassifier]:
    kwargs = dict(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    if interaction_constraints is not None:
        kwargs["interaction_constraints"] = interaction_constraints
    if classifier:
        model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", **kwargs)
    else:
        model = xgb.XGBRegressor(objective="reg:squarederror", **kwargs)

    sample_weight = None
    if sample_weight_has_triple is not None and sample_weight_has_triple > 1.0:
        if "financial_has_triple" not in train_df.columns:
            raise ValueError("sample_weight_has_triple requires column financial_has_triple")
        ht = pd.to_numeric(train_df["financial_has_triple"], errors="coerce").fillna(0).to_numpy()
        sample_weight = np.where(ht > 0.5, float(sample_weight_has_triple), 1.0)

    y_tr = train_df[label_column]
    y_va = val_df[label_column]
    eval_set = [(val_df[FEATURE_COLUMNS], y_va)]

    fit_kw = dict(verbose=False)
    if sample_weight is not None:
        model.fit(
            train_df[FEATURE_COLUMNS],
            y_tr,
            sample_weight=sample_weight,
            eval_set=eval_set,
            **fit_kw,
        )
    else:
        model.fit(train_df[FEATURE_COLUMNS], y_tr, eval_set=eval_set, **fit_kw)
    return model


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def build_portfolio(scores: pd.Series, top_k: int = DEFAULT_TOP_K) -> pd.Series:
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS} (rule)")
    chosen = scores.sort_values(ascending=False).head(top_k).copy()
    n = len(chosen)
    if n == 0:
        raise ValueError("build_portfolio: no scores (empty or all-NaN).")
    if n < MIN_STOCKS:
        raise ValueError(
            f"build_portfolio: only {n} usable names "
            f"(need >= {MIN_STOCKS} per competition rule)."
        )

    ranks = np.exp(np.linspace(2, 0, n))
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()

    assert abs(w.sum() - 1.0) < 1e-6, f"weights sum to {w.sum()}"
    assert (w <= MAX_WEIGHT + 1e-9).all(), "cap violated"
    assert (w > 0).sum() >= MIN_STOCKS, "too few names"
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(PRICES_CSI500_PARQUET))
    p.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date in data")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--interaction-constraints", default=None, metavar="GROUPS")
    p.add_argument(
        "--fresh-half-life-days",
        type=float,
        default=None,
        metavar="DAYS",
        help=f"default {DEFAULT_FRESH_HALF_LIFE_DAYS:g} if omitted",
    )
    p.add_argument(
        "--event-gate-calendar-days",
        type=int,
        default=None,
        metavar="N",
        help=f"default {DEFAULT_EVENT_GATE_CALENDAR_DAYS} if omitted",
    )
    p.add_argument(
        "--min-names-per-industry-day",
        type=int,
        default=None,
        metavar="K",
        help=f"default {DEFAULT_MIN_NAMES_PER_INDUSTRY_DAY} if omitted",
    )
    p.add_argument(
        "--sample-weight-has-triple",
        type=float,
        default=DEFAULT_SAMPLE_WEIGHT_HAS_TRIPLE,
        metavar="MULT",
    )
    p.add_argument("--train-from", default=None, metavar="DATE")
    p.add_argument(
        "--label-mode",
        choices=("topk_binary", "forward_return"),
        default="topk_binary",
        help="topk_binary：截面分位 + logistic；forward_return：回归 target_5d",
    )
    p.add_argument(
        "--top-pct-threshold",
        type=float,
        default=0.9,
        metavar="T",
        help="topk 正例：rank_pct≥T（默认 0.9≈前 10%%）",
    )
    args = p.parse_args()

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    prices = filter_prices_to_csi500_constituents(prices)
    print(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Building features")
    kw: dict = {}
    if args.fresh_half_life_days is not None:
        kw["fresh_half_life_days"] = args.fresh_half_life_days
    if args.event_gate_calendar_days is not None:
        kw["event_gate_calendar_days"] = args.event_gate_calendar_days
    if args.min_names_per_industry_day is not None:
        kw["min_names_per_industry_day"] = args.min_names_per_industry_day
    panel = build_features(prices, **kw)

    use_topk = args.label_mode == "topk_binary"
    if use_topk:
        panel = attach_cross_sectional_rank_labels(
            panel, top_pct_threshold=args.top_pct_threshold
        )
        print(f"   labels: {LABEL_TOPK_BINARY} (rank_pct>={args.top_pct_threshold:g})")

    train_label_col: str | None = LABEL_TOPK_BINARY if use_topk else None
    y_col = LABEL_TOPK_BINARY if use_topk else TARGET_COLUMN

    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = training_frame(panel, max_date=train_cutoff, label_column=train_label_col)

    if args.train_from:
        ts_from = pd.Timestamp(args.train_from)
        n_prev = len(train_pool)
        train_pool = train_pool[train_pool["date"] >= ts_from].copy()
        if train_pool["date"].nunique() == 0:
            raise RuntimeError(f"--train-from {args.train_from!r} yields no rows.")
        print(f"   train-from {ts_from.date()}: rows {len(train_pool):,}/{n_prev:,}")

    all_dates = np.sort(train_pool["date"].unique())
    if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 20:
        raise RuntimeError("Not enough dates to train; download more history.")
    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]

    print(f"   train: {len(train_df):,} rows up to {train_end.date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {val_start.date()}")

    ic_groups = _interaction_constraint_groups(FEATURE_COLUMNS, args.interaction_constraints)
    sw = args.sample_weight_has_triple
    if sw is not None and sw > 1.0:
        n_tri = int(
            (pd.to_numeric(train_df["financial_has_triple"], errors="coerce").fillna(0) > 0.5).sum()
        )
        print(f"   sample_weight: triple rows get x{sw:g} ({n_tri:,}/{len(train_df):,})")

    print(">> Training XGBoost")
    model = train_model(
        train_df,
        val_df,
        label_column=y_col,
        classifier=use_topk,
        interaction_constraints=ic_groups,
        sample_weight_has_triple=sw,
    )

    if use_topk:
        val_pred = model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    else:
        val_pred = model.predict(val_df[FEATURE_COLUMNS])
    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())
    print(f"   validation rank IC (pred vs {TARGET_COLUMN}): {ic:.4f}")

    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=args.as_of)
    if pred_df.empty:
        raise RuntimeError(f"No rows available for as_of={args.as_of}. Check data.")
    pred_date = pred_df["date"].iloc[0]
    print(f"   as of {pred_date.date()}, scoring {len(pred_df)} stocks")

    if use_topk:
        pred_scores = model.predict_proba(pred_df[FEATURE_COLUMNS])[:, 1]
    else:
        pred_scores = model.predict(pred_df[FEATURE_COLUMNS])
    pred_df = pred_df.assign(score=pred_scores)
    scores = pred_df.set_index("stock_code")["score"]
    weights = build_portfolio(scores, top_k=args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(out_path, index=False)
    print(f">> Wrote {len(out)} names to {out_path}")
    print(
        f"   weight summary: min={out['weight'].min():.4f} "
        f"max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()
