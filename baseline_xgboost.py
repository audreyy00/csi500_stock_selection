"""
XGBoost baseline for the CSI500 stock-selection competition.

Pipeline
--------
1. Load data/prices_csi500.parquet（仅 CSI500 成分，经 constituents_csi500 过滤）
2. Build features + 5-day forward target (features.py)
3. Train XGBoost on all but the last `EMBARGO_DAYS` training rows
   (默认 ``--sample-weight-has-triple`` 1.5=mild，三连季满配行小幅加权；设为 1 则均匀)
4. Validate on those held-out rows (reports rank IC as sanity check)
5. Predict on the most recent date
6. Build a portfolio: top-K names, score-weighted with the 10% cap

Usage
-----
  python baseline_xgboost.py                       # predict from latest data
  python baseline_xgboost.py --as-of 20260503      # predict as of a given date
  python baseline_xgboost.py --top-k 50 --out submissions/week1.csv
  python baseline_xgboost.py --as-of 20260414 --train-from 2026-01-01 \\
      --out submissions/baseline_recent.csv  # recent-window train floor
"""
from __future__ import annotations

import argparse
from pathlib import Path

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
    PRICES_CSI500_PARQUET,
    build_features,
    filter_prices_to_csi500_constituents,
    prediction_frame,
    training_frame,
)

EMBARGO_DAYS = 5            # gap between train end and val start (>= FORWARD_HORIZON
                            # so training labels don't leak into validation features)
VAL_DAYS = 40               # trailing trading days reserved for validation / early-stopping eval
MIN_TRAINING_DATES_EXTRA = 20
# 三连季可行样本加权：用户保留 mild_sw（弱化极少数行的支配力）
DEFAULT_SAMPLE_WEIGHT_HAS_TRIPLE = 1.5
MIN_STOCKS = 30             # rule: portfolio must hold >= 30 names
MAX_WEIGHT = 0.10           # rule: per-stock weight cap
DEFAULT_TOP_K = 50          # baseline picks top-50 by predicted score


def _interaction_constraint_groups(
    feature_names: list[str], groups_csv: str | None
) -> list[list[int]] | None:
    """Comma-separated groups: each group uses ``a|b|c`` inner list (interact allowed)."""
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
    interaction_constraints: list[list[int]] | None = None,
    sample_weight_has_triple: float | None = None,
) -> xgb.XGBRegressor:
    """若 ``sample_weight_has_triple>1``：训练损失对 ``financial_has_triple==1`` 的样本乘以该倍数；验证集不加权。"""
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
    model = xgb.XGBRegressor(**kwargs)
    sample_weight = None
    if sample_weight_has_triple is not None and sample_weight_has_triple > 1.0:
        if "financial_has_triple" not in train_df.columns:
            raise ValueError("sample_weight_has_triple requires column financial_has_triple")
        ht = pd.to_numeric(train_df["financial_has_triple"], errors="coerce").fillna(0).to_numpy()
        sample_weight = np.where(ht > 0.5, float(sample_weight_has_triple), 1.0)
    eval_set = [(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])]
    if sample_weight is not None:
        model.fit(
            train_df[FEATURE_COLUMNS],
            train_df[TARGET_COLUMN],
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=False,
        )
    else:
        model.fit(
            train_df[FEATURE_COLUMNS],
            train_df[TARGET_COLUMN],
            eval_set=eval_set,
            verbose=False,
        )
    return model


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    """Daily cross-sectional Spearman correlation, averaged over dates."""
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
    """Top-K names, weight proportional to (rank) then capped at MAX_WEIGHT.

    We use rank-weights rather than score-weights so pathological score scales
    do not produce a single dominant name.  After capping at 10% we redistribute
    spillover to uncapped names and iterate until feasible.
    """
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS} (rule)")
    chosen = scores.sort_values(ascending=False).head(top_k).copy()
    n = len(chosen)
    if n == 0:
        raise ValueError("build_portfolio: no scores (empty or all-NaN).")
    if n < MIN_STOCKS:
        raise ValueError(
            f"build_portfolio: only {n} usable names "
            f"(need >= {MIN_STOCKS} per competition rule). Often fixed by widening "
            f"feature coverage on predict day (fewer NaNs in inputs) — here top_k={top_k} "
            f"but the score series only had this many finite rows."
        )

    # Rank-based weights (best stock gets largest weight, then normalize).
    ranks = np.arange(n, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    # Iteratively cap at MAX_WEIGHT and redistribute to uncapped names.
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
    p.add_argument(
        "--interaction-constraints",
        default=None,
        metavar="GROUPS",
        help="Semicolon-separated feature groups; columns in a group use |. Example: "
             "'financial_accel|financial_fresh|analyst_growth'",
    )
    p.add_argument(
        "--fresh-half-life-days",
        type=float,
        default=None,
        metavar="DAYS",
        help=(
            f"financial_fresh exponential half-life (calendar days). "
            f"default {DEFAULT_FRESH_HALF_LIFE_DAYS:g} if omitted"
        ),
    )
    p.add_argument(
        "--event-gate-calendar-days",
        type=int,
        default=None,
        metavar="N",
        help=(
            "calendar window [0,N] for financial_evt_gate after notice. "
            f"default {DEFAULT_EVENT_GATE_CALENDAR_DAYS} if omitted"
        ),
    )
    p.add_argument(
        "--min-names-per-industry-day",
        type=int,
        default=None,
        metavar="K",
        help=(
            "min names per (date, industry) for financial_accel_industry_z. "
            f"default {DEFAULT_MIN_NAMES_PER_INDUSTRY_DAY} if omitted"
        ),
    )
    p.add_argument(
        "--sample-weight-has-triple",
        type=float,
        default=DEFAULT_SAMPLE_WEIGHT_HAS_TRIPLE,
        metavar="MULT",
        help=(
            "Train row weight multiplier when financial_has_triple==1. "
            f"Default {DEFAULT_SAMPLE_WEIGHT_HAS_TRIPLE:g} (mild). "
            "Use 1.0 for uniform / no uplift on triple rows."
        ),
    )
    p.add_argument(
        "--train-from",
        default=None,
        metavar="DATE",
        help=(
            "Lower bound on training-panel dates (YYYY-MM-DD or YYYYMMDD), applied after leakage cap. "
            "Use e.g. 2026-01-01 for 2026-only training."
        ),
    )
    args = p.parse_args()

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    prices = filter_prices_to_csi500_constituents(prices)
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
          f"dates {prices['date'].min().date()} to {prices['date'].max().date()}")

    print(">> Building features")
    kw: dict = {}
    if args.fresh_half_life_days is not None:
        kw["fresh_half_life_days"] = args.fresh_half_life_days
    if args.event_gate_calendar_days is not None:
        kw["event_gate_calendar_days"] = args.event_gate_calendar_days
    if args.min_names_per_industry_day is not None:
        kw["min_names_per_industry_day"] = args.min_names_per_industry_day
    if kw:
        print(
            "   finance overrides:",
            ", ".join(f"{k}={kw[k]}" for k in kw),
        )
    else:
        print(
            f"   finance defaults: half_life={DEFAULT_FRESH_HALF_LIFE_DAYS}, "
            f"gate_calendar={DEFAULT_EVENT_GATE_CALENDAR_DAYS}, "
            f"min_industry_n={DEFAULT_MIN_NAMES_PER_INDUSTRY_DAY}",
        )
    panel = build_features(prices, **kw)
    # Bound training data so backtesting with --as-of doesn't leak future rows.
    # Training uses features from date t with target = close(t+FORWARD_HORIZON),
    # so we cap training dates at as_of - FORWARD_HORIZON trading days.
    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = training_frame(panel, max_date=train_cutoff)

    if args.train_from:
        ts_from = pd.Timestamp(args.train_from)
        n_prev = len(train_pool)
        train_pool = train_pool[train_pool["date"] >= ts_from].copy()
        nd = train_pool["date"].nunique()
        if nd == 0:
            raise RuntimeError(
                f"--train-from {args.train_from!r} yields no rows (was {n_prev:,} before slice)."
            )
        print(f"   train-from {ts_from.date()}: rows {len(train_pool):,}/{n_prev:,}, {nd} dates")

    # Time-based split with embargo:
    #   [ ... train ... | embargo (discarded) | val (last VAL_DAYS) ]
    # The embargo prevents training labels (5-day forward) from reaching into
    # dates whose prices also feed the validation features.
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

    ic_groups = _interaction_constraint_groups(
        FEATURE_COLUMNS, args.interaction_constraints
    )
    sw = args.sample_weight_has_triple
    if sw is not None and sw > 1.0:
        n_tri = int((pd.to_numeric(train_df["financial_has_triple"], errors="coerce").fillna(0) > 0.5).sum())
        print(f"   sample_weight: triple rows get x{sw:g} ({n_tri:,}/{len(train_df):,})")
    print(">> Training XGBoost")
    model = train_model(
        train_df,
        val_df,
        interaction_constraints=ic_groups,
        sample_weight_has_triple=sw,
    )

    val_pred = model.predict(val_df[FEATURE_COLUMNS])
    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())
    print(f"   validation rank IC: {ic:.4f}")

    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=args.as_of)
    if pred_df.empty:
        raise RuntimeError(f"No rows available for as_of={args.as_of}. Check data.")
    pred_date = pred_df["date"].iloc[0]
    print(f"   as of {pred_date.date()}, scoring {len(pred_df)} stocks")

    pred_df = pred_df.assign(score=model.predict(pred_df[FEATURE_COLUMNS]))
    scores = pred_df.set_index("stock_code")["score"]
    weights = build_portfolio(scores, top_k=args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(out_path, index=False)
    print(f">> Wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} "
          f"max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")


if __name__ == "__main__":
    main()
