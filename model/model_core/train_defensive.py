"""
防御模型：与 ``baseline_xgboost`` 并列的流水线。

* 面板来自 ``features.build_features``；行情默认 **仅 CSI500**（``prices_csi500`` + ``constituents_csi500`` 过滤）。
* **本文件自拟** ``DEFENSIVE_FEATURE_COLUMNS`` / ``DEFENSIVE_FEATURES_NA_OK``，**不依赖**
  ``features.FEATURE_COLUMNS``；改 baseline 因子不会自动改防御模型。
* 标签口径仍关联 ``features`` 的 ``TARGET_COLUMN`` / ``FORWARD_HORIZON``。

用法（在项目根目录）::

    python train_defensive.py --train-end 2025-12-31 --out models/defensive_v1.json
    python train_defensive.py --as-of 20260424 --out submissions/eval_asof_20260424.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from baseline_xgboost import DEFAULT_TOP_K, build_portfolio
from consensus_logic import ANALYST_FEATURE_COLUMNS
from defensive_label import make_defensive_label
from financial_analysis import FINANCIAL_FEATURE_COLUMNS
from financial_extended import FINANCIAL_EXT_FEATURE_COLUMNS
from features import (
    TARGET_COLUMN,
    FORWARD_HORIZON,
    INDEX_CSI500_PARQUET,
    PRICES_CSI500_PARQUET,
    SEARCHLIGHT_COLUMNS,
    build_features,
    filter_prices_to_csi500_constituents,
    _replace_inf_nan_features,
)
from regime import REGIME_FEATURE_COLUMNS

# Searchlight：与 ``features.SEARCHLIGHT_COLUMNS`` 一致（斜率 + 加速度 + 斜率截面分位秩）。
DEFENSIVE_SEARCHLIGHT_COLUMNS: tuple[str, ...] = SEARCHLIGHT_COLUMNS

# ---------------------------------------------------------------------------
# 防御：`features.py` **只产出**列（含扩展风险）；送进 XGB 谁用哪列由本清单决定，
# **与 baseline 的 ``FEATURE_COLUMNS`` 独立演进**。
# ---------------------------------------------------------------------------
DEFENSIVE_FEATURE_COLUMNS: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "vol_20d",
    "volume_z_20d",
    "turnover_ma_20d",
    "close_over_ma20",
    "close_over_ma60",
    "rsi_14",

    "ret_3d_rank",
    "ret_5d_rank",
    "ret_20d_rank",
    "vol_20d_rank",

    "regime_bucket",
    "regime_corr_vp",
    "regime_compact_stable",
    "regime_vp_div",
    "regime_vp_div_vol",

    *DEFENSIVE_SEARCHLIGHT_COLUMNS,

    # "down_vol_20d",
    # "max_dd_20d",
    # "beta_20d",
    # "vol_60d",
    # "amp_20d",
    # "turnover_std_20d",
]

DEFENSIVE_FEATURES_NA_OK: frozenset[str] = (
    frozenset(ANALYST_FEATURE_COLUMNS) - frozenset({"has_coverage"})
) | (
    frozenset(FINANCIAL_FEATURE_COLUMNS) - frozenset({"financial_has_triple"})
) | (
    frozenset(FINANCIAL_EXT_FEATURE_COLUMNS) - frozenset({"financial_evt_gate"})
) | frozenset({"financial_days_since_notice"}) | (
    frozenset(REGIME_FEATURE_COLUMNS) - frozenset({"regime_bucket"})
) | frozenset(DEFENSIVE_SEARCHLIGHT_COLUMNS)


def _required_defensive_columns() -> list[str]:
    return [c for c in DEFENSIVE_FEATURE_COLUMNS if c not in DEFENSIVE_FEATURES_NA_OK]


def defensive_training_frame(
    panel: pd.DataFrame, min_date=None, max_date=None
) -> pd.DataFrame:
    """与 ``features.training_frame`` 同旨，但以防御侧列清单做 dropna / inf 清理。"""
    req = _required_defensive_columns() + [TARGET_COLUMN]
    missing = set(req) - set(panel.columns)
    if missing:
        raise ValueError(
            "Panel missing defensive training columns — add merges to ``build_features`` or "
            "remove names from DEFENSIVE_FEATURE_COLUMNS. Missing: "
            f"{sorted(missing)}"
        )
    df = panel.dropna(subset=req).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    _replace_inf_nan_features(df, list(DEFENSIVE_FEATURE_COLUMNS))
    return df


def defensive_prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """推断日单行，可用列与防御训练路径一致。"""
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    req = _required_defensive_columns()
    missing = set(req) - set(panel.columns)
    if missing:
        raise ValueError(f"Panel missing columns for defensive predict: {sorted(missing)}")
    df = panel[panel["date"] == as_of].dropna(subset=req).copy()
    _replace_inf_nan_features(df, list(DEFENSIVE_FEATURE_COLUMNS))
    return df


def _load_index_daily_returns(index_parquet: Path) -> pd.Series:
    if not index_parquet.is_file():
        raise FileNotFoundError(
            f"Index parquet not found: {index_parquet}. "
            "Use download_data.py or pass --index path to CSI500 daily data (close column)."
        )
    index_df = pd.read_parquet(index_parquet)
    if "date" not in index_df.columns or "close" not in index_df.columns:
        raise ValueError(f"{index_parquet} must contain columns: date, close")
    index_df = index_df.copy()
    index_df["date"] = pd.to_datetime(index_df["date"])
    index_df = index_df.sort_values("date")
    index_df["ret_1d"] = pd.to_numeric(index_df["close"], errors="coerce").pct_change(1)
    return index_df.set_index("date")["ret_1d"]


def train_defensive_model(
    *,
    prices_path: Path,
    index_path: Path,
    model_save_path: Path,
    train_start: str | None,
    train_end: str | None,
    as_of_for_train_cap: str | None,
    horizon: int,
    lambda_penalty: float,
    alpha_weight: float,
    n_estimators: int,
    seed: int,
) -> tuple[xgb.XGBRegressor, pd.DataFrame]:
    print(">> Loading index (CSI500 benchmark daily returns)")
    index_returns = _load_index_daily_returns(index_path)
    print(f"   index dates: {index_returns.index.min().date()} … {index_returns.index.max().date()}")

    print(">> Loading prices")
    prices = pd.read_parquet(prices_path)
    prices = filter_prices_to_csi500_constituents(prices)
    print(f"   rows={len(prices):,}（已按 constituents_csi500 过滤）")

    print(">> Building feature panel (``build_features``, defensive picks its own columns)")
    panel = build_features(prices)
    print(f"   panel rows={len(panel):,}")

    mf = set(DEFENSIVE_FEATURE_COLUMNS) - set(panel.columns)
    if mf:
        raise ValueError(
            "DEFENSIVE_FEATURE_COLUMNS 中有列不在面板上 — 请先扩展 ``build_features`` 或删掉: "
            f"{sorted(mf)}"
        )

    max_train_calendar: pd.Timestamp | None = None
    if as_of_for_train_cap is not None:
        as_ts = pd.Timestamp(as_of_for_train_cap)
        trading_dates = np.sort(panel["date"].unique())
        as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_ts)))
        cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
        max_train_calendar = pd.Timestamp(trading_dates[cutoff_idx])

    print(">> Defensive label (excess vs index − λ·downside)")
    panel["target_defensive"] = make_defensive_label(
        panel,
        index_returns=index_returns,
        horizon=horizon,
        lambda_penalty=lambda_penalty,
    )

    min_d = pd.Timestamp(train_start) if train_start else None
    max_d_user = pd.Timestamp(train_end) if train_end else None
    max_d = max_train_calendar
    if max_d is not None and max_d_user is not None:
        max_d = min(max_d, max_d_user)
    elif max_d is None:
        max_d = max_d_user

    print(
        ">> Training slice "
        "(dropna: ``DEFENSIVE_FEATURE_COLUMNS`` + ``target_5d``; y = ``target_defensive``)"
    )
    if max_train_calendar is not None:
        print(f"   max_train cap (no label peek past as-of): {max_train_calendar.date()}")
    train_df = defensive_training_frame(panel, min_date=min_d, max_date=max_d)

    if train_df.empty:
        raise RuntimeError(
            "defensive_training_frame is empty — widen dates or relax DEFENSIVE_FEATURES_NA_OK / columns."
        )

    y_train = pd.to_numeric(train_df["target_defensive"], errors="coerce")
    if not np.isfinite(y_train.to_numpy(dtype=float)).any():
        raise RuntimeError("target_defensive has no finite values after slice.")

    X_train = train_df[DEFENSIVE_FEATURE_COLUMNS]
    print(f"   samples: {len(X_train):,}  features: {len(DEFENSIVE_FEATURE_COLUMNS)}")
    finite_y = np.isfinite(y_train.to_numpy())
    print(f"   target_defensive mean={y_train.where(finite_y).mean():.4f} "
          f"std={y_train.where(finite_y).std():.4f}")

    iret = pd.to_numeric(train_df["date"].map(index_returns), errors="coerce")
    iret_finite = iret.notna()
    down = iret < 0
    sample_weight_arr = np.where(down.to_numpy(), 1.0 + float(alpha_weight), 1.0)
    sample_weight_arr = np.where(np.isfinite(iret.to_numpy()), sample_weight_arr, 1.0)
    sample_weight = pd.Series(sample_weight_arr, index=train_df.index, dtype=np.float64)
    down_share = float((iret[iret_finite] < 0).mean()) if iret_finite.any() else float("nan")
    print(
        f"   sample weights: index ret<0 rows x{1.0 + alpha_weight:.1f}, else 1.0; "
        f"down-day row share={down_share:.2%}"
    )

    print(">> Fit XGBRegressor")
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_save_path)
    print(f">> Saved model → {model_save_path}")

    importance = pd.DataFrame(
        {"feature": DEFENSIVE_FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nTop 10 features by gain-based importance:")
    print(importance.head(10).to_string(index=False))

    return model, panel


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Train defensive XGBoost (penalized-downside label); optional submission CSV when --as-of."
    )
    p.add_argument(
        "--prices",
        type=Path,
        default=PRICES_CSI500_PARQUET,
        help="默认 data/prices_csi500.parquet；读取后按 constituents_csi500 再过滤",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Without --as-of: XGBoost model JSON path (default: models/defensive_v1.json). "
        "With --as-of: submission CSV path (required).",
    )
    p.add_argument(
        "--model-save",
        type=Path,
        default=None,
        help="When --as-of: where to save the trained model JSON (default: models/defensive_v1.json). "
        "Ignored when only training without --as-of (use --out for model path).",
    )
    p.add_argument("--as-of", default=None, help="YYYYMMDD; build submission for this date after training")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="portfolio size (competition rule ≥30)")
    p.add_argument("--train-start", default=None, help="YYYY-MM-DD inclusive; omit = no lower bound")
    p.add_argument("--train-end", default=None, help="YYYY-MM-DD inclusive; omit = no upper bound")
    p.add_argument("--horizon", type=int, default=5, help="forward days; should match offensive FORWARD_HORIZON")
    p.add_argument(
        "--index",
        type=Path,
        default=None,
        help="CSI500 指数 parquet（date, close）。默认 data/index_csi500.parquet",
    )
    p.add_argument(
        "--lambda-penalty",
        type=float,
        default=1.0,
        help="Multiplier on max(0, -stock future ret) in defensive label.",
    )
    p.add_argument(
        "--alpha-weight",
        type=float,
        default=2.0,
        help="Extra train loss weight on rows where index daily ret < 0 (1 + alpha on those rows).",
    )
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    index_path = args.index if args.index is not None else INDEX_CSI500_PARQUET
    as_of_ts: pd.Timestamp | None = pd.Timestamp(args.as_of) if args.as_of else None
    if as_of_ts is not None:
        submission_out = args.out
        if submission_out is None:
            raise SystemExit("With --as-of, pass --out path for the submission CSV.")
        model_save = args.model_save if args.model_save is not None else root / "models" / "defensive_v1.json"
    else:
        submission_out = None
        model_save = args.out if args.out is not None else root / "models" / "defensive_v1.json"

    model, panel = train_defensive_model(
        prices_path=args.prices,
        index_path=index_path,
        model_save_path=model_save,
        train_start=args.train_start,
        train_end=args.train_end,
        as_of_for_train_cap=args.as_of,
        horizon=args.horizon,
        lambda_penalty=args.lambda_penalty,
        alpha_weight=args.alpha_weight,
        n_estimators=args.n_estimators,
        seed=args.seed,
    )

    if as_of_ts is not None:
        print(">> Predicting portfolio (defensive model)")
        pred_df = defensive_prediction_frame(panel, as_of=args.as_of)
        if pred_df.empty:
            raise RuntimeError(f"No rows for as_of={args.as_of}. Check data / feature dropna.")
        pred_df = pred_df.assign(score=model.predict(pred_df[DEFENSIVE_FEATURE_COLUMNS]))
        scores = pred_df.set_index("stock_code")["score"]
        weights = build_portfolio(scores, top_k=args.top_k)
        submission_out = Path(submission_out)
        submission_out.parent.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
        out.to_csv(submission_out, index=False)
        print(f">> Wrote {len(out)} names to {submission_out}")
        print(
            f"   weight summary: min={out['weight'].min():.4f} "
            f"max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}"
        )


if __name__ == "__main__":
    main()
