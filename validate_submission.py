"""
Validate a submission CSV against the competition rules.

Rules
-----
* File is CSV with exactly columns: stock_code, weight
* stock_code is a 6-digit string (zero-padded), every code is a current CSI500 constituent
* weight is a float, non-negative, sum to 1.0 within tolerance
* at least MIN_STOCKS distinct codes
* no single weight exceeds MAX_WEIGHT
* no duplicate stock_code

Usage
-----
  python validate_submission.py submission.csv
  python validate_submission.py submission.csv --constituents data/constituents_csi500.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MIN_STOCKS = 30
MAX_WEIGHT = 0.10
WEIGHT_SUM_TOL = 1e-4


def validate(path: Path, constituents_path: Path | None) -> list[str]:
    errors: list[str] = []

    try:
        df = pd.read_csv(path, dtype={"stock_code": str})
    except Exception as e:
        return [f"cannot read {path}: {e}"]

    if list(df.columns) != ["stock_code", "weight"]:
        errors.append(f"columns must be exactly ['stock_code','weight'], got {list(df.columns)}")
        return errors

    if df["stock_code"].isna().any() or df["weight"].isna().any():
        errors.append("found NaN values")

    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    if df["stock_code"].duplicated().any():
        dups = df.loc[df["stock_code"].duplicated(), "stock_code"].tolist()
        errors.append(f"duplicate stock_code rows: {dups[:5]}...")

    if not df["stock_code"].str.fullmatch(r"\d{6}").all():
        bad = df.loc[~df["stock_code"].str.fullmatch(r"\d{6}"), "stock_code"].head().tolist()
        errors.append(f"stock_code must be 6 digits, got e.g. {bad}")

    try:
        w = pd.to_numeric(df["weight"])
    except Exception as e:
        return errors + [f"weight column not numeric: {e}"]

    if (w < 0).any():
        errors.append(f"{(w < 0).sum()} negative weights (shorting not allowed)")

    total = w.sum()
    if abs(total - 1.0) > WEIGHT_SUM_TOL:
        errors.append(f"weights sum to {total:.6f}, must be 1.0 (tol {WEIGHT_SUM_TOL})")

    over_cap = (w > MAX_WEIGHT + 1e-9).sum()
    if over_cap:
        errors.append(f"{over_cap} weights exceed {MAX_WEIGHT:.0%} cap (max={w.max():.4f})")

    n = (w > 0).sum()
    if n < MIN_STOCKS:
        errors.append(f"only {n} names with positive weight; rule requires >= {MIN_STOCKS}")

    if constituents_path is not None and constituents_path.exists():
        cons = pd.read_csv(constituents_path, dtype={"stock_code": str})
        universe = set(cons["stock_code"].str.zfill(6))
        missing = [c for c in df["stock_code"] if c not in universe]
        if missing:
            errors.append(f"{len(missing)} codes not in CSI500 universe: e.g. {missing[:5]}")

    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("submission", help="path to submission CSV")
    p.add_argument("--constituents", default="data/constituents_csi500.csv")
    args = p.parse_args()

    sub_path = Path(args.submission)
    cons_path = Path(args.constituents)
    errors = validate(sub_path, cons_path if cons_path.exists() else None)

    if errors:
        print(f"FAIL: {sub_path}")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"OK: {sub_path} passes all checks")


if __name__ == "__main__":
    main()
