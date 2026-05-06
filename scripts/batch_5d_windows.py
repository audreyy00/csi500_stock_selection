"""
本地回测：固定 5 组 5 个交易日窗口，依次跑 ``baseline_xgboost`` + ``score_submission``。

约定与 README 一致：组合在 ``--as-of`` 收盘特征上产生，估值区间 ``[--start, --end]``。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (as_of YYYYMMDD, window_start, window_end) — 与用户选定的五组一致
DEFAULT_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("20260213", "20260224", "20260302"),
    ("20260311", "20260312", "20260318"),
    ("20260327", "20260330", "20260403"),
    ("20260415", "20260416", "20260422"),
    ("20260422", "20260423", "20260429"),
)


def _parse_excess(text: str) -> float:
    for line in text.splitlines():
        m = re.search(r"excess return\s*:\s*([+-]?[0-9.]+)%", line, re.I)
        if m:
            return float(m.group(1))
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "submissions" / "batch_5d_windows",
        help="where to write baseline_*.csv submissions",
    )
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, str, str, float]] = []
    for as_of, start, end in DEFAULT_WINDOWS:
        tag = f"w_{start}_{end}_asof_{as_of}"
        sub = args.out_dir / f"baseline_{tag}.csv"

        r1 = subprocess.run(
            [args.python, str(ROOT / "baseline_xgboost.py"), "--as-of", as_of, "--out", str(sub)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if r1.returncode != 0:
            print(f"FAIL {tag}: baseline\n{r1.stderr}", file=sys.stderr)
            continue

        r2 = subprocess.run(
            [
                args.python,
                str(ROOT / "score_submission.py"),
                str(sub),
                "--start",
                start,
                "--end",
                end,
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if r2.returncode != 0:
            print(f"FAIL {tag}: score\n{r2.stderr}", file=sys.stderr)
            continue

        results.append((as_of, start, end, _parse_excess(r2.stdout)))

    print(f"{'as_of':<10} {'start':<10} {'end':<10} {'excess%':>10}")
    print("-" * 48)
    xs: list[float] = []
    for as_of, start, end, ex in results:
        print(f"{as_of:<10} {start:<10} {end:<10} {ex:+10.3f}")
        if not (ex != ex):  # nan check
            xs.append(ex)
    if xs:
        print("-" * 48)
        print(f"mean excess: {sum(xs) / len(xs):+.3f}%  (n={len(xs)})")


if __name__ == "__main__":
    main()
