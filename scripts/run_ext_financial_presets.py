"""
批量试跑 financial / ext 关键超参预设，对比验证集 rank IC。

用法（仓库根目录，需已安装依赖；建议 csi500）::

  conda run -n csi500 python scripts/run_ext_financial_presets.py \\
      --as-of 20260414 --python "$(which python)"

预设列在 ``PRESETS``；每轮子进程跑 ``baseline_xgboost.py``，从 stdout 抓取
``validation rank IC``。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# baseline 默认已为 mild_sw（--sample-weight-has-triple 1.5）；「recommended」与之一致。
# (name, baseline_cli_args... as list of str)
PRESETS: tuple[tuple[str, list[str]], ...] = (
    (
        "recommended_mild_sw",
        [
            "--fresh-half-life-days",
            "75",
            "--event-gate-calendar-days",
            "5",
            "--min-names-per-industry-day",
            "17",
        ],
    ),
    (
        "gentler_sw_1p25",
        [
            "--fresh-half-life-days",
            "75",
            "--event-gate-calendar-days",
            "5",
            "--min-names-per-industry-day",
            "17",
            "--sample-weight-has-triple",
            "1.25",
        ],
    ),
    (
        "long_decay_90",
        [
            "--fresh-half-life-days",
            "90",
            "--event-gate-calendar-days",
            "5",
            "--min-names-per-industry-day",
            "17",
            "--sample-weight-has-triple",
            "2",
        ],
    ),
    (
        "strict_industry_20",
        [
            "--fresh-half-life-days",
            "75",
            "--event-gate-calendar-days",
            "5",
            "--min-names-per-industry-day",
            "20",
            "--sample-weight-has-triple",
            "2",
        ],
    ),
    (
        "uniform_no_triple_boost",
        [
            "--fresh-half-life-days",
            "75",
            "--event-gate-calendar-days",
            "5",
            "--min-names-per-industry-day",
            "17",
            "--sample-weight-has-triple",
            "1",
        ],
    ),
)


def _parse_val_ic(text: str) -> float:
    for line in text.splitlines():
        m = re.search(r"validation\s+rank\s+IC:\s*([-+0-9.eE]+)", line, re.I)
        if m:
            return float(m.group(1))
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None, help="passed to baseline_xgboost.py")
    ap.add_argument("--python", default=sys.executable, help="Python to run baseline")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "submissions" / "ext_preset_runs")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        args.python,
        str(ROOT / "baseline_xgboost.py"),
        *(["--as-of", args.as_of] if args.as_of else []),
    ]

    rows: list[tuple[str, float, str]] = []
    print(f"{'preset':<22} {'val_IC':>10}  (subprocess log tail on failure)")
    print("-" * 60)
    for name, extra in PRESETS:
        sub = args.out_dir / f"preset_{name}.csv"
        cmd = [*base_cmd, *extra, "--out", str(sub)]
        r = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        txt = (r.stdout or "") + "\n" + (r.stderr or "")
        ic = _parse_val_ic(txt)
        rows.append((name, ic, txt[-800:] if r.returncode != 0 else ""))
        if r.returncode != 0:
            print(f"{name:<22} {'FAIL':>10}  rc={r.returncode}")
            print(txt[-1200:])
            continue
        print(f"{name:<22} {ic:>10.4f}")

    print("-" * 60)
    feasible = [(n, ic) for n, ic, _ in rows if ic == ic]
    if feasible:
        best_name = max(feasible, key=lambda x: x[1])[0]
        print(f"best IC preset (this run): {best_name}")


if __name__ == "__main__":
    main()
