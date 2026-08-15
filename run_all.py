"""
One-command reproducibility entrypoint.

Runs every phase of the walk-forward + ML build in order, each as its
own subprocess (so a failure in one phase doesn't leave partially
imported state behind, and each phase remains independently runnable
via `python -m src.pipeline.<name>` too). Stops at the first failure —
every later phase depends on an earlier one's output files.

Usage:
    python run_all.py
    python run_all.py --skip-slow      # skip Phase 2/3/4 re-runs if
                                        # results/ already has them
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

PHASES = [
    ("Phase 1: confirm shortlist", "src.pipeline.confirm_shortlist"),
    ("Phase 2: walk-forward harness", "src.pipeline.run_walkforward"),
    ("Phase 3: combinations", "src.pipeline.run_combinations"),
    ("Phase 4: ML pipeline", "src.pipeline.run_ml"),
    ("Phase 5: benchmarks", "src.pipeline.run_benchmarks"),
    ("Phase 6: cost sensitivity", "src.pipeline.run_cost_sensitivity"),
    ("Phase 7: robustness", "src.pipeline.run_robustness"),
    ("Phase 8: charts + correlation matrix", "src.pipeline.build_charts"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated module suffixes to run, e.g. "
        "'run_benchmarks,run_cost_sensitivity'. Runs everything if omitted.",
    )
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None

    python = sys.executable

    for label, module in PHASES:
        module_suffix = module.rsplit(".", 1)[-1]

        if only is not None and module_suffix not in only:
            continue

        print()
        print("#" * 80)
        print(f"# {label}  ({module})")
        print("#" * 80)

        start = time.perf_counter()

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"

        result = subprocess.run(
            [python, "-m", module],
            cwd=PROJECT_ROOT,
            env=env,
        )

        elapsed = time.perf_counter() - start
        print(f"[{label}] finished in {elapsed / 60.0:.1f} min, exit={result.returncode}")

        if result.returncode != 0:
            print(f"\nFAILED at: {label} ({module}). Stopping.")
            sys.exit(result.returncode)

    print()
    print("All phases complete. See results/ and reports/report.md.")


if __name__ == "__main__":
    main()
