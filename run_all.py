"""
One-command reproducibility entrypoint.

Runs the ENTIRE build in order, each phase as its own subprocess (so a
failure in one phase doesn't leave partially imported state behind, and
each phase remains independently runnable via `python -m <module>` too).
Stops at the first failure — every later phase depends on an earlier
one's output files.

Phases 1-5 are the PIT data/backtest foundation (network-bound: ~650+
tickers from Yahoo Finance, rate-limit risk; budget well over an hour).
Phases 6-13 are the walk-forward/ML/benchmark build. On a genuinely
clean clone, `python run_all.py` with no flags runs all 13 phases and
needs nothing done by hand first, EXCEPT
`src/data/data/processed/security_mapping.csv` (hand-verified
renamed/delisted ticker mappings, committed to git — not regenerable by
any script, see `src/data/security_mapping.py`, so it can't be a phase
here).

Usage:
    python run_all.py                  # everything, phases 1-13
    python run_all.py --skip-data       # phases 6-13 only, if the data
                                         # foundation (phases 1-5's
                                         # output) already exists on disk
    python run_all.py --only run_benchmarks,run_cost_sensitivity
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# (label, module, is_data_phase) — order matters: each phase's input is
# an earlier phase's output.
PHASES = [
    ("Phase 1: PIT S&P 500 membership", "src.data.membership", True),
    ("Phase 2: download Yahoo price data", "src.data.download_all", True),
    ("Phase 3: build PIT daily dataset", "src.data.build_pit_dataset", True),
    ("Phase 4: build technical-indicator features", "src.features.build_features", True),
    ("Phase 5: in-sample parameter sweep", "src.backtest.run_sweep", True),
    ("Phase 6: confirm shortlist", "src.pipeline.confirm_shortlist", False),
    ("Phase 7: walk-forward harness", "src.pipeline.run_walkforward", False),
    ("Phase 8: combinations", "src.pipeline.run_combinations", False),
    ("Phase 9: ML pipeline", "src.pipeline.run_ml", False),
    ("Phase 10: benchmarks", "src.pipeline.run_benchmarks", False),
    ("Phase 11: cost sensitivity", "src.pipeline.run_cost_sensitivity", False),
    ("Phase 12: robustness", "src.pipeline.run_robustness", False),
    ("Phase 13: charts + correlation matrix", "src.pipeline.build_charts", False),
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
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip phases 1-5 (the PIT data/backtest foundation) and start "
        "at phase 6, e.g. because that foundation already exists on disk "
        "from a prior run and you only want to re-run the research build.",
    )
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None

    python = sys.executable

    for label, module, is_data_phase in PHASES:
        module_suffix = module.rsplit(".", 1)[-1]

        if only is not None and module_suffix not in only:
            continue

        if only is None and args.skip_data and is_data_phase:
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
