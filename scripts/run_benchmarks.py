"""
scripts/run_benchmarks.py — Unified entrypoint to execute model benchmarks.

Usage:
  python scripts/run_benchmarks.py --all
  python scripts/run_benchmarks.py --ultrasound
  python scripts/run_benchmarks.py --multi-cohort
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_ultrasound():
    print("\n=======================================================")
    print("  RUNNING EXPERIMENT 1 & 4: ULTRASOUND BENCHMARK ENGINE")
    print("=======================================================\n")
    import src.benchmark_ultrasound as bu
    bu.main()


def run_multi_cohort():
    print("\n=======================================================")
    print("  RUNNING EXPERIMENTS A, B, C, D: MULTI-COHORT ENGINE")
    print("=======================================================\n")
    import src.train_all as ta
    ta.main()


def main():
    parser = argparse.ArgumentParser(description="Gallstone AI Benchmark Runner")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks (Ultrasound + Multi-Cohort)")
    parser.add_argument("--ultrasound", action="store_true", help="Run Physical Ultrasound Benchmark (Exp 1 & Exp 4)")
    parser.add_argument("--multi-cohort", action="store_true", help="Run Multi-Cohort Benchmark (Exp A, B, C, D)")
    args = parser.parse_args()

    if not (args.all or args.ultrasound or args.multi_cohort):
        # Default to ultrasound benchmark
        print("No mode specified. Running physical ultrasound ground-truth benchmark by default.")
        run_ultrasound()
        return

    if args.ultrasound or args.all:
        run_ultrasound()

    if args.multi_cohort or args.all:
        run_multi_cohort()


if __name__ == "__main__":
    main()
