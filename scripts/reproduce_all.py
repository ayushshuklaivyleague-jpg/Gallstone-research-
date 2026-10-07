"""
scripts/reproduce_all.py — Unified Master Scientific Reproducibility Pipeline.

Single canonical entrypoint to regenerate every benchmark, statistical test, table,
figure, and manuscript artifact published in the paper.

Usage:
  python scripts/reproduce_all.py --all              # Run end-to-end replication
  python scripts/reproduce_all.py --benchmark        # Run ultrasound + cross-cohort benchmarks
  python scripts/reproduce_all.py --cv               # Run Stratified 5-Fold Cross-Validation
  python scripts/reproduce_all.py --tables           # Regenerate baseline characteristics & missingness
  python scripts/reproduce_all.py --stats            # Compute LRT & paired bootstrap tests
  python scripts/reproduce_all.py --figures          # Regenerate figures (calibration, ROC, PR, DCA)
  python scripts/reproduce_all.py --paper            # Rebuild publication PDF and HTML
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def step_tables():
    print("\n" + "=" * 78)
    print("  STEP 1: COMPUTING BASELINE & MISSINGNESS CHARACTERISTICS (TABLE 1 & 2)")
    print("=" * 78)
    import src.compute_all_scientific_analyses
    print("[✓] Tables 1 & 2 generated successfully.")


def step_stats():
    print("\n" + "=" * 78)
    print("  STEP 2: COMPUTING LIKELIHOOD RATIO TESTS & PAIRED BOOTSTRAP INFERENCE")
    print("=" * 78)
    import src.compute_exact_lrt_and_boot
    print("[✓] LRT & paired bootstrap statistics computed successfully.")


def step_figures():
    print("\n" + "=" * 78)
    print("  STEP 3: COMPUTING SUBGROUPS, SENSITIVITY, DCA & PUBLICATION FIGURES")
    print("=" * 78)
    import src.compute_models_and_figures
    print("[✓] Publication figures, subgroup analyses, and DCA generated successfully.")


def step_benchmarks(run_cv: bool = False, n_folds: int = 5):
    print("\n" + "=" * 78)
    print("  STEP 4: RUNNING CANONICAL ULTRASOUND BENCHMARK ENGINE")
    print("=" * 78)
    import src.benchmark_ultrasound as bu
    bu.run_all_benchmarks(run_cv=run_cv, n_folds=n_folds)
    print("[✓] Canonical ultrasound benchmark completed successfully.")


def step_paper():
    print("\n" + "=" * 78)
    print("  STEP 5: COMPILING PUBLICATION MANUSCRIPT (HTML & PDF)")
    print("=" * 78)
    import subprocess
    cmd = [sys.executable, str(ROOT / "convert_to_pdf.py")]
    subprocess.check_call(cmd)
    print("[✓] Manuscript compiled successfully to PAPER.pdf and PAPER_pub.html.")


def main():
    parser = argparse.ArgumentParser(description="Master Scientific Reproduction Pipeline")
    parser.add_argument("--all", action="store_true", help="Execute complete end-to-end scientific reproduction")
    parser.add_argument("--benchmark", action="store_true", help="Run ultrasound ground-truth benchmark suite")
    parser.add_argument("--cv", action="store_true", help="Run repeated Stratified K-Fold Cross-Validation")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--tables", action="store_true", help="Generate baseline and missingness tables")
    parser.add_argument("--stats", action="store_true", help="Run LRT deviance and paired bootstrap tests")
    parser.add_argument("--figures", action="store_true", help="Generate calibration curves, ROC/PR, and DCA")
    parser.add_argument("--paper", action="store_true", help="Compile manuscript into publication PDF and HTML")
    args = parser.parse_args()

    t_start = time.time()

    if not (args.all or args.benchmark or args.cv or args.tables or args.stats or args.figures or args.paper):
        print("No specific flag provided. Running complete scientific reproduction (--all)...")
        args.all = True

    if args.tables or args.all:
        step_tables()

    if args.stats or args.all:
        step_stats()

    if args.figures or args.all:
        step_figures()

    if args.benchmark or args.all:
        step_benchmarks(run_cv=args.cv, n_folds=args.folds)
    elif args.cv:
        step_benchmarks(run_cv=True, n_folds=args.folds)

    if args.paper or args.all:
        step_paper()

    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"  [COMPLETE] Reproduction pipeline finished successfully in {elapsed:.1f}s.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
