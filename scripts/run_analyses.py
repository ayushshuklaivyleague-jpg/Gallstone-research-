"""
scripts/run_analyses.py — Unified entrypoint to execute statistical analyses and figures.

Usage:
  python scripts/run_analyses.py --all
  python scripts/run_analyses.py --tables
  python scripts/run_analyses.py --stats
  python scripts/run_analyses.py --figures
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_tables():
    print("\n=======================================================")
    print("  COMPUTING BASELINE & MISSINGNESS TABLES")
    print("=======================================================\n")
    import src.compute_all_scientific_analyses


def run_stats():
    print("\n=======================================================")
    print("  COMPUTING LIKELIHOOD RATIO TESTS & PAIRED BOOTSTRAP")
    print("=======================================================\n")
    import src.compute_exact_lrt_and_boot


def run_figures():
    print("\n=======================================================")
    print("  COMPUTING SUBGROUPS, SENSITIVITY, DCA & FIGURES")
    print("=======================================================\n")
    import src.compute_models_and_figures


def main():
    parser = argparse.ArgumentParser(description="Gallstone AI Scientific Analyses Runner")
    parser.add_argument("--all", action="store_true", help="Run all analyses (Tables, Stats, Figures)")
    parser.add_argument("--tables", action="store_true", help="Generate Baseline & Missingness tables")
    parser.add_argument("--stats", action="store_true", help="Run LRT and Paired Bootstrap statistics")
    parser.add_argument("--figures", action="store_true", help="Generate publication figures & DCA")
    args = parser.parse_args()

    if not (args.all or args.tables or args.stats or args.figures):
        print("No specific mode specified. Running all scientific analyses by default.")
        run_tables()
        run_stats()
        run_figures()
        return

    if args.tables or args.all:
        run_tables()

    if args.stats or args.all:
        run_stats()

    if args.figures or args.all:
        run_figures()


if __name__ == "__main__":
    main()
