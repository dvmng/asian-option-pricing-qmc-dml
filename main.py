from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

EXPERIMENTS = {
    1: "01_market_data.py",
    2: "02_market_analysis.py",
    3: "03_model_calibration.py",
    4: "04_pricing_dataset.py",
    5: "05_ml_training.py",
    6: "06_ml_evaluation.py",
    7: "07_local_hedging.py",
    8: "08_dynamic_hedging.py",
}


def run_experiment(number: int, extra_args: list[str]) -> None:
    script = ROOT / "experiments" / EXPERIMENTS[number]
    command = [sys.executable, str(script), *extra_args]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entry point for the Asian-option pricing and hedging TFM."
    )
    parser.add_argument(
        "--checks",
        action="store_true",
        help="Run repository consistency checks.",
    )
    parser.add_argument(
        "--layout",
        action="store_true",
        help="Display the experimental workflow.",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Regenerate figures from stored results.",
    )
    parser.add_argument(
        "--tables",
        action="store_true",
        help="Regenerate tables from stored results.",
    )
    parser.add_argument(
        "--outputs",
        action="store_true",
        help="Regenerate figures and tables from stored results.",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=range(1, 9),
        help="Run one of the eight experimental stages.",
    )
    parser.add_argument("experiment_args", nargs=argparse.REMAINDER)
    return parser


def print_layout() -> None:
    print("\nTFM WORKFLOW")
    print("=" * 72)
    print("1. Market data              experiments/01_market_data.py")
    print("2. Market diagnostics       experiments/02_market_analysis.py")
    print("3. Log-OU calibration       experiments/03_model_calibration.py")
    print("4. RQMC pricing dataset     experiments/04_pricing_dataset.py")
    print("5. MLP/DML training         experiments/05_ml_training.py")
    print("6. Final evaluation         experiments/06_ml_evaluation.py")
    print("7. Local hedge validation   experiments/07_local_hedging.py")
    print("8. Dynamic forward stress   experiments/08_dynamic_hedging.py")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not any(
        [
            args.checks,
            args.layout,
            args.figures,
            args.tables,
            args.outputs,
            args.experiment,
        ]
    ):
        parser.print_help()
        return

    if args.checks:
        from checks.repository_check import run_checks

        run_checks(ROOT)

    if args.layout:
        print_layout()

    if args.figures or args.outputs:
        from tfm_project.paths import ProjectPaths
        from tfm_project.thesis_outputs import make_figures

        make_figures(ProjectPaths.discover(ROOT))

    if args.tables or args.outputs:
        from tfm_project.paths import ProjectPaths
        from tfm_project.thesis_outputs import make_tables

        make_tables(ProjectPaths.discover(ROOT))

    if args.experiment:
        extra_args = args.experiment_args
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]
        run_experiment(args.experiment, extra_args)


if __name__ == "__main__":
    main()
