"""
run.py
──────
Entry point for running AdaTextGrad experiments.

Usage:
    python run.py --method ada_textgrad --max_iter 5
    python run.py --method vanilla_textgrad --max_iter 5
    python run.py --method both --limit 3       # quick 3-problem test
    python run.py --compare                     # compare saved result files
"""

import argparse
import asyncio
import json
from pathlib import Path

from problems import (
    run_experiment,
    load_results,
    compute_metrics,
    print_comparison,
)


async def main(args):

    # auto-resume: count lines already written to the result file
    if args.resume and not args.compare:
        from pathlib import Path as _Path
        for method_name in (
            [args.method] if args.method != "both"
            else ["vanilla_textgrad", "ada_textgrad"]
        ):
            result_file = _Path("results") / f"{method_name}.jsonl"
            if result_file.exists():
                completed = sum(1 for line in result_file.open()
                                if line.strip())
                if completed > 0:
                    print(f"[resume] Found {completed} completed problems "
                          f"in {result_file}. Starting from index {completed}.")
                    args.start_from = completed
                else:
                    print(f"[resume] No completed problems found in {result_file}. "
                          f"Starting from the beginning.")
            else:
                print(f"[resume] No result file found for {method_name}. "
                      f"Starting from the beginning.")

    if args.compare:
        # compare two existing result files
        ada_path = Path("results/ada_textgrad.jsonl")
        tg_path  = Path("results/vanilla_textgrad.jsonl")

        if not ada_path.exists() or not tg_path.exists():
            print("Result files not found. Run experiments first.")
            print("  python run.py --method both")
            return

        ada_results = load_results(str(ada_path))
        tg_results  = load_results(str(tg_path))
        print_comparison(ada_results, tg_results)
        return

    if args.method in ("ada_textgrad", "both"):
        print("\n" + "="*55)
        print("  Running AdaTextGrad")
        print("="*55)
        ada_results = await run_experiment(
            method            = "ada_textgrad",
            max_iter          = args.max_iter,
            problems_path     = args.problems,
            limit             = args.limit,
            start_from        = args.start_from,
            cooldown_seconds  = args.cooldown,
        )
        ada_metrics = compute_metrics(ada_results)
        print(f"\n  AdaTextGrad solve rate: {ada_metrics['solve_rate']:.1%} "
              f"({ada_metrics['solved_count']}/{ada_metrics['total']})")

    if args.method in ("vanilla_textgrad", "both"):
        print("\n" + "="*55)
        print("  Running Vanilla TextGrad (baseline)")
        print("="*55)
        tg_results = await run_experiment(
            method            = "vanilla_textgrad",
            max_iter          = args.max_iter,
            problems_path     = args.problems,
            limit             = args.limit,
            start_from        = args.start_from,
            cooldown_seconds  = args.cooldown,
        )
        tg_metrics = compute_metrics(tg_results)
        print(f"\n  TextGrad solve rate: {tg_metrics['solve_rate']:.1%} "
              f"({tg_metrics['solved_count']}/{tg_metrics['total']})")

    if args.method == "both":
        print("\n")
        print_comparison(ada_results, tg_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaTextGrad experiment runner")

    parser.add_argument(
        "--method",
        choices=["ada_textgrad", "vanilla_textgrad", "both"],
        default="ada_textgrad",
        help="Which method to run. Default: ada_textgrad",
    )
    parser.add_argument(
        "--max_iter", type=int, default=5,
        help="Max optimization iterations per problem. Default: 5",
    )
    parser.add_argument(
        "--problems", type=str, default="leetcode_hard_39.json",
        help="Path to problems JSON file. Default: leetcode_hard_39.json",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run on first N problems (for quick testing). Default: all",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Print comparison table from saved result files.",
    )
    parser.add_argument(
        "--start_from", type=int, default=0,
        help="Skip first N problems (resume after crash). Default: 0",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Auto-resume: count completed problems in existing JSONL and skip them.",
    )
    parser.add_argument(
        "--cooldown", type=float, default=8.0,
        help="Seconds to sleep between problems (protect daily quota). Default: 8",
    )

    args = parser.parse_args()
    asyncio.run(main(args))