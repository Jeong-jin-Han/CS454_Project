#!/usr/bin/env python3
"""
GA-based parallel branch testing for fair comparison with Hill Climbing.

FAIR COMPARISON SETUP (Updated based on actual HC performance):
----------------------------------------------------------------
Hill Climbing (test_benchmark_parallel_csv.py):
  - max_trials_per_branch: 5
  - max_iterations: 10
  - basin_max_search: 1,000 (only triggers when stuck)
  - Actual performance: ~50-200 evals per trial (adaptive!)
  - Example (needle1): 212 total evals for 5 branches = 42 evals/branch

Genetic Algorithm (this file):
  - max_trials_per_branch: 5 (same as HC)
  - pop_size: 100
  - max_gen: 10
  - Budget: 1,000 evals per trial (fixed: 100 × 10)
  - Ratio: ~5-20x more than HC (fair for population-based method)

Why this is fair:
  • HC is adaptive: uses fewer evals when problem is easy
  • GA is fixed: always uses same budget regardless of difficulty
  • GA needs more evals than HC (population vs single-point search)
  • 1000 evals is reasonable minimum for GA to function properly

Both algorithms:
  - Skip unreachable branches (while-True False, for-loop False)
  - Use success_threshold: 0.0 (exact match)
  - Track NFE (Number of Fitness Evaluations) in CSV

Outputs to benchmark_log_ga/<file>.csv
"""

import os
import csv
import time
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path

from module.sbst_core import instrument_and_load, FitnessCalculator
from BASE.ga import ga


# ------------------------------------------------------------
# 1) Per-branch GA evaluation worker
# ------------------------------------------------------------
def _ga_worker(args):
    (
        file_path,
        func_name,
        lineno,
        branch_info,
        max_trials,
        success_threshold,
        pop_size,
        max_gen,
        tournament_k,
        elite_ratio,
        gene_mut_p,
        mutation_step_choices,
        ensure_mutation,
        seed_offset,
        target_outcome,  # ✅ Added target_outcome parameter
        use_biased_init,  # ✅ Added use_biased_init parameter
    ) = args

    result = {
        "function": func_name,
        "lineno": lineno,
        "outcome": target_outcome,  # ✅ Added outcome to result
        "convergence_speed": 0,
        "nfe": 0,
        "best_fitness": float("inf"),
        "best_solution": None,
        "success": False,
        "num_trials": 0,
        "error": None,
        "pid": os.getpid(),
        "adjusted_pop_size": pop_size,
        "adjusted_max_gen": max_gen,
    }
    
    # Suppress verbose output in workers to prevent buffer overflow
    worker_verbose = os.environ.get('WORKER_VERBOSE', '0') == '1'
    import sys
    import time
    old_stdout = sys.stdout
    if not worker_verbose:
        sys.stdout = open(os.devnull, 'w')

    try:
        source = open(file_path, "r", encoding="utf-8").read()
        namespace, traveler, record, _ = instrument_and_load(source)

        fitness_calc = FitnessCalculator(traveler, record, namespace)
        fitness_calc.evals = 0

        func_obj = namespace[func_name]
        parent_map = traveler.parent_map
        func_info = [f for f in traveler.functions if f.name == func_name][0]

        best_fitness = float("inf")
        best_solution = None
        total_nfe = 0
        total_generations = 0
        success = False

        target_branch_node = branch_info.node
        subject_node = branch_info.subject
        # target_outcome is now passed as a parameter

        # NO adaptive adjustment for fair comparison with Hill Climbing
        # Both algorithms will use their fixed parameters
        result["adjusted_pop_size"] = pop_size
        result["adjusted_max_gen"] = max_gen

        # Prepare biased initialization metadata
        var_constants = getattr(func_info, "var_constants", {}) or {} if use_biased_init else {}
        total_constants = list(getattr(func_info, "total_constants", set()) or []) if use_biased_init else []

        for trial in range(max_trials):
            seed = seed_offset + lineno * 1000 + trial
            rng = random.Random(seed)
            random.seed(seed)

            nfe_before = fitness_calc.evals

            ind, fit = ga(
                fitness_calc=fitness_calc,
                func_info=func_info,
                func_obj=func_obj,
                target_branch_node=target_branch_node,
                target_outcome=target_outcome,
                subject_node=subject_node,
                parent_map=parent_map,
                pop_size=pop_size,
                max_gen=max_gen,
                tournament_k=tournament_k,
                elite_ratio=elite_ratio,
                gene_mut_p=gene_mut_p,
                ensure_mutation=ensure_mutation,
                mutation_step_choices=mutation_step_choices,
                rng=rng,
                use_biased_init=use_biased_init,  # ✅ Pass biased init flag
                var_constants=var_constants,  # ✅ Pass variable-specific constants
                total_constants=total_constants,  # ✅ Pass all constants
            )

            nfe_after = fitness_calc.evals
            nfe_this = nfe_after - nfe_before

            total_nfe += nfe_this
            gens_est = int(round(nfe_this / float(max(1, pop_size))))
            total_generations += gens_est

            if fit is not None and fit < best_fitness:
                best_fitness = fit
                best_solution = ind

            result["num_trials"] += 1

            if fit is not None and fit <= success_threshold:
                success = True
                break

        result["convergence_speed"] = total_generations
        result["nfe"] = total_nfe
        result["best_fitness"] = best_fitness
        result["best_solution"] = best_solution
        result["success"] = success

    except Exception as e:
        result["error"] = str(e)
    finally:
        # Restore stdout
        if not worker_verbose:
            sys.stdout.close()
            sys.stdout = old_stdout

    return result


# ------------------------------------------------------------
# 2) Run GA on a single file → write CSV
# ------------------------------------------------------------
def run_parallel_test_with_csv(
    file_path: str,
    output_csv: str,
    max_trials_per_branch: int = 5,  # Match HC: 5 trials per branch
    success_threshold: float = 0.0,  # Match HC: exact threshold
    pop_size: int = 100,  # Small GA: 1000 evals per trial (100 × 10)
    max_gen: int = 10,    # Few generations for speed
    tournament_k: int = 3,
    elite_ratio: float = 0.1,
    gene_mut_p=None,
    mutation_step_choices=(-3, -2, -1, 1, 2, 3),
    ensure_mutation=True,
    num_workers=None,
    seed_offset: int = 0,
    skip_for_false: bool = True,  # ✅ Skip unreachable branches (same as HC)
    use_biased_init: bool = False,  # ✅ Toggle biased initialization on/off
):
    """
    Run GA-based testing on all branches in a file.
    
    Args:
        skip_for_false: If True, skip for-loop and while-True False branches (unreachable)
    """
    print(f"\n[GA] Analyzing file: {file_path}")

    source = open(file_path, "r", encoding="utf-8").read()
    namespace, traveler, record, _ = instrument_and_load(source)

    tasks = []
    for func_info in traveler.functions:
        func_name = func_info.name
        branches = traveler.branches.get(func_name, {})
        
        if not branches:
            print(f"  ⏭️  Skipping {func_name} (no branches)")
            continue
        
        print(f"  📝 Function: {func_name}, Branches: {list(branches.keys())}")

        for lineno, branch_info in branches.items():
            # Check if this branch is a for-loop or while-True
            is_for_loop = lineno in getattr(traveler.tx, 'loop_minlen', {})
            is_while_true = lineno in getattr(traveler.tx, 'while_always_true', {})
            
            # Create tasks for both True and False outcomes
            for target_outcome in [True, False]:
                # Skip while-True False branches (never exit) - unreachable
                if skip_for_false and is_while_true and target_outcome is False:
                    print(f"     ⏭️  Skipping while-True False: line {lineno} (unreachable)")
                    continue
                
                # Skip for-loop False branches (not entering loop) - usually unreachable
                if skip_for_false and is_for_loop and target_outcome is False:
                    print(f"     ⏭️  Skipping for-loop False: line {lineno} (often unreachable)")
                    continue
                
                tasks.append(
                    (
                        file_path,
                        func_name,
                        lineno,
                        branch_info,
                        max_trials_per_branch,
                        success_threshold,
                        pop_size,
                        max_gen,
                        tournament_k,
                        elite_ratio,
                        gene_mut_p,
                        mutation_step_choices,
                        ensure_mutation,
                        seed_offset,
                        target_outcome,  # ✅ Added target_outcome
                        use_biased_init,  # ✅ Added biased init flag
                    )
                )

    if not tasks:
        print("  (No branches found)")
        return []

    if num_workers is None:
        num_workers = cpu_count()

    # Run in parallel with proper cleanup
    pool = None
    try:
        pool = Pool(processes=num_workers)
        results = pool.map(_ga_worker, tasks)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        if pool:
            pool.terminate()
            pool.join()
        raise
    except Exception as e:
        print(f"\n❌ Error during parallel execution: {e}")
        if pool:
            pool.terminate()
            pool.join()
        raise
    finally:
        if pool:
            pool.close()
            pool.join()

    # ----- Write CSV -----
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=[
                "function",
                "lineno",
                "outcome",  # ✅ Added outcome column
                "convergence_speed",
                "nfe",
                "best_fitness",
                "best_solution",
                "success",
                "num_trials",
                "adjusted_pop_size",  # ✅ Track if parameters were adjusted
                "adjusted_max_gen",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "function": r["function"],
                    "lineno": r["lineno"],
                    "outcome": r["outcome"],  # ✅ Added outcome
                    "convergence_speed": r["convergence_speed"],
                    "nfe": r["nfe"],
                    "best_fitness": r["best_fitness"],
                    "best_solution": str(r["best_solution"]),
                    "success": r["success"],
                    "num_trials": r["num_trials"],
                    "adjusted_pop_size": r.get("adjusted_pop_size", ""),
                    "adjusted_max_gen": r.get("adjusted_max_gen", ""),
                    "error": r["error"],
                }
            )

    print(f"  → CSV saved: {output_csv}")
    return results


# ------------------------------------------------------------
# 3) Run GA on every Python file in a directory
# ------------------------------------------------------------
def run_directory_test(directory: str, skip_for_false: bool = True, use_biased_init: bool = False, output_dir: str = "benchmark_log_ga"):
    """
    Run GA testing on all Python files in a directory.
    
    Args:
        directory: Directory containing Python files to test
        skip_for_false: If True, skip for-loop and while-True False branches (unreachable)
        use_biased_init: If True, use biased initialization; if False, use pure random
        output_dir: Output directory for CSV files
    """
    directory = Path(directory)
    assert directory.exists(), f"{directory} does not exist."

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    py_files = sorted([p for p in directory.rglob("*.py") if "__pycache__" not in str(p)])

    print(f"\n[GA] Running for directory: {directory}")
    print(f"  Found {len(py_files)} files.\n")

    for f in py_files:
        out_csv = output_dir / (f.stem + ".csv")
        run_parallel_test_with_csv(
            file_path=str(f),
            output_csv=str(out_csv),
            skip_for_false=skip_for_false,  # ✅ Pass skip_for_false
            use_biased_init=use_biased_init,  # ✅ Pass biased init flag
        )


# ------------------------------------------------------------
# 4) CLI entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import multiprocessing

    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Genetic Algorithm - Branch Coverage Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # With biased initialization (default):
  python test_benchmark_ga_csv.py
  
  # With pure random initialization:
  python test_benchmark_ga_csv.py --random-init
  
  # Custom output directory:
  python test_benchmark_ga_csv.py --output benchmark_log_ga_biased
  python test_benchmark_ga_csv.py --random-init --output benchmark_log_ga_random
        '''
    )
    parser.add_argument('--random-init', action='store_true',
                       help='Use pure random initialization instead of biased (default: biased)')
    parser.add_argument('--output', '-o', type=str, default='benchmark_log_ga',
                       help='Output directory for CSV files (default: benchmark_log_ga)')
    parser.add_argument('--source', '-s', type=str, default='./benchmark',
                       help='Source directory containing benchmark files (default: ./benchmark)')
    
    args = parser.parse_args()

    if "fork" in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method("fork", force=True)
    else:
        multiprocessing.set_start_method("spawn", force=True)

    # Print configuration
    init_type = "RANDOM" if args.random_init else "BIASED"
    print(f"\n{'='*80}")
    print(f"🔧 CONFIGURATION: Genetic Algorithm")
    print(f"{'='*80}")
    print(f"Initialization: {init_type}")
    print(f"Source dir:     {args.source}")
    print(f"Output dir:     {args.output}")
    print(f"{'='*80}\n")

    # Run with configuration
    run_directory_test(
        directory=args.source,
        skip_for_false=True,
        use_biased_init=not args.random_init,  # Invert the flag (like HC)
        output_dir=args.output
    )
