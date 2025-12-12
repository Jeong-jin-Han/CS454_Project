
"""
GA-based parallel branch testing with time-based continuous evolution.

Time-Based Approach:
- Each branch gets 20 seconds (same as Hill Climbing)
- Initialize population ONCE, then continuously evolve until time limit
- No trial restarts - continuous evolution strategy
- Same CSV format as Hill Climbing for fair comparison

CSV columns (matching Hill Climbing):
- function, lineno, outcome, convergence_speed, nfe
- best_fitness, best_solution, success, num_trials
- total_time, time_to_solution
"""

import os
import csv
import json
import time
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path

from module.sbst_core import instrument_and_load, FitnessCalculator
from BASE.ga import ga


# ------------------------------------------------------------
# 1) Per-branch GA evaluation worker (Time-based continuous evolution)
# ------------------------------------------------------------
def _ga_worker(args):
    """
    Test one branch with GA using continuous evolution with strict time limit enforcement.
    
    Uses the same global random seed for fairness across all branches.
    Initializes population ONCE, then keeps evolving until:
    - Solution found (fitness 0.0) - early stopping
    - Time limit reached (checked before each individual evaluation)
    - Max generations reached (10000)
    
    Time limit is strictly enforced - the GA checks time before evaluating each individual
    and stops immediately if time limit is exceeded, ensuring it never goes over.
    
    Returns dict with:
    - convergence_speed: total generations evolved
    - nfe: total number of fitness evaluations
    - best_fitness: best fitness found
    - best_solution: solution achieving best fitness
    - num_trials: number of individuals examined (initial points) - equals NFE with early stopping
    - time_to_solution: time elapsed when solution was found (None if not solved)
    - total_time: total time spent on this branch
    """
    (
        file_path,
        func_name,
        lineno,
        branch_info,
        time_limit,
        random_seed,
        success_threshold,
        pop_size,
        tournament_k,
        elite_ratio,
        gene_mut_p,
        mutation_step_choices,
        ensure_mutation,
        target_outcome,
        use_biased_init,
    ) = args

    worker_pid = os.getpid()
    outcome_str = "True" if target_outcome else "False"
    
    result = {
        "function": func_name,
        "lineno": lineno,
        "outcome": target_outcome,
        "convergence_speed": 0,
        "nfe": 0,
        "best_fitness": float("inf"),
        "best_solution": None,
        "success": False,
        "num_trials": 0,
        "total_time": 0.0,
        "time_to_solution": None,
        "error": None,
        "worker_pid": worker_pid,
    }
    
    # Suppress verbose output in workers to prevent buffer overflow
    worker_verbose = os.environ.get('WORKER_VERBOSE', '0') == '1'
    import sys
    import time
    
    # Start timer for this branch
    branch_start_time = time.time()
    
    # Reset to same global seed for all branches (fairness)
    random.seed(random_seed)
    
    if worker_verbose:
        print(f"\n[Worker PID {worker_pid}] Starting branch {func_name}::{lineno} (outcome={outcome_str})")
        print(f"[Worker PID {worker_pid}] Time limit: {time_limit}s, Global seed: {random_seed}")
        sys.stdout.flush()
    
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
        total_individuals_examined = 0  # Track number of initial points (individuals examined)
        success = False
        time_to_solution = None

        target_branch_node = branch_info.node
        subject_node = branch_info.subject

        # Prepare biased initialization metadata
        var_constants = getattr(func_info, "var_constants", {}) or {} if use_biased_init else {}
        total_constants = list(getattr(func_info, "total_constants", set()) or []) if use_biased_init else []
        
        init_mode = "BIASED" if use_biased_init else "RANDOM"
        if worker_verbose:
            print(f"[Worker {worker_pid}] Initialization mode: {init_mode}")
            sys.stdout.flush()

        # Create RNG for GA
        rng = random.Random(random_seed)
        
        if worker_verbose:
            print(f"[Worker {worker_pid}] Starting continuous evolution...")
            sys.stdout.flush()

        nfe_before = fitness_calc.evals

        # Run GA once with high max_gen and strict time limit enforcement
        # The ga() function has:
        # - Built-in early stopping at fitness 0.0
        # - Strict time limit checking (stops before exceeding time_limit)
        ind, fit = ga(
            fitness_calc=fitness_calc,
            func_info=func_info,
            func_obj=func_obj,
            target_branch_node=target_branch_node,
            target_outcome=target_outcome,
            subject_node=subject_node,
            parent_map=parent_map,
            pop_size=pop_size,
            max_gen=10000,  # Very high - will be stopped by early stopping or time limit
            tournament_k=tournament_k,
            elite_ratio=elite_ratio,
            gene_mut_p=gene_mut_p,
            ensure_mutation=ensure_mutation,
            mutation_step_choices=mutation_step_choices,
            rng=rng,
            use_biased_init=use_biased_init,
            var_constants=var_constants,
            total_constants=total_constants,
            time_limit=time_limit,  # Strict time limit enforcement
            start_time=branch_start_time,  # Start time for this branch
        )

        nfe_after = fitness_calc.evals
        nfe_this = nfe_after - nfe_before

        total_nfe = nfe_this
        # Calculate which generation we're in (1-indexed for human readability)
        # 1-100 evals = Gen 1, 101-200 = Gen 2, 201-300 = Gen 3, etc.
        # So for 237 evals: (237-1)//100 + 1 = 236//100 + 1 = 2 + 1 = 3 (3rd generation)
        total_generations = ((nfe_this - 1) // pop_size + 1) if nfe_this > 0 else 0
        total_individuals_examined = nfe_this  # Each fitness eval = one individual examined
        
        best_fitness = fit if fit is not None else float("inf")
        best_solution = ind

        # Check success
        if fit is not None and fit <= success_threshold:
            time_to_solution = time.time() - branch_start_time
            if worker_verbose:
                print(f"[Worker {worker_pid}]  Branch {lineno} ({outcome_str}) succeeded!")
                print(f"[Worker {worker_pid}]   Time to solution: {time_to_solution:.3f}s")
                sys.stdout.flush()
            success = True

        total_time = time.time() - branch_start_time

        if worker_verbose:
            print(f"\n[Worker {worker_pid}]  Branch {lineno} ({outcome_str}) completed:")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Total generations evolved: {total_generations}")
            print(f"  Total individuals examined: {total_individuals_examined}")
            print(f"  Total NFE: {total_nfe}")
            print(f"  Best fitness: {best_fitness:.6g}")
            if time_to_solution is not None:
                print(f"    Time to solution: {time_to_solution:.3f}s")
            sys.stdout.flush()

        result["convergence_speed"] = total_generations
        result["nfe"] = total_nfe
        result["best_fitness"] = best_fitness
        result["best_solution"] = best_solution
        result["success"] = success
        result["num_trials"] = total_individuals_examined  # Number of individuals examined (initial points)
        result["total_time"] = total_time
        result["time_to_solution"] = time_to_solution

    except Exception as e:
        result["error"] = str(e)
    finally:
        # Restore stdout
        if not worker_verbose:
            sys.stdout.close()
            sys.stdout = old_stdout

    return result


# ------------------------------------------------------------
# 2) Run GA on a single file -> write CSV
# ------------------------------------------------------------
def run_parallel_test_with_csv(
    file_path: str,
    output_csv: str,
    time_limit_per_branch: float = 20.0,
    random_seed: int = 42,
    success_threshold: float = 0.0,
    pop_size: int = 100,
    tournament_k: int = 3,
    elite_ratio: float = 0.1,
    gene_mut_p=None,
    mutation_step_choices=(-3, -2, -1, 1, 2, 3),
    ensure_mutation=True,
    num_workers=None,
    skip_for_false: bool = True,
    use_biased_init: bool = True,
):
    """
    Run GA-based testing on all branches in a file with continuous evolution.
    
    Each branch is tested with the same global random seed for fairness.
    Population is initialized once, then evolves continuously until:
    - Solution found (fitness 0.0) - early stopping
    - max_gen reached (10000 generations)
    
    CSV columns (GA-specific format):
    - function: Function name
    - lineno: Branch line number
    - outcome: Target outcome (True/False)
    - convergence_speed: Total generations evolved
    - nfe: Total number of fitness evaluations
    - best_fitness: Best fitness achieved
    - best_solution: Solution achieving best fitness
    - success: Whether branch was solved
    - num_trials: Number of individuals examined (equals NFE)
    - generations: Total generations evolved (same as convergence_speed)
    - total_time: Total time spent on this branch
    - time_to_solution: Time elapsed when solution was found (None if not solved)
    
    Args:
        time_limit_per_branch: Time limit in seconds for each branch (default: 20.0)
        random_seed: Global random seed applied to all branches for fairness (default: 42)
        pop_size: Population size for GA (default: 100)
        skip_for_false: If True, skip for-loop and while-True False branches (unreachable)
    """
    print("\n" + "="*80)
    print(" PARALLEL GA TESTING WITH CSV OUTPUT (Time-based)")
    print("="*80)
    print(f"File: {file_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Workers: {num_workers if num_workers else cpu_count()}")
    print("="*80 + "\n")

    source = open(file_path, "r", encoding="utf-8").read()
    namespace, traveler, record, _ = instrument_and_load(source)

    tasks = []
    for func_info in traveler.functions:
        func_name = func_info.name
        branches = traveler.branches.get(func_name, {})
        
        if not branches:
            print(f"  Skipping {func_name} (no branches)")
            continue
        
        print(f" Function: {func_name}")
        print(f"   Branches: {list(branches.keys())}")

        for lineno, branch_info in branches.items():
            # Check if this branch is a for-loop or while-True
            is_for_loop = lineno in getattr(traveler.tx, 'loop_minlen', {})
            is_while_true = lineno in getattr(traveler.tx, 'while_always_true', {})
            
            # Create tasks for both True and False outcomes
            for target_outcome in [True, False]:
                # Skip while-True False branches (never exit) - unreachable
                if skip_for_false and is_while_true and target_outcome is False:
                    print(f"     Skipping while-True False: line {lineno} (unreachable)")
                    continue
                
                # Skip for-loop False branches (not entering loop) - usually unreachable
                if skip_for_false and is_for_loop and target_outcome is False:
                    print(f"     Skipping for-loop False: line {lineno} (often unreachable)")
                    continue
                
                tasks.append(
                    (
                        file_path,
                        func_name,
                        lineno,
                        branch_info,
                        time_limit_per_branch,
                        random_seed,
                        success_threshold,
                        pop_size,
                        tournament_k,
                        elite_ratio,
                        gene_mut_p,
                        mutation_step_choices,
                        ensure_mutation,
                        target_outcome,
                        use_biased_init,
                    )
                )
    
    print(f"\n Total branches to test: {len(tasks)}\n")

    if not tasks:
        print("(No branches found)")
        return [], output_csv

    if num_workers is None:
        num_workers = cpu_count()

    print(f" Starting {num_workers} worker processes...")
    print("="*80 + "\n")
    import sys
    sys.stdout.flush()

    # Run in parallel with proper cleanup
    start_time = time.time()
    
    pool = None
    try:
        pool = Pool(processes=num_workers)
        results = pool.map(_ga_worker, tasks)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        if pool:
            pool.terminate()
            pool.join()
        raise
    except Exception as e:
        print(f"\n Error during parallel execution: {e}")
        if pool:
            pool.terminate()
            pool.join()
        raise
    finally:
        if pool:
            pool.close()
            pool.join()
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print(" ALL BRANCHES COMPLETED")
    print("="*80)
    print(f"  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f" Total branches: {len(results)}")
    print("="*80 + "\n")
    sys.stdout.flush()

    # ----- Write CSV (same format as Hill Climbing) -----
    print(f" Writing results to {output_csv}...")
    
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as cf:
        fieldnames = [
            'function', 'lineno', 'outcome', 'convergence_speed', 'nfe',
            'best_fitness', 'best_solution', 'success', 'num_trials', 'generations',
            'total_time', 'time_to_solution'
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for r in results:
            if r["error"]:
                print(f"  Error in {r['function']}:{r['lineno']}: {r['error']}")
            
            writer.writerow({
                'function': r['function'],
                'lineno': r['lineno'],
                'outcome': r['outcome'],
                'convergence_speed': r['convergence_speed'],
                'nfe': r['nfe'],
                'best_fitness': r['best_fitness'],
                'best_solution': str(r['best_solution']),
                'success': r['success'],
                'num_trials': r['num_trials'],
                'generations': r['convergence_speed'],  # Same as convergence_speed for clarity
                'total_time': f"{r['total_time']:.3f}",
                'time_to_solution': f"{r['time_to_solution']:.3f}" if r['time_to_solution'] is not None else "N/A"
            })

    print(f" Results written to {output_csv}\n")
    sys.stdout.flush()
    
    # Print summary table (GA-specific format with generations)
    print("="*130)
    print(" RESULTS SUMMARY")
    print("="*130)
    print(f"{'Function':<20} {'Line':<6} {'Out':<5} {'InitPts':<8} {'Gens':<6} {'Time(s)':<10} {'Time2Sol':<10} "
          f"{'NFE':<10} {'Best Fitness':<15} {'Success'}")
    print("-"*130)
    
    for r in results:
        success_mark = "P" if r['success'] else "F"
        outcome_str = "T" if r['outcome'] else "F"
        time2sol_str = f"{r['time_to_solution']:.2f}s" if r['time_to_solution'] is not None else "N/A"
        print(f"{r['function']:<20} {r['lineno']:<6} {outcome_str:<5} "
              f"{r['num_trials']:<8} {r['convergence_speed']:<6} {r['total_time']:<10.2f} {time2sol_str:<10} "
              f"{r['nfe']:<10} {r['best_fitness']:<15.6g} {success_mark}")
    
    print("="*120 + "\n")
    
    # Summary statistics
    total_convergence = sum(r['convergence_speed'] for r in results)
    total_nfe = sum(r['nfe'] for r in results)
    total_individuals = sum(r['num_trials'] for r in results)
    successes = sum(1 for r in results if r['success'])
    
    print(" OVERALL STATISTICS")
    print("-"*80)
    print(f"Total generations evolved: {total_convergence}")
    print(f"Total NFE: {total_nfe}")
    print(f"Total individuals examined: {total_individuals}")
    print(f"Success rate: {successes}/{len(results)} ({100*successes/len(results):.1f}%)")
    print(f"Avg generations per branch: {total_convergence/len(results):.1f}")
    print(f"Avg NFE per branch: {total_nfe/len(results):.1f}")
    print(f"Avg individuals per branch: {total_individuals/len(results):.1f}")
    print("="*80 + "\n")
    sys.stdout.flush()
    
    return results, output_csv


# ------------------------------------------------------------
# 3) Run GA on every Python file in a directory
# ------------------------------------------------------------
def run_directory_test(
    source_dir,
    output_dir="benchmark_log_ga",
    time_limit_per_branch=20.0,
    random_seed=42,
    success_threshold=0.0,
    pop_size=100,
    num_workers=None,
    skip_for_false=True,
    use_biased_init=True
):
    """
    Test all .py files in source_dir and save results to output_dir with mirrored structure.
    
    Example:
        benchmark/arbitrary1.py -> benchmark_log_ga/arbitrary1.csv
        benchmark/HJJ/mixed_case.py -> benchmark_log_ga/HJJ/mixed_case.csv
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Find all .py files (excluding __pycache__ and other non-test files)
    py_files = [f for f in source_path.rglob("*.py") 
                if "__pycache__" not in str(f)]
    
    print("\n" + "="*80)
    print(f" Found {len(py_files)} Python files in {source_dir}")
    print("="*80)

    # Start overall timer
    overall_start_time = time.time()

    for py_file in py_files:
        # Compute relative path and output CSV path
        rel_path = py_file.relative_to(source_path)
        csv_file = output_path / rel_path.with_suffix('.csv')
        
        # Create output directory if needed
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n Testing: {py_file}")
        print(f" Output: {csv_file}")
        
        # Run test on this file
        try:
            results, _ = run_parallel_test_with_csv(
                file_path=str(py_file),
                output_csv=str(csv_file),
                time_limit_per_branch=time_limit_per_branch,
                random_seed=random_seed,
                success_threshold=success_threshold,
                pop_size=pop_size,
                num_workers=num_workers,
                skip_for_false=skip_for_false,
                use_biased_init=use_biased_init
            )
        except Exception as e:
            print(f" Error testing {py_file}: {e}")
            continue
    
    # Calculate total execution time
    total_execution_time = time.time() - overall_start_time
    
    # Save test configuration to JSON
    config_file = output_path / "test_config.json"
    config_data = {
        "algorithm": "Genetic Algorithm (Continuous Evolution)",
        "initialization": "biased" if use_biased_init else "random",
        "source_directory": str(source_dir),
        "output_directory": str(output_dir),
        "time_limit_per_branch": time_limit_per_branch,
        "random_seed": random_seed,
        "population_size": pop_size,
        "max_generations": 10000,
        "evolution_strategy": "continuous (no restarts, early stopping at fitness 0.0)",
        "total_execution_time_seconds": round(total_execution_time, 2),
        "total_execution_time_formatted": f"{int(total_execution_time // 60)}m {int(total_execution_time % 60)}s",
        "files_tested": len(py_files)
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print("\n" + "="*80)
    print(f"  TOTAL EXECUTION TIME: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
    print("="*80)
    print(f" ALL FILES TESTED! Results saved to {output_dir}/")
    print(f" Test configuration saved to {config_file}")
    print("="*80)


# ------------------------------------------------------------
# 4) CLI entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import multiprocessing

    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Genetic Algorithm - Branch Coverage Testing (Time-based)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # With biased initialization and default 20s time limit:
  python test_benchmark_ga_csv.py
  
  # With pure random initialization and custom time limit:
  python test_benchmark_ga_csv.py --random-init --time-limit 30
  
  # Custom seed for reproducibility:
  python test_benchmark_ga_csv.py --seed 123
  
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
    parser.add_argument('--time-limit', '-t', type=float, default=20.0,
                       help='Time limit in seconds per branch (default: 20.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()

    # Use 'fork' if available
    if "fork" in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method("fork", force=True)
    else:
        multiprocessing.set_start_method("spawn", force=True)

    # Print configuration
    init_type = "RANDOM" if args.random_init else "BIASED"
    print(f"\n{'='*80}")
    print(f" CONFIGURATION: Genetic Algorithm (Time-based)")
    print(f"{'='*80}")
    print(f"Initialization:      {init_type}")
    print(f"Time limit/branch:   {args.time_limit}s")
    print(f"Random seed:         {args.seed}")
    print(f"Source dir:          {args.source}")
    print(f"Output dir:          {args.output}")
    print(f"{'='*80}\n")

    # Run with configuration
    run_directory_test(
        source_dir=args.source,
        output_dir=args.output,
        time_limit_per_branch=args.time_limit,
        random_seed=args.seed,
        success_threshold=0.0,
        pop_size=10000, # 100
        num_workers=None,
        skip_for_false=True,
        use_biased_init=not args.random_init  # Invert the flag (like HC)
    )
