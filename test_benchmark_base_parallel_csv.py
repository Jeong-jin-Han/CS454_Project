
"""
Parallel baseline testing (NO compression) with CSV output.

Same metrics as compression version for fair comparison.
"""

import os
import sys
import time
import random
import csv
import json
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count
from module.sbst_core import instrument_and_load, FitnessCalculator
from hill_climb_multiD import hill_climb_simple_nd_code


def test_single_branch_baseline_with_metrics(args):
    """
    Test one branch with baseline (no compression) and return metrics.
    
    Uses the same global random seed for fairness across all branches.
    Runs multiple trials (different initial points) until time limit is reached.
    
    Returns dict with:
    - convergence_speed: sum of steps across all trials
    - nfe: total number of fitness evaluations
    - best_fitness: best fitness found
    - best_solution: solution achieving best fitness
    - num_trials_run: number of initial points tried within time limit
    - time_to_solution: time elapsed when solution was found (None if not solved)
    - total_time: total time spent on this branch
    """
    (file_path, func_name, lineno, branch_data,
     time_limit, random_seed, success_threshold, initial_low, initial_high,
     max_steps, target_outcome, use_biased_init) = args
    
    worker_pid = os.getpid()
    outcome_str = "True" if target_outcome else "False"
    
    # Reduce output verbosity
    worker_verbose = os.environ.get('WORKER_VERBOSE', '0') == '1'
    
    # Start timer for this branch (each thread has its own timer)
    branch_start_time = time.time()
    
    # Reset to same global seed for all branches (fairness)
    random.seed(random_seed)
    
    if worker_verbose:
        print(f"\n[Worker PID {worker_pid}] Starting branch {func_name}::{lineno} (outcome={outcome_str})")
        print(f"[Worker PID {worker_pid}] Time limit: {time_limit}s, Global seed: {random_seed}")
        sys.stdout.flush()
    
    # Load instrumented code
    source = open(file_path).read()
    namespace, traveler, record, _ = instrument_and_load(source)
    
    # Create new FitnessCalculator with its own eval counter
    fitness_calc = FitnessCalculator(traveler, record, namespace)
    fitness_calc.evals = 0
    
    parent_map = traveler.parent_map
    func_obj = namespace[func_name]
    
    # Get branch info
    branch_info = traveler.branches[func_name][lineno]
    target_branch_node = branch_info.node
    subject_node = branch_info.subject
    
    # Get function info
    func_info = [f for f in traveler.functions if f.name == func_name][0]
    dim = len(func_info.args)
    func_args = func_info.args

    # Prepare constant-based metadata for biased initialization
    var_constants = getattr(func_info, "var_constants", {}) or {}
    total_constants = list(getattr(func_info, "total_constants", set()) or [])

    def sample_initial_arg(arg_name: str, low: int, high: int) -> int:
        """
        Sample a single argument value using a mixture of:
          - uniform over [low, high] (if use_biased_init=False)
          - Gaussians centered at extracted constants (if use_biased_init=True)
        """
        # If biased initialization is disabled, use pure random
        if not use_biased_init:
            return random.randint(low, high)
        
        # Biased initialization: use constants if available
        # If we have no constants at all, fall back to uniform
        if not total_constants and not var_constants:
            return random.randint(low, high)

        # 20% uniform, 80% biased around constants
        if random.random() < 0.2:
            return random.randint(low, high)

        # Prefer per-variable constants if available, otherwise fall back to all constants
        const_list = list(var_constants.get(arg_name, []))
        if not const_list:
            const_list = total_constants
        if not const_list:
            return random.randint(low, high)

        center = random.choice(const_list)

        # Use a modest sigma so we stay reasonably close to interesting constants
        # but still explore around them.
        span = max(1, high - low)
        sigma = max(1, int(0.01 * span))  # 1% of span

        val = int(random.gauss(center, sigma))
        # Clamp to [low, high]
        if val < low:
            val = low
        elif val > high:
            val = high
        return val
    
    # Metrics
    total_steps = 0
    best_fitness = float('inf')
    best_solution = None
    branch_success = False
    time_to_solution = None  # Time when solution was found
    
    trial_results = []
    trial = 0  # Trial counter
    
    # Run trials until time limit is reached
    while True:
        elapsed_time = time.time() - branch_start_time
        
        # Check if time limit exceeded
        if elapsed_time >= time_limit:
            if worker_verbose:
                print(f"[Worker {worker_pid}] Time limit ({time_limit}s) reached after {trial} trials")
                sys.stdout.flush()
            break
        
        # Check if already succeeded
        if branch_success:
            if worker_verbose:
                print(f"[Worker {worker_pid}] Branch {lineno} succeeded, stopping trials")
                sys.stdout.flush()
            break
        
        if worker_verbose:
            print(f"[Worker {worker_pid}] Trial {trial+1} (elapsed: {elapsed_time:.2f}s/{time_limit}s)")
            sys.stdout.flush()
        
        # Random initial point (biased by extracted constants)
        initial = [
            sample_initial_arg(arg_name, initial_low, initial_high)
            for arg_name in func_args
        ]
        
        # Initial fitness
        init_fit = fitness_calc.fitness_for_candidate(
            func_obj, initial,
            target_branch_node, target_outcome,
            subject_node, parent_map
        )
        
        if worker_verbose:
            print(f"[Worker {worker_pid}] Branch {lineno} ({outcome_str}), Trial {trial+1}: init_f={init_fit:.4f}")
            sys.stdout.flush()
        
        # Run baseline hill climbing (NO compression) with strict time limit
        traj = hill_climb_simple_nd_code(
            fitness_calc, func_obj,
            target_branch_node, target_outcome,
            subject_node, parent_map,
            initial, dim,
            max_steps=max_steps,
            time_limit=time_limit,  # Pass time limit
            start_time=branch_start_time  # Pass start time
        )
        
        # Extract results
        final_point, final_f = traj[-1]
        steps_this_trial = len(traj)
        
        # Update metrics
        total_steps += steps_this_trial
        
        if final_f < best_fitness:
            best_fitness = final_f
            best_solution = list(final_point)
        
        trial_result = {
            "trial": trial,
            "initial_fitness": float(init_fit),
            "final_fitness": float(final_f),
            "steps": steps_this_trial
        }
        trial_results.append(trial_result)
        
        # Check success
        if final_f <= success_threshold:
            time_to_solution = time.time() - branch_start_time
            if worker_verbose:
                print(f"[Worker {worker_pid}]  Branch {lineno} ({outcome_str}) succeeded at trial {trial+1}")
                print(f"[Worker {worker_pid}]   Time to solution: {time_to_solution:.3f}s")
                sys.stdout.flush()
            branch_success = True
        
        # Increment trial counter
        trial += 1
    
    # Get total NFE
    total_nfe = fitness_calc.evals
    total_time = time.time() - branch_start_time
    
    if worker_verbose:
        print(f"\n[Worker {worker_pid}]  Branch {lineno} ({outcome_str}) completed:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Trials run: {len(trial_results)}")
        print(f"  Convergence speed: {total_steps}")
        print(f"  Total NFE: {total_nfe}")
        print(f"  Best fitness: {best_fitness:.6g}")
        if time_to_solution is not None:
            print(f"    Time to solution: {time_to_solution:.3f}s")
        sys.stdout.flush()
    
    return {
        'function': func_name,
        'lineno': lineno,
        'outcome': target_outcome,
        'convergence_speed': total_steps,
        'nfe': total_nfe,
        'best_fitness': best_fitness,
        'best_solution': best_solution,
        'success': branch_success,
        'num_trials_run': len(trial_results),
        'total_time': total_time,
        'time_to_solution': time_to_solution,  # None if not solved
        'worker_pid': worker_pid
    }


def run_parallel_test_baseline_with_csv(
    file_path,
    output_csv="results_baseline.csv",
    time_limit_per_branch=20.0,
    random_seed=42,
    success_threshold=0.0,
    initial_low=-100000,
    initial_high=10000,
    max_steps=2000,
    num_workers=None,
    skip_for_false=True,
    use_biased_init=True  # Toggle biased initialization on/off
):
    """
    Run baseline parallel testing and save to CSV.
    
    Each branch is tested with the same global random seed for fairness.
    The seed is reset at the start of each branch test.
    
    CSV columns (same format as compression version):
    - function: Function name
    - lineno: Branch line number
    - outcome: Target outcome (True/False)
    - convergence_speed: Sum of steps across all trials
    - nfe: Total number of fitness evaluations
    - best_fitness: Best fitness achieved
    - best_solution: Solution achieving best fitness
    - success: Whether branch was solved
    - num_trials: Number of initial points tried within time limit
    - total_time: Total time spent on this branch
    - time_to_solution: Time elapsed when solution was found (None if not solved)
    
    Args:
        time_limit_per_branch: Time limit in seconds for each branch (default: 20.0)
        random_seed: Global random seed applied to all branches for fairness (default: 42)
        skip_for_false: If True, skip for-loop and while-True False branches (unreachable)
    """
    
    print("\n" + "="*80)
    print(" PARALLEL BASELINE TESTING (No Compression) WITH CSV")
    print("="*80)
    print(f"File: {file_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Workers: {num_workers if num_workers else cpu_count()}")
    print("="*80 + "\n")
    
    # Load source
    source = open(file_path).read()
    namespace, traveler, record, _ = instrument_and_load(source)
    
    # Collect branch tasks
    branch_tasks = []
    
    for func_info in traveler.functions:
        func_name = func_info.name
        branches = traveler.branches.get(func_name, {})
        
        if not branches:
            print(f"  Skipping {func_name} (no branches)")
            continue
        
        print(f" Function: {func_name}")
        print(f"   Branches: {list(branches.keys())}")
        
        # Use function's range if it's wide enough (difference >= 10)
        # Otherwise use the provided initial_low/initial_high
        func_range = func_info.max_const - func_info.min_const
        if func_range >= 10:
            func_initial_low = func_info.min_const
            func_initial_high = func_info.max_const
            print(f"   Using function range: [{func_initial_low}, {func_initial_high}]")
        else:
            func_initial_low = initial_low
            func_initial_high = initial_high
            print(f"   Function range too narrow ({func_range}), using default: [{initial_low}, {initial_high}]")
        
        for lineno, branch_info in branches.items():
            # Check if this branch is a for-loop or while-True
            is_for_loop = lineno in getattr(traveler.tx, 'loop_minlen', {})
            is_while_true = lineno in getattr(traveler.tx, 'while_always_true', {})
            
            # Create tasks for both True and False outcomes
            for target_outcome in [True, False]:
                # Heuristic: skip while-True False branch (likely unreachable)
                if skip_for_false and is_while_true and target_outcome is False:
                    print(f"     Skipping while-True False: line {lineno} (unreachable)")
                    continue
                
                # Heuristic: skip for-loop False branch (often uninteresting / unreachable)
                if skip_for_false and is_for_loop and target_outcome is False:
                    print(f"     Skipping for-loop False: line {lineno} (often unreachable)")
                    continue
                
                task = (
                    file_path, func_name, lineno, branch_info,
                    time_limit_per_branch, random_seed, success_threshold,
                    func_initial_low, func_initial_high,
                    max_steps,
                    target_outcome,
                    use_biased_init  # Biased or random initialization
                )
                branch_tasks.append(task)
    
    print(f"\n Total branches to test: {len(branch_tasks)}\n")
    
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f" Starting {num_workers} worker processes...")
    print("="*80 + "\n")
    sys.stdout.flush()
    
    # Run in parallel
    start_time = time.time()
    
    pool = None
    try:
        pool = Pool(processes=num_workers)
        branch_results = pool.map(test_single_branch_baseline_with_metrics, branch_tasks)
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
    print(f" Total branches: {len(branch_results)}")
    print("="*80 + "\n")
    sys.stdout.flush()
    
    # Write to CSV
    print(f" Writing results to {output_csv}...")
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'function', 'lineno', 'outcome', 'convergence_speed', 'nfe',
            'best_fitness', 'best_solution', 'success', 'num_trials',
            'total_time', 'time_to_solution'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for result in branch_results:
            writer.writerow({
                'function': result['function'],
                'lineno': result['lineno'],
                'outcome': result['outcome'],
                'convergence_speed': result['convergence_speed'],
                'nfe': result['nfe'],
                'best_fitness': result['best_fitness'],
                'best_solution': str(result['best_solution']),
                'success': result['success'],
                'num_trials': result['num_trials_run'],
                'total_time': f"{result['total_time']:.3f}",
                'time_to_solution': f"{result['time_to_solution']:.3f}" if result['time_to_solution'] is not None else "N/A"
            })
    
    print(f" Results written to {output_csv}\n")
    sys.stdout.flush()
    
    # Print summary table
    print("="*120)
    print(" RESULTS SUMMARY")
    print("="*120)
    print(f"{'Function':<20} {'Line':<6} {'Out':<5} {'InitPts':<8} {'Time(s)':<10} {'Time2Sol':<10} "
          f"{'NFE':<10} {'Best Fitness':<15} {'Success'}")
    print("-"*120)
    
    for result in branch_results:
        success_mark = "P" if result['success'] else "F"
        outcome_str = "T" if result['outcome'] else "F"
        time2sol_str = f"{result['time_to_solution']:.2f}s" if result['time_to_solution'] is not None else "N/A"
        print(f"{result['function']:<20} {result['lineno']:<6} {outcome_str:<5} "
              f"{result['num_trials_run']:<8} {result['total_time']:<10.2f} {time2sol_str:<10} "
              f"{result['nfe']:<10} {result['best_fitness']:<15.6g} {success_mark}")
    
    print("="*120 + "\n")
    
    # Summary statistics
    total_convergence = sum(r['convergence_speed'] for r in branch_results)
    total_nfe = sum(r['nfe'] for r in branch_results)
    total_init_points = sum(r['num_trials_run'] for r in branch_results)
    successes = sum(1 for r in branch_results if r['success'])
    
    print(" OVERALL STATISTICS")
    print("-"*80)
    print(f"Total convergence speed: {total_convergence}")
    print(f"Total NFE: {total_nfe}")
    print(f"Total initial points tried: {total_init_points}")
    print(f"Success rate: {successes}/{len(branch_results)} ({100*successes/len(branch_results):.1f}%)")
    print(f"Avg convergence per branch: {total_convergence/len(branch_results):.1f}")
    print(f"Avg NFE per branch: {total_nfe/len(branch_results):.1f}")
    print(f"Avg initial points per branch: {total_init_points/len(branch_results):.1f}")
    print("="*80 + "\n")
    sys.stdout.flush()
    
    return branch_results, output_csv


def run_directory_test_baseline(
    source_dir,
    output_dir="benchmark_log",
    time_limit_per_branch=20.0,
    random_seed=42,
    success_threshold=0.0,
    initial_low=-100000,
    initial_high=10000,
    max_steps=2000,
    num_workers=None,
    skip_for_false=True,
    use_biased_init=True  # Toggle biased initialization on/off
):
    """
    Test all .py files in source_dir and save results to output_dir with mirrored structure.
    
    Example:
        benchmark/arbitrary1.py -> benchmark_log/arbitrary1.csv
        benchmark/HJJ/mixed_case.py -> benchmark_log/HJJ/mixed_case.csv
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
            results, _ = run_parallel_test_baseline_with_csv(
                file_path=str(py_file),
                output_csv=str(csv_file),
                time_limit_per_branch=time_limit_per_branch,
                random_seed=random_seed,
                success_threshold=success_threshold,
                initial_low=initial_low,
                initial_high=initial_high,
                max_steps=max_steps,
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
        "algorithm": "Baseline Hill Climbing (No Compression)",
        "initialization": "biased" if use_biased_init else "random",
        "source_directory": str(source_dir),
        "output_directory": str(output_dir),
        "time_limit_per_branch": time_limit_per_branch,
        "random_seed": random_seed,
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


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    import multiprocessing
    import argparse
    
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Baseline Hill Climbing (No Compression) - Branch Coverage Testing (Time-based)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # With biased initialization and default 20s time limit:
  python test_benchmark_base_parallel_csv.py
  
  # With pure random initialization and custom time limit:
  python test_benchmark_base_parallel_csv.py --random-init --time-limit 30
  
  # Custom seed for reproducibility:
  python test_benchmark_base_parallel_csv.py --seed 123
  
  # Custom output directory:
  python test_benchmark_base_parallel_csv.py --output benchmark_log_biased
  python test_benchmark_base_parallel_csv.py --random-init --output benchmark_log_random
        '''
    )
    parser.add_argument('--random-init', action='store_true',
                       help='Use pure random initialization instead of biased (default: biased)')
    parser.add_argument('--output', '-o', type=str, default='benchmark_log_2',
                       help='Output directory for CSV files (default: benchmark_log_2)')
    parser.add_argument('--source', '-s', type=str, default='./benchmark',
                       help='Source directory containing benchmark files (default: ./benchmark)')
    parser.add_argument('--time-limit', '-t', type=float, default=20.0,
                       help='Time limit in seconds per branch (default: 20.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Use 'fork' if available
    if 'fork' in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method('fork', force=True)
    else:
        multiprocessing.set_start_method('spawn', force=True)
    
    # Print configuration
    init_type = "RANDOM" if args.random_init else "BIASED"
    print(f"\n{'='*80}")
    print(f" CONFIGURATION: Baseline Hill Climbing (No Compression) (Time-based)")
    print(f"{'='*80}")
    print(f"Initialization:      {init_type}")
    print(f"Time limit/branch:   {args.time_limit}s")
    print(f"Random seed:         {args.seed}")
    print(f"Source dir:          {args.source}")
    print(f"Output dir:          {args.output}")
    print(f"{'='*80}\n")
    
    # Configuration: Test entire directory
    run_directory_test_baseline(
        source_dir=args.source,
        output_dir=args.output,
        time_limit_per_branch=args.time_limit,
        random_seed=args.seed,
        success_threshold=0.0,
        initial_low=-1000,
        initial_high=1000,
        max_steps=2000,
        num_workers=None,
        use_biased_init=not args.random_init  # Invert the flag
    )

