#!/usr/bin/env python3
"""
Parallel branch testing with CSV output for evaluation metrics.

Metrics per branch:
- convergence_speed: Sum of steps across all trials (to handle bad seeds fairly)
- nfe: Total number of fitness evaluations
- best_fitness: Best fitness achieved across all trials
- best_solution: Solution that achieved best fitness
"""

import os
import sys
import time
import random
import csv
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count
from module.sbst_core import instrument_and_load, FitnessCalculator
from compression_hc import hill_climb_with_compression_nd_code, CompressionManagerND


def test_single_branch_with_metrics(args):
    """
    Test one branch and return evaluation metrics.
    
    Returns dict with:
    - convergence_speed: sum of steps across all trials
    - nfe: total number of fitness evaluations
    - best_fitness: best fitness found
    - best_solution: solution achieving best fitness
    """
    (file_path, func_name, lineno, branch_data,
     max_trials, success_threshold, initial_low, initial_high,
     max_iterations, basin_max_search, target_outcome, use_biased_init) = args
    
    worker_pid = os.getpid()
    outcome_str = "True" if target_outcome else "False"
    
    # Reduce output verbosity in workers to prevent buffer overflow
    worker_verbose = os.environ.get('WORKER_VERBOSE', '0') == '1'
    
    if worker_verbose:
        print(f"\n[Worker PID {worker_pid}] Starting branch {func_name}::{lineno} (outcome={outcome_str})")
        sys.stdout.flush()
    
    # Each worker loads its own instrumented code
    source = open(file_path).read()
    namespace, traveler, record, _ = instrument_and_load(source)
    
    # ✅ Create new FitnessCalculator with its own eval counter
    fitness_calc = FitnessCalculator(traveler, record, namespace)
    fitness_calc.evals = 0  # Reset counter
    
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
    
    # Debug: Show initialization mode and available constants
    if worker_verbose:
        init_mode = "BIASED" if use_biased_init else "RANDOM"
        print(f"[Worker {worker_pid}] Initialization mode: {init_mode}")
        if use_biased_init:
            print(f"[Worker {worker_pid}] Available constants:")
            print(f"  Total constants: {total_constants[:10]}{'...' if len(total_constants) > 10 else ''}")
            if var_constants:
                for arg, consts in list(var_constants.items())[:3]:
                    const_list = list(consts)[:5]
                    print(f"  {arg}: {const_list}{'...' if len(consts) > 5 else ''}")
        sys.stdout.flush()

    def sample_initial_arg(arg_name: str, low: int, high: int) -> int:
        """
        Sample a single argument value using a mixture of:
          - uniform over [low, high] (if use_biased_init=False)
          - Gaussians centered at extracted constants (if use_biased_init=True)
        """
        # If biased initialization is disabled, use pure random
        if not use_biased_init:
            val = random.randint(low, high)
            if worker_verbose:
                print(f"    {arg_name}: {val} (random uniform)")
            return val
        
        # Biased initialization: use constants if available
        # If we have no constants at all, fall back to uniform
        if not total_constants and not var_constants:
            val = random.randint(low, high)
            if worker_verbose:
                print(f"    {arg_name}: {val} (random, no constants)")
            return val

        # 20% uniform, 80% biased around constants
        if random.random() < 0.2:
            val = random.randint(low, high)
            if worker_verbose:
                print(f"    {arg_name}: {val} (random 20%)")
            return val

        # Prefer per-variable constants if available, otherwise fall back to all constants
        const_list = list(var_constants.get(arg_name, []))
        if not const_list:
            const_list = total_constants
        if not const_list:
            val = random.randint(low, high)
            if worker_verbose:
                print(f"    {arg_name}: {val} (random fallback)")
            return val

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
        
        if worker_verbose:
            print(f"    {arg_name}: {val} (biased near constant {center}, sigma={sigma})")
        
        return val
    
    # Create ONE CompressionManagerND for this branch
    branch_cm = CompressionManagerND(dim, steepness=5.0)
    
    # Metrics to track
    total_steps = 0  # Convergence speed
    best_fitness = float('inf')
    best_solution = None
    branch_success = False
    
    trial_results = []
    
    # Run all trials
    for trial in range(max_trials):
        if branch_success:
            if worker_verbose:
                print(f"[Worker {worker_pid}] Branch {lineno} succeeded, skipping remaining trials")
                sys.stdout.flush()
            break
        
        # Random initial point (biased by extracted constants if enabled)
        random.seed(42 + lineno * 1000 + trial)
        
        if worker_verbose:
            init_mode = "BIASED" if use_biased_init else "RANDOM"
            print(f"[Worker {worker_pid}] Trial {trial+1}/{max_trials} - Generating {init_mode} initial point:")
            sys.stdout.flush()
        
        initial = [
            sample_initial_arg(arg_name, initial_low, initial_high)
            for arg_name in func_args
        ]
        
        # Initial fitness (counted in NFE)
        init_fit = fitness_calc.fitness_for_candidate(
            func_obj, initial,
            target_branch_node, target_outcome,
            subject_node, parent_map
        )
        
        if worker_verbose:
            print(f"[Worker {worker_pid}] Branch {lineno} ({outcome_str}), Trial {trial+1}/{max_trials}:")
            print(f"  Initial point: {initial}")
            print(f"  Initial fitness: {init_fit:.4f}")
            sys.stdout.flush()
        
        # Suppress verbose output from hill climbing in workers
        old_stdout = sys.stdout
        if not worker_verbose:
            sys.stdout = open(os.devnull, 'w')
        
        # Run hill climbing (all fitness evals counted automatically)
        try:
            traj, branch_cm = hill_climb_with_compression_nd_code(
                fitness_calc, func_obj,
                target_branch_node, target_outcome,
                subject_node, parent_map,
                initial, dim,
                max_iterations=max_iterations,
                basin_max_search=basin_max_search,
                global_min_threshold=1e-6,
                cm=branch_cm
            )
        finally:
            # Restore stdout
            if not worker_verbose:
                sys.stdout.close()
                sys.stdout = old_stdout
        
        # Extract results
        final_point, final_f, used_comp = traj[-1]
        steps_this_trial = len(traj)
        
        # Update metrics
        total_steps += steps_this_trial
        
        if final_f < best_fitness:
            best_fitness = final_f
            best_solution = list(final_point)
        
        trial_result = {
            "trial": trial,
            "initial_point": initial,
            "initial_fitness": float(init_fit),
            "final_point": list(final_point),
            "final_fitness": float(final_f),
            "steps": steps_this_trial
        }
        trial_results.append(trial_result)
        
        # Check success
        if final_f <= success_threshold:
            if worker_verbose:
                print(f"[Worker {worker_pid}] 🎉 Branch {lineno} ({outcome_str}) succeeded at trial {trial}")
                sys.stdout.flush()
            branch_success = True
    
    # Get total NFE from fitness calculator
    total_nfe = fitness_calc.evals
    
    if worker_verbose:
        print(f"\n[Worker {worker_pid}] ✅ Branch {lineno} ({outcome_str}) completed:")
        print(f"  Convergence speed (total steps): {total_steps}")
        print(f"  Total NFE: {total_nfe}")
        print(f"  Best fitness: {best_fitness:.6g}")
        print(f"  Best solution: {best_solution}")
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
        'worker_pid': worker_pid,
        'trial_details': trial_results  # Keep for detailed analysis
    }


def run_parallel_test_with_csv(
    file_path,
    output_csv="results.csv",
    max_trials_per_branch=20,
    success_threshold=0.0,
    initial_low=-100000,
    initial_high=10000,
    max_iterations=100,
    basin_max_search=100000,
    num_workers=None,
    skip_for_false=True,
    use_biased_init=True  # Toggle biased initialization on/off
):
    """
    Run parallel branch testing and save metrics to CSV.
    Tests both True and False outcomes for each branch.
    
    CSV columns:
    - function: Function name
    - lineno: Branch line number
    - outcome: Target outcome (True/False)
    - convergence_speed: Sum of steps across all trials
    - nfe: Total number of fitness evaluations
    - best_fitness: Best fitness achieved
    - best_solution: Solution achieving best fitness
    - success: Whether branch was solved
    - num_trials: Number of trials run
    
    Args:
        skip_for_false: If True, skip for-loop and while-True False branches (unreachable)
    """
    
    print("\n" + "="*80)
    print("🚀 PARALLEL BRANCH TESTING WITH CSV OUTPUT")
    print("="*80)
    print(f"File: {file_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Workers: {num_workers if num_workers else cpu_count()}")
    print("="*80 + "\n")
    
    # Load source to discover branches
    source = open(file_path).read()
    namespace, traveler, record, _ = instrument_and_load(source)
    
    # Collect all branch tasks
    branch_tasks = []
    
    for func_info in traveler.functions:
        func_name = func_info.name
        branches = traveler.branches.get(func_name, {})
        
        if not branches:
            print(f"⏭️  Skipping {func_name} (no branches)")
            continue
        
        print(f"📝 Function: {func_name}")
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
                # Skip while-True False branches (never exit) - unreachable
                if skip_for_false and is_while_true and target_outcome is False:
                    print(f"   ⏭️  Skipping while-True False: line {lineno} (unreachable)")
                    continue
                
                # Skip for-loop False branches (not entering loop) - usually unreachable
                if skip_for_false and is_for_loop and target_outcome is False:
                    print(f"   ⏭️  Skipping for-loop False: line {lineno} (often unreachable)")
                    continue
                
                task = (
                    file_path, func_name, lineno, branch_info,
                    max_trials_per_branch, success_threshold,
                    func_initial_low, func_initial_high,
                    max_iterations, basin_max_search,
                    target_outcome,  # Target outcome
                    use_biased_init  # Biased or random initialization
                )
                branch_tasks.append(task)
    
    print(f"\n📊 Total branches to test: {len(branch_tasks)}\n")
    
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"🔧 Starting {num_workers} worker processes...")
    print("="*80 + "\n")
    sys.stdout.flush()
    
    # Run in parallel
    start_time = time.time()
    
    pool = None
    try:
        pool = Pool(processes=num_workers)
        # Use map with timeout to prevent indefinite hanging
        branch_results = pool.map(test_single_branch_with_metrics, branch_tasks)
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
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("✅ ALL BRANCHES COMPLETED")
    print("="*80)
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"📊 Total branches: {len(branch_results)}")
    print("="*80 + "\n")
    sys.stdout.flush()
    
    # Write to CSV
    print(f"📝 Writing results to {output_csv}...")
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'function', 'lineno', 'outcome', 'convergence_speed', 'nfe',
            'best_fitness', 'best_solution', 'success', 'num_trials'
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
                'num_trials': result['num_trials_run']
            })
    
    print(f"✅ Results written to {output_csv}\n")
    sys.stdout.flush()
    
    # Print summary table
    print("="*80)
    print("📈 RESULTS SUMMARY")
    print("="*80)
    print(f"{'Function':<20} {'Line':<6} {'Out':<6} {'Conv.Speed':<12} {'NFE':<10} {'Best Fitness':<15} {'Success'}")
    print("-"*80)
    
    for result in branch_results:
        success_mark = "✅" if result['success'] else "❌"
        outcome_str = "T" if result['outcome'] else "F"
        print(f"{result['function']:<20} {result['lineno']:<6} {outcome_str:<6} "
              f"{result['convergence_speed']:<12} {result['nfe']:<10} "
              f"{result['best_fitness']:<15.6g} {success_mark}")
    
    print("="*80 + "\n")
    
    # Summary statistics
    total_convergence = sum(r['convergence_speed'] for r in branch_results)
    total_nfe = sum(r['nfe'] for r in branch_results)
    successes = sum(1 for r in branch_results if r['success'])
    
    print("📊 OVERALL STATISTICS")
    print("-"*80)
    print(f"Total convergence speed: {total_convergence}")
    print(f"Total NFE: {total_nfe}")
    print(f"Success rate: {successes}/{len(branch_results)} ({100*successes/len(branch_results):.1f}%)")
    print(f"Avg convergence per branch: {total_convergence/len(branch_results):.1f}")
    print(f"Avg NFE per branch: {total_nfe/len(branch_results):.1f}")
    print("="*80 + "\n")
    sys.stdout.flush()
    
    return branch_results, output_csv


def run_directory_test(
    source_dir,
    output_dir="benchmark_log",
    max_trials_per_branch=20,
    success_threshold=0.0,
    initial_low=-100000,
    initial_high=10000,
    max_iterations=100,
    basin_max_search=100000,
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
    print(f"🔍 Found {len(py_files)} Python files in {source_dir}")
    print("="*80)
    
    for py_file in py_files:
        # Compute relative path and output CSV path
        rel_path = py_file.relative_to(source_path)
        csv_file = output_path / rel_path.with_suffix('.csv')
        
        # Create output directory if needed
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📝 Testing: {py_file}")
        print(f"📊 Output: {csv_file}")
        
        # Run test on this file
        try:
            results, _ = run_parallel_test_with_csv(
                file_path=str(py_file),
                output_csv=str(csv_file),
                max_trials_per_branch=max_trials_per_branch,
                success_threshold=success_threshold,
                initial_low=initial_low,
                initial_high=initial_high,
                max_iterations=max_iterations,
                basin_max_search=basin_max_search,
                num_workers=num_workers,
                skip_for_false=skip_for_false,
                use_biased_init=use_biased_init
            )
        except Exception as e:
            print(f"❌ Error testing {py_file}: {e}")
            continue
    
    print("\n" + "="*80)
    print(f"✅ ALL FILES TESTED! Results saved to {output_dir}/")
    print("="*80)


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    import multiprocessing
    import argparse
    
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Hill Climbing with Compression - Branch Coverage Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # With biased initialization (default):
  python test_benchmark_parallel_csv.py
  
  # With pure random initialization:
  python test_benchmark_parallel_csv.py --random-init
  
  # Custom output directory:
  python test_benchmark_parallel_csv.py --output benchmark_log_biased
  python test_benchmark_parallel_csv.py --random-init --output benchmark_log_random
        '''
    )
    parser.add_argument('--random-init', action='store_true',
                       help='Use pure random initialization instead of biased (default: biased)')
    parser.add_argument('--output', '-o', type=str, default='benchmark_log_1',
                       help='Output directory for CSV files (default: benchmark_log_1)')
    parser.add_argument('--source', '-s', type=str, default='./benchmark',
                       help='Source directory containing benchmark files (default: ./benchmark)')
    
    args = parser.parse_args()
    
    # Use 'fork' if available
    if 'fork' in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method('fork', force=True)
    else:
        multiprocessing.set_start_method('spawn', force=True)
    
    # Print configuration
    init_type = "RANDOM" if args.random_init else "BIASED"
    print(f"\n{'='*80}")
    print(f"🔧 CONFIGURATION: Hill Climbing with Compression")
    print(f"{'='*80}")
    print(f"Initialization: {init_type}")
    print(f"Source dir:     {args.source}")
    print(f"Output dir:     {args.output}")
    print(f"{'='*80}\n")
    
    # Configuration: Test entire directory
    run_directory_test(
        source_dir=args.source,
        output_dir=args.output,
        max_trials_per_branch=5,
        success_threshold=0.0,
        initial_low=-1000,
        initial_high=1000,
        max_iterations=10,
        basin_max_search=1000,
        num_workers=None,
        use_biased_init=not args.random_init  # Invert the flag
    )

