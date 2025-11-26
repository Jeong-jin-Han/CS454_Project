#!/usr/bin/env python3
"""
Parallel baseline testing (NO compression) with CSV output.

Same metrics as compression version for fair comparison.
"""

import os
import time
import random
import csv
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count
from module.sbst_core import instrument_and_load, FitnessCalculator
from hill_climb_multiD import hill_climb_simple_nd_code


def test_single_branch_baseline_with_metrics(args):
    """
    Test one branch with baseline (no compression) and return metrics.
    """
    (file_path, func_name, lineno, branch_data,
     max_trials, success_threshold, initial_low, initial_high,
     max_steps) = args
    
    worker_pid = os.getpid()
    print(f"\n[Worker PID {worker_pid}] Starting branch {func_name}::{lineno}")
    
    # Load instrumented code
    source = open(file_path).read()
    namespace, traveler, record, _ = instrument_and_load(source)
    
    # ✅ Create new FitnessCalculator with its own eval counter
    fitness_calc = FitnessCalculator(traveler, record, namespace)
    fitness_calc.evals = 0
    
    parent_map = traveler.parent_map
    func_obj = namespace[func_name]
    
    # Get branch info
    branch_info = traveler.branches[func_name][lineno]
    target_branch_node = branch_info.node
    subject_node = branch_info.subject
    target_outcome = True
    
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
          - uniform over [low, high]
          - Gaussians centered at extracted constants for this variable
        """
        # If we have no constants at all, fall back to uniform
        if not total_constants and not var_constants:
            return random.randint(low, high)

        # 50% uniform, 50% biased around constants
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
    
    trial_results = []
    
    # Run all trials
    for trial in range(max_trials):
        if branch_success:
            print(f"[Worker {worker_pid}] Branch {lineno} succeeded, skipping remaining trials")
            break
        
        # Random initial point (biased by extracted constants)
        random.seed(42 + lineno * 1000 + trial)
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
        
        print(f"[Worker {worker_pid}] Branch {lineno}, Trial {trial+1}/{max_trials}: init_f={init_fit:.4f}")
        
        # Run baseline hill climbing (NO compression)
        traj = hill_climb_simple_nd_code(
            fitness_calc, func_obj,
            target_branch_node, target_outcome,
            subject_node, parent_map,
            initial, dim,
            max_steps=max_steps
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
            print(f"[Worker {worker_pid}] 🎉 Branch {lineno} succeeded at trial {trial}")
            branch_success = True
    
    # Get total NFE
    total_nfe = fitness_calc.evals
    
    print(f"\n[Worker {worker_pid}] ✅ Branch {lineno} completed:")
    print(f"  Convergence speed: {total_steps}")
    print(f"  Total NFE: {total_nfe}")
    print(f"  Best fitness: {best_fitness:.6g}")
    
    return {
        'function': func_name,
        'lineno': lineno,
        'convergence_speed': total_steps,
        'nfe': total_nfe,
        'best_fitness': best_fitness,
        'best_solution': best_solution,
        'success': branch_success,
        'num_trials_run': len(trial_results),
        'worker_pid': worker_pid
    }


def run_parallel_test_baseline_with_csv(
    file_path,
    output_csv="results_baseline.csv",
    max_trials_per_branch=20,
    success_threshold=0.0,
    initial_low=-100000,
    initial_high=10000,
    max_steps=2000,
    num_workers=None
):
    """
    Run baseline parallel testing and save to CSV.
    """
    
    print("\n" + "="*80)
    print("🚀 PARALLEL BASELINE TESTING (No Compression) WITH CSV")
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
            task = (
                file_path, func_name, lineno, branch_info,
                max_trials_per_branch, success_threshold,
                func_initial_low, func_initial_high,
                max_steps
            )
            branch_tasks.append(task)
    
    print(f"\n📊 Total branches to test: {len(branch_tasks)}\n")
    
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"🔧 Starting {num_workers} worker processes...")
    print("="*80 + "\n")
    
    # Run in parallel
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        branch_results = pool.map(test_single_branch_baseline_with_metrics, branch_tasks)
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("✅ ALL BRANCHES COMPLETED")
    print("="*80)
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"📊 Total branches: {len(branch_results)}")
    print("="*80 + "\n")
    
    # Write to CSV
    print(f"📝 Writing results to {output_csv}...")
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'function', 'lineno', 'convergence_speed', 'nfe',
            'best_fitness', 'best_solution', 'success', 'num_trials'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for result in branch_results:
            writer.writerow({
                'function': result['function'],
                'lineno': result['lineno'],
                'convergence_speed': result['convergence_speed'],
                'nfe': result['nfe'],
                'best_fitness': result['best_fitness'],
                'best_solution': str(result['best_solution']),
                'success': result['success'],
                'num_trials': result['num_trials_run']
            })
    
    print(f"✅ Results written to {output_csv}\n")
    
    # Print summary table
    print("="*80)
    print("📈 RESULTS SUMMARY")
    print("="*80)
    print(f"{'Function':<20} {'Line':<6} {'Conv.Speed':<12} {'NFE':<10} {'Best Fitness':<15} {'Success'}")
    print("-"*80)
    
    for result in branch_results:
        success_mark = "✅" if result['success'] else "❌"
        print(f"{result['function']:<20} {result['lineno']:<6} "
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
    
    return branch_results, output_csv


def run_directory_test_baseline(
    source_dir,
    output_dir="benchmark_log",
    max_trials_per_branch=20,
    success_threshold=0.0,
    initial_low=-100000,
    initial_high=10000,
    max_steps=2000,
    num_workers=None
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
            results, _ = run_parallel_test_baseline_with_csv(
                file_path=str(py_file),
                output_csv=str(csv_file),
                max_trials_per_branch=max_trials_per_branch,
                success_threshold=success_threshold,
                initial_low=initial_low,
                initial_high=initial_high,
                max_steps=max_steps,
                num_workers=num_workers
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
    
    # Use 'fork' if available
    if 'fork' in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method('fork', force=True)
    else:
        multiprocessing.set_start_method('spawn', force=True)
    
    # Configuration: Test entire directory
    run_directory_test_baseline(
        source_dir="./benchmark",
        output_dir="benchmark_log_2",
        max_trials_per_branch=5,
        success_threshold=0.0,
        initial_low=-1000,      # Reduced from -100000
        initial_high=1000,      # Reduced from 10000
        max_steps=2000,
        num_workers=None  # Use all CPU cores
    )

