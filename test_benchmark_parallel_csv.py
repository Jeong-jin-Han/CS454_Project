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
import time
import random
import csv
from multiprocessing import Pool, cpu_count
from module.sbst_core import instrument_and_load, FitnessCalculator
from hill_climb_multiD import hill_climb_with_compression_nd_code, CompressionManagerND


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
     max_iterations, basin_max_search) = args
    
    worker_pid = os.getpid()
    print(f"\n[Worker PID {worker_pid}] Starting branch {func_name}::{lineno}")
    
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
    target_outcome = True
    
    # Get function info
    func_info = [f for f in traveler.functions if f.name == func_name][0]
    dim = len(func_info.args)
    func_args = func_info.args
    
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
            print(f"[Worker {worker_pid}] Branch {lineno} succeeded, skipping remaining trials")
            break
        
        # Random initial point
        random.seed(42 + lineno * 1000 + trial)
        initial = [random.randint(initial_low, initial_high) for _ in func_args]
        
        # Initial fitness (counted in NFE)
        init_fit = fitness_calc.fitness_for_candidate(
            func_obj, initial,
            target_branch_node, target_outcome,
            subject_node, parent_map
        )
        
        print(f"[Worker {worker_pid}] Branch {lineno}, Trial {trial+1}/{max_trials}: init_f={init_fit:.4f}")
        
        # Run hill climbing (all fitness evals counted automatically)
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
            print(f"[Worker {worker_pid}] 🎉 Branch {lineno} succeeded at trial {trial}")
            branch_success = True
    
    # Get total NFE from fitness calculator
    total_nfe = fitness_calc.evals
    
    print(f"\n[Worker {worker_pid}] ✅ Branch {lineno} completed:")
    print(f"  Convergence speed (total steps): {total_steps}")
    print(f"  Total NFE: {total_nfe}")
    print(f"  Best fitness: {best_fitness:.6g}")
    print(f"  Best solution: {best_solution}")
    
    return {
        'function': func_name,
        'lineno': lineno,
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
    num_workers=None
):
    """
    Run parallel branch testing and save metrics to CSV.
    
    CSV columns:
    - function: Function name
    - lineno: Branch line number
    - convergence_speed: Sum of steps across all trials
    - nfe: Total number of fitness evaluations
    - best_fitness: Best fitness achieved
    - best_solution: Solution achieving best fitness
    - success: Whether branch was solved
    - num_trials: Number of trials run
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
        
        func_initial_low = func_info.min_const
        func_initial_high = func_info.max_const
        
        for lineno, branch_info in branches.items():
            task = (
                file_path, func_name, lineno, branch_info,
                max_trials_per_branch, success_threshold,
                func_initial_low, func_initial_high,
                max_iterations, basin_max_search
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
        branch_results = pool.map(test_single_branch_with_metrics, branch_tasks)
    
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
    
    # Configuration
    file_path = "./benchmark/count_divisor_2.py"
    output_csv = "results_parallel.csv"
    
    results, csv_path = run_parallel_test_with_csv(
        file_path=file_path,
        output_csv=output_csv,
        max_trials_per_branch=20,
        success_threshold=0.0,
        max_iterations=100,
        basin_max_search=100000,
        num_workers=None  # Use all CPU cores
    )
    
    print(f"\n🏁 TESTING COMPLETE! Results saved to: {csv_path}\n")

