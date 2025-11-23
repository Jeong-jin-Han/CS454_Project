#!/usr/bin/env python3
"""
Parallel version of test_benchmark_base.py (WITHOUT compression)

Strategy: Same as test_benchmark_parallel.py but uses baseline hill climbing
"""

import os
import time
import random
from multiprocessing import Pool, cpu_count
from module.sbst_core import instrument_and_load, FitnessCalculator
from hill_climb_multiD import hill_climb_simple_nd_code


def test_single_branch_baseline(args):
    """
    Test one branch with baseline (no compression) hill climbing.
    """
    (file_path, func_name, lineno, branch_data,
     max_trials, success_threshold, initial_low, initial_high,
     max_steps) = args
    
    worker_pid = os.getpid()
    print(f"\n[Worker PID {worker_pid}] Starting branch {func_name}::{lineno}")
    
    # Load instrumented code
    source = open(file_path).read()
    namespace, traveler, record, _ = instrument_and_load(source)
    
    fitness_calc = FitnessCalculator(traveler, record, namespace)
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
    
    branch_results = []
    best_result = None
    branch_success = False
    
    # Run all trials
    for trial in range(max_trials):
        if branch_success:
            print(f"[Worker {worker_pid}] Branch {lineno} already succeeded, skipping remaining trials")
            break
        
        # Random initial point
        random.seed(42 + lineno * 1000 + trial)
        initial = [random.randint(initial_low, initial_high) for _ in func_args]
        
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
        
        trial_result = {
            "function": func_name,
            "lineno": lineno,
            "trial": trial,
            "target_outcome": target_outcome,
            "initial_point": initial,
            "initial_fitness": float(init_fit),
            "final_point": list(final_point),
            "final_fitness": float(final_f),
            "steps": len(traj),
            "worker_pid": worker_pid
        }
        branch_results.append(trial_result)
        
        # Update best
        if best_result is None or final_f < best_result["final_fitness"]:
            best_result = trial_result
        
        # Check success
        if final_f <= success_threshold:
            print(f"[Worker {worker_pid}] 🎉 Branch {lineno} succeeded at trial {trial} with f={final_f:.6g}")
            branch_success = True
    
    print(f"\n[Worker {worker_pid}] ✅ Branch {lineno} completed:")
    print(f"  Trials run: {len(branch_results)}")
    print(f"  Success: {branch_success}")
    print(f"  Best fitness: {best_result['final_fitness']:.6g}")
    
    return {
        'branch_key': (func_name, lineno),
        'results': branch_results,
        'best_result': best_result,
        'success': branch_success,
        'worker_pid': worker_pid
    }


def run_parallel_test_baseline(
    file_path,
    max_trials_per_branch=20,
    success_threshold=0.0,
    initial_low=-100000,
    initial_high=10000,
    max_steps=2000,
    num_workers=None
):
    """
    Run baseline branch testing in parallel (no compression).
    """
    
    print("\n" + "="*80)
    print("🚀 PARALLEL BASELINE TESTING (No Compression)")
    print("="*80)
    print(f"File: {file_path}")
    print(f"Workers: {num_workers if num_workers else cpu_count()}")
    print(f"Max trials per branch: {max_trials_per_branch}")
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
        
        func_initial_low = func_info.min_const
        func_initial_high = func_info.max_const
        
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
        branch_results = pool.map(test_single_branch_baseline, branch_tasks)
    
    elapsed_time = time.time() - start_time
    
    # Collect results
    all_results = []
    for branch_result in branch_results:
        all_results.extend(branch_result['results'])
    
    # Print summary
    print("\n" + "="*80)
    print("✅ ALL BRANCHES COMPLETED")
    print("="*80)
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"📊 Total branches: {len(branch_results)}")
    print(f"📊 Total trials: {len(all_results)}")
    print("="*80 + "\n")
    
    # Branch summary
    print("="*80)
    print("📈 BRANCH SUMMARY")
    print("="*80)
    
    for br in branch_results:
        func_name, lineno = br['branch_key']
        success_str = "✅" if br['success'] else "❌"
        
        print(f"\n{success_str} {func_name}::{lineno} (Worker PID {br['worker_pid']})")
        print(f"   Trials: {len(br['results'])}")
        print(f"   Best fitness: {br['best_result']['final_fitness']:.6g}")
        print(f"   Best point: {br['best_result']['final_point']}")
    
    print("\n" + "="*80)
    
    # Detailed results
    print("\n" + "="*80)
    print("📋 DETAILED TRIAL RESULTS")
    print("="*80)
    
    for r in all_results:
        print(
            f"func={r['function']}, line={r['lineno']}, trial={r['trial']}: "
            f"init_f={r['initial_fitness']:.3g} → final_f={r['final_fitness']:.3g}, "
            f"steps={r['steps']}, PID={r['worker_pid']}"
        )
    
    print("\n" + "="*80)
    
    return all_results, branch_results


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
    
    results, branch_summaries = run_parallel_test_baseline(
        file_path=file_path,
        max_trials_per_branch=20,
        success_threshold=0.0,
        max_steps=2000,
        num_workers=None  # Use all CPU cores
    )
    
    print("\n🏁 BASELINE TESTING COMPLETE!\n")

