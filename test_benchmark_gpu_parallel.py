"""
Parallelized branch testing with compression-based hill climbing.

This version parallelizes across branches and trials to fully utilize CPU resources.
Note: GPU acceleration is not applicable here since we're executing Python AST code,
but we can use multiprocessing to parallelize independent tests.
"""

from module.sbst_core import instrument_and_load, FitnessCalculator
import ast
import random
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import time

from hill_climb_multiD_gpu import hill_climb_with_compression_nd_code, CompressionManagerND

# ============================================================================
# Worker function for parallel trial execution
# ============================================================================
def run_single_trial(args_tuple):
    """
    Run a single trial for a specific branch.
    This function will be executed in parallel by worker processes.
    
    Returns:
    --------
    dict with trial results
    """
    (trial_idx, func_name, lineno, target_outcome, 
     dim, initial_point, file_path, 
     max_iterations, basin_max_search, 
     success_threshold) = args_tuple
    
    # Each worker needs its own instrumented code instance
    source = open(file_path).read()
    namespace, traveler, record, instrumented_tree = instrument_and_load(source)
    
    fitness_calc = FitnessCalculator(traveler, record, namespace)
    parent_map = traveler.parent_map
    
    # Find the function and branch info
    func_obj = namespace[func_name]
    branches = traveler.branches.get(func_name, {})
    branch_info = branches[lineno]
    
    target_branch_node = branch_info.node
    subject_node = branch_info.subject
    
    # Create fresh compression manager for this trial
    cm = CompressionManagerND(dim, steepness=5.0)
    
    # Calculate initial fitness
    init_fit = fitness_calc.fitness_for_candidate(
        func_obj, initial_point,
        target_branch_node, target_outcome,
        subject_node, parent_map
    )
    
    print(f"[Worker PID={id(cm)%10000}] func={func_name}, line={lineno}, trial={trial_idx}: init_f={init_fit:.4f}, start={initial_point}")
    
    # Run hill climbing with compression
    traj, cm = hill_climb_with_compression_nd_code(
        fitness_calc, func_obj,
        target_branch_node, target_outcome,
        subject_node, parent_map,
        initial_point,
        dim,
        max_iterations=max_iterations,
        basin_max_search=basin_max_search,
        global_min_threshold=1e-6,
        cm=cm
    )
    
    # Extract final results
    final_point, final_f, used_comp = traj[-1]
    
    print(f"[Worker PID={id(cm)%10000}] func={func_name}, line={lineno}, trial={trial_idx}: final_f={final_f:.6g}, steps={len(traj)}")
    
    return {
        "function": func_name,
        "lineno": lineno,
        "trial": trial_idx,
        "target_outcome": target_outcome,
        "initial_point": list(initial_point),
        "initial_fitness": float(init_fit),
        "final_point": list(final_point),
        "final_fitness": float(final_f),
        "steps": len(traj),
        "success": final_f <= success_threshold
    }


# ============================================================================
# Main parallel testing function
# ============================================================================
def test_branches_parallel(
    file_path,
    max_trials_per_branch=20,
    success_threshold=0.0,
    initial_low=-100000,
    initial_high=10000,
    max_iterations=100,
    basin_max_search=100000,
    num_workers=None,
    filter_lineno=None  # Optional: only test specific line numbers
):
    """
    Test all branches in parallel using multiprocessing.
    
    Parameters:
    -----------
    file_path : str
        Path to the Python file to test
    max_trials_per_branch : int
        Maximum number of trials per branch
    success_threshold : float
        Fitness threshold for success
    initial_low, initial_high : int
        Range for random initial points
    max_iterations : int
        Max compression iterations per trial
    basin_max_search : int
        Basin detection search range
    num_workers : int or None
        Number of parallel workers (default: CPU count)
    filter_lineno : set or None
        If provided, only test branches at these line numbers
    
    Returns:
    --------
    results : list of dicts
        All trial results
    """
    
    print("\n" + "="*80)
    print("🚀 PARALLEL BRANCH TESTING WITH COMPRESSION")
    print("="*80)
    print(f"📁 File: {file_path}")
    print(f"🔢 Workers: {num_workers if num_workers else cpu_count()}")
    print(f"🎯 Max trials per branch: {max_trials_per_branch}")
    print(f"✅ Success threshold: {success_threshold}")
    print("="*80 + "\n")
    
    # Load source once to discover functions and branches
    source = open(file_path).read()
    namespace, traveler, record, instrumented_tree = instrument_and_load(source)
    
    random.seed(42)
    
    # Collect all tasks (branch, trial combinations)
    all_tasks = []
    branch_info_map = {}  # For summary reporting
    
    for func_info in traveler.functions:
        func_name = func_info.name
        func_args = func_info.args
        func_dims = func_info.args_dim
        dim = len(func_args)
        
        # Use function-specific bounds if available
        func_initial_low = func_info.min_const
        func_initial_high = func_info.max_const
        
        branches = traveler.branches.get(func_name, {})
        
        if not branches:
            print(f"⏭️  Skipping {func_name} (no branches)")
            continue
        
        print(f"\n📝 Function: {func_name}, args={func_args}, dim={dim}")
        print(f"   Branches: {list(branches.keys())}")
        
        for lineno, branch_info in branches.items():
            # Optional filter
            if filter_lineno is not None and lineno not in filter_lineno:
                continue
            
            branch_key = (func_name, lineno)
            branch_info_map[branch_key] = {
                'func_args': func_args,
                'dim': dim,
                'initial_low': func_initial_low,
                'initial_high': func_initial_high
            }
            
            # Generate random initial points for all trials
            target_outcome = True
            
            for trial in range(max_trials_per_branch):
                initial_point = [
                    random.randint(func_initial_low, func_initial_high) 
                    for _ in func_args
                ]
                
                task = (
                    trial, func_name, lineno, target_outcome,
                    dim, initial_point, file_path,
                    max_iterations, basin_max_search,
                    success_threshold
                )
                all_tasks.append(task)
    
    print(f"\n📊 Total tasks to execute: {len(all_tasks)}")
    print(f"   (branches × trials)\n")
    
    # Execute all tasks in parallel
    if num_workers is None:
        num_workers = cpu_count()
    
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_single_trial, all_tasks)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ ALL PARALLEL TASKS COMPLETED")
    print("="*80)
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
    print(f"📊 Total trials: {len(results)}")
    print("="*80 + "\n")
    
    # Summarize results by branch
    branch_summaries = {}
    
    for result in results:
        branch_key = (result['function'], result['lineno'])
        
        if branch_key not in branch_summaries:
            branch_summaries[branch_key] = {
                'total_trials': 0,
                'successes': 0,
                'best_fitness': float('inf'),
                'best_point': None,
                'total_steps': 0
            }
        
        summary = branch_summaries[branch_key]
        summary['total_trials'] += 1
        if result['success']:
            summary['successes'] += 1
        if result['final_fitness'] < summary['best_fitness']:
            summary['best_fitness'] = result['final_fitness']
            summary['best_point'] = result['final_point']
        summary['total_steps'] += result['steps']
    
    # Print branch summaries
    print("\n" + "="*80)
    print("📈 BRANCH SUMMARY")
    print("="*80)
    
    for branch_key, summary in sorted(branch_summaries.items()):
        func_name, lineno = branch_key
        success_rate = 100.0 * summary['successes'] / summary['total_trials']
        avg_steps = summary['total_steps'] / summary['total_trials']
        
        print(f"\n🔍 {func_name}::{lineno}")
        print(f"   Success rate: {summary['successes']}/{summary['total_trials']} ({success_rate:.1f}%)")
        print(f"   Best fitness: {summary['best_fitness']:.6g}")
        print(f"   Best point: {summary['best_point']}")
        print(f"   Avg steps: {avg_steps:.1f}")
    
    print("\n" + "="*80)
    
    return results, branch_summaries


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    # Configuration
    file_path = "./benchmark/count_divisor_2.py"
    
    # Optional: filter to specific line numbers
    filter_lineno = {28}  # Set to None to test all branches
    
    results, summaries = test_branches_parallel(
        file_path=file_path,
        max_trials_per_branch=20,
        success_threshold=0.0,
        initial_low=-100000,
        initial_high=10000,
        max_iterations=100,
        basin_max_search=100000,
        num_workers=None,  # Use all CPU cores
        filter_lineno=filter_lineno
    )
    
    # Print detailed results
    print("\n" + "="*80)
    print("📋 DETAILED RESULTS")
    print("="*80)
    
    for r in results:
        print(
            f"func={r['function']}, line={r['lineno']}, trial={r['trial']}: "
            f"init_f={r['initial_fitness']:.3g} → final_f={r['final_fitness']:.3g}, "
            f"steps={r['steps']}, success={r['success']}"
        )
    
    print("\n" + "="*80)
    print("🏁 PARALLEL TESTING COMPLETE")
    print("="*80)

