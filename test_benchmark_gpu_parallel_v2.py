"""
Advanced parallelized branch testing with SHARED compression metadata.

This version:
1. Parallelizes ACROSS branches (each branch runs independently)
2. WITHIN each branch, trials run sequentially but share compression metadata
3. This allows later trials to benefit from basins discovered in earlier trials

This is a hybrid approach: branch-level parallelism + trial-level metadata sharing.
"""

from module.sbst_core import instrument_and_load, FitnessCalculator
import ast
import random
from multiprocessing import Pool, cpu_count
import time

from hill_climb_multiD_gpu import hill_climb_with_compression_nd_code, CompressionManagerND


# ============================================================================
# Worker function for a single BRANCH (all trials for that branch)
# ============================================================================
def test_single_branch_all_trials(args_tuple):
    """
    Test a single branch with multiple trials, sharing compression metadata.
    
    This function runs ALL trials for one branch sequentially, but different
    branches run in parallel across workers.
    
    Returns:
    --------
    list of dicts (one per trial)
    """
    (func_name, lineno, target_outcome, dim, func_args,
     initial_low, initial_high, file_path,
     max_trials, max_iterations, basin_max_search,
     success_threshold, random_seed) = args_tuple
    
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
    
    # Create ONE compression manager for this branch (shared across all trials)
    branch_cm = CompressionManagerND(dim, steepness=5.0)
    
    print(f"\n{'='*60}")
    print(f"[Worker] Testing {func_name}::line{lineno}")
    print(f"         {max_trials} trials with SHARED metadata")
    print(f"{'='*60}\n")
    
    random.seed(random_seed)
    trial_results = []
    branch_success = False
    
    for trial in range(max_trials):
        if branch_success:
            print(f"[{func_name}::{lineno}] Skipping remaining trials (already succeeded)")
            break
        
        print(f"\n[{func_name}::{lineno}] Trial {trial+1}/{max_trials}")
        print(f"{'─'*60}")
        
        # Generate random initial point
        initial_point = [
            random.randint(initial_low, initial_high) 
            for _ in func_args
        ]
        
        # Calculate initial fitness
        init_fit = fitness_calc.fitness_for_candidate(
            func_obj, initial_point,
            target_branch_node, target_outcome,
            subject_node, parent_map
        )
        
        print(f"  Initial: {initial_point}, fitness={init_fit:.4f}")
        
        # Run hill climbing with SHARED compression manager
        traj, branch_cm = hill_climb_with_compression_nd_code(
            fitness_calc, func_obj,
            target_branch_node, target_outcome,
            subject_node, parent_map,
            initial_point,
            dim,
            max_iterations=max_iterations,
            basin_max_search=basin_max_search,
            global_min_threshold=1e-6,
            cm=branch_cm  # ✅ Reuse metadata across trials
        )
        
        # Extract final results
        final_point, final_f, used_comp = traj[-1]
        success = final_f <= success_threshold
        
        # Count compressions accumulated so far
        total_compressions = sum(len(branch_cm.dim_compressions[d]) for d in range(dim))
        
        print(f"  Final: {final_point}, fitness={final_f:.6g}")
        print(f"  Steps: {len(traj)}, Compressions: {total_compressions}")
        print(f"  Success: {'✅' if success else '❌'}")
        
        trial_result = {
            "function": func_name,
            "lineno": lineno,
            "trial": trial,
            "target_outcome": target_outcome,
            "initial_point": list(initial_point),
            "initial_fitness": float(init_fit),
            "final_point": list(final_point),
            "final_fitness": float(final_f),
            "steps": len(traj),
            "total_compressions": total_compressions,
            "success": success
        }
        trial_results.append(trial_result)
        
        if success:
            print(f"\n  🎉 SUCCESS at trial {trial+1}!")
            branch_success = True
    
    print(f"\n{'='*60}")
    print(f"[Worker] Completed {func_name}::line{lineno}")
    print(f"         Trials: {len(trial_results)}, Success: {branch_success}")
    print(f"{'='*60}\n")
    
    return trial_results


# ============================================================================
# Main parallel testing function
# ============================================================================
def test_branches_parallel_v2(
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
    Test all branches in parallel (branch-level parallelism).
    Within each branch, trials run sequentially with shared metadata.
    
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
    branch_summaries : dict
        Summary statistics per branch
    """
    
    print("\n" + "="*80)
    print("🚀 PARALLEL BRANCH TESTING (V2: Shared Metadata)")
    print("="*80)
    print(f"📁 File: {file_path}")
    print(f"🔢 Workers: {num_workers if num_workers else cpu_count()}")
    print(f"🎯 Max trials per branch: {max_trials_per_branch}")
    print(f"✅ Success threshold: {success_threshold}")
    print(f"♻️  Compression metadata: SHARED within each branch")
    print("="*80 + "\n")
    
    # Load source once to discover functions and branches
    source = open(file_path).read()
    namespace, traveler, record, instrumented_tree = instrument_and_load(source)
    
    # Collect all branch-level tasks
    branch_tasks = []
    branch_info_map = {}
    
    for func_info in traveler.functions:
        func_name = func_info.name
        func_args = func_info.args
        func_dims = func_info.args_dim
        dim = len(func_args)
        
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
            
            # Create one task per branch (task will run all trials)
            target_outcome = True
            random_seed = 42 + lineno  # Deterministic but different per branch
            
            task = (
                func_name, lineno, target_outcome, dim, func_args,
                func_initial_low, func_initial_high, file_path,
                max_trials_per_branch, max_iterations, basin_max_search,
                success_threshold, random_seed
            )
            branch_tasks.append(task)
    
    print(f"\n📊 Total branches to test: {len(branch_tasks)}")
    print(f"   (each branch will run up to {max_trials_per_branch} trials)\n")
    
    # Execute all branch tasks in parallel
    if num_workers is None:
        num_workers = min(cpu_count(), len(branch_tasks))
    
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        # Each worker returns a list of trial results
        branch_results_list = pool.map(test_single_branch_all_trials, branch_tasks)
    
    elapsed_time = time.time() - start_time
    
    # Flatten results
    all_results = []
    for branch_results in branch_results_list:
        all_results.extend(branch_results)
    
    print("\n" + "="*80)
    print("✅ ALL PARALLEL BRANCH TESTS COMPLETED")
    print("="*80)
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
    print(f"📊 Total branches: {len(branch_tasks)}")
    print(f"📊 Total trials: {len(all_results)}")
    print("="*80 + "\n")
    
    # Summarize results by branch
    branch_summaries = {}
    
    for result in all_results:
        branch_key = (result['function'], result['lineno'])
        
        if branch_key not in branch_summaries:
            branch_summaries[branch_key] = {
                'total_trials': 0,
                'successes': 0,
                'best_fitness': float('inf'),
                'best_point': None,
                'total_steps': 0,
                'final_compressions': 0
            }
        
        summary = branch_summaries[branch_key]
        summary['total_trials'] += 1
        if result['success']:
            summary['successes'] += 1
        if result['final_fitness'] < summary['best_fitness']:
            summary['best_fitness'] = result['final_fitness']
            summary['best_point'] = result['final_point']
        summary['total_steps'] += result['steps']
        summary['final_compressions'] = result['total_compressions']  # Last trial's count
    
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
        print(f"   Compressions accumulated: {summary['final_compressions']}")
    
    print("\n" + "="*80)
    
    return all_results, branch_summaries


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    # Configuration
    file_path = "./benchmark/count_divisor_2.py"
    
    # Optional: filter to specific line numbers
    filter_lineno = {28}  # Set to None to test all branches
    
    results, summaries = test_branches_parallel_v2(
        file_path=file_path,
        max_trials_per_branch=20,
        success_threshold=0.0,
        initial_low=-100000,
        initial_high=10000,
        max_iterations=100,
        basin_max_search=100000,
        num_workers=None,  # Use optimal number of workers
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
            f"steps={r['steps']}, compressions={r['total_compressions']}, "
            f"success={r['success']}"
        )
    
    print("\n" + "="*80)
    print("🏁 PARALLEL TESTING COMPLETE")
    print("="*80)

