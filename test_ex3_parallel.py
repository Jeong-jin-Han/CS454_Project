#!/usr/bin/env python3
"""
Quick test with ex3.py to debug parallelization
"""

import os
import time
import random
from multiprocessing import Pool, cpu_count, current_process
from module.sbst_core import instrument_and_load, FitnessCalculator
from hill_climb_multiD_gpu import hill_climb_with_compression_nd_code, CompressionManagerND

def run_one_trial(args):
    """Run a single trial - with PID printing"""
    trial_idx, file_path, func_name, lineno = args
    
    pid = os.getpid()
    process_name = current_process().name
    
    print(f"\n[WORKER {process_name} PID={pid}] Starting trial {trial_idx}")
    
    # Load and instrument
    source = open(file_path).read()
    namespace, traveler, record, _ = instrument_and_load(source)
    
    fitness_calc = FitnessCalculator(traveler, record, namespace)
    parent_map = traveler.parent_map
    
    # Get function and branch
    func_obj = namespace[func_name]
    branch_info = traveler.branches[func_name][lineno]
    func_info = [f for f in traveler.functions if f.name == func_name][0]
    dim = len(func_info.args)
    
    # Random initial point
    random.seed(42 + trial_idx)
    initial = [random.randint(-100, 100) for _ in func_info.args]
    
    # Run hill climbing
    cm = CompressionManagerND(dim, steepness=5.0)
    
    traj, cm = hill_climb_with_compression_nd_code(
        fitness_calc, func_obj,
        branch_info.node, True,
        branch_info.subject, parent_map,
        initial, dim,
        max_iterations=10,
        basin_max_search=1000,
        global_min_threshold=1e-6,
        cm=cm
    )
    
    final_point, final_f, used_comp = traj[-1]
    
    print(f"[WORKER {process_name} PID={pid}] Trial {trial_idx} done: f={final_f:.4f}")
    
    return {
        'trial': trial_idx,
        'pid': pid,
        'process_name': process_name,
        'fitness': final_f,
        'steps': len(traj)
    }


def test_parallel():
    """Test parallel execution with ex3.py"""
    print("\n" + "="*80)
    print("🔍 PARALLEL TEST WITH EX3.PY")
    print("="*80)
    print(f"Main PID: {os.getpid()}")
    print(f"CPU cores: {cpu_count()}")
    print("="*80 + "\n")
    
    file_path = "./benchmark/ex3.py"
    
    # Get function info
    source = open(file_path).read()
    namespace, traveler, record, _ = instrument_and_load(source)
    
    func_name = traveler.functions[0].name
    branches = traveler.branches[func_name]
    first_lineno = list(branches.keys())[0]
    
    print(f"Testing function: {func_name}")
    print(f"First branch at line: {first_lineno}")
    print(f"Number of branches: {len(branches)}\n")
    
    # Create tasks
    num_trials = 16
    tasks = [(i, file_path, func_name, first_lineno) for i in range(num_trials)]
    
    print(f"Running {num_trials} trials...")
    print("="*80)
    
    # Test 1: Sequential (1 worker)
    print("\n📊 TEST 1: SEQUENTIAL (1 worker)")
    print("─"*80)
    start1 = time.time()
    with Pool(processes=1) as pool:
        results1 = pool.map(run_one_trial, tasks)
    time1 = time.time() - start1
    
    pids1 = set(r['pid'] for r in results1)
    print(f"\nSequential time: {time1:.2f}s")
    print(f"Unique PIDs: {len(pids1)}")
    print(f"PIDs: {pids1}")
    
    # Test 2: Parallel (4 workers)
    print("\n\n📊 TEST 2: PARALLEL (4 workers)")
    print("─"*80)
    print("⚠️  Watch htop NOW - you should see 4-5 Python processes!")
    print("─"*80)
    
    start2 = time.time()
    with Pool(processes=4) as pool:
        results2 = pool.map(run_one_trial, tasks, chunksize=1)
    time2 = time.time() - start2
    
    pids2 = set(r['pid'] for r in results2)
    print(f"\nParallel time: {time2:.2f}s")
    print(f"Unique PIDs: {len(pids2)}")
    print(f"PIDs: {pids2}")
    
    # Analysis
    speedup = time1 / time2
    
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    print(f"Sequential:  {time1:.2f}s  (1 PID)")
    print(f"Parallel:    {time2:.2f}s  ({len(pids2)} PIDs)")
    print(f"Speedup:     {speedup:.2f}x")
    print("="*80)
    
    if len(pids2) > 1:
        print("\n✅ SUCCESS: Multiple processes used!")
        print(f"   {len(pids2)} different worker PIDs detected")
    else:
        print("\n❌ FAILURE: Only 1 process used!")
        print("   Parallelization is NOT working!")
    
    if speedup > 1.5:
        print(f"\n✅ SPEEDUP: {speedup:.2f}x is good!")
    elif speedup > 1.0:
        print(f"\n⚠️  SPEEDUP: {speedup:.2f}x is marginal")
    else:
        print(f"\n❌ NO SPEEDUP: {speedup:.2f}x means slower!")
    
    print("\n" + "="*80)
    
    # Show per-worker distribution
    print("\n📋 Worker Distribution:")
    from collections import Counter
    worker_counts = Counter(r['process_name'] for r in results2)
    for worker, count in sorted(worker_counts.items()):
        print(f"   {worker}: {count} trials")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import multiprocessing
    
    # Try fork first (faster), fall back to spawn
    start_method = 'fork' if 'fork' in multiprocessing.get_all_start_methods() else 'spawn'
    multiprocessing.set_start_method(start_method, force=True)
    
    print(f"\nUsing start method: {start_method}")
    
    test_parallel()

