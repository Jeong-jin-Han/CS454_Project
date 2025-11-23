#!/usr/bin/env python3
"""
DEBUG version to find why parallelization is not working.
"""

import os
import time
from multiprocessing import Pool, current_process, cpu_count
from module.sbst_core import instrument_and_load, FitnessCalculator
import random

def worker_function(args):
    """Debug worker to see if it's actually running in parallel"""
    task_id, sleep_time = args
    pid = os.getpid()
    process_name = current_process().name
    
    print(f"[PID {pid}] {process_name}: Starting task {task_id}")
    time.sleep(sleep_time)
    print(f"[PID {pid}] {process_name}: Finished task {task_id}")
    
    return task_id, pid

def test_basic_multiprocessing():
    """Test if basic multiprocessing works"""
    print("\n" + "="*80)
    print("TEST 1: Basic Multiprocessing Test")
    print("="*80)
    print(f"Main PID: {os.getpid()}")
    print(f"CPU count: {cpu_count()}")
    print("\nStarting 8 tasks on 4 workers...")
    print("If parallelization works, you should see multiple PIDs in htop\n")
    
    tasks = [(i, 2) for i in range(8)]  # 8 tasks, each sleeps 2 seconds
    
    start = time.time()
    with Pool(processes=4) as pool:
        results = pool.map(worker_function, tasks)
    elapsed = time.time() - start
    
    print(f"\nResults: {results}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Expected: ~4s (8 tasks / 4 workers * 2s)")
    print(f"Speedup: {(8*2)/elapsed:.2f}x")
    
    unique_pids = len(set(r[1] for r in results))
    print(f"Unique PIDs: {unique_pids} (should be 4)")
    
    if elapsed < 6 and unique_pids > 1:
        print("\n✅ Basic multiprocessing WORKS!")
        return True
    else:
        print("\n❌ Basic multiprocessing FAILED!")
        return False


def run_single_trial_debug(args):
    """Debug version of trial runner with extensive logging"""
    trial_idx, file_path = args
    
    pid = os.getpid()
    print(f"[Worker PID {pid}] Starting trial {trial_idx}")
    
    # Import and instrument
    source = open(file_path).read()
    namespace, traveler, record, instrumented_tree = instrument_and_load(source)
    
    fitness_calc = FitnessCalculator(traveler, record, namespace)
    parent_map = traveler.parent_map
    
    # Get first function and branch
    func_info = traveler.functions[0]
    func_obj = namespace[func_info.name]
    
    branches = traveler.branches.get(func_info.name, {})
    if not branches:
        return trial_idx, pid, None
    
    first_branch_lineno = list(branches.keys())[0]
    branch_info = branches[first_branch_lineno]
    
    # Random initial point
    random.seed(42 + trial_idx)
    initial = [random.randint(-100, 100) for _ in func_info.args]
    
    # Calculate fitness
    fitness = fitness_calc.fitness_for_candidate(
        func_obj, initial,
        branch_info.node, True,
        branch_info.subject, parent_map
    )
    
    print(f"[Worker PID {pid}] Trial {trial_idx} fitness: {fitness:.4f}")
    
    return trial_idx, pid, fitness


def test_actual_workload():
    """Test with actual branch testing workload"""
    print("\n" + "="*80)
    print("TEST 2: Actual Workload Test")
    print("="*80)
    print(f"Testing with ex3.py\n")
    
    file_path = "./benchmark/ex3.py"
    num_trials = 12  # 12 trials on 4 workers = 3 per worker
    
    tasks = [(i, file_path) for i in range(num_trials)]
    
    print(f"Running {num_trials} trials on 4 workers...")
    print("Watch htop for multiple Python processes!\n")
    
    start = time.time()
    with Pool(processes=4) as pool:
        results = pool.map(run_single_trial_debug, tasks)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.2f}s")
    
    unique_pids = len(set(r[1] for r in results if r[1]))
    print(f"Unique worker PIDs: {unique_pids} (should be 4)")
    
    if unique_pids > 1:
        print("\n✅ Parallel execution WORKS!")
        return True
    else:
        print("\n❌ Parallel execution FAILED - only 1 process used!")
        return False


def test_with_chunksize():
    """Test with explicit chunksize to force distribution"""
    print("\n" + "="*80)
    print("TEST 3: Testing with Explicit Chunksize")
    print("="*80)
    
    file_path = "./benchmark/ex3.py"
    num_trials = 16
    
    tasks = [(i, file_path) for i in range(num_trials)]
    
    print(f"Running {num_trials} trials on 4 workers with chunksize=1")
    print("This forces Pool to distribute work across workers\n")
    
    start = time.time()
    with Pool(processes=4) as pool:
        results = pool.map(run_single_trial_debug, tasks, chunksize=1)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.2f}s")
    
    unique_pids = len(set(r[1] for r in results if r[1]))
    print(f"Unique worker PIDs: {unique_pids}")


if __name__ == "__main__":
    import multiprocessing
    
    # Try different start methods
    print("\n" + "="*80)
    print("🐛 DEBUGGING PARALLELIZATION")
    print("="*80)
    print(f"Python multiprocessing start method: {multiprocessing.get_start_method()}")
    print(f"Available methods: {multiprocessing.get_all_start_methods()}")
    
    # Force fork if available (faster than spawn)
    if 'fork' in multiprocessing.get_all_start_methods():
        print("\n⚠️  Trying 'fork' method (faster than 'spawn')")
        multiprocessing.set_start_method('fork', force=True)
    else:
        print("\n⚠️  Using 'spawn' method (fork not available)")
        multiprocessing.set_start_method('spawn', force=True)
    
    # Run tests
    test1_ok = test_basic_multiprocessing()
    
    if test1_ok:
        print("\n" + "─"*80)
        input("Press Enter to continue to actual workload test...")
        test2_ok = test_actual_workload()
        
        if test2_ok:
            print("\n✅ Parallelization is working correctly!")
        else:
            print("\n❌ Something wrong with actual workload distribution")
            print("\nTrying with explicit chunksize...")
            test_with_chunksize()
    else:
        print("\n❌ Basic multiprocessing is broken on this system!")
        print("Check:")
        print("  1. Python version (should be 3.7+)")
        print("  2. OS/kernel supports multiprocessing")
        print("  3. No restrictions on process creation")

