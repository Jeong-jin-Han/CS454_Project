#!/usr/bin/env python3
"""
Realistic workload test to demonstrate parallelization speedup.

This test uses:
- Multiple branches or more trials
- Larger workload that benefits from parallelization
"""

import time
import sys
from test_benchmark_gpu_parallel import test_branches_parallel
from test_benchmark_gpu_parallel_v2 import test_branches_parallel_v2

def test_with_larger_workload():
    """Test with workload large enough to see speedup"""
    
    print("\n" + "="*80)
    print("🧪 REALISTIC WORKLOAD TEST")
    print("="*80)
    print("Testing with larger workload to demonstrate speedup...")
    print("This will take a few minutes - watch your CPU usage in htop!")
    print("="*80 + "\n")
    
    file_path = "./benchmark/count_divisor_2.py"
    
    # Configuration for realistic test
    config = {
        'max_trials_per_branch': 20,  # More trials
        'success_threshold': 0.0,
        'max_iterations': 50,
        'basin_max_search': 10000,
        'filter_lineno': None  # Test ALL branches, not just one
    }
    
    print("📋 Configuration:")
    print(f"   Trials per branch: {config['max_trials_per_branch']}")
    print(f"   Testing: ALL branches")
    print(f"   Workers: Will test with 1, 2, 4, and all cores\n")
    
    # Test with different worker counts to show scaling
    results_summary = []
    
    for num_workers in [1, 2, 4, None]:
        worker_label = f"{num_workers} workers" if num_workers else "all cores"
        
        print("\n" + "─"*80)
        print(f"🔄 Testing with {worker_label}")
        print("─"*80)
        
        start = time.time()
        results, summaries = test_branches_parallel(
            file_path=file_path,
            num_workers=num_workers,
            **config
        )
        elapsed = time.time() - start
        
        total_successes = sum(r['success'] for r in results)
        
        print(f"\n✅ Completed with {worker_label}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Total trials: {len(results)}")
        print(f"   Successes: {total_successes}/{len(results)}")
        
        results_summary.append({
            'workers': num_workers or 'all',
            'time': elapsed,
            'trials': len(results),
            'successes': total_successes
        })
    
    # Print comparison
    print("\n" + "="*80)
    print("📊 SCALING COMPARISON")
    print("="*80)
    print(f"{'Workers':<12} {'Time':<12} {'Speedup':<12} {'Efficiency'}")
    print("-"*80)
    
    baseline_time = results_summary[0]['time']
    
    for r in results_summary:
        speedup = baseline_time / r['time']
        workers = r['workers'] if isinstance(r['workers'], int) else 8  # Estimate
        efficiency = speedup / workers * 100 if workers > 0 else 0
        
        worker_str = str(r['workers'])
        time_str = f"{r['time']:.2f}s"
        speedup_str = f"{speedup:.2f}x"
        efficiency_str = f"{efficiency:.1f}%"
        
        print(f"{worker_str:<12} {time_str:<12} {speedup_str:<12} {efficiency_str}")
    
    print("="*80)
    print("\n💡 Tips:")
    print("   - Speedup should increase with more workers")
    print("   - Efficiency shows how well cores are utilized")
    print("   - Monitor CPU usage with: htop")
    print("   - 0% GPU usage is CORRECT (this is CPU-only work)")
    print("="*80 + "\n")
    
    return results_summary


def test_single_branch_many_trials():
    """Test single branch with many trials to reduce overhead"""
    
    print("\n" + "="*80)
    print("🎯 SINGLE BRANCH, MANY TRIALS TEST")
    print("="*80)
    print("Testing one branch with many trials...")
    print("="*80 + "\n")
    
    file_path = "./benchmark/count_divisor_2.py"
    
    # Test with 1 worker (sequential)
    print("🔄 Sequential (1 worker)...")
    start = time.time()
    results_seq, _ = test_branches_parallel(
        file_path=file_path,
        max_trials_per_branch=40,  # Many trials
        num_workers=1,
        filter_lineno={28}  # Just one branch
    )
    time_seq = time.time() - start
    print(f"✅ Sequential: {time_seq:.2f}s")
    
    # Test with 4 workers
    print("\n🔄 Parallel (4 workers)...")
    start = time.time()
    results_par, _ = test_branches_parallel(
        file_path=file_path,
        max_trials_per_branch=40,  # Same number of trials
        num_workers=4,
        filter_lineno={28}  # Same branch
    )
    time_par = time.time() - start
    print(f"✅ Parallel: {time_par:.2f}s")
    
    # Compare
    speedup = time_seq / time_par
    
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    print(f"Sequential time:  {time_seq:.2f}s")
    print(f"Parallel time:    {time_par:.2f}s")
    print(f"Speedup:          {speedup:.2f}x")
    print(f"Expected speedup: ~3-4x (for 4 workers)")
    print("="*80 + "\n")
    
    if speedup < 1.5:
        print("⚠️  Low speedup suggests:")
        print("   - Branch is too simple/fast")
        print("   - Multiprocessing overhead dominates")
        print("   - Try testing more complex branches")
    elif speedup > 2.5:
        print("✅ Good speedup! Parallelization is working well.")
    else:
        print("📊 Moderate speedup - expected for this workload.")
    
    return speedup


def quick_test():
    """Quick test to verify things work"""
    print("\n" + "="*80)
    print("⚡ QUICK VERIFICATION TEST")
    print("="*80 + "\n")
    
    file_path = "./benchmark/count_divisor_2.py"
    
    print("Testing with moderate workload (this takes ~30 seconds)...")
    
    start = time.time()
    results, summaries = test_branches_parallel(
        file_path=file_path,
        max_trials_per_branch=10,
        num_workers=None,  # Use all cores
        filter_lineno={28}
    )
    elapsed = time.time() - start
    
    print(f"\n✅ Test completed in {elapsed:.2f}s")
    print(f"   Trials run: {len(results)}")
    print(f"   Successes: {sum(r['success'] for r in results)}/{len(results)}")
    
    print("\n💡 While this runs, check CPU usage:")
    print("   $ htop")
    print("   You should see multiple Python processes at ~100% CPU")
    print("   GPU usage will stay at 0% (this is correct)")
    print("="*80 + "\n")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_test()
        elif sys.argv[1] == "single":
            test_single_branch_many_trials()
        elif sys.argv[1] == "full":
            test_with_larger_workload()
        else:
            print(f"Usage: {sys.argv[0]} [quick|single|full]")
            print("  quick  - Fast verification test (~30s)")
            print("  single - Single branch, many trials (~2min)")
            print("  full   - All branches, scaling test (~5-10min)")
    else:
        print("\n" + "="*80)
        print("🎯 PARALLELIZATION TEST SUITE")
        print("="*80)
        print("\nChoose a test:\n")
        print("  1. Quick test (recommended first)")
        print("     python test_realistic_workload.py quick")
        print("")
        print("  2. Single branch with many trials")
        print("     python test_realistic_workload.py single")
        print("")
        print("  3. Full scaling test (all branches)")
        print("     python test_realistic_workload.py full")
        print("\n" + "="*80 + "\n")

