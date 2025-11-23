#!/usr/bin/env python3
"""
Heavy workload test that WILL show clear parallelization speedup.

This test uses:
- ALL branches (not just one)
- OR harder branches with more trials
- OR increased difficulty settings
"""

import time
import multiprocessing
from test_benchmark_gpu_parallel import test_branches_parallel

def test_all_branches():
    """Test ALL branches to get enough work"""
    print("\n" + "="*80)
    print("⚡ HEAVY WORKLOAD TEST - ALL BRANCHES")
    print("="*80)
    print("\nThis will test ALL branches in the file.")
    print("This should take 5-10 minutes and show clear speedup.")
    print("\n💡 Open htop in another terminal to see CPU usage!")
    print("="*80 + "\n")
    
    file_path = "./benchmark/count_divisor_2.py"
    
    # Test 1: Sequential (1 worker)
    print("🔄 Test 1: Sequential (1 worker) - ALL branches")
    print("─"*80)
    print("This will take several minutes...")
    
    start1 = time.time()
    results1, summaries1 = test_branches_parallel(
        file_path=file_path,
        max_trials_per_branch=20,
        num_workers=1,  # Sequential
        filter_lineno=None,  # ✅ ALL branches!
        success_threshold=0.0,
        max_iterations=100,
        basin_max_search=50000
    )
    time1 = time.time() - start1
    
    success1 = sum(r['success'] for r in results1)
    branches1 = len(summaries1)
    
    print(f"\n✅ Sequential completed:")
    print(f"   Time: {time1:.2f} seconds ({time1/60:.1f} minutes)")
    print(f"   Branches tested: {branches1}")
    print(f"   Total trials: {len(results1)}")
    print(f"   Successes: {success1}/{len(results1)}")
    
    # Test 2: Parallel (4 workers)
    print("\n\n🔄 Test 2: Parallel (4 workers) - ALL branches")
    print("─"*80)
    print("💡 NOW check htop - you should see 4 Python processes at 100%!")
    
    start2 = time.time()
    results2, summaries2 = test_branches_parallel(
        file_path=file_path,
        max_trials_per_branch=20,
        num_workers=4,  # Parallel
        filter_lineno=None,  # ✅ ALL branches!
        success_threshold=0.0,
        max_iterations=100,
        basin_max_search=50000
    )
    time2 = time.time() - start2
    
    success2 = sum(r['success'] for r in results2)
    branches2 = len(summaries2)
    
    print(f"\n✅ Parallel completed:")
    print(f"   Time: {time2:.2f} seconds ({time2/60:.1f} minutes)")
    print(f"   Branches tested: {branches2}")
    print(f"   Total trials: {len(results2)}")
    print(f"   Successes: {success2}/{len(results2)}")
    
    # Comparison
    speedup = time1 / time2
    efficiency = (speedup / 4) * 100
    
    print("\n\n" + "="*80)
    print("📊 RESULTS COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<25} {'Sequential':<20} {'Parallel (4x)':<20}")
    print("─"*80)
    print(f"{'Total time':<25} {f'{time1:.2f}s ({time1/60:.1f}m)':<20} {f'{time2:.2f}s ({time2/60:.1f}m)':<20}")
    print(f"{'Branches tested':<25} {branches1:<20} {branches2:<20}")
    print(f"{'Total trials':<25} {len(results1):<20} {len(results2):<20}")
    print(f"{'Successes':<25} {success1:<20} {success2:<20}")
    print("─"*80)
    print(f"\n{'🚀 SPEEDUP:':<25} {speedup:.2f}x")
    print(f"{'⚡ EFFICIENCY:':<25} {efficiency:.1f}% (ideal: 100%)")
    print(f"{'⏱️  TIME SAVED:':<25} {time1-time2:.2f}s ({(time1-time2)/60:.1f}m)")
    print("="*80)
    
    if speedup >= 3.0:
        print("\n   ✅ Excellent speedup! Parallelization is very effective.")
    elif speedup >= 2.0:
        print("\n   ✅ Good speedup! Parallelization is working well.")
    else:
        print("\n   ⚠️  Lower than expected. Check if all branches were tested.")


def test_single_hard_branch():
    """Test one branch but with MANY trials to ensure enough work"""
    print("\n" + "="*80)
    print("⚡ HEAVY WORKLOAD TEST - MANY TRIALS")
    print("="*80)
    print("\nThis will test ONE branch with MANY trials.")
    print("100 trials should provide enough work to see speedup.")
    print("\n💡 Open htop in another terminal!")
    print("="*80 + "\n")
    
    file_path = "./benchmark/count_divisor_2.py"
    num_trials = 100  # Much more trials
    
    # Test 1: Sequential
    print("🔄 Test 1: Sequential (1 worker) - 100 trials")
    print("─"*80)
    
    start1 = time.time()
    results1, _ = test_branches_parallel(
        file_path=file_path,
        max_trials_per_branch=num_trials,
        num_workers=1,
        filter_lineno={28},
        success_threshold=0.0,
        max_iterations=100,
        basin_max_search=50000
    )
    time1 = time.time() - start1
    
    print(f"\n✅ Sequential: {time1:.2f}s ({len(results1)} trials)")
    
    # Test 2: Parallel
    print("\n\n🔄 Test 2: Parallel (8 workers) - 100 trials")
    print("─"*80)
    print("💡 Check htop for 8 Python processes!")
    
    start2 = time.time()
    results2, _ = test_branches_parallel(
        file_path=file_path,
        max_trials_per_branch=num_trials,
        num_workers=8,  # More workers
        filter_lineno={28},
        success_threshold=0.0,
        max_iterations=100,
        basin_max_search=50000
    )
    time2 = time.time() - start2
    
    print(f"\n✅ Parallel: {time2:.2f}s ({len(results2)} trials)")
    
    # Results
    speedup = time1 / time2
    
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    print(f"Sequential (1 worker):  {time1:.2f}s")
    print(f"Parallel (8 workers):   {time2:.2f}s")
    print(f"Speedup:                {speedup:.2f}x")
    print(f"Expected:               ~6-7x (for 8 workers)")
    print("="*80)


def test_harder_benchmark():
    """Test with a more complex benchmark file"""
    print("\n" + "="*80)
    print("⚡ TESTING HARDER BENCHMARK")
    print("="*80)
    print("\nTesting with a more complex function that takes longer to solve.")
    print("="*80 + "\n")
    
    # Try different benchmark files
    benchmark_files = [
        "./benchmark/collatz_step.py",
        "./benchmark/prime_check.py",
        "./benchmark/HJJ/rugged_case.py",
    ]
    
    for file_path in benchmark_files:
        try:
            print(f"\n📁 Testing: {file_path}")
            print("─"*80)
            
            # Quick test with 4 workers
            start = time.time()
            results, summaries = test_branches_parallel(
                file_path=file_path,
                max_trials_per_branch=20,
                num_workers=4,
                filter_lineno=None,
                success_threshold=0.0,
                max_iterations=100,
                basin_max_search=50000
            )
            elapsed = time.time() - start
            
            avg_time_per_trial = elapsed / len(results) if results else 0
            
            print(f"✅ Completed in {elapsed:.2f}s")
            print(f"   Trials: {len(results)}")
            print(f"   Avg time per trial: {avg_time_per_trial:.2f}s")
            
            if avg_time_per_trial > 0.5:
                print(f"   ✅ Good candidate! Trials are slow enough for parallelization.")
                return file_path
            else:
                print(f"   ⚠️  Too fast (avg {avg_time_per_trial:.2f}s per trial)")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return None


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            test_all_branches()
        elif sys.argv[1] == "many":
            test_single_hard_branch()
        elif sys.argv[1] == "harder":
            test_harder_benchmark()
        else:
            print(f"Usage: {sys.argv[0]} [all|many|harder]")
    else:
        print("\n" + "="*80)
        print("⚡ HEAVY WORKLOAD TESTS")
        print("="*80)
        print("\nYour previous test was too light (0.04s per trial)!")
        print("Choose a heavier test:\n")
        print("  1. Test ALL branches (recommended)")
        print("     python test_heavy_workload.py all")
        print("     Time: ~5-10 minutes, will show clear speedup\n")
        print("  2. Test ONE branch with 100 trials")
        print("     python test_heavy_workload.py many")
        print("     Time: ~3-5 minutes\n")
        print("  3. Test harder benchmark files")
        print("     python test_heavy_workload.py harder")
        print("     Time: ~2-5 minutes\n")
        print("="*80)
        print("\n💡 Recommendation: Run option 1 (all branches)")
        print("   This will give you the clearest speedup demonstration!")
        print("\n" + "="*80 + "\n")

