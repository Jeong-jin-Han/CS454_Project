#!/usr/bin/env python3
"""
Simple demonstration of parallelization speedup.
This will clearly show the difference between 1 worker and multiple workers.
"""

import time
import multiprocessing
from test_benchmark_gpu_parallel import test_branches_parallel

def main():
    print("\n" + "="*80)
    print("⚡ SIMPLE SPEEDUP DEMONSTRATION")
    print("="*80)
    print("\nThis test will:")
    print("  1. Run with 1 worker (sequential)")
    print("  2. Run with 4 workers (parallel)")
    print("  3. Show you the speedup")
    print("\n💡 While running, open another terminal and run: htop")
    print("   You'll see multiple Python processes at 100% CPU!")
    print("\n⏱️  This will take about 2-3 minutes total...")
    print("="*80 + "\n")
    
    file_path = "./benchmark/count_divisor_2.py"
    trials = 30  # Enough to see speedup, not too slow
    
    # Test 1: Sequential (1 worker)
    print("🔄 Test 1: Sequential (1 worker)")
    print("─"*80)
    start1 = time.time()
    results1, summaries1 = test_branches_parallel(
        file_path=file_path,
        max_trials_per_branch=trials,
        num_workers=1,  # Sequential
        filter_lineno={28},  # One branch
        success_threshold=0.0,
        max_iterations=50,
        basin_max_search=10000
    )
    time1 = time.time() - start1
    
    success1 = sum(r['success'] for r in results1)
    
    print(f"\n✅ Sequential completed:")
    print(f"   Time: {time1:.2f} seconds")
    print(f"   Trials: {len(results1)}")
    print(f"   Successes: {success1}/{len(results1)}")
    print(f"   Avg time per trial: {time1/len(results1):.2f}s")
    
    # Test 2: Parallel (4 workers)
    print("\n\n🔄 Test 2: Parallel (4 workers)")
    print("─"*80)
    print("💡 NOW check htop - you should see 4 Python processes!")
    
    start2 = time.time()
    results2, summaries2 = test_branches_parallel(
        file_path=file_path,
        max_trials_per_branch=trials,
        num_workers=4,  # Parallel
        filter_lineno={28},  # Same branch
        success_threshold=0.0,
        max_iterations=50,
        basin_max_search=10000
    )
    time2 = time.time() - start2
    
    success2 = sum(r['success'] for r in results2)
    
    print(f"\n✅ Parallel completed:")
    print(f"   Time: {time2:.2f} seconds")
    print(f"   Trials: {len(results2)}")
    print(f"   Successes: {success2}/{len(results2)}")
    print(f"   Avg time per trial: {time2/len(results2):.2f}s")
    
    # Comparison
    speedup = time1 / time2
    efficiency = (speedup / 4) * 100  # 4 workers
    
    print("\n\n" + "="*80)
    print("📊 RESULTS COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<25} {'Sequential':<20} {'Parallel (4x)':<20}")
    print("─"*80)
    print(f"{'Total time':<25} {f'{time1:.2f}s':<20} {f'{time2:.2f}s':<20}")
    print(f"{'Trials completed':<25} {len(results1):<20} {len(results2):<20}")
    print(f"{'Successes':<25} {success1:<20} {success2:<20}")
    print(f"{'Avg time per trial':<25} {f'{time1/len(results1):.2f}s':<20} {f'{time2/len(results2):.2f}s':<20}")
    print("─"*80)
    print(f"\n{'🚀 SPEEDUP:':<25} {speedup:.2f}x")
    print(f"{'⚡ EFFICIENCY:':<25} {efficiency:.1f}% (ideal: 100%)")
    print("="*80)
    
    # Interpretation
    print("\n📈 INTERPRETATION:")
    if speedup >= 3.0:
        print("   ✅ Excellent speedup! Parallelization is very effective.")
        print("   ✅ Your CPU cores are being utilized well.")
    elif speedup >= 2.0:
        print("   ✅ Good speedup! Parallelization is working.")
        print("   💡 Some overhead, but overall beneficial.")
    elif speedup >= 1.3:
        print("   ⚠️  Moderate speedup. Some benefit from parallelization.")
        print("   💡 Overhead is significant but still faster.")
    else:
        print("   ⚠️  Low speedup. Workload might be too small.")
        print("   💡 Try increasing trials or testing more branches.")
    
    print("\n💡 ABOUT GPU USAGE:")
    print("   • GPU usage at 0% is CORRECT and EXPECTED")
    print("   • This workload uses CPU multiprocessing, not GPU")
    print("   • Python AST execution cannot run on GPU")
    print("   • Monitor CPU usage (htop) instead of GPU")
    
    print("\n🎯 KEY TAKEAWAY:")
    if speedup >= 2.0:
        print(f"   Parallelization gave you {speedup:.1f}x speedup!")
        print(f"   You're saving {time1-time2:.1f} seconds ({(1-time2/time1)*100:.0f}% time reduction)")
    else:
        print(f"   For this workload size, speedup was {speedup:.1f}x")
        print(f"   Test with more trials/branches for better speedup")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()

