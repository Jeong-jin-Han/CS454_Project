#!/usr/bin/env python3
"""
Quick demo script to test both parallel versions.
"""

import sys
import time
from test_benchmark_gpu_parallel import test_branches_parallel
from test_benchmark_gpu_parallel_v2 import test_branches_parallel_v2

def demo_version_1():
    """Test Version 1: Full parallelization"""
    print("\n" + "🔷"*40)
    print("TESTING VERSION 1: Full Parallelization")
    print("🔷"*40 + "\n")
    
    start = time.time()
    results, summaries = test_branches_parallel(
        file_path="./benchmark/count_divisor_2.py",
        max_trials_per_branch=5,  # Small number for demo
        success_threshold=0.0,
        num_workers=4,  # Limit workers for demo
        filter_lineno={28}
    )
    elapsed = time.time() - start
    
    print(f"\n✅ Version 1 completed in {elapsed:.2f} seconds")
    print(f"   Total results: {len(results)}")
    return results, summaries, elapsed

def demo_version_2():
    """Test Version 2: Hybrid with metadata sharing"""
    print("\n" + "🔶"*40)
    print("TESTING VERSION 2: Hybrid (Metadata Sharing)")
    print("🔶"*40 + "\n")
    
    start = time.time()
    results, summaries = test_branches_parallel_v2(
        file_path="./benchmark/count_divisor_2.py",
        max_trials_per_branch=5,  # Small number for demo
        success_threshold=0.0,
        num_workers=4,  # Limit workers for demo
        filter_lineno={28}
    )
    elapsed = time.time() - start
    
    print(f"\n✅ Version 2 completed in {elapsed:.2f} seconds")
    print(f"   Total results: {len(results)}")
    return results, summaries, elapsed

def compare_versions():
    """Run both versions and compare"""
    print("\n" + "="*80)
    print("🏁 PARALLEL VERSION COMPARISON DEMO")
    print("="*80)
    
    # Run V1
    try:
        r1, s1, t1 = demo_version_1()
        v1_ok = True
    except Exception as e:
        print(f"\n❌ Version 1 failed: {e}")
        import traceback
        traceback.print_exc()
        v1_ok = False
        t1 = None
    
    # Run V2
    try:
        r2, s2, t2 = demo_version_2()
        v2_ok = True
    except Exception as e:
        print(f"\n❌ Version 2 failed: {e}")
        import traceback
        traceback.print_exc()
        v2_ok = False
        t2 = None
    
    # Compare
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    if v1_ok and v2_ok:
        print(f"\n✅ Both versions completed successfully!")
        print(f"\n⏱️  Timing:")
        print(f"   Version 1 (Full):   {t1:.2f}s")
        print(f"   Version 2 (Hybrid): {t2:.2f}s")
        print(f"   Speedup ratio:      {t2/t1:.2f}x")
        
        print(f"\n📈 Success rates:")
        for key in s1.keys():
            rate1 = s1[key]['successes'] / s1[key]['total_trials'] * 100
            rate2 = s2[key]['successes'] / s2[key]['total_trials'] * 100
            print(f"   {key}:")
            print(f"      V1: {s1[key]['successes']}/{s1[key]['total_trials']} ({rate1:.1f}%)")
            print(f"      V2: {s2[key]['successes']}/{s2[key]['total_trials']} ({rate2:.1f}%)")
    elif v1_ok:
        print(f"\n⚠️  Only Version 1 completed")
        print(f"   Time: {t1:.2f}s")
    elif v2_ok:
        print(f"\n⚠️  Only Version 2 completed")
        print(f"   Time: {t2:.2f}s")
    else:
        print(f"\n❌ Both versions failed")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    import multiprocessing
    
    # Required for multiprocessing on some systems
    multiprocessing.set_start_method('spawn', force=True)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "v1":
            demo_version_1()
        elif sys.argv[1] == "v2":
            demo_version_2()
        else:
            print(f"Usage: {sys.argv[0]} [v1|v2|compare]")
    else:
        compare_versions()

