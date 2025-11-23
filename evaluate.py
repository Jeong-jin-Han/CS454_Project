import os
import sys
import csv
import time
import glob
import importlib.util
import itertools
import numpy as np
from scipy.special import expit as sigmoid, logit
from typing import List, Tuple, Optional, Dict

# ==============================================================================
# 1. 핵심 알고리즘 정의 (제공해주신 코드 포함)
# ==============================================================================

class SigmoidWarping:
    def __init__(self, node_index, length, steepness=5.0):
        self.node_start = int(node_index)
        self.length = int(length)
        self.node_end = self.node_start + self.length
        self.steepness = float(steepness)
        self.shift = self.length - 1 

    def forward(self, node):
        node = np.atleast_1d(node).astype(float)
        position = np.zeros(len(node), dtype=float)
        for i, n in enumerate(node):
            if n < self.node_start: position[i] = float(n)
            elif n > self.node_end: position[i] = float(n) - self.shift
            else:
                if n == self.node_start: position[i] = float(self.node_start)
                elif n == self.node_end: position[i] = float(self.node_start + 1)
                else:
                    t = (n - self.node_start) / self.length
                    x = self.steepness * (t - 0.5)
                    position[i] = self.node_start + sigmoid(x)
        return position[0] if len(node) == 1 else position

    def inverse(self, position):
        position = np.atleast_1d(position).astype(float)
        node = np.zeros(len(position), dtype=int)
        for i, pos in enumerate(position):
            if pos < self.node_start: node[i] = int(np.floor(pos))
            elif pos >= self.node_start + 1.0: node[i] = int(np.ceil(pos + self.shift))
            else:
                s = np.clip(pos - self.node_start, 0.0, 1.0)
                if s <= 0.01: node[i] = self.node_start
                elif s >= 0.99: node[i] = self.node_end
                else:
                    x = logit(s)
                    t = x / self.steepness + 0.5
                    node_f = self.node_start + t * self.length
                    node[i] = int(round(node_f))
        return node[0] if len(node) == 1 else node

class MetadataCompressionOriginalSpace:
    def __init__(self, compressions_x_space=None, steepness=5.0):
        self.metadata_x = sorted(compressions_x_space or [], key=lambda x: x[0])
        self.steepness = float(steepness)
        self.warpings = []
        if self.metadata_x: self._build_warpings()

    def _build_warpings(self):
        cumulative_shift = 0
        for i, (x_start, x_length) in enumerate(self.metadata_x):
            z_start = x_start - cumulative_shift
            z_length = x_length
            warping = SigmoidWarping(z_start, z_length, self.steepness)
            self.warpings.append(warping)
            cumulative_shift += (z_length - 1)

    def forward(self, node):
        position = node
        for warping in self.warpings: position = warping.forward(position)
        return position

    def inverse(self, position):
        node = position
        for warping in reversed(self.warpings): node = warping.inverse(node)
        return node

def detect_compression_basin(fitness_func, local_min_x, max_search=100):
    """
    MODIFIED: Removes all internal prints to prevent console buffering issues.
    Only returns (start, length) or None.
    """
    local_min_fitness = fitness_func(local_min_x)
    
    # LEFT search
    left_boundary = local_min_x
    farthest_left_equal = local_min_x
    found_left_equal = False
    found_left_exit = False
    for i in range(1, max_search + 1):
        current_x = local_min_x - i
        current_fitness = fitness_func(current_x)
        if abs(current_fitness - local_min_fitness) < 1e-9:
            farthest_left_equal = current_x
            found_left_equal = True
        elif current_fitness > local_min_fitness: pass
        else:
            left_boundary = current_x + 1
            found_left_exit = True
            break
    else: left_boundary = local_min_x - max_search
    if not found_left_exit and found_left_equal: left_boundary = farthest_left_equal

    # RIGHT search
    right_boundary = local_min_x
    farthest_right_equal = local_min_x
    found_right_equal = False
    found_right_exit = False
    for i in range(1, max_search + 1):
        current_x = local_min_x + i
        current_fitness = fitness_func(current_x)
        if abs(current_fitness - local_min_fitness) < 1e-9:
            farthest_right_equal = current_x
            found_right_equal = True
        elif current_fitness > local_min_fitness: pass
        else:
            right_boundary = current_x - 1
            found_right_exit = True
            break
    else: right_boundary = local_min_x + max_search
    if not found_right_exit and found_right_equal: right_boundary = farthest_right_equal

    basin_length = right_boundary - left_boundary + 1
    if basin_length < 2: return None
    if not (found_left_equal or found_right_equal or found_left_exit or found_right_exit): return None
    return (left_boundary, basin_length)

def merge_overlapping_compressions(compressions):
    if not compressions: return []
    sorted_comps = sorted(compressions, key=lambda x: x[0])
    merged = []
    for start, length in sorted_comps:
        end = start + length - 1
        if not merged:
            merged.append((start, length))
            continue
        last_start, last_length = merged[-1]
        last_end = last_start + last_length - 1
        if start <= last_end + 1:
            new_start = min(start, last_start)
            new_end = max(end, last_end)
            new_length = new_end - new_start + 1
            merged[-1] = (new_start, new_length)
        else:
            merged.append((start, length))
    return merged

class CompressionManagerND:
    def __init__(self, dim, steepness=5.0):
        self.dim = dim
        self.steepness = float(steepness)
        self.dim_compressions = [{} for _ in range(dim)]
        self.dim_systems = [{} for _ in range(dim)]
    
    def update_dimension(self, vary_dim, fixed_coords, basin):
        if basin is None: return
        comps = self.dim_compressions[vary_dim].get(fixed_coords, [])
        comps.append(basin)
        comps = merge_overlapping_compressions(comps)
        self.dim_compressions[vary_dim][fixed_coords] = comps
        self.dim_systems[vary_dim][fixed_coords] = MetadataCompressionOriginalSpace(comps, self.steepness)
    
    def get_system(self, vary_dim, fixed_coords):
        return self.dim_systems[vary_dim].get(fixed_coords, None)

def detect_basin_along_dimension(fitness_func_nd, point, vary_dim, max_search=100):
    """
    MODIFIED: Prints result locally before returning.
    """
    def f1d(val):
        new_point = list(point)
        new_point[vary_dim] = int(val)
        return fitness_func_nd(tuple(new_point))
    
    basin = detect_compression_basin(f1d, local_min_x=int(point[vary_dim]), max_search=max_search)
    
    # Log the result here to ensure a full line print
    min_val = int(point[vary_dim])
    min_f = fitness_func_nd(point)
    if basin:
        start, length = basin
        print(f"    [Dim {vary_dim}] (1D Detect @ x={min_val}, f={min_f:.2f}) -> Basin: [{start}, {start+length-1}] (len={length})")
    else:
        print(f"    [Dim {vary_dim}] (1D Detect @ x={min_val}, f={min_f:.2f}) -> No compressible basin found.")
        
    return basin

# ------------------------------------------------------------------------------
# Hill Climbing Algorithms (User Provided Logic)
# ------------------------------------------------------------------------------
def hill_climb_with_compression_nd(fitness_func_nd, start_point, dim, max_iterations=10, basin_max_search=100, global_min_threshold=1e-6):
    traj = []
    cm = CompressionManagerND(dim, steepness=5.0)
    point = tuple(int(x) for x in start_point)
    f = fitness_func_nd(point)
    traj.append((point, f, False))
    
    print(f"\n🚀 {dim}D COMPRESSION climb start at {point}, f={f:.4f}\n")

    for it in range(max_iterations):
        print(f"================================================================================")
        print(f"🔄 Iteration {it+1}/{max_iterations}")
        print(f"================================================================================")
        
        while True:
            candidates = []
            # 1. O(D) Axis-aligned Neighbors
            for d in range(dim):
                fixed_coords = tuple(point[i] for i in range(dim) if i != d)
                comp_sys = cm.get_system(d, fixed_coords)
                if comp_sys is not None:
                    z = comp_sys.forward(point[d])
                    nm, np_ = comp_sys.inverse(z - 1), comp_sys.inverse(z + 1)
                else:
                    nm, np_ = point[d] - 1, point[d] + 1
                
                pm = list(point); pm[d] = nm
                pp = list(point); pp[d] = np_
                candidates.append((tuple(pm), fitness_func_nd(tuple(pm))))
                candidates.append((tuple(pp), fitness_func_nd(tuple(pp))))
            
            # 2. O(D^2) Diagonal Neighbors
            # Note: This logic seems slightly off (offset2 should be +/- 1), but we keep the user's O(D^2) logic for now.
            # The original user's intent was to check neighbors in *all* directions.
            for d1, d2 in itertools.combinations(range(dim), 2):
                for o1 in [1, -1]:
                    for o2 in [1, -1]: # Correcting based on standard steepest ascent/user intent for diagonal check
                        np_ = list(point); np_[d1] += o1; np_[d2] += o2
                        candidates.append((tuple(np_), fitness_func_nd(tuple(np_))))
            
            best_point, best_f = point, f
            for cp, cf in candidates:
                if cf < best_f - 1e-9: best_point, best_f = cp, cf
            
            if best_f < f - 1e-9:
                point, f = best_point, best_f
                used_comp = any(cm.get_system(d, tuple(point[i] for i in range(dim) if i != d)) is not None for d in range(dim))
                traj.append((point, f, used_comp))
            else:
                print(f"  📍 Climbed to local optimum at {point}, f={f:.6g}")
                break
        
        if abs(f) < global_min_threshold:
            print("\n🎉 SUCCESS: reached near-global minimum")
            break
        
        # Basin Detection
        print(f"\n⚠️ STUCK at local minimum {point}, f={f:.6g}")
        print(f"  🔍 Detecting basins along all {dim} dimensions...")
        basins = {}
        for d in range(dim):
            basin = detect_basin_along_dimension(fitness_func_nd, point, d, max_search=basin_max_search)
            if basin:
                fixed_coords = tuple(point[i] for i in range(dim) if i != d)
                cm.update_dimension(d, fixed_coords, basin)
                basins[d] = basin
        
        # Restart Logic
        restart_candidates = []
        for d, (b_start, b_len) in basins.items():
            b_end = b_start + b_len - 1
            rp1 = list(point); rp1[d] = b_start - 1
            rp2 = list(point); rp2[d] = b_end + 1
            restart_candidates.append((tuple(rp1), fitness_func_nd(tuple(rp1))))
            restart_candidates.append((tuple(rp2), fitness_func_nd(tuple(rp2))))
        
        if restart_candidates:
            best_r_p, best_r_f = min(restart_candidates, key=lambda t: t[1])
            if best_r_f < f - 1e-9:
                print(f"  ➡️ Restarting from {best_r_p}, f={best_r_f:.4f}")
                point, f = best_r_p, best_r_f
                traj.append((point, f, True))
            else:
                print("\n  ⚠️ Restart candidates didn't improve fitness. Stopping.")
                break
        else:
            break
            
    print(f"\n🏁 FINAL {dim}D COMPRESSION RESULTS 🏁")
    print(f"  Final position: {point}, Final fitness: {f:.6g}, Total steps: {len(traj)}")
    return traj

def hill_climb_simple_nd(fitness_func_nd, start_point, dim, max_steps=2000):
    point = tuple(int(x) for x in start_point)
    f = fitness_func_nd(point)
    traj = [(point, f)]
    print(f"\n🚀 {dim}D BASELINE climb start at {point}, f={f:.4f}\n")

    for _ in range(max_steps):
        candidates = []
        # 1. O(D)
        for d in range(dim):
            nm = list(point); nm[d] -= 1
            np_ = list(point); np_[d] += 1
            candidates.append((tuple(nm), fitness_func_nd(tuple(nm))))
            candidates.append((tuple(np_), fitness_func_nd(tuple(np_))))
        # 2. O(D^2) Diagonal
        for d1, d2 in itertools.combinations(range(dim), 2):
            for o1 in [1, -1]:
                for o2 in [1, -1]:
                    np_ = list(point); np_[d1] += o1; np_[d2] += o2
                    candidates.append((tuple(np_), fitness_func_nd(tuple(np_))))

        best_point, best_f = point, f
        for cp, cf in candidates:
            if cf < best_f - 1e-9: best_point, best_f = cp, cf
        
        if best_f < f - 1e-9:
            point, f = best_point, best_f
            traj.append((point, f))
        else:
            break
            
    print(f"🏁 FINAL {dim}D BASELINE RESULTS 🏁")
    print(f"  Final position: {point}, Final fitness: {f:.6g}, Total steps: {len(traj)}")
    return traj


# ==============================================================================
# 3. Test Runner & CSV Reporting
# ==============================================================================

def load_module_from_path(path):
    """Dynamically loads a python module from a file path."""
    # File name (without extension) used as module name
    module_name = os.path.basename(path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # Add module to sys.modules so it can be accessed
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_all_tests(test_folder="test_cases", output_csv="test_results.csv"):
    """
    1. Finds all .py files in test_folder
    2. Loads configurations (func, dim, start, optimal)
    3. Runs both algos
    4. Logs to console & CSV
    """
    
    # 1. Setup CSV
    csv_headers = [
        "Test Name", "Dim", "Algorithm", "Success", "Final Fitness", "Steps", "Time(s)", "Start Point"
    ]
    
    # Create file and write headers if not exists
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    # 2. Find test files
    test_files = glob.glob(os.path.join(test_folder, "*.py"))
    test_files.sort()
    
    print(f"📂 Found {len(test_files)} test files in '{test_folder}'")

    for file_path in test_files:
        file_name = os.path.basename(file_path)
        
        try:
            # 3. Dynamic Load
            mod = load_module_from_path(file_path)
            
            # Required attributes in test file
            fitness_func = mod.FITNESS_FUNC
            config = mod.TEST_CONFIG
            
            dim = config['dim']
            start_point = config['start_point']
            optimal_val = config['optimal_val']
            threshold = config.get('threshold', 1e-3)
            max_iter = config.get('max_iterations', 10)
            basin_search = config.get('basin_max_search', 30)
            max_steps_base = config.get('max_steps_baseline', 2000)
            
            print("\n" + "="*80)
            print(f"🧪 TEST FILE: {file_name} | Function: {fitness_func.__name__} (Dim={dim})")
            print(f"   Target: {optimal_val} (Threshold: {threshold})")
            print("="*80)

            results_to_save = []

            # --- A. Run Compression Algo ---
            print("\n🔹 RUNNING WITH COMPRESSION...")
            t0 = time.time()
            # The result of hill_climb_with_compression_nd is (traj, cm)
            traj_comp = hill_climb_with_compression_nd(
                fitness_func, start_point, dim, max_iter, basin_search, threshold
            )
            t_comp = time.time() - t0
            
            # Check if execution finished successfully (not an empty list from a crash)
            if traj_comp and traj_comp[-1]:
                f_comp = traj_comp[-1][1]
                succ_comp = abs(f_comp - optimal_val) < threshold
                steps_comp = len(traj_comp)
                final_point_comp = traj_comp[-1][0]
            else:
                 f_comp, succ_comp, steps_comp, final_point_comp = np.inf, False, 0, start_point # Handle potential non-improvement stop
                
            results_to_save.append({
                "Test Name": file_name, "Dim": dim, "Algorithm": "Compression",
                "Success": succ_comp, "Final Fitness": f_comp, "Steps": steps_comp,
                "Time(s)": round(t_comp, 4), "Start Point": str(start_point)
            })

            # --- B. Run Baseline Algo ---
            print("\n🔹 RUNNING BASELINE...")
            t0 = time.time()
            traj_base = hill_climb_simple_nd(
                fitness_func, start_point, dim, max_steps_base
            )
            t_base = time.time() - t0
            
            if traj_base and traj_base[-1]:
                f_base = traj_base[-1][1]
                succ_base = abs(f_base - optimal_val) < threshold
                steps_base = len(traj_base)
            else:
                f_base, succ_base, steps_base = np.inf, False, 0
            
            results_to_save.append({
                "Test Name": file_name, "Dim": dim, "Algorithm": "Baseline",
                "Success": succ_base, "Final Fitness": f_base, "Steps": steps_base,
                "Time(s)": round(t_base, 4), "Start Point": str(start_point)
            })

            # --- C. Print Summary Table ---
            print("\n" + "="*80)
            print("📊 COMPARISON SUMMARY")
            print("="*80)
            print(f"{'Metric':<25} {'Compression':<20} {'Baseline':<20}")
            print("-" * 70)
            print(f"{'Success':<25} {str(succ_comp):<20} {str(succ_base):<20}")
            print(f"{'Final Fitness':<25} {f_comp:<20.6g} {f_base:<20.6g}")
            print(f"{'Steps':<25} {steps_comp:<20} {steps_base:<20}")
            print(f"{'Time(s)':<25} {t_comp:<20.4f} {t_base:<20.4f}")
            print("="*80)

            # --- D. Save to CSV ---
            with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                for row in results_to_save:
                    writer.writerow(row)
                    
        except Exception as e:
            # Catching the case where traj_comp might be ([], cm) if it immediately fails/stops
            if file_name.endswith('.py'):
                print(f"❌ Error processing {file_name}: {e}")
            
    print(f"\n✅ All tests completed. Results saved to {output_csv}")

if __name__ == "__main__":
    if not os.path.exists("test_cases"):
        os.makedirs("test_cases")
        
    # The existing files must be saved before running the runner
    # We assume 'test_cases/test1.py' is correctly created and the runner is saved as 'evaluate.py'
    
    # We run the main function from the currently executed script
    run_all_tests()