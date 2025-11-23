import numpy as np
from scipy.special import expit as sigmoid, logit
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
import itertools

# print("✅ Imports loaded!")
# print("📍 Focus: Hill climbing with adaptive compression")

# ===============================
# Warping
# ===============================
class SigmoidWarping:
    """Compress integer region to length-1 interval."""
    def __init__(self, node_index, length, steepness=5.0):
        self.node_start = int(node_index)
        self.length = int(length)
        assert self.length >= 2, "Compression length must be >= 2."
        self.node_end = self.node_start + self.length
        self.steepness = float(steepness)
        self.shift = self.length - 1  # how much right side is pulled left

    def forward(self, node):
        """X → Z"""
        node = np.atleast_1d(node).astype(float)
        position = np.zeros(len(node), dtype=float)
        for i, n in enumerate(node):
            if n < self.node_start:
                position[i] = float(n)
            elif n > self.node_end:
                position[i] = float(n) - self.shift
            else:
                if n == self.node_start:
                    position[i] = float(self.node_start)
                elif n == self.node_end:
                    position[i] = float(self.node_start + 1)
                else:
                    t = (n - self.node_start) / self.length
                    x = self.steepness * (t - 0.5)
                    s = sigmoid(x)
                    position[i] = self.node_start + s
        return position[0] if len(node) == 1 else position

    def inverse(self, position):
        """Z → X"""
        position = np.atleast_1d(position).astype(float)
        node = np.zeros(len(position), dtype=int)
        for i, pos in enumerate(position):
            if pos < self.node_start:
                node[i] = int(np.floor(pos))
            elif pos >= self.node_start + 1.0:
                node[i] = int(np.ceil(pos + self.shift))
            else:
                s = np.clip(pos - self.node_start, 0.0, 1.0)
                if s <= 0.01:
                    node[i] = int(self.node_start)
                elif s >= 0.99:
                    node[i] = int(self.node_end)
                else:
                    x = logit(s)
                    t = x / self.steepness + 0.5
                    node_f = self.node_start + t * self.length
                    node[i] = int(round(node_f))

        # Always return plain Python ints to avoid NumPy scalar types leaking
        # into the SBST fitness calculator (which compares `is True` / `is False`
        # and is sensitive to NumPy's bool_/int64 types).
        if len(position) == 1:
            return int(node[0])
        else:
            # For multi-D, keep the array shape but convert elements to Python int
            return np.vectorize(int)(node)

class MetadataCompressionOriginalSpace:
    """Metadata ALWAYS in ORIGINAL X-space."""
    def __init__(self, compressions_x_space=None, steepness=5.0):
        self.metadata_x = sorted(compressions_x_space or [], key=lambda x: x[0])
        self.steepness = float(steepness)
        self.warpings = []
        self.z_positions = []
        if self.metadata_x:
            print(f"\n{'='*80}")
            print(f"📦 METADATA (Original X-space): {self.metadata_x}")
            print(f"{'='*80}")
            self._build_warpings()
            print(f"✅ Built {len(self.warpings)} compressions")
            print(f"{'='*80}\n")

    def _build_warpings(self):
        self.warpings = []
        self.z_positions = []
        cumulative_shift = 0
        for i, (x_start, x_length) in enumerate(self.metadata_x):
            assert x_length >= 2, "Each compression length must be >= 2."
            x_end = x_start + x_length
            z_start = x_start - cumulative_shift
            z_length = x_length
            print(f"  Compression #{i+1}:")
            print(f"    Original X[{x_start}, {x_end}] → Z[{z_start}, {z_start + z_length}]")
            print(f"    Saves {z_length - 1} nodes")
            warping = SigmoidWarping(z_start, z_length, self.steepness)
            self.warpings.append(warping)
            self.z_positions.append((z_start, z_length))
            cumulative_shift += (z_length - 1)

    def forward(self, node):
        """X → Z"""
        position = node
        for warping in self.warpings:
            position = warping.forward(position)
        return position

    def inverse(self, position):
        """Z → X"""
        node = position
        for warping in reversed(self.warpings):
            node = warping.inverse(node)
        return node


# ===============================
# Bidirectional Basin Detector (FIXED)
# ===============================
def detect_compression_basin(fitness_func, local_min_x, max_search=100, verbose = False):
    """
    Bidirectional basin detection with CORRECT priority logic:

    Rule 1: if current_fit == local_fit, continue to search (record farthest point)
    Rule 2: if current_fit > local_fit, keep searching within limited length
    Rule 3: if current_fit < local_fit, STOP (exited basin) - USE THIS EXIT POINT

    PRIORITY: Use Rule 3 exit point if found, otherwise use farthest equal-fitness point from Rule 1.

    Search both LEFT and RIGHT from local_min_x.

    Returns:
    --------
    (start_x, length) or None if no compression needed
    """
    def debug_print(msg):
        if verbose:
            print(msg)


    local_min_fitness = fitness_func(local_min_x)

    debug_print(f"  🔍 Detecting basin from local min: x={local_min_x}, fitness={local_min_fitness:.2f}")

    # LEFT search
    left_boundary = local_min_x
    farthest_left_equal = local_min_x
    found_left_equal = False
    found_left_exit = False

    for i in range(1, max_search + 1):
        current_x = local_min_x - i
        current_fitness = fitness_func(current_x)

        if abs(current_fitness - local_min_fitness) < 1e-9:  # Rule 1
            farthest_left_equal = current_x
            found_left_equal = True
            debug_print(f"    LEFT: x={current_x}, fitness={current_fitness:.2f} == {local_min_fitness:.2f} (continue)")
        elif current_fitness > local_min_fitness:  # Rule 2
            debug_print(f"    LEFT: x={current_x}, fitness={current_fitness:.2f} > {local_min_fitness:.2f} (plateau/hill)")
        else:  # Rule 3
            debug_print(f"    LEFT: x={current_x}, fitness={current_fitness:.2f} < {local_min_fitness:.2f} (EXIT!)")
            left_boundary = current_x + 1
            found_left_exit = True
            break
    else:
        left_boundary = local_min_x - max_search

    # Priority: Use Rule 3 exit point if found, otherwise farthest equal-fitness
    if found_left_exit:
        debug_print(f"    ✅ LEFT: Using Rule 3 exit boundary: x={left_boundary}")
    elif found_left_equal:
        left_boundary = farthest_left_equal
        debug_print(f"    ✅ LEFT: Using farthest equal-fitness point (no exit found): x={left_boundary}")

    # RIGHT search
    right_boundary = local_min_x
    farthest_right_equal = local_min_x
    found_right_equal = False
    found_right_exit = False

    for i in range(1, max_search + 1):
        current_x = local_min_x + i
        current_fitness = fitness_func(current_x)

        if abs(current_fitness - local_min_fitness) < 1e-9:  # Rule 1
            farthest_right_equal = current_x
            found_right_equal = True
            debug_print(f"    RIGHT: x={current_x}, fitness={current_fitness:.2f} == {local_min_fitness:.2f} (continue)")
        elif current_fitness > local_min_fitness:  # Rule 2
            debug_print(f"    RIGHT: x={current_x}, fitness={current_fitness:.2f} > {local_min_fitness:.2f} (plateau/hill)")
        else:  # Rule 3
            debug_print(f"    RIGHT: x={current_x}, fitness={current_fitness:.2f} < {local_min_fitness:.2f} (EXIT!)")
            right_boundary = current_x - 1
            found_right_exit = True
            break
    else:
        right_boundary = local_min_x + max_search

    # Priority: Use Rule 3 exit point if found, otherwise farthest equal-fitness
    if found_right_exit:
        debug_print(f"    ✅ RIGHT: Using Rule 3 exit boundary: x={right_boundary}")
    elif found_right_equal:
        right_boundary = farthest_right_equal
        debug_print(f"    ✅ RIGHT: Using farthest equal-fitness point (no exit found): x={right_boundary}")

    basin_length = right_boundary - left_boundary + 1

    if basin_length < 2:
        debug_print(f"  ⚠️ No compression: basin too small (length={basin_length})")
        return None

    if not (found_left_equal or found_right_equal or found_left_exit or found_right_exit):
        debug_print(f"  ⚠️ No compression: only Rule 2 exists")
        return None

    debug_print(f"  ✅ Basin detected: X[{left_boundary}, {right_boundary}], length={basin_length}")
    return (left_boundary, basin_length)

# print("✅ Bidirectional basin detection defined (WITH PRIORITY FIX!)")

# ===============================
# Metadata Overlap Handler
# ===============================
def merge_overlapping_compressions(compressions):
    """
    Handle overlapping compressions in ORIGINAL X-space.
    """
    if not compressions:
        return []

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
            print(f"  🔄 Merged: ({start},{length}) + ({last_start},{last_length}) → ({new_start},{new_length})")
        else:
            merged.append((start, length))

    return merged



# ===============================
# Multi-D Compression Manager
# ===============================
class CompressionManagerND:
    """
    Manage 1D compressions along each dimension in N-D space.
    
    For each dimension i, we maintain compressions indexed by
    the fixed values of all other dimensions.
    
    Example for 3D:
    - dim_compressions[0][(y_val, z_val)] = list of (x_start, length) compressions
    - dim_compressions[1][(x_val, z_val)] = list of (y_start, length) compressions
    - dim_compressions[2][(x_val, y_val)] = list of (z_start, length) compressions
    """
    def __init__(self, dim, steepness=5.0):
        self.dim = dim
        self.steepness = float(steepness)
        # dim_compressions[i] = dict mapping fixed_coords → list of compressions
        self.dim_compressions = [{} for _ in range(dim)]
        # dim_systems[i] = dict mapping fixed_coords → MetadataCompressionOriginalSpace
        self.dim_systems = [{} for _ in range(dim)]
    
    def update_dimension(self, vary_dim, fixed_coords, basin):
        """
        Add compression along dimension vary_dim.
        
        Parameters:
        -----------
        vary_dim : int
            Which dimension to compress (0 to dim-1)
        fixed_coords : tuple
            Values of all other dimensions (length = dim-1)
        basin : tuple (start, length) or None
            Basin detected along vary_dim
        """
        if basin is None:
            return
        
        # Get existing compressions for this slice
        comps = self.dim_compressions[vary_dim].get(fixed_coords, [])
        comps.append(basin)
        comps = merge_overlapping_compressions(comps)
        
        # Store updated compressions
        self.dim_compressions[vary_dim][fixed_coords] = comps
        self.dim_systems[vary_dim][fixed_coords] = MetadataCompressionOriginalSpace(
            compressions_x_space=comps,
            steepness=self.steepness
        )
    
    def get_system(self, vary_dim, fixed_coords):
        """Get compression system for a specific dimension slice."""
        return self.dim_systems[vary_dim].get(fixed_coords, None)


def detect_basin_along_dimension(fitness_func_nd, point, vary_dim, max_search=100, verbose=False):
    """
    Detect basin along one dimension while fixing all others.
    
    Parameters:
    -----------
    fitness_func_nd : callable
        N-D fitness function taking tuple/list of coordinates
    point : tuple/list
        Current point in N-D space
    vary_dim : int
        Which dimension to vary (0 to dim-1)
    max_search : int
        Maximum search distance
    
    Returns:
    --------
    (start, length) or None
    """
    def f1d(val):
        # Create point with val at vary_dim, fixed elsewhere
        new_point = list(point)
        new_point[vary_dim] = int(val)
        return fitness_func_nd(tuple(new_point))
    
    return detect_compression_basin(f1d, local_min_x=int(point[vary_dim]), max_search=max_search, verbose=verbose)


# ===============================
# N-D Hill-climb with Compression
# ===============================
def hill_climb_with_compression_nd(
    fitness_func_nd,
    start_point,
    dim,
    max_iterations=10,
    basin_max_search=100,
    global_min_threshold=1e-6
):
    """
    N-D hill climbing with axis-aligned 1D compressions.
    
    Parameters:
    -----------
    fitness_func_nd : callable
        Fitness function taking tuple/list of dim coordinates
    start_point : tuple/list
        Starting point (length = dim)
    dim : int
        Number of dimensions
    max_iterations : int
        Maximum compression iterations
    basin_max_search : int
        Maximum basin search distance per dimension
    global_min_threshold : float
        Threshold for global minimum detection
    
    Returns:
    --------
    traj : list of (point, fitness, used_compression)
    cm : CompressionManagerND
    """
    traj = []
    cm = CompressionManagerND(dim, steepness=5.0)

    # Initialize point
    point = tuple(int(x) for x in start_point)
    f = fitness_func_nd(point)
    traj.append((point, f, False))

    print(f"\n🚀 {dim}D hill climbing start at {point}, f={f:.4f}\n")

    for it in range(max_iterations):
        print("="*80)
        print(f"🔄 Iteration {it+1}/{max_iterations}")
        print("="*80)

        step_count = 0
        while True:
            # ----- Propose neighbors in all directions -----
            candidates = []

            # 1) Axis-aligned neighbors (O(D))
            for d in range(dim):
                # Get compression system for this dimension
                fixed_coords = tuple(point[i] for i in range(dim) if i != d)
                comp_sys = cm.get_system(d, fixed_coords)

                if comp_sys is not None:
                    # Use compressed space
                    z = comp_sys.forward(point[d])
                    neighbor_minus = comp_sys.inverse(z - 1)
                    neighbor_plus = comp_sys.inverse(z + 1)
                else:
                    # No compression, use regular neighbors
                    neighbor_minus = point[d] - 1
                    neighbor_plus = point[d] + 1

                # Create neighbor points
                point_minus = list(point)
                point_minus[d] = neighbor_minus
                cand_minus = tuple(point_minus)
                candidates.append((cand_minus, fitness_func_nd(cand_minus)))

                point_plus = list(point)
                point_plus[d] = neighbor_plus
                cand_plus = tuple(point_plus)
                candidates.append((cand_plus, fitness_func_nd(cand_plus)))

            # 2) Diagonal neighbors (O(D^2))
            if dim >= 2:
                for d1, d2 in itertools.combinations(range(dim), 2):
                    # Compression system for d1
                    fixed1 = tuple(point[i] for i in range(dim) if i != d1)
                    comp1 = cm.get_system(d1, fixed1)
                    if comp1 is not None:
                        z1 = comp1.forward(point[d1])
                        n1_vals = [comp1.inverse(z1 - 1), comp1.inverse(z1 + 1)]
                    else:
                        n1_vals = [point[d1] - 1, point[d1] + 1]

                    # Compression system for d2
                    fixed2 = tuple(point[i] for i in range(dim) if i != d2)
                    comp2 = cm.get_system(d2, fixed2)
                    if comp2 is not None:
                        z2 = comp2.forward(point[d2])
                        n2_vals = [comp2.inverse(z2 - 1), comp2.inverse(z2 + 1)]
                    else:
                        n2_vals = [point[d2] - 1, point[d2] + 1]

                    # Combine offsets in both dimensions
                    for v1 in n1_vals:
                        for v2 in n2_vals:
                            diag_point = list(point)
                            diag_point[d1] = v1
                            diag_point[d2] = v2
                            cand = tuple(diag_point)
                            candidates.append((cand, fitness_func_nd(cand)))

            # Pick best neighbor (steepest descent)
            best_point, best_f = point, f
            for cand_point, cand_f in candidates:
                if cand_f < best_f:
                    best_point, best_f = cand_point, cand_f

            if best_f < f:
                point, f = best_point, best_f
                used_comp = any(cm.get_system(d, tuple(point[i] for i in range(dim) if i != d)) is not None 
                              for d in range(dim))
                traj.append((point, f, used_comp))
                step_count += 1
            else:
                print(f"  📍 Climbed {step_count} steps, now at {point}, f={f:.6g}")
                break

        # Check convergence / global min
        if abs(f) < global_min_threshold:
            print("\n🎉 SUCCESS: reached near-global minimum")
            break

        print(f"\n⚠️ STUCK at local minimum {point}, f={f:.6g}")
        print(f"  🔍 Detecting basins along all {dim} dimensions...")

        # ----- Detect basins along each dimension -----
        basins = {}  # dimension -> (start, length)
        for d in range(dim):
            basin = detect_basin_along_dimension(fitness_func_nd, point, d, max_search=basin_max_search)
            if basin:
                fixed_coords = tuple(point[i] for i in range(dim) if i != d)
                print(f"  ✅ Dim {d} basin: {basin}")
                cm.update_dimension(d, fixed_coords, basin)
                basins[d] = basin

        if not basins:
            print("\n  ⚠️ No compressible basin found in any dimension. Stopping.")
            break

        # ----- Choose restart point after compression -----
        # Try both ends of each detected basin
        restart_candidates = []
        
        for d, (b_start, b_len) in basins.items():
            b_end = b_start + b_len - 1
            
            # Try left end (start - 1)
            restart_point = list(point)
            restart_point[d] = b_start - 1
            restart_candidates.append((tuple(restart_point), fitness_func_nd(tuple(restart_point))))
            
            # Try right end (end + 1)
            restart_point = list(point)
            restart_point[d] = b_end + 1
            restart_candidates.append((tuple(restart_point), fitness_func_nd(tuple(restart_point))))

        # Pick best restart candidate
        if restart_candidates:
            restart_point, restart_f = min(restart_candidates, key=lambda t: t[1])
            print(f"\n  ➡️ Restarting from {restart_point}, f={restart_f:.4f}")
            point, f = restart_point, restart_f
            traj.append((point, f, True))
        else:
            print("\n  ⚠️ No valid restart candidate. Stopping.")
            break

    print("\n" + "="*80)
    print(f"🏁 FINAL {dim}D RESULTS")
    print("="*80)
    print(f"  Final position: {point}")
    print(f"  Final fitness: {f:.6g}")
    print(f"  Total steps:   {len(traj)}")
    total_compressions = sum(len(cm.dim_compressions[d]) for d in range(dim))
    print(f"  Total compressions: {total_compressions}")
    print("="*80 + "\n")

    return traj, cm


# ===============================
# Baseline: N-D Hill Climb WITHOUT Compression
# ===============================
def hill_climb_simple_nd(
    fitness_func_nd,
    start_point,
    dim,
    max_steps=2000
):
    """
    Simple N-D hill climbing WITHOUT compression (for comparison).
    
    Returns:
    --------
    traj : list of (point, fitness)
    """
    point = tuple(int(x) for x in start_point)

    
    f = fitness_func_nd(point)
    traj = [(point, f)]

    for _ in range(max_steps):
        # Try 2*dim neighbors (±1 in each dimension)
        candidates = []
        for d in range(dim):
            # -1 in dimension d
            neighbor = list(point)
            neighbor[d] -= 1
            candidates.append((tuple(neighbor), fitness_func_nd(tuple(neighbor))))
            
            # +1 in dimension d
            neighbor = list(point)
            neighbor[d] += 1
            candidates.append((tuple(neighbor), fitness_func_nd(tuple(neighbor))))

        # Pick best
        best_point, best_f = point, f
        for cand_point, cand_f in candidates:
            if cand_f < best_f:
                best_point, best_f = cand_point, cand_f

        if best_f < f:
            point, f = best_point, best_f
            traj.append((point, f))
        else:
            break  # Stuck at local minimum

    return traj

from module.sbst_core import instrument_and_load, FitnessCalculator, hill_climbing_search
import ast
import random


# ===============================
# N-D Hill-climb with Compression
# ===============================
def hill_climb_with_compression_nd_code(
    # fitness_func_nd_code,
    fitness_calc, func_obj, target_branch_node, target_outcome, subject_node, parent_map,
    start_point,
    dim,
    max_iterations=10,
    basin_max_search=100,
    global_min_threshold=1e-6,
    verbose = False,
    cm = None,  # Optional: reuse compression manager across trials
):
    """
    N-D hill climbing with axis-aligned 1D compressions.
    
    Parameters:
    -----------
    fitness_func_nd : callable
        Fitness function taking tuple/list of dim coordinates
    start_point : tuple/list
        Starting point (length = dim)
    dim : int
        Number of dimensions
    max_iterations : int
        Maximum compression iterations
    basin_max_search : int
        Maximum basin search distance per dimension
    global_min_threshold : float
        Threshold for global minimum detection
    cm : CompressionManagerND or None
        Optional compression manager to reuse metadata across trials
    
    Returns:
    --------
    traj : list of (point, fitness, used_compression)
    cm : CompressionManagerND
    """
    def fitness_func_nd_code(x):
        return fitness_calc.fitness_for_candidate(func_obj, x, target_branch_node, target_outcome, subject_node, parent_map)


    traj = []
    # Reuse existing compression manager if provided, otherwise create new one
    if cm is None:
        cm = CompressionManagerND(dim, steepness=5.0)
        print("📦 Created NEW CompressionManagerND for this search")
    else:
        print("♻️ REUSING existing CompressionManagerND with accumulated metadata")

    # Initialize point
    point = tuple(int(x) for x in start_point)
    f = fitness_func_nd_code(point)
    traj.append((point, f, False))

    print(f"\n🚀 {dim}D hill climbing start at {point}, f={f:.4f}\n")

    # ✅ Early success check: if already at goal, return immediately
    if abs(f) < global_min_threshold:
        print("🎉 INITIAL POINT IS ALREADY AT GOAL! Returning immediately.")
        print("\n" + "="*80)
        print(f"🏁 FINAL {dim}D RESULTS")
        print("="*80)
        print(f"  Final position: {point}")
        print(f"  Final fitness: {f:.6g}")
        print(f"  Total steps:   {len(traj)}")
        total_compressions = sum(len(cm.dim_compressions[d]) for d in range(dim))
        print(f"  Total compressions: {total_compressions}")
        print("="*80 + "\n")
        return traj, cm

    for it in range(max_iterations):
        print("="*80)
        print(f"🔄 Iteration {it+1}/{max_iterations}")
        print("="*80)
        
        # ✅ Check if already at goal before starting this iteration
        if abs(f) < global_min_threshold:
            print("🎉 SUCCESS: Already at goal at start of iteration!")
            break

        step_count = 0
        max_steps_per_iteration = 10000  # Safety limit to prevent infinite loops
        
        while step_count < max_steps_per_iteration:
            # ----- Propose neighbors in all directions -----
            candidates = []

            # 1) Axis-aligned neighbors (O(D))
            for d in range(dim):
                # Get compression system for this dimension
                fixed_coords = tuple(point[i] for i in range(dim) if i != d)
                comp_sys = cm.get_system(d, fixed_coords)

                if comp_sys is not None:
                    # Use compressed space
                    z = comp_sys.forward(point[d])
                    neighbor_minus = comp_sys.inverse(z - 1)
                    neighbor_plus = comp_sys.inverse(z + 1)
                else:
                    # No compression, use regular neighbors
                    neighbor_minus = point[d] - 1
                    neighbor_plus = point[d] + 1

                # Create neighbor points
                point_minus = list(point)
                point_minus[d] = neighbor_minus
                cand_minus = tuple(point_minus)
                candidates.append((cand_minus, fitness_func_nd_code(cand_minus)))

                point_plus = list(point)
                point_plus[d] = neighbor_plus
                cand_plus = tuple(point_plus)
                candidates.append((cand_plus, fitness_func_nd_code(cand_plus)))

            # 2) Diagonal neighbors (O(D^2))
            if dim >= 2:
                for d1, d2 in itertools.combinations(range(dim), 2):
                    # Compression system for d1
                    fixed1 = tuple(point[i] for i in range(dim) if i != d1)
                    comp1 = cm.get_system(d1, fixed1)
                    if comp1 is not None:
                        z1 = comp1.forward(point[d1])
                        n1_vals = [comp1.inverse(z1 - 1), comp1.inverse(z1 + 1)]
                    else:
                        n1_vals = [point[d1] - 1, point[d1] + 1]

                    # Compression system for d2
                    fixed2 = tuple(point[i] for i in range(dim) if i != d2)
                    comp2 = cm.get_system(d2, fixed2)
                    if comp2 is not None:
                        z2 = comp2.forward(point[d2])
                        n2_vals = [comp2.inverse(z2 - 1), comp2.inverse(z2 + 1)]
                    else:
                        n2_vals = [point[d2] - 1, point[d2] + 1]

                    # Combine offsets in both dimensions
                    for v1 in n1_vals:
                        for v2 in n2_vals:
                            diag_point = list(point)
                            diag_point[d1] = v1
                            diag_point[d2] = v2
                            cand = tuple(diag_point)
                            candidates.append((cand, fitness_func_nd_code(cand)))

            # Pick best neighbor (steepest descent)
            best_point, best_f = point, f
            for cand_point, cand_f in candidates:
                if cand_f < best_f:
                    best_point, best_f = cand_point, cand_f

            if best_f < f:
                point, f = best_point, best_f
                used_comp = any(cm.get_system(d, tuple(point[i] for i in range(dim) if i != d)) is not None 
                              for d in range(dim))
                traj.append((point, f, used_comp))
                step_count += 1
            else:
                print(f"  📍 Climbed {step_count} steps, now at {point}, f={f:.6g}")
                break
        
        # Safety check: warn if hit the step limit
        if step_count >= max_steps_per_iteration:
            print(f"  ⚠️ WARNING: Hit maximum step limit ({max_steps_per_iteration}) in iteration {it+1}")

        # Check convergence / global min
        if abs(f) < global_min_threshold:
            print("\n🎉 SUCCESS: reached near-global minimum")
            break

        print(f"\n⚠️ STUCK at local minimum {point}, f={f:.6g}")
        print(f"  🔍 Detecting basins along all {dim} dimensions...")

        # ----- Detect basins along each dimension -----
        basins = {}  # dimension -> (start, length)
        for d in range(dim):
            basin = detect_basin_along_dimension(fitness_func_nd_code, point, d, max_search=basin_max_search)
            if basin:
                fixed_coords = tuple(point[i] for i in range(dim) if i != d)
                print(f"  ✅ Dim {d} basin: {basin}")
                cm.update_dimension(d, fixed_coords, basin)
                basins[d] = basin

        if not basins:
            print("\n  ⚠️ No compressible basin found in any dimension. Stopping.")
            break

        # ----- Choose restart point after compression -----
        # Try both ends of each detected basin
        restart_candidates = []
        
        for d, (b_start, b_len) in basins.items():
            b_end = b_start + b_len - 1
            
            # Try left end (start - 1)
            restart_point = list(point)
            restart_point[d] = b_start - 1
            restart_candidates.append((tuple(restart_point), fitness_func_nd_code(tuple(restart_point))))
            
            # Try right end (end + 1)
            restart_point = list(point)
            restart_point[d] = b_end + 1
            restart_candidates.append((tuple(restart_point), fitness_func_nd_code(tuple(restart_point))))

        # Pick best restart candidate
        if restart_candidates:
            restart_point, restart_f = min(restart_candidates, key=lambda t: t[1])
            print(f"\n  ➡️ Restarting from {restart_point}, f={restart_f:.4f}")
            point, f = restart_point, restart_f
            traj.append((point, f, True))
            
            # ✅ Check if restart point already reached the goal
            if abs(f) < global_min_threshold:
                print("🎉 RESTART POINT IS ALREADY AT GOAL! Stopping iterations.")
                break
        else:
            print("\n  ⚠️ No valid restart candidate. Stopping.")
            break

    print("\n" + "="*80)
    print(f"🏁 FINAL {dim}D RESULTS")
    print("="*80)
    print(f"  Final position: {point}")
    print(f"  Final fitness: {f:.6g}")
    print(f"  Total steps:   {len(traj)}")
    total_compressions = sum(len(cm.dim_compressions[d]) for d in range(dim))
    print(f"  Total compressions: {total_compressions}")
    print("="*80 + "\n")

    return traj, cm



def hill_climb_simple_nd_code(
    # fitness_func_nd_code,
    fitness_calc, func_obj, target_branch_node, target_outcome, subject_node, parent_map,
    start_point,
    dim,
    max_steps=2000
):
    """
    Simple N-D hill climbing WITHOUT compression (for comparison).
    
    Returns:
    --------
    traj : list of (point, fitness)
    """
    point = tuple(int(x) for x in start_point)

    def fitness_func_nd_code(x):
        return fitness_calc.fitness_for_candidate(func_obj, x, target_branch_node, target_outcome, subject_node, parent_map)

    f = fitness_func_nd_code(point)
    traj = [(point, f)]

    for _ in range(max_steps):
        # Try 2*dim neighbors (±1 in each dimension)
        candidates = []
        for d in range(dim):
            # -1 in dimension d
            neighbor = list(point)
            neighbor[d] -= 1
            candidates.append((tuple(neighbor), fitness_func_nd_code(tuple(neighbor))))
            
            # +1 in dimension d
            neighbor = list(point)
            neighbor[d] += 1
            candidates.append((tuple(neighbor), fitness_func_nd_code(tuple(neighbor))))

        # Pick best
        best_point, best_f = point, f
        for cand_point, cand_f in candidates:
            if cand_f < best_f:
                best_point, best_f = cand_point, cand_f

        if best_f < f:
            point, f = best_point, best_f
            traj.append((point, f))
        else:
            break  # Stuck at local minimum

    return traj