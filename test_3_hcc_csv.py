#!/usr/bin/env python3
import os
import csv
import time
import random
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from benchmark.test_3.fitness import (
    fitness_needle,
    fitness_rugged,
    fitness_plateau,
    fitness_combined,
)

from compression_hc import CompressionManagerND, detect_basin_along_dimension
from test_3_plot import plot_fitness_landscape


# ===============================
# N-D Hill-climb with Compression
# ===============================
def hill_climb_with_compression_nd_code(
    fitness_fn,            # <---- 단순 callable((tuple))->float
    start_point,
    dim,
    max_iterations=10,
    basin_max_search=100,
    global_min_threshold=1e-6,
    verbose=True,
    cm=None,               # Optional: compression manager reuse
    time_limit=None,
    start_time=None,
):
    """
    N-D hill climbing with axis-aligned 1D compressions (Test3 version)

    This version:
      - DOES NOT use FitnessCalculator / branches / func_obj
      - fitness_fn(point) must return a non-negative scalar
      - Supports strict time-limit enforcement before every eval
    """

    import time as time_module
    import itertools

    # Wrap fitness to ensure integer and consistent calling
    def fitness_func_nd_code(x):
        return float(fitness_fn(tuple(int(v) for v in x)))

    # For dimension deactivation
    deactivation_patience = 20

    traj = []
    
    # Track visited points to detect loops
    visited_points = set()
    compression_jumps = []  # Track when we made compression-based jumps
    no_basin_events = []    # Track when no basins were detected

    # ---------------------------
    # CompressionManager setup
    # ---------------------------
    if cm is None:
        cm = CompressionManagerND(dim, steepness=5.0)
        if verbose:
            print("📦 Created NEW CompressionManagerND for this search")
    else:
        if verbose:
            print("♻️ REUSING existing CompressionManagerND with accumulated metadata")

    active_dims = list(range(dim))
    dim_stagnation = {d: 0 for d in range(dim)}

    # -------------------------------------------------------
    # Time check BEFORE initial evaluation
    # -------------------------------------------------------
    if time_limit is not None and start_time is not None:
        if time_module.time() - start_time >= time_limit:
            point = tuple(int(x) for x in start_point)
            traj.append((point, float("inf"), False))
            return traj, cm

    # Initialize point
    point = tuple(int(x) for x in start_point)
    f = fitness_func_nd_code(point)
    traj.append((point, f, False))
    visited_points.add(point)

    if verbose:
        print(f"\n🚀 {dim}D hill climbing start at {point}, f={f:.6g}\n")

    # Early success
    if abs(f) < global_min_threshold:
        if verbose:
            print("🎉 INITIAL POINT IS ALREADY A GOAL")
        return traj, cm

    # ============================================================
    # MAIN ITERATIONS
    # ============================================================
    for it in range(max_iterations):
        if not active_dims:
            if verbose:
                print("All dimensions deactivated. Stopping.")
            return traj, cm

        if verbose:
            print("=" * 80)
            print(f"🔄 Iteration {it+1}/{max_iterations}")
            print("=" * 80)

        # Check early
        if abs(f) < global_min_threshold:
            if verbose:
                print("🎉 SUCCESS at iteration start")
            break

        step_count = 0
        max_steps_per_iteration = 10000

        # -----------------------------------------------------------
        # STEEPEST DESCENT LOOP
        # -----------------------------------------------------------
        while step_count < max_steps_per_iteration:

            # TIME CHECK
            if time_limit is not None and start_time is not None:
                if time_module.time() - start_time >= time_limit:
                    if verbose:
                        print("⏱️ Time limit reached inside climbing loop → stop")
                    return traj, cm

            best_point = point
            best_f = f
            candidates = []
            meaningful_dims = set()

            # ------------------------
            # AXIS NEIGHBORS
            # ------------------------
            for d in active_dims:

                # time check
                if time_limit is not None and start_time is not None:
                    if time_module.time() - start_time >= time_limit:
                        return traj, cm

                fixed = tuple(point[i] for i in range(dim) if i != d)
                comp_sys = cm.get_system(d, fixed)

                if comp_sys:
                    z = comp_sys.forward(point[d])
                    neigh_vals = [comp_sys.inverse(z - 1), comp_sys.inverse(z + 1)]
                else:
                    neigh_vals = [point[d] - 1, point[d] + 1]

                for val in neigh_vals:
                    cand = list(point)
                    cand[d] = val
                    cand_t = tuple(cand)

                    # strict time check BEFORE eval
                    if time_limit and (time_module.time() - start_time) >= time_limit:
                        return traj, cm

                    cand_f = fitness_func_nd_code(cand_t)
                    candidates.append((cand_t, cand_f, [d]))

            # ------------------------
            # DIAGONAL NEIGHBORS: with compression metadata reuse
            # ------------------------
            if dim >= 2:
                import itertools
                
                for d1, d2 in itertools.combinations(range(dim), 2):
                    # Skip if dimensions not active
                    if d1 not in active_dims or d2 not in active_dims:
                        continue
                    
                    # time check
                    if time_limit and (time_module.time() - start_time) >= time_limit:
                        return traj, cm
                    
                    # Generate 4 diagonal neighbors: (±1, ±1)
                    for offset1 in [-1, 1]:
                        for offset2 in [-1, 1]:
                            # Move diagonally in X-space
                            diag_point = list(point)
                            diag_point[d1] += offset1
                            diag_point[d2] += offset2
                            diag_point_tuple = tuple(diag_point)
                            
                            # time check before eval
                            if time_limit and (time_module.time() - start_time) >= time_limit:
                                return traj, cm
                            
                            # Evaluate fitness at diagonal neighbor
                            diag_f = fitness_func_nd_code(diag_point_tuple)
                            candidates.append((diag_point_tuple, diag_f, [d1, d2]))
            # ------------------------

            # ------------------------
            # Select steepest descent
            # ------------------------
            for cand_point, cand_f, modified in candidates:
                if cand_f < best_f:
                    best_point, best_f = cand_point, cand_f
                if cand_f != f:
                    for d in modified:
                        meaningful_dims.add(d)

            # Update stagnation
            for d in range(dim):
                if d not in meaningful_dims:
                    dim_stagnation[d] += 1
                else:
                    dim_stagnation[d] = 0

            # Remove stagnant dims
            for d in list(active_dims):
                if dim_stagnation[d] >= deactivation_patience:
                    active_dims.remove(d)
                    if verbose:
                        print(f"Deactivating dim {d} due to stagnation")

            # If no improvement → stop climb
            if best_f < f:
                point, f = best_point, best_f
                used_comp = any(
                    cm.get_system(d, tuple(point[i] for i in range(dim) if i != d))
                    is not None for d in range(dim)
                )
                traj.append((point, f, used_comp))
                visited_points.add(point)
                step_count += 1
            else:
                if verbose:
                    print(f"📍 Stuck after {step_count} steps at {point}, f={f:.6g}")
                break

        # After full climbing iteration:
        if abs(f) < global_min_threshold:
            if verbose:
                print("🎉 SUCCESS after climbing")
            break

        # -----------------------------------------------------------
        # BASIN DETECTION: Search from ALL neighbors (comprehensive)
        # -----------------------------------------------------------
        if verbose:
            print(f"\n⚠️ Stuck at {point}, detecting basins from ALL neighbors…")

        # Generate all neighbors (axis-aligned + diagonal)
        neighbor_points = []
        
        # Axis-aligned neighbors
        for d in active_dims:
            for offset in [-1, 1]:
                neighbor = list(point)
                neighbor[d] += offset
                neighbor_points.append(tuple(neighbor))
        
        # Diagonal neighbors (if 2D or higher)
        if dim >= 2:
            import itertools
            for d1, d2 in itertools.combinations(active_dims, 2):
                for offset1 in [-1, 1]:
                    for offset2 in [-1, 1]:
                        neighbor = list(point)
                        neighbor[d1] += offset1
                        neighbor[d2] += offset2
                        neighbor_points.append(tuple(neighbor))
        
        if verbose:
            print(f"  🔍 Checking {len(neighbor_points)} neighbors × {len(active_dims)} dimensions = {len(neighbor_points) * len(active_dims)} basin searches")
            print(f"     Neighbors: {neighbor_points}")
        
        # Collect all basin detections and restart candidates
        all_restart_candidates = []
        basins_found_count = 0
        
        # Track for debugging: which neighbors find basins
        neighbor_basin_info = {}
        
        for neighbor in neighbor_points:
            if time_limit and (time_module.time() - start_time) >= time_limit:
                return traj, cm
            
            # Basin detection along each dimension from this neighbor
            for d in active_dims:
                # Enable verbose for last few iterations to debug
                is_last_iterations = (it >= max_iterations - 3)
                basin = detect_basin_along_dimension(
                    fitness_func_nd_code, neighbor, d, basin_max_search, verbose=is_last_iterations
                )
                
                if basin:
                    basins_found_count += 1
                    b_start, b_len = basin
                    b_end = b_start + b_len - 1
                    
                    # Track basin info for debugging
                    if neighbor not in neighbor_basin_info:
                        neighbor_basin_info[neighbor] = []
                    neighbor_basin_info[neighbor].append((d, b_start, b_len))
                    
                    # Update compression metadata
                    fixed = tuple(neighbor[i] for i in range(dim) if i != d)
                    cm.update_dimension(d, fixed, basin)
                    
                    # Add restart candidates at basin boundaries
                    restart_left = list(neighbor)
                    restart_left[d] = b_start - 1
                    restart_left_tuple = tuple(restart_left)
                    restart_left_f = fitness_func_nd_code(restart_left_tuple)
                    all_restart_candidates.append((restart_left_tuple, restart_left_f))
                    
                    restart_right = list(neighbor)
                    restart_right[d] = b_end + 1
                    restart_right_tuple = tuple(restart_right)
                    restart_right_f = fitness_func_nd_code(restart_right_tuple)
                    all_restart_candidates.append((restart_right_tuple, restart_right_f))
        
        if verbose:
            print(f"  ✅ Found {basins_found_count} basins, generated {len(all_restart_candidates)} restart candidates")
            
            # Show details for last few iterations
            is_last_iterations = (it >= max_iterations - 3)
            if is_last_iterations and neighbor_basin_info:
                print(f"\n  📊 BASIN DETECTION DETAILS (Iteration {it+1}):")
                for neighbor, basins_list in neighbor_basin_info.items():
                    print(f"     Neighbor {neighbor}:")
                    for d, b_start, b_len in basins_list:
                        print(f"       Dim {d}: basin [{b_start}, {b_start+b_len-1}] (length={b_len})")
                
                # Show top 5 unique restart candidates
                if all_restart_candidates:
                    unique_candidates = {}
                    for pt, fit in all_restart_candidates:
                        if pt not in unique_candidates:
                            unique_candidates[pt] = fit
                    sorted_unique = sorted(unique_candidates.items(), key=lambda x: x[1])[:5]
                    print(f"\n  🎯 Top 5 restart candidates:")
                    for pt, fit in sorted_unique:
                        same_as_stuck = " ⚠️ SAME AS STUCK!" if pt == point else ""
                        print(f"     {pt}: f={fit:.6f}{same_as_stuck}")
                print()
        
        if not all_restart_candidates:
            # ==========================================
            # NO BASINS DETECTED - DETAILED DIAGNOSTICS
            # ==========================================
            if verbose:
                print("\n" + "="*80)
                print("⚠️ ⚠️ ⚠️  NO BASINS DETECTED - DIAGNOSTIC REPORT  ⚠️ ⚠️ ⚠️")
                print("="*80)
                
                # 1. Current state
                print(f"\n📍 CURRENT STATE:")
                print(f"   Point: {point}")
                print(f"   Fitness: {f:.6g}")
                print(f"   Iteration: {it+1}/{max_iterations}")
                print(f"   Trajectory length: {len(traj)}")
                
                # 2. Was this point reached via compression?
                was_compression_jump = False
                if len(traj) >= 2:
                    prev_point, prev_f, prev_used_comp = traj[-2]
                    was_compression_jump = prev_used_comp or traj[-1][2]  # Check last entry
                
                print(f"\n🔄 HOW DID WE GET HERE?")
                print(f"   Reached via compression jump: {was_compression_jump}")
                if len(traj) >= 2:
                    prev_point, prev_f, _ = traj[-2]
                    print(f"   Previous point: {prev_point}, f={prev_f:.6g}")
                    print(f"   Fitness change: {f - prev_f:.6g}")
                
                # 3. Loop detection
                is_revisit = point in list(visited_points)[:-1]  # Exclude current point
                print(f"\n🔁 LOOP DETECTION:")
                print(f"   Revisiting previous point: {is_revisit}")
                print(f"   Total unique points visited: {len(visited_points)}")
                
                # 4. Evaluate all neighbors to see the landscape
                print(f"\n🔍 NEIGHBOR LANDSCAPE ANALYSIS:")
                print(f"   Total neighbors checked: {len(neighbor_points)}")
                
                neighbor_fitness_values = []
                for neighbor in neighbor_points[:8]:  # Show first 8 neighbors
                    if time_limit and (time_module.time() - start_time) >= time_limit:
                        break
                    neighbor_f = fitness_func_nd_code(neighbor)
                    neighbor_fitness_values.append((neighbor, neighbor_f))
                    improvement = "✓ BETTER" if neighbor_f < f else ("= SAME" if neighbor_f == f else "✗ WORSE")
                    print(f"   {neighbor}: f={neighbor_f:.6g} {improvement}")
                
                if len(neighbor_points) > 8:
                    print(f"   ... and {len(neighbor_points) - 8} more neighbors")
                
                # 5. Compression state
                print(f"\n📦 COMPRESSION STATE:")
                has_any_compression = False
                for d in range(dim):
                    fixed = tuple(point[i] for i in range(dim) if i != d)
                    comp_sys = cm.get_system(d, fixed)
                    if comp_sys:
                        has_any_compression = True
                        print(f"   Dim {d}: HAS compression system")
                        print(f"      Basins: {len(comp_sys.basins) if hasattr(comp_sys, 'basins') else 'N/A'}")
                    else:
                        print(f"   Dim {d}: No compression system")
                
                if not has_any_compression:
                    print(f"   ⚠️ NO active compression systems at this point!")
                
                # 6. Summary and recommendation
                print(f"\n📊 DIAGNOSTIC SUMMARY:")
                if is_revisit:
                    print(f"   ⚠️ CIRCULAR LOOP DETECTED - revisiting point {point}")
                if was_compression_jump and not all_restart_candidates:
                    print(f"   ⚠️ COMPRESSION JUMP LED TO DEAD END")
                if neighbor_fitness_values:
                    best_neighbor_f = min(nf for _, nf in neighbor_fitness_values)
                    if best_neighbor_f >= f:
                        print(f"   ⚠️ ALL NEIGHBORS ARE WORSE OR EQUAL (true local minimum)")
                    else:
                        print(f"   ℹ️ Better neighbors exist, but no basins detected")
                
                print("="*80 + "\n")
                
                # Record this event for later analysis
                no_basin_event = {
                    "iteration": it,
                    "point": point,
                    "fitness": f,
                    "was_compression_jump": was_compression_jump,
                    "is_revisit": is_revisit,
                    "neighbor_count": len(neighbor_points),
                    "has_compression": has_any_compression,
                }
                no_basin_events.append(no_basin_event)
            
            break

        # -----------------------------------------------------------
        # RESTART FROM BEST BASIN BOUNDARY (from all neighbors)
        # -----------------------------------------------------------
        restart_point, restart_f = min(all_restart_candidates, key=lambda t: t[1])

        if verbose:
            print(f"➡️ Restarting from {restart_point}, f={restart_f:.6g}")
        
        # Track this compression jump
        compression_jump_info = {
            "iteration": it,
            "from_point": point,
            "from_fitness": f,
            "to_point": restart_point,
            "to_fitness": restart_f,
            "fitness_change": restart_f - f,
            "basins_found": basins_found_count,
        }
        compression_jumps.append(compression_jump_info)

        point, f = restart_point, restart_f
        traj.append((point, f, True))
        visited_points.add(point)

        if abs(f) < global_min_threshold:
            if verbose:
                print("🎉 Restart hit goal")
            break

    # ==========================================
    # FINAL DIAGNOSTIC SUMMARY
    # ==========================================
    if verbose and (compression_jumps or no_basin_events):
        print("\n" + "="*80)
        print("📊 FINAL DIAGNOSTIC SUMMARY")
        print("="*80)
        
        print(f"\n🎯 SEARCH RESULTS:")
        print(f"   Final point: {point}")
        print(f"   Final fitness: {f:.6g}")
        print(f"   Success (fitness < 1e-6): {abs(f) < global_min_threshold}")
        print(f"   Total trajectory length: {len(traj)}")
        print(f"   Unique points visited: {len(visited_points)}")
        
        if compression_jumps:
            print(f"\n🔄 COMPRESSION JUMPS MADE: {len(compression_jumps)}")
            for i, jump in enumerate(compression_jumps):
                print(f"   Jump {i+1} (iter {jump['iteration']}):")
                print(f"      From: {jump['from_point']}, f={jump['from_fitness']:.6g}")
                print(f"      To:   {jump['to_point']}, f={jump['to_fitness']:.6g}")
                change_symbol = "⬇️" if jump['fitness_change'] < 0 else ("➡️" if jump['fitness_change'] == 0 else "⬆️")
                print(f"      Change: {jump['fitness_change']:.6g} {change_symbol}")
                print(f"      Basins found: {jump['basins_found']}")
        
        if no_basin_events:
            print(f"\n⚠️ NO-BASIN EVENTS: {len(no_basin_events)}")
            for i, event in enumerate(no_basin_events):
                print(f"   Event {i+1} (iter {event['iteration']}):")
                print(f"      Point: {event['point']}, f={event['fitness']:.6g}")
                print(f"      Via compression: {event['was_compression_jump']}")
                print(f"      Revisiting point: {event['is_revisit']}")
                print(f"      Has compression systems: {event['has_compression']}")
        
        print("="*80 + "\n")
    
    return traj, cm


# ============================================================
# Fitness mapping
# ============================================================
fitness_map = {
    "needle": fitness_needle,
    "rugged": fitness_rugged,
    "plateau": fitness_plateau,
    "combined": fitness_combined,
}


# ============================================================
# Worker (single experiment, RANDOM INIT ONLY)
# ============================================================
def _hcc_worker_test3(args):
    (
        fitness_name,
        num_args,
        time_limit,
        random_seed,
        init_low,
        init_high,
        max_iterations,
        basin_max_search,
    ) = args

    rng = random.Random(random_seed)
    fitness_fn = fitness_map[fitness_name]

    # Always RANDOM initialization
    start_point = tuple(rng.randint(init_low, init_high) for _ in range(num_args))

    start_time = time.time()

    traj, cm = hill_climb_with_compression_nd_code(
        fitness_fn=fitness_fn,
        start_point=start_point,
        dim=num_args,
        time_limit=time_limit,
        start_time=start_time,
        max_iterations=max_iterations,
        basin_max_search=basin_max_search,
    )

    generations = len(traj)
    best_point, best_fitness, _ = traj[-1]

    total_time = time.time() - start_time
    success = abs(best_fitness) < 1e-6

    time_to_solution = None
    if success:
        for gen, (pt, f, _) in enumerate(traj):
            if abs(f) < 1e-6:
                time_to_solution = gen
                break

    return {
        "history": [(i, pt, f) for i, (pt, f, _) in enumerate(traj)],
        "best_ind": best_point,
        "best_fitness": best_fitness,
        "generations": generations,
        "nfe": generations,
        "success": success,
        "total_time": total_time,
        "time_to_solution": time_to_solution,
        "compression_manager": cm,  # Pass compression manager for visualization
    }


# ============================================================
# Plotting utilities
# ============================================================
def plot_landscape_and_path(fitness_name, num_args, history, out_png, value_range=(-150, 150)):
    fitness_fn = fitness_map[fitness_name]

    if num_args == 1:
        lo, hi = value_range   # value_range = (-150, 150)
        xs = np.linspace(lo, hi, 500)
        ys = [fitness_fn((int(x),)) for x in xs]

        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys, "gray", alpha=0.7)

        traj_x = [pt[0] for (_, pt, ft) in history]
        traj_f = [ft for (_, pt, ft) in history]
        colors = np.linspace(0.3, 1.0, len(history))

        for i in range(len(history)):
            plt.scatter(traj_x[i], traj_f[i], c=[[0, 0, 0, colors[i]]], s=40)

        plt.title(f"HC+C Trajectory: {fitness_name} ({num_args}D)")
        plt.xlabel("x")
        plt.ylabel("fitness")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Saved 1D plot: {out_png}")

    elif num_args == 2:
        # Create contour plot (top-down projection view)
        fig, ax = plt.subplots(figsize=(10, 8))

        lo, hi = value_range
        X = np.linspace(lo, hi, 200)
        Y = np.linspace(lo, hi, 200)
        Xg, Yg = np.meshgrid(X, Y)

        Z = np.zeros_like(Xg)
        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):
                Z[i, j] = fitness_fn((int(Xg[i, j]), int(Yg[i, j])))

        # Plot contours
        contour = ax.contourf(Xg, Yg, Z, levels=30, cmap="viridis", alpha=0.8)
        ax.contour(Xg, Yg, Z, levels=30, colors='black', alpha=0.2, linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Fitness', rotation=270, labelpad=20)

        # Plot trajectory
        traj_x = [pt[0] for (_, pt, _) in history]
        traj_y = [pt[1] for (_, pt, _) in history]
        
        # Draw trajectory line with gradient color
        for i in range(len(history) - 1):
            ax.plot(traj_x[i:i+2], traj_y[i:i+2], 
                   color='red', alpha=0.6, linewidth=2, zorder=5)
        
        # Plot trajectory points with varying sizes
        colors = np.linspace(0.3, 1.0, len(history))
        for i in range(len(history)):
            ax.scatter(traj_x[i], traj_y[i], 
                      c='red', s=50, alpha=colors[i], 
                      edgecolors='white', linewidths=1, zorder=10)
        
        # Mark start and end points
        ax.scatter(traj_x[0], traj_y[0], 
                  c='lime', s=200, marker='*', 
                  edgecolors='black', linewidths=2, 
                  label='Start', zorder=15)
        ax.scatter(traj_x[-1], traj_y[-1], 
                  c='cyan', s=200, marker='*', 
                  edgecolors='black', linewidths=2, 
                  label='End', zorder=15)

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(f"HC+C Trajectory: {fitness_name} ({num_args}D) - Contour View", fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_png, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved 2D contour plot: {out_png}")

    else:
        print("Plotting only supported for 1D or 2D.")


# ============================================================
# run_single
# ============================================================
def run_parallel_test3_hcc(
    fitness_name,
    num_args,
    output_dir,
    time_limit=10.0,
    random_seed=42,
    init_low=-150,
    init_high=150,
    max_iterations=10,
    basin_max_search=1000,
):
    args = (
        fitness_name,
        num_args,
        time_limit,
        random_seed,
        init_low,
        init_high,
        max_iterations,
        basin_max_search,
    )

    result = _hcc_worker_test3(args)

    os.makedirs(output_dir, exist_ok=True)

    tag = f"{fitness_name}_{num_args}d"
    ind_csv = os.path.join(output_dir, f"{tag}_ind.csv")
    res_csv = os.path.join(output_dir, f"{tag}_result.csv")
    plot_png = os.path.join(output_dir, f"{tag}.png")

    # Save ind.csv
    with open(ind_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_ind", "best_fitness"])
        for gen, pt, ft in result["history"]:
            writer.writerow([gen, list(pt), ft])

    # Save result.csv
    with open(res_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "convergence_speed",
                "nfe",
                "best_fitness",
                "best_solution",
                "success",
                "num_trials",
                "generations",
                "total_time",
                "time_to_solution",
            ]
        )
        writer.writerow(
            [
                result["generations"],
                result["nfe"],
                result["best_fitness"],
                list(result["best_ind"]),
                result["success"],
                result["nfe"],
                result["generations"],
                result["total_time"],
                result["time_to_solution"],
            ]
        )
    # Get value range for plotting
    initial_low = init_low
    initial_high = init_high
    # Plot trajectory with compression regions
    fitness_fn = fitness_map[fitness_name]
    plot_fitness_landscape(
        fitness_fn, 
        result["history"], 
        fitness_name, 
        num_args, 
        value_range=(initial_low, initial_high), 
        save_path=plot_png,
        compression_manager=result.get("compression_manager", None)
    )
    return result


# ============================================================
# Run all
# ============================================================
def run_directory_test3_hcc(
    fitness_list=("needle", "rugged", "plateau", "combined"),
    dims=(1, 2),
    time_limit=20.0,
    initial_low=-150,
    initial_high=150,
    max_iterations=10,
    basin_max_search=1000,
    random_seed=42,
):
    output_base = f"benchmark_log_test3_hcc_test"

    for fname in fitness_list:
        for d in dims:
            output_dir = f"{output_base}/{fname}_{d}D"
            print(f"\n🚀 Running HC+C on {fname} ({d}D)")
            result = run_parallel_test3_hcc(
                fitness_name=fname,
                num_args=d,
                output_dir=output_dir,
                time_limit=time_limit,
                random_seed=random_seed,
                init_low=initial_low,
                init_high=initial_high,
                max_iterations=max_iterations,
                basin_max_search=basin_max_search,
            )
            print(
                f"   → Done. Success={result['success']}, Best Fit={result['best_fitness']}"
            )


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hill Climb + Compression (Test3)")

    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--initial-low", type=int, default=-150)
    parser.add_argument("--initial-high", type=int, default=150)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--basin-max-search", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--dims", type=int, nargs="+", default=[1, 2])
    # parser.add_argument(
    #     "--fitness",
    #     type=str,
    #     nargs="+",
    #     default=["needle", "rugged", "plateau", "combined"],
    # )
    parser.add_argument("--dims", type=int, nargs="+", default=[2])
    parser.add_argument(
        "--fitness",
        type=str,
        nargs="+",
        default=["rugged"],
    )

    args = parser.parse_args()

    run_directory_test3_hcc(
        fitness_list=args.fitness,
        dims=args.dims,
        time_limit=args.time_limit,
        initial_low=args.initial_low,
        initial_high=args.initial_high,
        max_iterations=args.max_iterations,
        basin_max_search=args.basin_max_search,
        random_seed=args.seed,
    )
