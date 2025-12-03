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
    verbose=False,
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
            # DIAGONAL NEIGHBORS
            # ------------------------
            if len(active_dims) >= 2:
                for d1, d2 in itertools.combinations(active_dims, 2):

                    # time check
                    if time_limit and (time_module.time() - start_time) >= time_limit:
                        return traj, cm

                    # compression for d1
                    fixed1 = tuple(point[i] for i in range(dim) if i != d1)
                    comp1 = cm.get_system(d1, fixed1)
                    if comp1:
                        z1 = comp1.forward(point[d1])
                        n1_vals = [comp1.inverse(z1 - 1), comp1.inverse(z1 + 1)]
                    else:
                        n1_vals = [point[d1] - 1, point[d1] + 1]

                    # compression for d2
                    fixed2 = tuple(point[i] for i in range(dim) if i != d2)
                    comp2 = cm.get_system(d2, fixed2)
                    if comp2:
                        z2 = comp2.forward(point[d2])
                        n2_vals = [comp2.inverse(z2 - 1), comp2.inverse(z2 + 1)]
                    else:
                        n2_vals = [point[d2] - 1, point[d2] + 1]

                    for v1 in n1_vals:
                        for v2 in n2_vals:
                            cand = list(point)
                            cand[d1] = v1
                            cand[d2] = v2
                            cand_t = tuple(cand)

                            if time_limit and (time_module.time() - start_time) >= time_limit:
                                return traj, cm

                            cand_f = fitness_func_nd_code(cand_t)
                            candidates.append((cand_t, cand_f, [d1, d2]))

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
        # BASIN DETECTION
        # -----------------------------------------------------------
        if verbose:
            print(f"\n⚠️ Stuck at {point}, detecting basins…")

        basins = {}
        for d in active_dims:

            if time_limit and (time_module.time() - start_time) >= time_limit:
                return traj, cm

            basin = detect_basin_along_dimension(
                fitness_func_nd_code, point, d, basin_max_search
            )
            if basin:
                fixed = tuple(point[i] for i in range(dim) if i != d)
                cm.update_dimension(d, fixed, basin)
                basins[d] = basin

        if not basins:
            if verbose:
                print("No basins found → stopping")
            break

        # -----------------------------------------------------------
        # RESTART FROM BASIN BOUNDARY
        # -----------------------------------------------------------
        restart_candidates = []

        for d, (b_start, b_len) in basins.items():

            if time_limit and (time_module.time() - start_time) >= time_limit:
                return traj, cm

            b_end = b_start + b_len - 1

            # Left boundary
            left = list(point)
            left[d] = b_start - 1
            left_t = tuple(left)
            restart_candidates.append((left_t, fitness_func_nd_code(left_t)))

            # Right boundary
            right = list(point)
            right[d] = b_end + 1
            right_t = tuple(right)
            restart_candidates.append((right_t, fitness_func_nd_code(right_t)))

        restart_point, restart_f = min(restart_candidates, key=lambda t: t[1])

        if verbose:
            print(f"➡️ Restarting from {restart_point}, f={restart_f:.6g}")

        point, f = restart_point, restart_f
        traj.append((point, f, True))

        if abs(f) < global_min_threshold:
            if verbose:
                print("🎉 Restart hit goal")
            break

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
        fitness_fn=fitness_rugged,
        start_point=(5, -20),
        dim=2,
        time_limit=5.0,
        start_time=time.time(),
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
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        lo, hi = value_range
        X = np.linspace(lo, hi, 100)
        Y = np.linspace(lo, hi, 100)
        Xg, Yg = np.meshgrid(X, Y)

        Z = np.zeros_like(Xg)
        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):
                Z[i, j] = fitness_fn((int(Xg[i, j]), int(Yg[i, j])))

        ax.plot_surface(Xg, Yg, Z, cmap="viridis", alpha=0.7)

        traj_x = [pt[0] for (_, pt, _) in history]
        traj_y = [pt[1] for (_, pt, _) in history]
        traj_z = [ft for (_, pt, ft) in history]

        ax.plot(traj_x, traj_y, traj_z, color="red", marker="o")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("fitness")
        ax.set_title(f"HC+C Trajectory: {fitness_name} ({num_args}D)")

        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Saved 2D plot: {out_png}")

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
    # Plot trajectory
    # save path
    # plot_fitness_landscape(fitness_name, num_args, result["history"], plot_png, value_range=(initial_low, initial_high))
    fitness_fn = fitness_map[fitness_name]
    plot_fitness_landscape(fitness_fn, result["history"], fitness_name, num_args, value_range=(initial_low, initial_high), save_path=plot_png)
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
    parser.add_argument("--dims", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--fitness",
        type=str,
        nargs="+",
        default=["needle", "rugged", "plateau", "combined"],
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
