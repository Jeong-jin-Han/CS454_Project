#!/usr/bin/env python3
import os
import csv
import time
import random
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import itertools

from benchmark.test_3.fitness import (
    fitness_needle,
    fitness_rugged,
    fitness_plateau,
    fitness_combined,
)

from compression_hc import CompressionManagerND, detect_basin_along_dimension
from test_3_plot import plot_fitness_landscape


# ===============
# N-D Hill Climb
# ===============
# def hill_climb_with_compression_nd_code(
#     fitness_fn,  # <---- 단순 callable((tuple))->float
#     start_point,
#     dim,
#     max_iterations=10,
#     basin_max_search=1000,
#     global_min_threshold=1e-6,
#     verbose=False,
#     cm=None,  # Optional: compression manager reuse
#     time_limit=None,
#     start_time=None,
# ):
#     """
#     N-D hill climbing with axis-aligned 1D compressions (Test3 version)

#     This version:
#       - DOES NOT use FitnessCalculator / branches / func_obj
#       - fitness_fn(point) must return a non-negative scalar
#       - Supports strict time-limit enforcement before every eval
#     """

#     # Wrap fitness to ensure integer and consistent calling
#     def fitness_func_nd_code(x):
#         return float(fitness_fn(tuple(int(v) for v in x)))

#     # For dimension deactivation
#     deactivation_patience = 20

#     traj = []

#     # ---------------------------
#     # CompressionManager setup
#     # ---------------------------
#     if cm is None:
#         cm = CompressionManagerND(dim, steepness=5.0)
#         if verbose:
#             print("📦 Created NEW CompressionManagerND for this search")
#     else:
#         if verbose:
#             print("♻️ REUSING existing CompressionManagerND with accumulated metadata")

#     active_dims = list(range(dim))
#     dim_stagnation = {d: 0 for d in range(dim)}

#     # -------------------------------------------------------
#     # Time check BEFORE initial evaluation
#     # -------------------------------------------------------
#     if time_limit is not None and start_time is not None:
#         if time.time() - start_time >= time_limit:
#             print("⏱️ Time limit reached before start → stop")
#             point = tuple(int(x) for x in start_point)
#             traj.append((point, float("inf"), False))
#             return traj, cm

#     # Initialize point
#     point = tuple(int(x) for x in start_point)
#     f = fitness_func_nd_code(point)
#     traj.append((point, f, False))

#     if verbose:
#         print(f"\n🚀 {dim}D hill climbing start at {point}, f={f:.6g}\n")

#     # Early success
#     if abs(f) < global_min_threshold:
#         if verbose:
#             print("🎉 INITIAL POINT IS ALREADY A GOAL")
#         return traj, cm

#     # ============================================================
#     # MAIN ITERATIONS
#     # ============================================================
#     for it in range(max_iterations):
#         # print(f"it={it}, point={point}, f={f}, active_dims={active_dims}")
#         if not active_dims:
#             if verbose:
#                 print("All dimensions deactivated. Stopping.")
#             return traj, cm

#         if verbose:
#             print("=" * 80)
#             print(f"🔄 Iteration {it+1}/{max_iterations}")
#             print("=" * 80)

#         # Check early
#         if abs(f) < global_min_threshold:
#             if verbose:
#                 print("🎉 SUCCESS at iteration start")
#             break

#         step_count = 0
#         max_steps_per_iteration = 10000

#         # -----------------------------------------------------------
#         # STEEPEST DESCENT LOOP
#         # -----------------------------------------------------------
#         while step_count < max_steps_per_iteration:

#             # TIME CHECK
#             if time_limit is not None and start_time is not None:
#                 if time.time() - start_time >= time_limit:
#                     if verbose:
#                         print("⏱️ Time limit reached inside climbing loop → stop")
#                     return traj, cm

#             best_point = point
#             best_f = f
#             candidates = []
#             meaningful_dims = set()

#             # ------------------------
#             # AXIS NEIGHBORS
#             # ------------------------
#             for d in active_dims:

#                 # time check
#                 if time_limit is not None and start_time is not None:
#                     if time.time() - start_time >= time_limit:
#                         return traj, cm

#                 fixed = tuple(point[i] for i in range(dim) if i != d)
#                 comp_sys = cm.get_system(d, fixed)

#                 if comp_sys:
#                     z = comp_sys.forward(point[d])
#                     neigh_vals = [comp_sys.inverse(z - 1), comp_sys.inverse(z + 1)]
#                 else:
#                     neigh_vals = [point[d] - 1, point[d] + 1]

#                 for val in neigh_vals:
#                     cand = list(point)
#                     cand[d] = val
#                     cand_t = tuple(cand)

#                     # strict time check BEFORE eval
#                     if time_limit and (time.time() - start_time) >= time_limit:
#                         return traj, cm

#                     cand_f = fitness_func_nd_code(cand_t)
#                     candidates.append((cand_t, cand_f, [d]))

#             # ------------------------
#             # DIAGONAL NEIGHBORS
#             # ------------------------
#             if len(active_dims) >= 2:
#                 for d1, d2 in itertools.combinations(active_dims, 2):

#                     # time check
#                     if time_limit and (time.time() - start_time) >= time_limit:
#                         return traj, cm

#                     # compression for d1
#                     fixed1 = tuple(point[i] for i in range(dim) if i != d1)
#                     comp1 = cm.get_system(d1, fixed1)
#                     if comp1:
#                         z1 = comp1.forward(point[d1])
#                         n1_vals = [comp1.inverse(z1 - 1), comp1.inverse(z1 + 1)]
#                     else:
#                         n1_vals = [point[d1] - 1, point[d1] + 1]

#                     # compression for d2
#                     fixed2 = tuple(point[i] for i in range(dim) if i != d2)
#                     comp2 = cm.get_system(d2, fixed2)
#                     if comp2:
#                         z2 = comp2.forward(point[d2])
#                         n2_vals = [comp2.inverse(z2 - 1), comp2.inverse(z2 + 1)]
#                     else:
#                         n2_vals = [point[d2] - 1, point[d2] + 1]

#                     for v1 in n1_vals:
#                         for v2 in n2_vals:
#                             cand = list(point)
#                             cand[d1] = v1
#                             cand[d2] = v2
#                             cand_t = tuple(cand)

#                             if time_limit and (time.time() - start_time) >= time_limit:
#                                 return traj, cm

#                             cand_f = fitness_func_nd_code(cand_t)
#                             candidates.append((cand_t, cand_f, [d1, d2]))

#             # ------------------------
#             # Select steepest descent
#             # ------------------------
#             for cand_point, cand_f, modified in candidates:
#                 if cand_f < best_f:
#                     best_point, best_f = cand_point, cand_f
#                 if cand_f != f:
#                     for d in modified:
#                         meaningful_dims.add(d)

#             # Update stagnation
#             for d in range(dim):
#                 if d not in meaningful_dims:
#                     dim_stagnation[d] += 1
#                 else:
#                     dim_stagnation[d] = 0

#             # Remove stagnant dims
#             for d in list(active_dims):
#                 if dim_stagnation[d] >= deactivation_patience:
#                     active_dims.remove(d)
#                     if verbose:
#                         print(f"Deactivating dim {d} due to stagnation")

#             # If no improvement → stop climb
#             if best_f < f:
#                 point, f = best_point, best_f
#                 used_comp = any(
#                     cm.get_system(d, tuple(point[i] for i in range(dim) if i != d))
#                     is not None
#                     for d in range(dim)
#                 )
#                 traj.append((point, f, used_comp))
#                 step_count += 1
#             else:
#                 if verbose:
#                     print(f"📍 Stuck after {step_count} steps at {point}, f={f:.6g}")
#                 # print(
#                 #     "📍 Stuck at {}, f={:.6g} after {} steps".format(
#                 #         point, f, step_count
#                 #     )
#                 # )
#                 break

#         # After full climbing iteration:
#         if abs(f) < global_min_threshold:
#             if verbose:
#                 print("🎉 SUCCESS after climbing")
#             # print("🎉 SUCCESS")
#             break

#         # -----------------------------------------------------------
#         # BASIN DETECTION
#         # -----------------------------------------------------------
#         if verbose:
#             print(f"\n⚠️ Stuck at {point}, detecting basins…")

#         basins = {}
#         for d in active_dims:
#             if time_limit and (time.time() - start_time) >= time_limit:
#                 return traj, cm

#             basin = detect_basin_along_dimension(
#                 fitness_func_nd_code, point, d, basin_max_search
#             )
#             if basin:
#                 fixed = tuple(point[i] for i in range(dim) if i != d)
#                 cm.update_dimension(d, fixed, basin)
#                 basins[d] = basin

#         if not basins:
#             if verbose:
#                 print("No basins found → stopping")
#             # print("❌ NO BASINS FOUND → STOP")
#             break

#         # -----------------------------------------------------------
#         # RESTART FROM BASIN BOUNDARY
#         # -----------------------------------------------------------
#         restart_candidates = []

#         for d, (b_start, b_len) in basins.items():

#             if time_limit and (time.time() - start_time) >= time_limit:
#                 return traj, cm

#             b_end = b_start + b_len - 1

#             # Left boundary
#             left = list(point)
#             left[d] = b_start - 1
#             left_t = tuple(left)
#             restart_candidates.append((left_t, fitness_func_nd_code(left_t)))

#             # Right boundary
#             right = list(point)
#             right[d] = b_end + 1
#             right_t = tuple(right)
#             restart_candidates.append((right_t, fitness_func_nd_code(right_t)))

#         restart_point, restart_f = min(restart_candidates, key=lambda t: t[1])

#         if verbose:
#             print(f"➡️ Restarting from {restart_point}, f={restart_f:.6g}")

#         point, f = restart_point, restart_f
#         traj.append((point, f, True))

#         if abs(f) < global_min_threshold:
#             if verbose:
#                 print("🎉 Restart hit goal")
#             # print("🎉 SUCCESS after restart")
#             break
#     return traj, cm


def hill_climb_simple_nd_code(
    fitness_fn,  # <---- 단순 callable((tuple))->float
    start_point,
    dim,
    max_steps=2000,
    time_limit=None,
    start_time=None,
):
    point = tuple(int(x) for x in start_point)

    def fitness_func_nd_code(x):
        return float(fitness_fn(tuple(int(v) for v in x)))

    # ⏱️ Check time before initial evaluation
    if time_limit is not None and start_time is not None:
        if time.time() - start_time >= time_limit:
            return [(point, float("inf"))]  # Return immediately if time exceeded

    f = fitness_func_nd_code(point)
    traj = [(point, f)]

    for _ in range(max_steps):
        # ⏱️ Check time limit before evaluating neighbors
        if time_limit is not None and start_time is not None:
            if time.time() - start_time >= time_limit:
                return traj  # Return best found so far

        # Try 2*dim neighbors (±1 in each dimension)
        candidates = []
        for d in range(dim):
            # ⏱️ Check time before each evaluation
            if time_limit is not None and start_time is not None:
                if time.time() - start_time >= time_limit:
                    return traj

            # -1 in dimension d
            neighbor = list(point)
            neighbor[d] -= 1
            candidates.append((tuple(neighbor), fitness_func_nd_code(tuple(neighbor))))

            # ⏱️ Check time before next evaluation
            if time_limit is not None and start_time is not None:
                if time.time() - start_time >= time_limit:
                    return traj

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
        
        if time.time() - start_time >= time_limit:
            return traj  # Return if time limit exceeded

    return traj


# ============================================================
# Fitness mapping
# ============================================================
fitness_map = {
    "needle": fitness_needle,
    "rugged": fitness_rugged,
    "plateau": fitness_plateau,
    "combined": fitness_combined,
}


def test3_single_fitness_with_metrics(
    fitness_fn,
    num_args,
    time_limit,
    random_seed,
    init_low,
    init_high,
    max_steps,
    basin_max_search,
    success_threshold=0.0,
):
    start_time = time.time()
    random.seed(random_seed)

    total_steps = 0
    best_fitness = float("inf")
    best_solution = None
    success = False
    time_to_solution = None

    history = []
    trial = 0

    while True:

        # 전체 timeout 체크
        if time.time() - start_time >= time_limit:
            break

        if success:
            break

        initial = tuple(random.randint(init_low, init_high) for _ in range(num_args))

        try:
            traj = hill_climb_simple_nd_code(
                fitness_fn=fitness_fn,
                start_point=initial,
                dim=num_args,
                max_steps=max_steps,
                time_limit=time_limit,  # ✔ 전체 제한 전달
                start_time=start_time,  # ✔ 전체 시작 시간 전달
            )
        except Exception as e:
            print(f"Error during hill climbing: {e}")
            break

        history += [
            (trial, gen_idx, pt, fitness_value)
            for gen_idx, (pt, fitness_value) in enumerate(traj)
        ]

        final_point, final_f = traj[-1]
        total_steps += len(traj)

        if final_f < best_fitness:
            best_fitness = final_f
            best_solution = list(final_point)

        if final_f <= success_threshold:
            time_to_solution = time.time() - start_time
            success = True

        trial += 1

    total_time = time.time() - start_time

    return {
        "history": history,
        "convergence_speed": total_steps,
        "nfe": total_steps,
        "best_fitness": best_fitness,
        "best_solution": best_solution,
        "success": success,
        "num_trials_run": trial,
        "total_time": total_time,
        "time_to_solution": time_to_solution,
    }


# ============================================================
# Run all
# ============================================================
def run_directory_test3_hc(
    fitness_list=("needle", "rugged", "plateau", "combined"),
    dims=(1, 2),
    time_limit=20.0,
    initial_low=-150,
    initial_high=150,
    max_steps=2000,
    basin_max_search=1000,
    random_seed=42,
):
    output_base = "benchmark_log_test3_hc_test"

    for fname in fitness_list:
        for d in dims:
            print(f"\n=== Test3: fitness={fname}, dim={d}D ===")

            fitness_fn = fitness_map[fname]
            output_dir = f"{output_base}/{fname}_{d}D"
            os.makedirs(output_dir, exist_ok=True)

            # ----- Run the multi-trial evaluation -----
            result = test3_single_fitness_with_metrics(
                fitness_fn=fitness_fn,
                num_args=d,
                time_limit=time_limit,
                random_seed=random_seed,
                init_low=initial_low,
                init_high=initial_high,
                max_steps=max_steps,
                basin_max_search=basin_max_search,
                success_threshold=0.0,
            )
            
            print(f"Hill Climb completed in {result['total_time']:.2f}s over {result['num_trials_run']} trials.")

            # ----- Save summary CSV -----
            tag = f"{fname}_{d}d"
            ind_csv = os.path.join(output_dir, f"{tag}_ind.csv")

            with open(ind_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "trial_id",
                        "generation_index",
                        "point",
                        "fitness_value",
                    ]
                )
                for record in result["history"]:
                    writer.writerow(
                        [
                            record[0],
                            record[1],
                            record[2],
                            record[3],
                        ]
                    )

            res_csv = os.path.join(output_dir, f"{tag}_result.csv")

            with open(res_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "convergence_speed",
                        "nfe",
                        "best_fitness",
                        "best_solution",
                        "success",
                        "num_trials_run",
                        "total_time",
                        "time_to_solution",
                    ]
                )
                writer.writerow(
                    [
                        result["convergence_speed"],
                        result["nfe"],
                        result["best_fitness"],
                        result["best_solution"],
                        result["success"],
                        result["num_trials_run"],
                        f"{result['total_time']:.6f}",
                        result["time_to_solution"],
                    ]
                )

            # ----- Plot -----
            plot_path = os.path.join(output_dir, f"{tag}.png")
            plot_fitness_landscape(
                fitness_fn,
                result["history"],
                fname,
                d,
                value_range=(initial_low, initial_high),
                save_path=plot_path,
                is_hcc=False,
                is_hc=True,
            )

            print(
                f" → Done. Success={result['success']}, Best Fit={result['best_fitness']}"
            )


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hill Climb (Test3)")

    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--initial-low", type=int, default=-150)
    parser.add_argument("--initial-high", type=int, default=150)
    parser.add_argument("--max-steps", type=int, default=2000)
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

    run_directory_test3_hc(
        fitness_list=args.fitness,
        dims=args.dims,
        time_limit=args.time_limit,
        initial_low=args.initial_low,
        initial_high=args.initial_high,
        max_steps=args.max_steps,
        basin_max_search=args.basin_max_search,
        random_seed=args.seed,
    )
