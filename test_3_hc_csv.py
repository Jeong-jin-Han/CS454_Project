
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


def hill_climb_simple_nd_code(
    fitness_fn,
    start_point,
    dim,
    max_steps=2000,
    time_limit=None,
    start_time=None,
):
    point = tuple(int(x) for x in start_point)

    def fitness_func_nd_code(x):
        return float(fitness_fn(tuple(int(v) for v in x)))

    # Check time before initial evaluation
    if time_limit is not None and start_time is not None:
        if time.time() - start_time >= time_limit:
            return [(point, float("inf"))]  # Return immediately if time exceeded

    f = fitness_func_nd_code(point)
    traj = [(point, f)]

    for _ in range(max_steps):
        # Check time limit before evaluating neighbors
        if time_limit is not None and start_time is not None:
            if time.time() - start_time >= time_limit:
                return traj  # Return best found so far

        # Try 2*dim neighbors (±1 in each dimension)
        candidates = []
        for d in range(dim):
            # Check time before each evaluation
            if time_limit is not None and start_time is not None:
                if time.time() - start_time >= time_limit:
                    return traj

            # -1 in dimension d
            neighbor = list(point)
            neighbor[d] -= 1
            candidates.append((tuple(neighbor), fitness_func_nd_code(tuple(neighbor))))

            # Check time before next evaluation
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
                time_limit=time_limit,
                start_time=start_time,
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

            print(
                f"Hill Climb completed in {result['total_time']:.2f}s over {result['num_trials_run']} trials."
            )

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
                f" -> Done. Success={result['success']}, Best Fit={result['best_fitness']}"
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
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--basin-max-search", type=int, default=100)
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
