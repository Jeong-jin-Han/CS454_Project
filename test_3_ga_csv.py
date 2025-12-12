
"""
GA Testing for Arbitrary Fitness Landscapes (Test 3 Project)

- Supports random / biased initialization
- Time-limit continuous evolution
- Input parameters:
    - Fitness functions: needle, rugged, plateau, combined
    - Dimensions: 1D, 2D
    - Initialization: biased / random
    - Population size
    - Time limit

- Output folder:
      benchmark_log_test3_ga_biased_test/
      benchmark_log_test3_ga_random_test/
        - Saves CSV logs and plots
            - Individual trajectory (ind.csv)
            - Summary results (result.csv)
            - Plots fitness landscape (1D / 2D)

Fitness functions imported from benchmark/test_3/fitness.py
"""

import os
import csv
import time
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from benchmark.test_3.fitness import (
    fitness_needle,
    fitness_rugged,
    fitness_plateau,
    fitness_combined,
)

from test_3_plot import plot_fitness_landscape

# -------------------------------------------------------------------
# Fitness mapping
# -------------------------------------------------------------------
FITNESS_MAP = {
    "needle": fitness_needle,
    "rugged": fitness_rugged,
    "plateau": fitness_plateau,
    "combined": fitness_combined,
}


# -------------------------------------------------------------------
# GA Implementation for test3
# -------------------------------------------------------------------
def ga_test3(
    fitness_fn,
    num_args,
    pop_size=10000,
    mutation_p=None,
    mutation_step_choices=(-3, -2, -1, 1, 2, 3),
    max_gen=10000,
    time_limit=None,
    start_time=None,
    value_range=(-150, 150),
    rng=None,
):
    """
    GA with strict time limit:
    - Checks time *before every fitness evaluation*
    - Stops immediately when time limit is exceeded.
    """

    if rng is None:
        rng = random.Random()

    lo, hi = value_range
    dim = num_args

    if mutation_p is None:
        mutation_p = 1.0 / max(1, dim)

    def init_individual():
        return tuple(rng.randint(lo, hi) for _ in range(dim))

    # ============================================================
    # INITIAL POPULATION
    # ============================================================
    population = []
    fits = []

    for _ in range(pop_size):
        if time_limit and (time.time() - start_time) >= time_limit:
            # No individuals? -> produce empty result
            if not population:
                return None, float("inf"), []
            break

        ind = init_individual()
        f = float(fitness_fn(ind))

        population.append(ind)
        fits.append(f)

    def best_of_population():
        idx, best_fit = min(list(enumerate(fits)), key=lambda x: x[1])
        return population[idx], best_fit

    best_ind, best_fit = best_of_population()
    history = [(0, best_ind, best_fit)]

    # ============================================================
    # Operators
    # ============================================================
    def crossover(p1, p2):
        return tuple(p1[i] if rng.random() < 0.5 else p2[i] for i in range(dim))

    def mutate(ind):
        vec = list(ind)
        mutated = False
        for i in range(dim):
            if rng.random() < mutation_p:
                vec[i] = max(lo, min(hi, vec[i] + rng.choice(mutation_step_choices)))
                mutated = True

        if not mutated:
            j = rng.randrange(dim)
            vec[j] = max(lo, min(hi, vec[j] + rng.choice(mutation_step_choices)))

        return tuple(vec)

    def selection():
        i1, i2 = rng.randrange(len(population)), rng.randrange(len(population))
        return population[i1] if fits[i1] < fits[i2] else population[i2]

    # ============================================================
    # GA MAIN LOOP (strict time-check)
    # ============================================================
    gen = 1
    while gen <= max_gen:

        if time_limit and (time.time() - start_time) >= time_limit:
            break

        new_population = []
        new_fits = []

        # --------------------------
        # Evaluate each child strictly
        # --------------------------
        for _ in range(pop_size):
            if time_limit and (time.time() - start_time) >= time_limit:
                break

            p1, p2 = selection(), selection()
            child = crossover(p1, p2)
            child = mutate(child)

            # STRICT TIME CHECK before evaluation
            if time_limit and (time.time() - start_time) >= time_limit:
                break

            child_fit = float(fitness_fn(child))

            new_population.append(child)
            new_fits.append(child_fit)

        if not new_population:
            break

        population = new_population
        fits = new_fits

        best_ind, best_fit = best_of_population()
        history.append((gen, best_ind, best_fit))

        if best_fit == 0.0:
            break

        gen += 1

    return best_ind, best_fit, history


# -------------------------------------------------------------------
# Worker function (like _ga_worker)
# -------------------------------------------------------------------
def _ga_worker_test3(args):
    (
        fitness_name,
        num_args,
        time_limit,
        random_seed,
        pop_size,
    ) = args

    rng = random.Random(random_seed)
    fitness_fn = FITNESS_MAP[fitness_name]

    start_time = time.time()

    best_ind, best_fit, history = ga_test3(
        fitness_fn=fitness_fn,
        num_args=num_args,
        pop_size=pop_size,
        time_limit=time_limit,
        start_time=start_time,
        rng=rng,
    )

    total_time = time.time() - start_time
    nfe = len(history) * pop_size  # approx evaluations
    generations = len(history) - 1
    success = best_fit == 0.0

    time_to_solution = None
    if success:
        for gen, ind, fit in history:
            if fit == 0.0:
                time_to_solution = gen
                break

    return {
        "history": history,
        "best_ind": best_ind,
        "best_fitness": best_fit,
        "generations": generations,
        "nfe": nfe,
        "success": success,
        "total_time": total_time,
        "time_to_solution": time_to_solution,
    }


# -------------------------------------------------------------------
# Run single GA test and save CSV
# -------------------------------------------------------------------
def run_parallel_test3_ga(
    fitness_name,
    num_args,
    output_dir,
    time_limit=10.0,
    pop_size=100,
    random_seed=42,
    value_range=(-150, 150),
):
    from benchmark.test_3.fitness import (
        fitness_needle,
        fitness_rugged,
        fitness_plateau,
        fitness_combined,
    )

    fitness_map = {
        "needle": fitness_needle,
        "rugged": fitness_rugged,
        "plateau": fitness_plateau,
        "combined": fitness_combined,
    }

    fitness_fn = fitness_map[fitness_name]
    constants = list(range(-5, 6))  # constants for biased init

    # ============================================================
    # Run GA using worker
    # ============================================================
    result = _ga_worker_test3(
        (
            fitness_name,
            num_args,
            time_limit,
            random_seed,
            pop_size,
        )
    )

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Filenames
    # ----------------------------------------------------------
    tag = f"{fitness_name}_{num_args}d"

    ind_csv = os.path.join(output_dir, f"{tag}_ind.csv")
    res_csv = os.path.join(output_dir, f"{tag}_result.csv")
    plot_png = os.path.join(output_dir, f"{tag}.png")

    # ----------------------------------------------------------
    # Save ind.csv
    # ----------------------------------------------------------
    with open(ind_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_ind", "best_fitness"])
        for gen, ind, fit in result["history"]:
            writer.writerow([gen, list(ind), fit])

    # ----------------------------------------------------------
    # Save result.csv
    # ----------------------------------------------------------
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

    # ============================================================
    # Plotting (fitness landscape + trajectory)
    # ============================================================
    plot_fitness_landscape(
        fitness_fn=fitness_fn,
        history=result["history"],
        fitness_name=fitness_name,
        num_args=num_args,
        value_range=value_range,
        save_path=plot_png,
    )

    return result


def run_directory_test3_ga(
    fitness_list, dims, time_limit=10.0, pop_size=100, random_seed=42
):
    output_base = f"benchmark_log_test3_ga_test"

    for fitness_name in fitness_list:
        for num_args in dims:

            output_dir = f"{output_base}/{fitness_name}_{num_args}D"
            result = run_parallel_test3_ga(
                fitness_name,
                num_args,
                output_dir,
                time_limit,
                pop_size,
                random_seed,
            )


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time-limit",
        type=float,
        default=20.0,
    )
    parser.add_argument("--pop", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dims", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--fitness",
        type=str,
        nargs="+",
        default=["needle", "rugged", "plateau", "combined"],
    )

    args = parser.parse_args()

    run_directory_test3_ga(
        fitness_list=args.fitness,
        dims=args.dims,
        time_limit=args.time_limit,
        pop_size=args.pop,
        random_seed=args.seed,
    )
