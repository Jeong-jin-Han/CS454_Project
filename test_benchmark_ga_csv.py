#!/usr/bin/env python3
"""
GA-based parallel branch testing following the same structure as:
- run_parallel_test_with_csv
- run_directory_test
- main()

Outputs to benchmark_log_ga/<file>.csv
"""

import os
import csv
import time
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path

from module.sbst_core import instrument_and_load, FitnessCalculator
from BASE.ga import ga


# ------------------------------------------------------------
# 1) Per-branch GA evaluation worker
# ------------------------------------------------------------
def _ga_worker(args):
    (
        file_path,
        func_name,
        lineno,
        branch_info,
        max_trials,
        success_threshold,
        pop_size,
        max_gen,
        tournament_k,
        elite_ratio,
        gene_mut_p,
        mutation_step_choices,
        ensure_mutation,
        seed_offset,
    ) = args

    result = {
        "function": func_name,
        "lineno": lineno,
        "convergence_speed": 0,
        "nfe": 0,
        "best_fitness": float("inf"),
        "best_solution": None,
        "success": False,
        "num_trials": 0,
        "error": None,
        "pid": os.getpid(),
    }

    try:
        source = open(file_path, "r", encoding="utf-8").read()
        namespace, traveler, record, _ = instrument_and_load(source)

        fitness_calc = FitnessCalculator(traveler, record, namespace)
        fitness_calc.evals = 0

        func_obj = namespace[func_name]
        parent_map = traveler.parent_map
        func_info = [f for f in traveler.functions if f.name == func_name][0]

        best_fitness = float("inf")
        best_solution = None
        total_nfe = 0
        total_generations = 0
        success = False

        target_branch_node = branch_info.node
        subject_node = branch_info.subject
        target_outcome = True

        for trial in range(max_trials):
            seed = seed_offset + lineno * 1000 + trial
            rng = random.Random(seed)
            random.seed(seed)

            nfe_before = fitness_calc.evals

            ind, fit = ga(
                fitness_calc=fitness_calc,
                func_info=func_info,
                func_obj=func_obj,
                target_branch_node=target_branch_node,
                target_outcome=target_outcome,
                subject_node=subject_node,
                parent_map=parent_map,
                pop_size=pop_size,
                max_gen=max_gen,
                tournament_k=tournament_k,
                elite_ratio=elite_ratio,
                gene_mut_p=gene_mut_p,
                ensure_mutation=ensure_mutation,
                mutation_step_choices=mutation_step_choices,
                rng=rng,
            )

            nfe_after = fitness_calc.evals
            nfe_this = nfe_after - nfe_before

            total_nfe += nfe_this
            gens_est = int(round(nfe_this / float(max(1, pop_size))))
            total_generations += gens_est

            if fit is not None and fit < best_fitness:
                best_fitness = fit
                best_solution = ind

            result["num_trials"] += 1

            if fit is not None and fit <= success_threshold:
                success = True
                break

        result["convergence_speed"] = total_generations
        result["nfe"] = total_nfe
        result["best_fitness"] = best_fitness
        result["best_solution"] = best_solution
        result["success"] = success

    except Exception as e:
        result["error"] = str(e)

    return result


# ------------------------------------------------------------
# 2) Run GA on a single file → write CSV
# ------------------------------------------------------------
def run_parallel_test_with_csv(
    file_path: str,
    output_csv: str,
    max_trials_per_branch: int = 5,
    success_threshold: float = 0.0,
    pop_size: int = 10000,
    max_gen: int = 50,
    tournament_k: int = 3,
    elite_ratio: float = 0.1,
    gene_mut_p=None,
    mutation_step_choices=(-3, -2, -1, 1, 2, 3),
    ensure_mutation=True,
    num_workers=None,
    seed_offset: int = 0,
):
    print(f"\n[GA] Analyzing file: {file_path}")

    source = open(file_path, "r", encoding="utf-8").read()
    namespace, traveler, record, _ = instrument_and_load(source)

    tasks = []
    for func_info in traveler.functions:
        func_name = func_info.name
        branches = traveler.branches.get(func_name, {})

        for lineno, branch_info in branches.items():
            tasks.append(
                (
                    file_path,
                    func_name,
                    lineno,
                    branch_info,
                    max_trials_per_branch,
                    success_threshold,
                    pop_size,
                    max_gen,
                    tournament_k,
                    elite_ratio,
                    gene_mut_p,
                    mutation_step_choices,
                    ensure_mutation,
                    seed_offset,
                )
            )

    if not tasks:
        print("  (No branches found)")
        return []

    if num_workers is None:
        num_workers = cpu_count()

    with Pool(processes=num_workers) as pool:
        results = pool.map(_ga_worker, tasks)

    # ----- Write CSV -----
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=[
                "function",
                "lineno",
                "convergence_speed",
                "nfe",
                "best_fitness",
                "best_solution",
                "success",
                "num_trials",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "function": r["function"],
                    "lineno": r["lineno"],
                    "convergence_speed": r["convergence_speed"],
                    "nfe": r["nfe"],
                    "best_fitness": r["best_fitness"],
                    "best_solution": str(r["best_solution"]),
                    "success": r["success"],
                    "num_trials": r["num_trials"],
                    "error": r["error"],
                }
            )

    print(f"  → CSV saved: {output_csv}")
    return results


# ------------------------------------------------------------
# 3) Run GA on every Python file in a directory
# ------------------------------------------------------------
def run_directory_test(directory: str):
    directory = Path(directory)
    assert directory.exists(), f"{directory} does not exist."

    output_dir = Path("benchmark_log_ga")
    output_dir.mkdir(exist_ok=True)

    py_files = sorted([p for p in directory.rglob("*.py")])

    print(f"\n[GA] Running for directory: {directory}")
    print(f"  Found {len(py_files)} files.\n")

    for f in py_files:
        out_csv = output_dir / (f.stem + ".csv")
        run_parallel_test_with_csv(
            file_path=str(f),
            output_csv=str(out_csv),
        )


# ------------------------------------------------------------
# 4) CLI entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import multiprocessing

    if "fork" in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method("fork", force=True)
    else:
        multiprocessing.set_start_method("spawn", force=True)

    run_directory_test("./benchmark")
