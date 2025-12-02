import ast, os, random
from typing import Tuple
from module.sbst_core import instrument_and_load, FitnessCalculator
import ast, random


def get_random(
    num_args: int,
    value_range: Tuple[int, int],
) -> Tuple[int, ...]:
    # Get random arguments
    return tuple(
        random.randint(value_range[0], value_range[1]) for _ in range(num_args)
    )


fail = 0


def ga(
    fitness_calc: FitnessCalculator,
    func_info,
    func_obj,
    target_branch_node,
    target_outcome: bool,
    subject_node,
    parent_map,
    # hyperparameters
    pop_size: int = 500,
    max_gen: int = 500,
    tournament_k: int = 3,
    elite_ratio: float = 0.1,
    gene_mut_p: float = None,  # If None, set to 1/num_args
    ensure_mutation: bool = True,
    mutation_step_choices: Tuple[int, ...] = (-3, -2, -1, 1, 2, 3),
    rng: random.Random = random.Random(),
    # biased initialization parameters
    use_biased_init: bool = False,
    var_constants: dict = None,
    total_constants: list = None,
):
    value_range = (func_info.min_const, func_info.max_const)
    num_args = func_info.args_dim
    func_args = func_info.args
    
    # Initialize biased init parameters if not provided
    if var_constants is None:
        var_constants = {}
    if total_constants is None:
        total_constants = []
    
    # --- Helper functions ---
    if gene_mut_p is None:
        gene_mut_p = 1.0 / max(1, num_args)

    def clip(v: int) -> int:
        return max(value_range[0], min(value_range[1], v))

    def init_individual() -> Tuple[int, ...]:
        """Initialize individual with optional biased initialization."""
        if not use_biased_init:
            # Pure random initialization
            return tuple(rng.randint(*value_range) for _ in range(num_args))
        
        # Biased initialization: sample near extracted constants
        result = []
        for i in range(num_args):
            arg_name = func_args[i] if i < len(func_args) else f"arg{i}"
            
            # If no constants available, fall back to uniform
            if not total_constants and not var_constants:
                result.append(rng.randint(*value_range))
                continue
            
            # 20% uniform, 80% biased around constants
            if rng.random() < 0.2:
                result.append(rng.randint(*value_range))
                continue
            
            # Prefer per-variable constants if available
            const_list = list(var_constants.get(arg_name, []))
            if not const_list:
                const_list = total_constants
            if not const_list:
                result.append(rng.randint(*value_range))
                continue
            
            # Sample near a randomly chosen constant
            center = rng.choice(const_list)
            span = max(1, value_range[1] - value_range[0])
            sigma = max(1, int(0.01 * span))  # 1% of span
            
            # Sample from Gaussian and clip to range
            val = int(rng.gauss(center, sigma))
            val = clip(val)
            result.append(val)
        
        return tuple(result)

    def selection(pop, fits, k=tournament_k):
        idxs = rng.sample(range(len(pop)), k)
        best_idx = min(idxs, key=lambda i: fits[i][0])
        return pop[best_idx]

    def crossover(p1: Tuple[int, ...], p2: Tuple[int, ...]) -> Tuple[int, ...]:
        # uniform crossover
        return tuple(p1[i] if rng.random() < 0.5 else p2[i] for i in range(num_args))

    def mutate(ind: Tuple[int, ...]) -> Tuple[int, ...]:
        # integer perturbation; per-gene mutation with probability gene_mut_p
        vec = list(ind)
        any_flip = False
        for i in range(num_args):
            if rng.random() < gene_mut_p:
                vec[i] = clip(vec[i] + rng.choice(mutation_step_choices))
                any_flip = True
        if ensure_mutation and not any_flip:
            i = rng.randrange(num_args)
            vec[i] = clip(vec[i] + rng.choice(mutation_step_choices))
        return tuple(vec)

    def dedup(pop):
        seen, out = set(), []
        for ind in pop:
            if ind not in seen:
                out.append(ind)
                seen.add(ind)
        return out

    # Initialization
    population = [init_individual() for _ in range(pop_size)]

    # optional: show one record entry
    if fitness_calc._record.get_trace():
        k = fitness_calc._record.get_trace()[-1]

    # Evaluate initial population with early stopping
    fit_cache = []
    for ind in population:
        fitness = fitness_calc.fitness_for_candidate(
            func_obj,
            ind,
            target_branch_node,
            target_outcome,
            subject_node,
            parent_map,
        )
        fit_cache.append((fitness, ind))
        
        # ✅ Early stop if solution found during initialization
        if fitness == 0.0:
            print(f"[GA] ✅ Solution found during initialization after {len(fit_cache)} evaluations!")
            best_ind, best_fit = ind, 0.0
            return best_ind, best_fit
    
    print(
        f"[GA] Starting GA for target branch at line {target_branch_node.lineno if hasattr(target_branch_node, 'lineno') else 'N/A'} with population size {pop_size}, max generations {max_gen}"
    )
    
    # Initialize best from initial population
    best_fitness_init = min(fit for fit, _ in fit_cache)
    best_ind_init = [ind for fit, ind in fit_cache if fit == best_fitness_init][0]
    print(f"[GA] Initial best fitness: {best_fitness_init:.4f}")
    best_ind, best_fit = best_ind_init, best_fitness_init
    for gen in range(max_gen):

        ranked = sorted(zip(population, fit_cache), key=lambda x: x[1][0])
        elites_n = max(1, int(pop_size * elite_ratio))
        elites = [ind for ind, _ in ranked[:elites_n]]

        # Check if solution already found from previous generation
        if ranked[0][1][0] == 0.0:
            best_ind, best_fit = ranked[0][0], 0.0
            print(f"[GA] ✅ Solution found! Stopping at generation {gen}")
            break

        # Next generation creation
        next_gen = elites[:]
        while len(next_gen) < pop_size:
            p1 = selection(population, fit_cache)
            p2 = selection(population, fit_cache)
            child = mutate(crossover(p1, p2))
            next_gen.append(child)
        next_gen = dedup(next_gen)
        # Next generation evaluation with early stopping
        population = next_gen
        fit_cache = []
        
        for ind in population:
            fitness = fitness_calc.fitness_for_candidate(
                func_obj,
                ind,
                target_branch_node,
                target_outcome,
                subject_node,
                parent_map,
            )
            fit_cache.append((fitness, ind))
            
            # Update best solution
            if abs(fitness) < abs(best_fit):
                best_fit = fitness
                best_ind = ind
            
            # ✅ Early stop if solution found during generation evaluation
            if fitness == 0.0:
                print(f"[GA] ✅ Solution found in Gen {gen} after {len(fit_cache)} evaluations in this generation!")
                print(f"[GA] Total evaluations so far: {fitness_calc.evals}")
                return best_ind, best_fit

        print(
            f"[GA] End of Gen {gen}, best ind: {best_ind}, best fitness: {best_fit:.4f}"
        )

        if gen == max_gen - 1:
            print(
                f"[GA] Reached max generations ({max_gen}) without finding optimal solution."
            )
            global fail
            fail += 1
    # Return best solution even if target is not covered; targets remain as is
    return best_ind, best_fit


# file -> instrument AST -> ga_search_all -> output tests
def main(target_path: str):
    # 0. Make test case output directory outside the target file directory
    target_dir = os.path.dirname(target_path)
    project_root = os.path.dirname(target_dir)
    TESTS_DIR = os.path.join(project_root, "generated_inputs")
    if not os.path.exists(TESTS_DIR):
        os.makedirs(TESTS_DIR)

    # 1. Read code
    source = open(target_path).read()

    # 2. Parse & instrument
    tree = ast.parse(source)
    namespace, traveler, record, instrumented_tree = instrument_and_load(source)
    fitness_calc = FitnessCalculator(traveler, record, namespace)
    print(f"[+] Instrumentation complete for {target_path}")

    # 3. Extract function definitions
    func_infos = [fn for fn in traveler.functions]
    print(f"[+] Found {len(func_infos)} function(s) in the target file.")

    # 4. For each function, find branch leaves and run GA search
    for func_info in func_infos:
        print(f"\n=== Processing function: {func_info.name} ===")
        tests = {}
        func_name = func_info.name
        func_obj = namespace[func_name]

        # Iterate over all branch nodes in the function
        branches = traveler.branches.get(func_name, {})
        parent_map = traveler.parent_map

        total = len(branches) * 2
        found_count = 0
        print(
            f"[+] Branches: {list(branches.items())}, total targets to cover (T/F): {total}"
        )
        for target_lineno, branch_info in branches.items():
            target_branch_node = branch_info.node
            subject_node = branch_info.subject  # Match

            test_found = ga(
                fitness_calc,
                func_info,
                func_obj,
                target_branch_node,
                True,
                subject_node,
                parent_map,
            )
            test_key = (target_lineno, target_branch_node, True)
            tests[test_key] = test_found
            print(f"Test found for target {test_key}: args={test_found}")
            found_count += 1
            print(f"Checked {found_count} / {total} targets so far.")

            test_found = ga(
                fitness_calc,
                func_info,
                func_obj,
                target_branch_node,
                False,
                subject_node,
                parent_map,
            )
            test_key = (target_lineno, target_branch_node, False)
            tests[test_key] = test_found
            print(f"Test found for target {test_key}: args={test_found}")
            found_count += 1
            print(f"Checked {found_count} / {total} targets so far.")

        print("All tests checked for function", func_name)
        print(f"Failed {fail} targets.")
        print("Tests:", tests)

        # 4-2. Write test suite to file
        target_module = os.path.basename(target_path).removesuffix(".py")
        test_file_path = os.path.join(TESTS_DIR, f"test_ga_{func_name}.py")
        lines = [f"from benchmark import {target_module}", ""]

        for idx, test in enumerate(tests.items(), start=1):
            (lineno, target_branch_node, desired), (args_tuple, fit) = test

            args_str = ", ".join(repr(a) for a in args_tuple)
            lines.append(f"def test_{func_name}_{idx}():")
            lines.append(f"    {target_module}.{func_name}({args_str})")
            lines.append("")

        with open(test_file_path, "w") as f:
            f.write("\n".join(lines))

        print("Test suite generated in", test_file_path)
        print("Input range:", (func_info.min_const, func_info.max_const))

    print(f"[GA] Number of failures: {fail}")
