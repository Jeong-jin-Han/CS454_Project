import ast, os, random
from typing import Tuple
from module.sbst_core import (instrument_and_load, FitnessCalculator, hill_climbing_search)
import ast, random

def get_random(
    num_args: int,
    value_range: Tuple[int, int],
) -> Tuple[int, ...]:
    # Get random arguments
    return tuple(random.randint(value_range[0], value_range[1]) for _ in range(num_args))

# def ga_search_all(
#     tree: ast.AST,
#     func_name: str,
#     targets: List[Tuple[int, str, bool]],
#     num_args: int,
#     node_info: Dict[int, Tuple[int, str, Any, bool]],
#     **ga_kwargs,
# ):
#     """
#     Cover leaf targets sequentially using GA.
#     If other targets are covered in the process, record and remove them immediately.
#     Returns: tests = [(args_tuple, target_tuple), ...]
#     """
#     tests: List[Tuple[Tuple[int, ...], Tuple[int, str, bool]]] = []

#     while targets:
#         current_target = targets[0]
#         best_args, fit, targets, covered_cases = ga(
#             tree=tree,
#             func_name=func_name,
#             targets=targets.copy(),
#             target=current_target,
#             num_args=num_args,
#             node_info=node_info,
#             **ga_kwargs,
#         )

#         tests.append((best_args, current_target))
#         if current_target in targets:
#             # Preventing infinite loop: Even if not found, proceed to the next target
#             targets.remove(current_target)
#         if fit == 0.0:
#             print(f"[GA] Target {current_target} covered with args {best_args}, remaining targets: {len(targets)}")
#         else:
#             print(f"[GA] Target {current_target} NOT covered, Best fit={fit:.4f}, args={best_args}, remaining targets: {len(targets)}")
#         for args_tuple, covered_t in covered_cases:
#             tests.append((args_tuple, covered_t))
#         print(f"[GA] Covered {len(covered_cases)} additional targets in this run")
#         print(f"[GA] Remaining targets to cover: {len(targets)}")

#     return tests

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
    pop_size: int = 200,
    max_gen: int = 100,
    tournament_k: int = 3,
    elite_ratio: float = 0.1,
    gene_mut_p: float = None,   # If None, set to 1/num_args
    ensure_mutation: bool = True,
    mutation_step_choices: Tuple[int, ...] = (-3, -2, -1, 1, 2, 3),
    rng: random.Random = random.Random(),
):
    value_range = (func_info.min_const, func_info.max_const)
    num_args = func_info.args_dim
    # --- Helper functions ---
    if gene_mut_p is None:
        gene_mut_p = 1.0 / max(1, num_args)
        
    def clip(v: int) -> int:
        return max(value_range[0], min(value_range[1], v))

    def init_individual() -> Tuple[int, ...]:
        return tuple(rng.randint(*value_range) for _ in range(num_args))

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
        
    fit_cache = [(fitness_calc.fitness_for_candidate(func_obj, ind, target_branch_node, target_outcome, subject_node, parent_map), ind) for ind in population]
    print(f"[GA] Starting GA for target branch at line {target_branch_node.lineno if hasattr(target_branch_node, 'lineno') else 'N/A'} with population size {pop_size}, max generations {max_gen}")
    print(f"[GA] Initial best fitness: {min(fit for fit, _ in fit_cache):.4f}")
    best_ind, best_fit = None, float("inf")
    for gen in range(max_gen):
        
        ranked = sorted(zip(population, fit_cache), key=lambda x: x[1][0])
        elites_n = max(1, int(pop_size * elite_ratio))
        elites = [ind for ind, _ in ranked[:elites_n]]
        
        if ranked[0][1][0] == 0.0:
            best_ind, best_fit = ranked[0][0], 0.0
            break
        
        # Next generation creation
        next_gen = elites[:]
        while len(next_gen) < pop_size:
            p1 = selection(population, fit_cache)
            p2 = selection(population, fit_cache)
            child = mutate(crossover(p1, p2))
            next_gen.append(child)
        next_gen = dedup(next_gen)
        # Next generation evaluation
        population = next_gen
        fit_cache = [(fitness_calc.fitness_for_candidate(func_obj, ind, target_branch_node, target_outcome, subject_node, parent_map), ind) for ind in population]
        
        # Update best solution
        for ind, (fval, _rec) in zip(population, fit_cache):
            if abs(fval) < abs(best_fit):
                best_fit = fval
                best_ind = ind
                
        print(f"[GA] End of Gen {gen}, best ind: {best_ind}, best fitness: {best_fit:.4f}")
        
        # Early stopping: break loop if optimal solution found
        if best_fit == 0.0:
            break
        
        if gen == max_gen - 1:
            print(f"[GA] Reached max generations ({max_gen}) without finding optimal solution.")
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
        print(f"[+] Branches: {list(branches.items())}, total targets to cover (T/F): {total}")
        for target_lineno, branch_info in branches.items():
            target_branch_node = branch_info.node
            subject_node = branch_info.subject # Match
            
            test_found = ga(fitness_calc, func_info, func_obj, target_branch_node, True, subject_node, parent_map)
            test_key = (target_lineno, target_branch_node, True)
            tests[test_key] = test_found
            print(f"Test found for target {test_key}: args={test_found}")
            found_count += 1
            print(f"Checked {found_count} / {total} targets so far.")

            test_found = ga(fitness_calc, func_info, func_obj, target_branch_node, False, subject_node, parent_map)
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
        test_file_path = os.path.join(
                TESTS_DIR, f"test_ga_{func_name}.py"
        )
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