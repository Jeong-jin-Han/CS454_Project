from module.sbst_core import instrument_and_load, FitnessCalculator
import random

from hill_climb_multiD import hill_climb_simple_nd_code

# file_path = "./benchmark/collatz_step.py"
file_path = "./benchmark/HJJ/needle_case.py" # path to target
file_path = "./benchmark/HJJ/rugged_case.py" # path to target
file_path = "./benchmark/HJJ/plateau_case.py" # path to target
file_path = "./benchmark/HJJ/mixed_case.py" # path to target

source = open(file_path).read()
namespace, traveler, record, instrumented_tree = instrument_and_load(source)

# 하이퍼파라미터
MAX_TRIALS_PER_BRANCH = 20
SUCCESS_THRESHOLD = 0.0
INITIAL_LOW, INITIAL_HIGH = -100000, 10000

# fitness calculator
fitness_calc = FitnessCalculator(traveler, record, namespace)
parent_map = traveler.parent_map

results = []
random.seed(42)

# Iterate over ALL functions in the file
for func_info in traveler.functions:
    func_name = func_info.name
    func_args = func_info.args
    func_dims = func_info.args_dim
    func_obj = namespace[func_name]
    
    print("\n" + "=" * 80)
    print(f"📝 Testing function: {func_name}")
    print(func_info)
    
    # Get branches for this function
    branches = traveler.branches.get(func_name, {})
    print(f"Branches (by lineno): {list(branches.keys())}")
    
    # Skip functions with no branches
    if not branches:
        print(f"⏭️  Skipping {func_name} (no branches to test)")
        continue
    
    dim = len(func_args)
    
    # Test each branch in this function
    for lineno, branch_info in branches.items():
        print("\n" + "=" * 80)
        print(f"🔎 Branch at lineno={lineno}")
        print(branch_info)

        target_branch_node = branch_info.node
        subject_node = branch_info.subject
        target_outcome = True

        best_result_for_branch = None
        branch_success = False

        for trial in range(MAX_TRIALS_PER_BRANCH):
            print("\n" + "-" * 60)
            print(f"[lineno={lineno}] Trial {trial+1}/{MAX_TRIALS_PER_BRANCH}")

            initial = [random.randint(INITIAL_LOW, INITIAL_HIGH) for _ in func_args]

            init_fit = fitness_calc.fitness_for_candidate(
                func_obj, initial,
                target_branch_node, target_outcome,
                subject_node, parent_map
            )
            print(f"[lineno={lineno}][trial={trial}] initial fitness: {init_fit} for {initial}")

            # 2) Hill-climb 실행 (NO compression baseline)
            traj = hill_climb_simple_nd_code(
                fitness_calc, func_obj,
                target_branch_node, target_outcome,
                subject_node, parent_map,
                initial,
                dim,
            )

            # 3) 마지막 상태
            final_point, final_f = traj[-1]

            print(f"===== WITHOUT COMPRESSION (lineno={lineno}, trial={trial}) =====")
            print(f"End:   {final_point}, f={final_f:.6g}")
            print(f"Trajectory length: {len(traj)}")
            print("=============================================")

            trial_result = {
                "function": func_name,
                "lineno": lineno,
                "trial": trial,
                "target_outcome": target_outcome,
                "initial_point": initial,
                "initial_fitness": init_fit,
                "final_point": list(final_point),
                "final_fitness": final_f,
                "steps": len(traj),
            }
            results.append(trial_result)

            if best_result_for_branch is None or final_f < best_result_for_branch["final_fitness"]:
                best_result_for_branch = trial_result

            if final_f <= SUCCESS_THRESHOLD:
                print(f"🎉 Branch lineno={lineno} succeeded at trial {trial} with f={final_f:.6g}")
                branch_success = True
                break

        print("\n" + "=" * 80)
        print(">>> SUMMARY for branch lineno={}: success={}, best_f={:.6g}, best_x={}".format(
            lineno,
            branch_success,
            best_result_for_branch["final_fitness"] if best_result_for_branch else float("inf"),
            best_result_for_branch["final_point"] if best_result_for_branch else None
        ))
        print("=" * 80)

print("\n===== GLOBAL SUMMARY OVER ALL BRANCHES & TRIALS =====")
for r in results:
    print(
        f"func={r['function']}, line={r['lineno']}, trial={r['trial']}, outcome={r['target_outcome']}: "
        f"init_f={r['initial_fitness']:.3g}, "
        f"final_f={r['final_fitness']:.3g}, "
        f"steps={r['steps']}, "
        f"final_x={r['final_point']}"
    )
