from module.sbst_core import instrument_and_load, FitnessCalculator, hill_climbing_search
import ast
import random

from hill_climb_multiD import hill_climb_with_compression_nd_code, CompressionManagerND

# file_path = "./benchmark/ex5.py" # path to target
file_path = "./benchmark/collatz_step.py" # path to target
# file_path = "./benchmark/HJJ/rugged_case.py" # path to target
# file_path = "./benchmark/count_divisor_2.py" # path to target
# file_path = "./benchmark/HJJ/needle_case.py" # path to target

source = open(file_path).read()
namespace, traveler, record, instrumented_tree = instrument_and_load(source)

# 하이퍼파라미터
MAX_TRIALS_PER_BRANCH = 20      # 브랜치당 최대 시도 횟수
SUCCESS_THRESHOLD = 0.0         # 이 값 이하이면 "성공"으로 간주
INITIAL_LOW, INITIAL_HIGH = -100000, 10000

# fitness calculator 준비
fitness_calc = FitnessCalculator(traveler, record, namespace)
parent_map = traveler.parent_map

results = []
random.seed(42)

# Iterate over ALL functions in the file
for func_info in traveler.functions:
    func_name = func_info.name
    func_args = func_info.args
    func_dims = func_info.args_dim

    INITIAL_LOW, INITIAL_HIGH = func_info.min_const, func_info.max_const
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

        best_result_for_branch = None  # 이 브랜치에서 가장 좋은 결과
        branch_success = False
        
        # ✅ Create ONE CompressionManagerND per branch to reuse metadata across trials
        branch_cm = CompressionManagerND(dim, steepness=5.0)
        print(f"\n📦 Created CompressionManagerND for branch {lineno} (will be reused across all trials)\n")

        for trial in range(MAX_TRIALS_PER_BRANCH):
            print("\n" + "-" * 60)
            print(f"[lineno={lineno}] Trial {trial+1}/{MAX_TRIALS_PER_BRANCH}")

            # (선택) 재현성을 위해 seed를 브랜치/트라이얼마다 다르게 고정
            # random.seed(42 + lineno * 1000 + trial)

            # 1) 랜덤 초기 해
            initial = [random.randint(INITIAL_LOW, INITIAL_HIGH) for _ in func_args]

            init_fit = fitness_calc.fitness_for_candidate(
                func_obj, initial,
                target_branch_node, target_outcome,
                subject_node, parent_map
            )
            print(f"[lineno={lineno}][trial={trial}] initial fitness: {init_fit} for {initial}")

            # 2) 압축 Hill-climb 실행 (with reused compression manager)
            traj, cm = hill_climb_with_compression_nd_code(
                fitness_calc, func_obj,
                target_branch_node, target_outcome,
                subject_node, parent_map,
                initial,
                dim,
                max_iterations=100,
                basin_max_search=100000,
                global_min_threshold=1e-6,
                cm=branch_cm  # ✅ Pass the branch-level compression manager to reuse metadata
            )

            # 3) 마지막 상태 추출
            final_point, final_f, used_comp = traj[-1]

            print(f"===== WITH COMPRESSION (lineno={lineno}, trial={trial}) =====")
            print(f"End:   {final_point}, f={final_f:.6g}")
            print(f"Trajectory length: {len(traj)}")
            print(f"Used compression in last step? {used_comp}")
            print("=============================================")

            # 4) 이 trial의 결과 구조화
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

            # 5) 브랜치 내 최적 해 갱신
            if best_result_for_branch is None or final_f < best_result_for_branch["final_fitness"]:
                best_result_for_branch = trial_result

            # 6) 성공 여부 체크: fitness가 threshold 이하이면 브랜치 탐색 종료
            if final_f <= SUCCESS_THRESHOLD:
                print(f"🎉 Branch lineno={lineno} succeeded at trial {trial} with f={final_f:.6g}")
                branch_success = True
                break

        # 브랜치별 요약 출력
        total_compressions = sum(len(branch_cm.dim_compressions[d]) for d in range(dim))
        print("\n" + "=" * 80)
        print(">>> SUMMARY for branch lineno={}: success={}, best_f={:.6g}, best_x={}".format(
            lineno,
            branch_success,
            best_result_for_branch["final_fitness"] if best_result_for_branch else float("inf"),
            best_result_for_branch["final_point"] if best_result_for_branch else None
        ))
        print(f">>> Total metadata compressions accumulated: {total_compressions}")
        print("=" * 80)

# 전체 요약
print("\n===== GLOBAL SUMMARY OVER ALL BRANCHES & TRIALS =====")
for r in results:
    print(
        f"func={r['function']}, line={r['lineno']}, trial={r['trial']}, outcome={r['target_outcome']}: "
        f"init_f={r['initial_fitness']:.3g}, "
        f"final_f={r['final_fitness']:.3g}, "
        f"steps={r['steps']}, "
        f"final_x={r['final_point']}"
    )
