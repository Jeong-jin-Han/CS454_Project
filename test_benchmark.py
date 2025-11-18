from module.sbst_core import instrument_and_load, FitnessCalculator, hill_climbing_search
import ast
import random

from hill_climb_multiD import hill_climb_with_compression_nd_code

# file_path = "./benchmark/ex2.py" # path to target
file_path = "./benchmark/collatz_step.py" # path to target

source = open(file_path).read()
namespace, traveler, record, instrumented_tree = instrument_and_load(source)

# 1) 타겟 함수 정보 가져오기
func_info = traveler.functions[0]   # 예: 첫 번째 함수만 사용
func_name = func_info.name
func_args = func_info.args
func_dims = func_info.args_dim
func_obj = namespace[func_name]

print("Target function:", func_name)
print(func_info)

# 2) 브랜치 목록 가져오기
branches = traveler.branches.get(func_name, {})
print("Branches (by lineno):", list(branches.keys()))

# 3) fitness calculator 준비
fitness_calc = FitnessCalculator(traveler, record, namespace)

parent_map = traveler.parent_map
dim = len(func_args)

# ----------------------------------------------------------------------
# 모든 브랜치에 대해 압축 힐 클라이밍 수행
# ----------------------------------------------------------------------
results = []  # 나중에 CSV나 요약용으로 쓸 수 있게 저장

for lineno, branch_info in branches.items():
    print("\n" + "=" * 80)
    print(f"🔎 Branch at lineno={lineno}")
    print(branch_info)

    target_branch_node = branch_info.node
    subject_node = branch_info.subject

    # 원하는 outcome을 정합니다. (True: taken, False: not taken)
    # 필요하면 [True, False] 둘 다 돌리는 루프를 추가할 수 있습니다.
    target_outcome = True

    # 1) 랜덤 초기 해
    initial = [random.randint(-100000, 10000) for _ in func_args]

    init_fit = fitness_calc.fitness_for_candidate(
        func_obj, initial,
        target_branch_node, target_outcome,
        subject_node, parent_map
    )
    print(f"[lineno={lineno}] initial fitness: {init_fit} for {initial}")

    # 2) 압축 Hill-climb 실행
    traj, cm = hill_climb_with_compression_nd_code(
        fitness_calc, func_obj,
        target_branch_node, target_outcome,
        subject_node, parent_map,
        initial,
        dim,                 # 차원 수 = 인자 개수
        max_iterations=10,
        basin_max_search=100,
        global_min_threshold=1e-6
    )

    # 3) 마지막 상태 추출
    final_point, final_f, used_comp = traj[-1]

    print(f"\n===== WITH COMPRESSION (lineno={lineno}) =====")
    print(f"End:   {final_point}, f={final_f:.6g}")
    print(f"Trajectory length: {len(traj)}")
    print(f"Used compression in last step? {used_comp}")
    print("=============================================")

    # 4) 결과 저장
    results.append({
        "lineno": lineno,
        "target_outcome": target_outcome,
        "initial_point": initial,
        "initial_fitness": init_fit,
        "final_point": list(final_point),
        "final_fitness": final_f,
        "steps": len(traj),
    })

# 필요하면 여기서 results를 출력/로그/CSV 저장 등
print("\n===== SUMMARY OVER ALL BRANCHES =====")
for r in results:
    print(
        f"line {r['lineno']}, outcome={r['target_outcome']}: "
        f"init_f={r['initial_fitness']:.3g}, "
        f"final_f={r['final_fitness']:.3g}, "
        f"steps={r['steps']}, "
        f"final_x={r['final_point']}"
    )
