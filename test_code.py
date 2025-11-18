from module.sbst_core import instrument_and_load, FitnessCalculator, hill_climbing_search
import ast
import random

from hill_climb_multiD import hill_climb_with_compression_nd_code

# file_path = "./benchmark/ex1.py" # path to target
file_path = "./benchmark/collatz_step.py" # path to target

source = open(file_path).read()   
namespace, traveler, record, instrumented_tree = instrument_and_load(source)

# Example: pick first function
func_info = traveler.functions[0]
func_name = func_info.name
func_args = func_info.args
func_dims = func_info.args_dim
func_obj = namespace[func_name]

print(func_info)

# list branches in the function
branches = traveler.branches.get(func_name, {})
print("Branches:", branches.keys())


# list branches in the function
branches = traveler.branches.get(func_name, {})
print("Branches:", branches.keys())

# Choose a branch lineno and an outcome to target (True: taken, False: not taken)
target_lineno = next(iter(branches.keys())) # e.g., Choose the first branch
branch_info = branches[2]
target_branch_node = branch_info.node
target_outcome = True   # e.g. try to make branch taken
subject_node = branch_info.subject
parent_map = traveler.parent_map

print(branch_info)


# create fitness calculator
fitness_calc = FitnessCalculator(traveler, record, namespace)

# make an initial candidate (random ints for each arg)
initial = [random.randint(-20, 20) for _ in func_args]

fitness = fitness_calc.fitness_for_candidate(func_obj, initial, target_branch_node, target_outcome, subject_node, parent_map)
print("initial fitness:", fitness, "for", initial)


traj, cm = hill_climb_with_compression_nd_code(
    # fitness_func_nd_code,
    fitness_calc, func_obj, target_branch_node, target_outcome, subject_node, parent_map,
    initial,
    len(func_args), # dim,
    max_iterations=10,
    basin_max_search=100,
    global_min_threshold=1e-6
)


base_final_point, base_final_f, _ = traj[-1]
print(f"\n===== WITH COMPRESSION =====")
print(f"End:   {base_final_point}, f={base_final_f:.6g}")
print(f"Trajectory length: {len(traj)}")