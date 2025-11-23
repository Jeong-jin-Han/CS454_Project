# Waffle Plateau Test
# Global Min at (50, 50) = 0.0

def fitness_1d_plateau(x, local_min_x, global_min_x, fitness_level):
    if 5 <= x <= 15: return fitness_level
    if 45 <= x <= 55: return 0.0
    if x < 30: return fitness_level + 0.1 * (x - local_min_x)**2
    else: return 0.0 + 0.1 * (x - global_min_x)**2

def test1_func(point):
    x0, x1 = int(point[0]), int(point[1])
    f0 = fitness_1d_plateau(x0, 10, 50, 10.0)
    f1 = fitness_1d_plateau(x1, 20, 50, 20.0)
    return f0 + f1

# [필수] 이 이름으로 정의해야 Runner가 인식함
FITNESS_FUNC = test1_func

# [필수] 설정값
TEST_CONFIG = {
    'dim': 2,
    'start_point': (10, 20),
    'optimal_val': 0.0,
    'threshold': 1.0,
    'max_iterations': 10,
    'basin_max_search': 30,
    'max_steps_baseline': 2000
}