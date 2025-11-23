# test_needle_coupled.py

TARGET_SUM = 100
TARGET_PROD = 32768

# 1. [Target Code] 시스템 상태 검증
def verify_system_state(point: tuple) -> bool:
    x, y, z = point
    
    # 세 변수의 합과 곱이 동시에 특정 값을 만족해야 함
    if (x + y + z == TARGET_SUM) and (x * y * z == TARGET_PROD):
        return True # Target Branch
    return False

# 2. [Fitness Function]
def fitness_coupled(point: tuple) -> int:
    x, y, z = point
    
    current_sum = x + y + z
    current_prod = x * y * z
    
    # Branch Distance Calculation
    dist_sum = abs(current_sum - TARGET_SUM)
    dist_prod = abs(current_prod - TARGET_PROD)
    
    # Needle 효과 강화:
    # 둘 다 정확히 맞지 않으면, 거리 정보를 매우 약하게 주거나 증폭시켜서
    # "하나만 맞추는 것"이 의미 없게 만듦.
    if dist_sum == 0 and dist_prod == 0:
        return 0
    else:
        # 힌트가 거의 없는 고원(High Plateau) 형성
        return 10000 + dist_sum + dist_prod

FITNESS_FUNC = fitness_coupled
TEST_CONFIG = {
    'dim': 3,
    'start_point': (1, 1, 1),
    'optimal_val': 0,
    'threshold': 0,
    'max_iterations': 100,
    'basin_max_search': 500, 
    'max_steps_baseline': 10000
}