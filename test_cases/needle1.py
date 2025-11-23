# test_needle_coupled.py

# 목표: x + y + z = 100 이면서 x * y * z = 32768 (예: 32, 32, 36 근처의 조합)
TARGET_SUM = 100
TARGET_PROD = 32768

def coupled_needle(point: tuple) -> int:
    x, y, z = point
    
    current_sum = x + y + z
    current_prod = x * y * z
    
    # [Complex Needle]
    # 두 조건(합, 곱)을 동시에 정확히 맞춰야 0을 반환.
    # 하나라도 틀리면 아주 큰 상수(Flat Penalty)를 반환하여 힌트를 주지 않음.
    
    if current_sum == TARGET_SUM and current_prod == TARGET_PROD:
        return 0
    else:
        # 힌트가 없음 (Gradient 0)
        # x를 1 늘렸을 때 합은 맞을 수 있어도 곱이 틀어지므로
        # 탐색기는 "여기도 아니네" 하고 제자리걸음 할 가능성이 높음
        return 10000

FITNESS_FUNC = coupled_needle
TEST_CONFIG = {
    'dim': 3,
    'start_point': (1, 1, 1),
    'optimal_val': 0,
    'threshold': 0,
    'max_iterations': 100,
    # 변수들이 얽혀있어서 단순 평지 압축으로 해결될지, 
    # 아니면 우연히 세 변수의 조합이 맞을 때까지 점프해야 할지 테스트
    'basin_max_search': 500, 
    'max_steps_baseline': 10000
}