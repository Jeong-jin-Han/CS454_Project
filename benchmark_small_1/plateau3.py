# test_plateau_lock.py

SECRET_CODE = [10, 20, 30, 40]

# 1. [Target Code] 실제 테스트 대상 로직
# SBST 프레임워크가 4개의 정수 인자를 개별적으로 전달하도록 인터페이스를 변경
def unlock_door(d0: int, d1: int, d2: int, d3: int) -> bool:
    # 각 자릿수가 정확히 일치해야만 열림
    if d0 == SECRET_CODE[0]:
        if d1 == SECRET_CODE[1]:
            if d2 == SECRET_CODE[2]:
                if d3 == SECRET_CODE[3]:
                    return True  # Target Branch
    return False

# 2. [Fitness Function] SBSE 관점의 거리 계산
# SBST가 4개의 정수 인자를 그대로 사용할 수 있도록 시그니처를 맞춤
def fitness_digital_lock(d0: int, d1: int, d2: int, d3: int) -> int:
    # Branch Distance Strategy: Hamming Distance
    # 값이 "얼마나 가까운가(abs)"는 무시하고 "맞았나(0)/틀렸나(1)"만 계산하여
    # 의도적으로 Plateau 지형을 형성함 (Gradient 소실 시뮬레이션)

    cost = 0

    # 4개의 중첩된 if문을 통과해야 함 (Approach Level)
    # 여기서는 순서 상관없이 틀린 개수 당 1000점 페널티로 단순화
    if d0 != SECRET_CODE[0]:
        cost += 1000
    if d1 != SECRET_CODE[1]:
        cost += 1000
    if d2 != SECRET_CODE[2]:
        cost += 1000
    if d3 != SECRET_CODE[3]:
        cost += 1000

    return cost

# FITNESS_FUNC = fitness_digital_lock
# TEST_CONFIG = {
#     'dim': 4,
#     'start_point': (0, 0, 0, 0),
#     'optimal_val': 0,
#     'threshold': 0,
#     'max_iterations': 100,
#     'basin_max_search': 500,
#     'max_steps_baseline': 10000
# }