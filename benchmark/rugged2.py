# test_rugged_checksum.py

TARGET_HASH = 0xCAFEBABE

# 1. [Target Code] 실제 테스트 대상 로직
def verify_packet(packet_id: int) -> bool:
    # 입력값 변조 로직 (Checksum 계산)
    temp = packet_id
    temp = (temp ^ (temp << 13)) & 0xFFFFFFFF
    temp = (temp ^ (temp >> 17)) & 0xFFFFFFFF
    temp = (temp ^ (temp << 5))  & 0xFFFFFFFF
    
    # 조건문: 계산된 해시가 타겟과 일치하는가?
    if temp == TARGET_HASH:
        return True # Target Branch
    else:
        return False

# 2. [Fitness Function] 거리 계산
def fitness_check_integrity(point: tuple) -> int:
    packet_id = point[0]
    
    # 로직 내부의 상태를 시뮬레이션하여 거리 계산
    temp = packet_id
    temp = (temp ^ (temp << 13)) & 0xFFFFFFFF
    temp = (temp ^ (temp >> 17)) & 0xFFFFFFFF
    temp = (temp ^ (temp << 5))  & 0xFFFFFFFF
    
    # Branch Distance: abs(LHS - RHS)
    # 비트 연산 때문에 입력값이 조금만 변해도 거리가 요동침 (Rugged)
    distance = abs(temp - TARGET_HASH)
    
    return distance

FITNESS_FUNC = fitness_check_integrity
TEST_CONFIG = {
    'dim': 1,
    'start_point': (12345,), 
    'optimal_val': 0,
    'threshold': 0,
    'max_iterations': 50,
    'basin_max_search': 100,
    'max_steps_baseline': 5000
}