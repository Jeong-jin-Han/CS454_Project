# test_rugged_checksum.py

# 목표: 단순 비교가 아니라, 비트 연산된 결과가 특정 값과 일치해야 함
TARGET_HASH = 0xCAFEBABE

def check_integrity(packet_id: int) -> int:
    """
    [Target Branch]
    if (scrambled_id == TARGET_HASH):
        return 0  # Success
    """
    
    # 입력값을 변형하는 로직 (해시 함수 흉내)
    # 이 로직 때문에 x가 1만 변해도 결과값(diff)은 요동칩니다.
    temp = packet_id
    temp = (temp ^ (temp << 13)) & 0xFFFFFFFF
    temp = (temp ^ (temp >> 17)) & 0xFFFFFFFF
    temp = (temp ^ (temp << 5))  & 0xFFFFFFFF
    
    # Branch Distance 계산: 변형된 값과 타겟 값의 차이
    # SBSE에서 'x == y' 조건의 거리는 abs(x - y)로 정의됨
    distance = abs(temp - TARGET_HASH)
    
    return distance

FITNESS_FUNC = check_integrity
TEST_CONFIG = {
    'dim': 1,
    'start_point': (12345,), 
    'optimal_val': 0,
    'threshold': 0,
    'max_iterations': 50,
    'basin_max_search': 100,  # 해시 충돌 느낌의 Basin 탐색 필요
    'max_steps_baseline': 5000
}