# test_plateau_calendar.py

# 목표: "수요일"이면서 "14시"인 시간대를 찾아라 (Unix Timestamp 기준)
# 1일 = 86400초, 1시간 = 3600초

def calendar_scheduler(timestamp_tuple: tuple) -> int:
    timestamp = timestamp_tuple[0] # 입력은 초 단위 정수
    
    # 1. Day Check (Plateau 크기: 86400)
    # 1970.1.1(목) 기준 -> timestamp // 86400 은 "일수"
    days_passed = timestamp // 86400
    # 요일 계산 (0:목, 1:금, ... 6:수) -> 목표는 나머지 6
    weekday = days_passed % 7
    
    # 2. Hour Check (Plateau 크기: 3600)
    # 해당 날짜 안에서의 시간
    seconds_in_day = timestamp % 86400
    hour = seconds_in_day // 3600
    
    fitness = 0
    
    # 요일이 틀리면 페널티 (수요일이 목표)
    if weekday != 6:
        fitness += 10000
    
    # 시간이 틀리면 페널티 (14시가 목표)
    if hour != 14:
        fitness += abs(14 - hour) * 100
        
    # [설명]
    # timestamp가 1초 변해도 요일, 시간은 안 변함.
    # 최소 3600초가 변해야 시간 로직이 반응하고,
    # 86400초가 변해야 요일 로직이 반응함.
    return fitness

FITNESS_FUNC = calendar_scheduler
TEST_CONFIG = {
    'dim': 1,
    'start_point': (0,), # 1970년 1월 1일 0시
    'optimal_val': 0,
    'threshold': 0,
    'max_iterations': 50,
    # 하루(86400초)라는 거대한 평지를 압축할 수 있는지 테스트
    # basin_max_search가 매우 커야 하거나, 
    # 알고리즘이 반복적으로 basin을 확장하며 탐색해야 함
    'basin_max_search': 100000, 
    'max_steps_baseline': 10000
}