# test_plateau_calendar.py

# 1. [Target Code] 스케줄러 로직
def check_schedule(timestamp_tuple: tuple) -> bool:
    timestamp = timestamp_tuple[0]
    
    # 조건 1: 수요일인가? (86400초 단위 변화)
    # (timestamp // 86400) % 7 == 6
    if (timestamp // 86400) % 7 == 6:
        
        # 조건 2: 14시인가? (3600초 단위 변화)
        # (timestamp % 86400) // 3600 == 14
        if (timestamp % 86400) // 3600 == 14:
            return True # Target Branch
            
    return False

# 2. [Fitness Function] 거리 계산
def fitness_calendar(timestamp_tuple: tuple) -> int:
    timestamp = timestamp_tuple[0]
    fitness = 0
    
    # Logic 1: Day Check
    days_passed = timestamp // 86400
    weekday = days_passed % 7
    
    if weekday != 6:
        # Approach Level Penalty (10000) + Branch Distance
        # 요일이 틀리면 시간은 보지도 않으므로 큰 페널티
        fitness += 10000 + abs(6 - weekday) * 100
    else:
        # Logic 2: Hour Check (요일은 맞음, Approach Level 통과)
        seconds_in_day = timestamp % 86400
        hour = seconds_in_day // 3600
        
        if hour != 14:
            # Branch Distance for Hour
            fitness += abs(14 - hour) * 100
            
    return fitness

FITNESS_FUNC = fitness_calendar
TEST_CONFIG = {
    'dim': 1,
    'start_point': (0,), 
    'optimal_val': 0,
    'threshold': 0,
    'max_iterations': 50,
    'basin_max_search': 100000, # 86400초(하루)를 넘어야 함
    'max_steps_baseline': 10000
}