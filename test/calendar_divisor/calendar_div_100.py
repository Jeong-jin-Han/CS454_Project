
def check_schedule(timestamp: int) -> bool:
    # 조건 1: 수요일인가? (86400초 단위 변화)
    # (timestamp // 86400) % 7 == 6
    if (timestamp // 86400) % 7 == 6:
        
        # 조건 2: 14시인가? (3600초 단위 변화)
        # (timestamp % 86400) // 3600 == 14
        if (timestamp % 86400) // 3600 == 14:
            return True # Target Branch
            
    return False
