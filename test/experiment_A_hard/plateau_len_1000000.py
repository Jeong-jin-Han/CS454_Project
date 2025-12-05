
def plateau_1000000(x: int):
    # Target: 0
    # Trap (Plateau): [1, 1000000] (Cost = 50)
    # Wall (Penalty): x < 0 or x > 1000000 (Cost = 1000 + dist)
    
    target = 0
    
    if x != target:
        return 1000.0 + abs(x)
    elif 1 <= x <= 1000000:
        return 50.0  # Perfect Flat Trap
    else:
        # Plateau를 벗어나면 오히려 페널티를 주어 가둬버림
        return 0.0
