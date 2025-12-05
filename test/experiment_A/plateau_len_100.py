
def plateau_100(x: int):
    # Target is at 0
    # Plateau range: [0, 100] returns constant cost
    # Outside: gradient exists
    
    target = 0
    dist = abs(x - target)
    
    # Plateau Region: If distance is within [1, length], return flat fitness
    if 0 < dist <= 100:
        return 100.0  # Perfectly Flat
    elif dist == 0:
        return 0.0    # Success
    else:
        # Gradient outside the plateau
        return 100.0 + (dist - 100)
