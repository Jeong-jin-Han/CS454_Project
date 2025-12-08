SX = 2000
SY = -500

def plateau_case(x: int, y: int) -> int:
    dx = abs(x - SX)
    dy = abs(y - SY)
    d = dx + dy

    if d > 5000:
        return 1000

    if d > 50:
        return 500

    if 10 <= d <= 20:
        return 100

    if 1 <= d < 10:
        return 400

    return 0
