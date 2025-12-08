SX = 5000
SY = -1000
SZ = 42

def _scramble_y(y: int) -> int:
    z = y
    z ^= (z << 7)
    z ^= (z >> 9)
    z ^= (z << 8)
    return z

def mixed(x: int, y: int, z: int) -> int:

    dx = abs(x - SX)
    dy = abs(y - SY)
    d = dx + dy

    if d > 5000:
        cost_xy = 1000
    elif d > 50:
        cost_xy = 500
    elif 10 <= d <= 20:
        cost_xy = 100
    elif 1 <= d < 10:
        cost_xy = 400
    else:
        cost_xy = 0

    hy = _scramble_y(y)
    noise_y = abs(hy % 20)
    cost_xy += noise_y

    dz = abs(z - SZ)
    if dz == 0:
        cost_z = 0
    elif dz <= 3:
        cost_z = dz * 50
    else:
        cost_z = 10000

    total_cost = cost_xy + cost_z
    return total_cost
