# mixed_case.py

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
    """
    target branch 예:
        if x == SX and y == SY and z == SZ:
            # TARGET

    branch distance는 아래 세 축 비용을 합친 cost로 정의할 수 있습니다.
    """

    # (1) x: plateau
    dx = abs(x - SX)
    if dx > 1000:
        cost_x = 1000   # 멀리서는 전부 같음 → plateau
    else:
        cost_x = dx     # 근처에서만 slope

    # (2) y: rugged
    dy = abs(y - SY)
    hy = _scramble_y(y)
    noise_y = abs(hy % 200)   # 0~199
    cost_y = dy + noise_y

    # (3) z: needle-like
    dz = abs(z - SZ)
    if dz == 0:
        cost_z = 0
    elif dz <= 3:
        cost_z = dz * 50   # 아주 좁은 basin
    else:
        cost_z = 10000     # 거의 어디서나 huge penalty

    total_cost = cost_x + cost_y + cost_z
    return total_cost
