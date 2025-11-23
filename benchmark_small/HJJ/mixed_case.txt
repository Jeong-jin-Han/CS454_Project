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

    - (x, y): plateau_hard 패턴 + y축에 약한 rugged 노이즈
    - z     : needle-like penalty
    """

    # -------------------------------
    # (1) x, y에 대한 plateau_hard 기반 비용
    # -------------------------------
    dx = abs(x - SX)
    dy = abs(y - SY)
    d = dx + dy  # Manhattan 거리

    # 기본 plateau 구조 (plateau_hard와 동일한 형식)
    if d > 5000:
        cost_xy = 1000   # 바깥 큰 plateau
    elif d > 50:
        cost_xy = 500    # 그 안쪽 plateau
    elif 10 <= d <= 20:
        cost_xy = 100    # 고리형 local-minimum plateau (함정)
    elif 1 <= d < 10:
        cost_xy = 400    # 안쪽 벽: 오히려 더 나빠지는 구간
    else:  # d == 0
        cost_xy = 0      # 진짜 TARGET 지점

    # -------------------------------
    # (2) y 방향 rugged 노이즈 추가
    # -------------------------------
    # 노이즈 크기를 작게 잡아서, plateau_hard의 큰 구조는 유지하되
    # 세부적으로 울퉁불퉁하게 만들기
    hy = _scramble_y(y)
    noise_y = abs(hy % 20)   # 0~19
    cost_xy += noise_y

    # -------------------------------
    # (3) z에 대한 needle-like 비용
    # -------------------------------
    dz = abs(z - SZ)
    if dz == 0:
        cost_z = 0
    elif dz <= 3:
        cost_z = dz * 50   # 아주 좁은 basin
    else:
        cost_z = 10000     # 대부분 huge penalty

    # 최종 비용
    total_cost = cost_xy + cost_z
    return total_cost
