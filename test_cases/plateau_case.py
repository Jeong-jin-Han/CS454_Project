# plateau_case.py

SX = 2000
SY = -500

def plateau_case(x: int, y: int) -> int:
    """
    타깃 브랜치:
        if x == SX and y == SY:
            # TARGET
    라고 가정.

    순수 hill climb이 쉽게 TARGET까지 못 가도록,
    고리 모양 plateau(local minimum)를 만든 버전입니다.
    """
    dx = abs(x - SX)
    dy = abs(y - SY)
    d = dx + dy  # Manhattan 거리

    # 1) 아주 멀리서는 완전 flat: cost = 1000
    if d > 5000:
        return 1000

    # 2) 중간 범위: 약간 더 좋은 plateau: cost = 500
    if d > 50:
        return 500

    # 3) d ∈ [10, 20] 구간: "함정 plateau" (local minimum ring)
    if 10 <= d <= 20:
        return 100

    # 4) d ∈ [1, 9] 구간: 오히려 더 나빠짐 (wall 구간)
    if 1 <= d < 10:
        return 400

    # 5) d == 0 : 진짜 TARGET
    return 0