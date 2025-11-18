# plateau_case.py

SX = 2000
SY = -500

def plateau(x: int, y: int) -> int:
    """
    예: target branch
        if x == SX and y == SY:
            # TARGET
    로 가정.
    branch distance는 아래 cost를 그대로 써도 되고, 변형해서 써도 됩니다.
    """
    dx = abs(x - SX)
    dy = abs(y - SY)
    d = dx + dy

    # 멀리서는 완전 flat: fitness = 300
    if d > 300:
        return 300

    # target 근처에서만 slope가 생김
    return d
