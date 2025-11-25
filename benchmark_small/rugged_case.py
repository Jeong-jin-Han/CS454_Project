# rugged_case.py

TARGET = 31415

def _scramble(x: int) -> int:
    # 간단한 xorshift 스타일 비트 스크램블
    z = x
    z ^= (z << 13)
    z ^= (z >> 17)
    z ^= (z << 5)
    return z

def rugged(x: int) -> int:
    """
    목표 브랜치를 예를 들어:
        if x == TARGET:
            # TARGET
    로 잡고, branch distance를 아래 cost와 연동해서 쓸 수 있습니다.
    """
    d = abs(x - TARGET)

    # base distance
    base = d

    # hash noise: x가 1 증가해도 완전 다른 값이 나옴
    h = _scramble(x)
    noise = abs(h % 100)   # 0~99

    # distance + noise → rugged
    cost = base + noise

    # 진짜 타깃 근처(예: ±5)는 조금 완만하게 해서 global basin만 살짝 smooth
    if d <= 5:
        cost = d

    return cost
