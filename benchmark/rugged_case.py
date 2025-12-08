TARGET = 31415

def _scramble(x: int) -> int:
    z = x
    z ^= (z << 13)
    z ^= (z >> 17)
    z ^= (z << 5)
    return z

def rugged(x: int) -> int:
    d = abs(x - TARGET)

    base = d

    h = _scramble(x)
    noise = abs(h % 100)

    cost = base + noise

    if d <= 5:
        cost = d

    return cost
