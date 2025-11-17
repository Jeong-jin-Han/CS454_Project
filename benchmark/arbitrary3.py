def f(x: int, y: int, z: int) -> int:
    v = x * y - z * z

    if v > 10:
        v += 3
        if x == y:
            v += 10
    elif v > 0:
        v -= 5
    else:
        v = -v

    if (x > 2 and z < 2) or (y == 3 and x != 1):
        v += 7
    else:
        v -= 4

    return v
