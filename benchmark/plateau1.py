def plateau1(x: int, y: int, z: int) -> int:
    v = x + 2*y - z

    if 0 <= x <= 10 and 5 <= y <= 15 and 2 <= z <= 8:
        v = 50
    elif 10 < x <= 20 and 15 < y <= 25 and 8 < z <= 15:
        v = 75
    else:
        v = x * y - z * z

    if v == 50:
        if (x + y + z) % 3 == 0:
            v += 1
        else:
            v -= 1

    return v
