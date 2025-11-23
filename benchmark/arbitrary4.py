def f(a: int, b: int, c: int) -> int:
    t = a + b * 2 - c

    if t == 3:
        t += 50
    elif t == 4:
        t -= 40
    else:
        if a == b:
            t += 7
        if b == c:
            t -= 8

    return t
