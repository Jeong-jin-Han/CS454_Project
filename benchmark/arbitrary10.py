def f(a: int, b: int, c: int) -> int:
    s = a + b + c

    if 3 <= s <= 5:
        s += 20
        if a == 1 and b == 1:
            s += 5
        if c == 3:
            s -= 7
    else:
        if s < 3:
            s -= 10
        else:
            s += 10

    return s
