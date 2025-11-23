def f(a: int, b: int) -> int:
    r = 0

    if a > 1:
        for i in range(a):
            r += i
        if b < 2:
            r += 10
        else:
            r -= 10
    else:
        while b > 0:
            r += b
            b -= 1

    if r % 2 == 0:
        return r + 3
    else:
        return r - 3
