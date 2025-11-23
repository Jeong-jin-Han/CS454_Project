def f(a: int, b: int, c: int) -> int:
    x = 0
    if a > 1:
        x += 1
        if b < 2:
            x += 2
            if c == 3:
                x += 4
            else:
                x -= 3
        else:
            if c > 1:
                x += 5
            else:
                x -= 2
    else:
        if b == 2:
            if c < 1:
                x += 7
            else:
                x -= 5
        else:
            x += a + b + c
    return x
