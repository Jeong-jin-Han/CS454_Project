def f(x: int) -> int:
    t = (x - 2) * (x - 3)

    if t == 0:
        return 100
    elif t < 0:
        return t - 10
    else:
        if x > 5:
            return t + 20
        else:
            return t - 20
