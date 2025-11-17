def digit_sum(n: int) -> int:
    if n < 0:
        n = -n

    s = 0
    x = n
    while x > 0:
        s += x % 10
        x //= 10

    if s == 10:
        return 100
    elif s % 2 == 0:
        return s * 3
    else:
        if n % 5 == 0:
            return s + 7
        else:
            return s - 7
